"""C题第一、二问：原始记录审计、状态建模、观测质量和遮挡回放。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models import CVKalman, cv_matrices
from real_data_pipeline import attach_split, load_sequences, load_split


# 第二问的实验条件在所有比较方法间保持一致，并在查看测试结果前固定。
MASK_DURATIONS = (0.25, 0.5, 1.0, 1.5, 2.0)
RETURN_DELAY_S = 1.0
REQUEST_BUDGET = 3
REQUEST_COST = 1.0
FIXED_REQUEST_INTERVAL_S = 0.5
NOMINAL_COVERAGE = 0.90


def frame_features(seq: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造逐帧质量特征、坏帧标签、标签可用性和帧间时间差。

    运动特征只由当前及过去坐标计算。关键点形状统计只使用当前帧的21个
    原始关键点，不读取未来帧；中位点坐标有效性仍作为硬性质量门槛。
    """
    position = np.asarray(seq["position"], dtype=float)
    timestamp = np.asarray(seq["time"], dtype=float)
    bad = np.asarray(seq["bad"], dtype=bool)
    known = np.asarray(seq["quality_known"], dtype=bool)

    dt = np.r_[np.nan, np.diff(timestamp)]
    positive_dt = dt[1:][np.isfinite(dt[1:]) & (dt[1:] > 0)]
    typical_dt = float(np.median(positive_dt)) if len(positive_dt) else 1.0 / 30.0
    dt[0] = typical_dt
    dt = np.where((dt > 0) & np.isfinite(dt), dt, typical_dt)

    valid = np.isfinite(position).all(axis=1)
    step = np.zeros(len(timestamp), dtype=float)
    if len(timestamp) > 1:
        pair_valid = valid[1:] & valid[:-1]
        distance = np.linalg.norm(np.diff(position, axis=0), axis=1)
        step[1:] = np.where(pair_valid, distance / np.maximum(dt[1:], 1e-8), 0.0)
    acceleration = np.zeros(len(timestamp), dtype=float)
    if len(timestamp) > 1:
        acceleration[1:] = np.abs(np.diff(step)) / np.maximum(dt[1:], 1e-8)

    cloud = seq.get("keypoints")
    if cloud is None:
        finite_fraction = np.ones(len(position), dtype=float)
        spread = np.zeros(len(position), dtype=float)
        spread90 = np.zeros(len(position), dtype=float)
    else:
        cloud = np.asarray(cloud, dtype=float)
        joint_valid = np.isfinite(cloud).all(axis=2)
        finite_fraction = joint_valid.mean(axis=1)
        center = np.nanmedian(cloud, axis=1)
        radius = np.linalg.norm(cloud - center[:, None, :], axis=2)
        spread = np.nanmedian(radius, axis=1)
        spread90 = np.nanquantile(radius, 0.90, axis=1)
    spread_change = np.r_[0.0, np.abs(np.diff(spread))] if len(spread) else spread
    x = np.column_stack((
        np.log1p(dt), np.log1p(step), np.log1p(acceleration),
        1.0 - finite_fraction, np.log1p(spread), np.log1p(spread90),
        np.log1p(spread_change),
    ))
    x[~np.isfinite(x)] = 0.0
    label_bad = bad | ~valid
    return x, label_bad.astype(int), known, dt


def build_quality_table(sequences: list[dict]) -> pd.DataFrame:
    """把每只手的逐帧质量信息整理成长表，保留环境和语义阶段。"""
    rows = []
    for seq in sequences:
        x, label, known, dt = frame_features(seq)
        for i in range(len(label)):
            rows.append({
                "dataset": seq["dataset"], "episode_uuid": seq["episode_uuid"],
                "environment_id": seq["environment_id"], "split": seq["split"],
                "side": seq["side"], "stage": seq["stage"][i], "frame": i,
                "quality_known": bool(known[i]), "finite": bool(np.isfinite(seq["position"][i]).all()),
                "bad": int(label[i]), "dt": float(dt[i]),
                "log_dt": float(x[i, 0]), "log_speed": float(x[i, 1]),
                "log_acceleration": float(x[i, 2]),
                "missing_joint_fraction": float(x[i, 3]),
                "log_joint_spread": float(x[i, 4]),
                "log_joint_spread90": float(x[i, 5]),
                "log_joint_spread_change": float(x[i, 6]),
            })
    return pd.DataFrame(rows)


def tune_bad_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """仅用验证集选坏帧阈值；测试集不参与阈值选择。"""
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best])


def classification_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    """按帧报告坏帧指标，同时给出多数类基线，避免类别不平衡造成误读。"""
    d = frame[frame["quality_known"]].copy()
    if d.empty:
        return {"n_frames": 0}
    y = d["bad"].to_numpy(dtype=int)
    score = d["prob_bad"].to_numpy(dtype=float)
    pred = (score >= threshold).astype(int)
    row = {
        "n_frames": int(len(d)), "bad_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "bad_precision": float(precision_score(y, pred, zero_division=0)),
        "bad_recall": float(recall_score(y, pred, zero_division=0)),
        "bad_f1": float(f1_score(y, pred, zero_division=0)),
        "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
        "brier": float(brier_score_loss(y, score)),
    }
    if len(np.unique(y)) == 2:
        row["auc"] = float(roc_auc_score(y, score))
    return row


def fit_quality_model(sequences: list[dict], output: Path) -> tuple[pd.DataFrame, dict, float]:
    """第一问观测可靠性模型：训练集拟合，验证集定阈值，测试集只评估。

    输出逐帧审计表、分组指标和阈值。训练样本仅取官方训练环境中
    标签已知且坐标完整的帧；坐标不完整时由硬性规则直接标记为低质量。
    """
    table = build_quality_table(sequences)
    feature_cols = [
        "log_dt", "log_speed", "log_acceleration", "missing_joint_fraction",
        "log_joint_spread", "log_joint_spread90", "log_joint_spread_change",
    ]
    train = table[(table.split == "train") & table.quality_known & table.finite]
    if train.bad.nunique() < 2:
        raise ValueError("训练集质量标记不足两个类别，无法训练逻辑回归")

    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=7))
    model.fit(train[feature_cols], train.bad)
    known_finite = table.quality_known & table.finite
    table["prob_bad"] = 1.0
    table.loc[known_finite, "prob_bad"] = model.predict_proba(
        table.loc[known_finite, feature_cols])[:, 1]

    validation = table[(table.split == "validation") & table.quality_known & table.finite]
    threshold = tune_bad_threshold(validation.bad.to_numpy(), validation.prob_bad.to_numpy())
    table["pred_bad"] = (~table.finite) | (table.prob_bad >= threshold)

    rows = []
    for split_name in ("validation", "test"):
        part = table[table.split == split_name]
        rows.append({"split": split_name, "threshold_from_validation": threshold,
                     **classification_metrics(part, threshold)})
    pd.DataFrame(rows).to_csv(output / "q1_quality_metrics.csv", index=False)
    table.to_csv(output / "q1_observation_audit.csv", index=False)

    # 环境、数据源和语义阶段是解释性分组，不把高基数文本标签硬塞进分类器。
    groups = []
    for keys, part in table[table.quality_known].groupby(
            ["split", "dataset", "environment_id", "stage"], dropna=False):
        split_name, dataset, environment, stage = keys
        groups.append({"split": split_name, "dataset": dataset,
                       "environment_id": environment, "stage": stage,
                       "n_frames": len(part), "bad_rate": float(part.bad.mean()),
                       "complete_rate": float(part.finite.mean())})
    pd.DataFrame(groups).to_csv(output / "q1_factor_summary.csv", index=False)
    return table, rows[-1], threshold


def good_runs(seq: dict) -> list[np.ndarray]:
    """返回连续、坐标有效、未被质量标记且时间严格递增的区间。"""
    position = np.asarray(seq["position"], dtype=float)
    timestamp = np.asarray(seq["time"], dtype=float)
    good = np.isfinite(position).all(axis=1) & ~np.asarray(seq["bad"], dtype=bool)
    good &= np.isfinite(timestamp)
    indices = np.flatnonzero(good)
    if not len(indices):
        return []
    breaks = []
    for j in range(1, len(indices)):
        left, right = indices[j - 1], indices[j]
        if right != left + 1 or timestamp[right] <= timestamp[left]:
            breaks.append(j)
    return [run for run in np.split(indices, breaks) if len(run) >= 3]


def estimate_motion_parameters(sequences: list[dict]) -> tuple[float, float, dict]:
    """只用训练环境估计常速度卡尔曼模型的过程/观测残差尺度。

    观测残差尺度并不等同于传感器真实噪声；这里的参考轨迹也不是物理真值。
    按训练环境进行重抽样，给出参数的经验区间。
    """
    by_environment: dict[str, list[float]] = {}
    cadence = []
    second_diff = []
    for seq in sequences:
        if seq["split"] != "train":
            continue
        p = np.asarray(seq["position"], dtype=float)
        t = np.asarray(seq["time"], dtype=float)
        for run in good_runs(seq):
            dt = np.diff(t[run])
            cadence.extend(dt[dt > 0].tolist())
            velocity = np.diff(p[run], axis=0) / np.maximum(dt[:, None], 1e-9)
            if len(velocity) > 1:
                midpoint_dt = 0.5 * (dt[1:] + dt[:-1])
                acc = np.diff(velocity, axis=0) / np.maximum(midpoint_dt[:, None], 1e-9)
                by_environment.setdefault(seq["environment_id"], []).extend(acc.ravel().tolist())
                second_diff.extend(np.diff(p[run], n=2, axis=0).ravel().tolist())

    positive_dt = np.asarray(cadence, dtype=float)
    typical_dt = float(np.median(positive_dt[positive_dt > 0])) if np.any(positive_dt > 0) else 1/30
    acceleration_values = np.asarray([x for values in by_environment.values() for x in values], dtype=float)
    accel_var = max(float(np.var(acceleration_values)), 1e-8) if len(acceleration_values) else 1e-4
    # 二阶差分仅提供建模尺度，不能解释为独立的测量噪声真值。
    measurement_var = max(float(np.var(second_diff) / 6), 1e-8) if second_diff else 1e-6

    rng = np.random.default_rng(2026)
    envs = list(by_environment)
    estimates = []
    if envs:
        for _ in range(300):
            sampled = rng.choice(envs, size=len(envs), replace=True)
            values = np.concatenate([np.asarray(by_environment[e]) for e in sampled])
            estimates.append(np.var(values))
    ci = ([float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]
          if estimates else [None, None])
    return accel_var, measurement_var, {
        "median_frame_interval_s": typical_dt,
        "acceleration_residual_variance": accel_var,
        "acceleration_variance_cluster_bootstrap_95ci": ci,
        "reference_second_difference_variance_over_6": measurement_var,
        "train_environment_count": len(envs),
    }


def make_filter(p0, p1, dt, accel_var, measurement_var) -> CVKalman:
    """用两个可见历史点初始化六维位置—速度状态。"""
    velocity = (np.asarray(p1) - np.asarray(p0)) / max(float(dt), 1e-8)
    state = np.r_[p1, velocity]
    covariance = np.diag([measurement_var] * 3 + [measurement_var / max(dt * dt, 1e-8)] * 3)
    return CVKalman(state, covariance, accel_var, measurement_var)


def one_step_state_replay(sequences: list[dict], accel_var: float,
                          measurement_var: float, output: Path) -> pd.DataFrame:
    """第一问状态变化模型：逐段预测下一帧，比较末值、线性外推和卡尔曼。

    每条记录单独计算误差，再按数据集和环境汇总，避免把逐帧样本当作
    相互独立的实验样本。验证/测试参考仅用于评价，预测更新后才能读取。
    """
    rows = []
    for seq in sequences:
        if seq["split"] not in ("validation", "test"):
            continue
        p, t = np.asarray(seq["position"], float), np.asarray(seq["time"], float)
        for run_id, run in enumerate(good_runs(seq)):
            filt = make_filter(p[run[0]], p[run[1]], t[run[1]] - t[run[0]],
                               accel_var, measurement_var)
            for k in range(2, len(run)):
                ix0, ix1, ix = run[k-2], run[k-1], run[k]
                dt = float(t[ix] - t[ix1])
                if dt <= 0:
                    continue
                pred_kalman, covariance = filt.predict(dt)
                velocity = (p[ix1] - p[ix0]) / max(t[ix1] - t[ix0], 1e-8)
                estimates = {
                    "last_value": p[ix1],
                    "constant_velocity": p[ix1] + velocity * dt,
                    "cv_kalman": pred_kalman,
                }
                sigma = float(np.sqrt(np.trace(covariance) / 3))
                for method, estimate in estimates.items():
                    rows.append({"dataset": seq["dataset"], "episode_uuid": seq["episode_uuid"],
                                 "environment_id": seq["environment_id"], "split": seq["split"],
                                 "side": seq["side"], "stage": seq["stage"][ix],
                                 "run": run_id, "timestamp": t[ix], "method": method,
                                 "error": float(np.linalg.norm(p[ix] - estimate)),
                                 "sigma": sigma if method == "cv_kalman" else np.nan})
                # 真正状态只在当前时间点已到达后用于滤波器更新。
                filt.update(p[ix])
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "q1_state_replay.csv", index=False)
    summary = detail.groupby(["split", "dataset", "environment_id", "method"], as_index=False).agg(
        n_points=("error", "size"), mean_error=("error", "mean"),
        p95_error=("error", lambda x: float(np.quantile(x, 0.95))),
        mean_sigma=("sigma", "mean"))
    summary.to_csv(output / "q1_state_metrics.csv", index=False)
    stage = detail.groupby(["split", "stage", "method"], as_index=False).agg(
        n_points=("error", "size"), mean_error=("error", "mean"),
        p95_error=("error", lambda x: float(np.quantile(x, 0.95))))
    stage.to_csv(output / "q1_state_by_stage.csv", index=False)
    return detail


def validation_calibration(state_detail: pd.DataFrame) -> tuple[float, float]:
    """用验证集卡尔曼误差/标准差比值校准名义90%区间和请求容差。"""
    val = state_detail[(state_detail.split == "validation") & (state_detail.method == "cv_kalman")]
    ratio = val.error.to_numpy() / np.maximum(val.sigma.to_numpy(), 1e-8)
    factor = float(np.quantile(ratio, NOMINAL_COVERAGE))
    tolerance = float(np.quantile(val.error.to_numpy(), NOMINAL_COVERAGE))
    return max(factor, 1e-8), max(tolerance, 1e-8)


def predict_hidden_window(position: np.ndarray, timestamp: np.ndarray, start: int,
                          end: int, method: str, accel_var: float,
                          measurement_var: float, interval_factor: float,
                          tolerance: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """因果地重放一个遮挡窗。

    窗口内部坐标只由评估器保存。预测器仅能在请求返回时读取指定的真实点；
    返回时间到达前不更新状态。每种方法使用相同预算、延迟和单次成本。
    """
    p, t = np.asarray(position, float), np.asarray(timestamp, float)
    local_p, local_t = p[start-2:end], t[start-2:end]
    filt = make_filter(local_p[0], local_p[1], local_t[1] - local_t[0],
                       accel_var, measurement_var)
    estimates, radii = [], []
    requests, wait_total, cost = 0, 0.0, 0.0
    pending = None
    last_request_time = float(local_t[1])
    window_end_time = float(local_t[-1])

    for i in range(2, len(local_t)):
        dt = max(float(local_t[i] - local_t[i-1]), 1e-8)
        estimate, covariance = filt.predict(dt)

        # 只有返回时间已到，才能使用该请求对应的真实记录点。
        if pending is not None and i == pending["index"]:
            filt.update(local_p[i])
            estimate = filt.state[:3].copy()
            covariance = filt.covariance[:3, :3].copy()
            pending = None

        sigma = float(np.sqrt(np.trace(covariance) / 3))
        radius = interval_factor * sigma
        if method in ("adaptive_request", "fixed_interval") and requests < REQUEST_BUDGET and pending is None:
            if method == "adaptive_request":
                should_request = radius > tolerance
            else:
                should_request = local_t[i] - last_request_time >= FIXED_REQUEST_INTERVAL_S
            # 若在窗口结束前无法返回，这次请求没有可评价的收益，不发出请求。
            if should_request and local_t[i] + RETURN_DELAY_S <= window_end_time:
                return_time = local_t[i] + RETURN_DELAY_S
                response_index = int(np.searchsorted(local_t, return_time, side="left"))
                if response_index > i and response_index < len(local_t):
                    pending = {"index": response_index, "requested_at": float(local_t[i])}
                    requests += 1
                    cost += REQUEST_COST
                    wait_total += float(local_t[response_index] - local_t[i])
                    last_request_time = float(local_t[i])

        estimates.append(estimate)
        radii.append(radius)

    return np.asarray(estimates), np.asarray(radii), {
        "requests": requests, "observation_cost": cost, "waiting_s": wait_total,
    }


def run_mask_replay(sequences: list[dict], accel_var: float, measurement_var: float,
                    interval_factor: float, tolerance: float, output: Path) -> tuple[pd.DataFrame, float]:
    """第二问：固定遮挡窗回放，比较末值、常速度、固定复核和自适应复核。"""
    rows = []
    started = time.perf_counter()
    for seq in sequences:
        if seq["split"] != "test":
            continue
        p, t = np.asarray(seq["position"], float), np.asarray(seq["time"], float)
        for run_id, run in enumerate(good_runs(seq)):
            rp, rt = p[run], t[run]
            if len(run) < 10:
                continue
            for duration in MASK_DURATIONS:
                start = 2
                window_no = 0
                while start < len(run) - 1:
                    end = int(np.searchsorted(rt, rt[start] + duration, side="left"))
                    end = min(max(end, start + 2), len(run))
                    if end - start < 2:
                        break
                    truth = rp[start:end]
                    times = rt[start:end]
                    last = rp[start-1]
                    dt_prev = max(float(rt[start-1] - rt[start-2]), 1e-8)
                    velocity = (last - rp[start-2]) / dt_prev
                    baseline_predictions = {
                        "last_value": (np.repeat(last[None, :], len(truth), axis=0), None, {}),
                        "constant_velocity": (np.asarray([last + velocity * (x - rt[start-1]) for x in times]), None, {}),
                    }
                    # 两种需要复核的策略共用状态方程和返回机制，只改变请求时机。
                    for method in ("adaptive_request", "fixed_interval"):
                        estimate, radius, usage = predict_hidden_window(
                            rp, rt, start, end, method, accel_var, measurement_var,
                            interval_factor, tolerance)
                        baseline_predictions[method] = (estimate, radius, usage)

                    for method, (estimate, interval_radii, method_usage) in baseline_predictions.items():
                        err = np.linalg.norm(truth - estimate, axis=1)
                        is_model_interval = method in ("adaptive_request", "fixed_interval")
                        rows.append({
                            "episode_uuid": seq["episode_uuid"], "environment_id": seq["environment_id"],
                            "dataset": seq["dataset"], "side": seq["side"],
                            "stage": str(seq["stage"][run[start]]), "run": run_id,
                            "window": window_no, "duration_s": duration, "method": method,
                            "n_points": len(err), "mae": float(np.mean(err)),
                            "p95_error": float(np.quantile(err, 0.95)),
                            "coverage_90": float(np.mean(err <= interval_radii)) if is_model_interval else np.nan,
                            "mean_interval_radius": float(np.mean(interval_radii)) if is_model_interval else np.nan,
                            "requests": method_usage["requests"] if method in ("adaptive_request", "fixed_interval") else 0,
                            "observation_cost": method_usage["observation_cost"] if method in ("adaptive_request", "fixed_interval") else 0.0,
                            "waiting_s": method_usage["waiting_s"] if method in ("adaptive_request", "fixed_interval") else 0.0,
                        })
                    window_no += 1
                    start = end

    detail = pd.DataFrame(rows)
    detail.to_csv(output / "q2_test_replay.csv", index=False)
    summary = detail.groupby(["duration_s", "method"], as_index=False).agg(
        records=("episode_uuid", "nunique"), windows=("window", "size"),
        points=("n_points", "sum"), mae=("mae", "mean"),
        p95_error=("p95_error", "mean"), coverage_90=("coverage_90", "mean"),
        mean_interval_radius=("mean_interval_radius", "mean"),
        requests=("requests", "mean"), observation_cost=("observation_cost", "mean"),
        waiting_s=("waiting_s", "mean"))
    summary.to_csv(output / "q2_summary.csv", index=False)
    stage_summary = detail.groupby(["duration_s", "stage", "method"], as_index=False).agg(
        windows=("window", "size"), mae=("mae", "mean"),
        p95_error=("p95_error", "mean"), requests=("requests", "mean"),
        waiting_s=("waiting_s", "mean"))
    stage_summary.to_csv(output / "q2_by_stage.csv", index=False)
    return detail, time.perf_counter() - started


def audit_quality_runs(sequences: list[dict], output: Path) -> pd.DataFrame:
    """统计真实低质量标记连续区间及恢复时间；不把其参考坐标误差当真值。"""
    events = []
    for seq in sequences:
        t = np.asarray(seq["time"], float)
        bad = np.asarray(seq["bad"], bool) & np.asarray(seq["quality_known"], bool)
        ids = np.flatnonzero(bad)
        if not len(ids):
            continue
        breaks = np.flatnonzero(np.diff(ids) > 1) + 1
        for run_no, event in enumerate(np.split(ids, breaks)):
            start, end = int(event[0]), int(event[-1])
            next_good = next((i for i in range(end + 1, len(t))
                              if seq["quality_known"][i] and not seq["bad"][i]
                              and np.isfinite(seq["position"][i]).all()), None)
            reasons = [str(x) for x in np.asarray(seq["bad_reason"], dtype=object)[event]
                       if str(x).strip()]
            reason_text = "|".join(sorted(set(reasons))) if reasons else "unspecified"
            lower_reason = reason_text.lower()
            if "occlusion" in lower_reason:
                reason_group = "occlusion"
            elif "near edge" in lower_reason or "near_edge" in lower_reason:
                reason_group = "near_edge"
            else:
                reason_group = "other_or_unspecified"
            positive_dt = np.diff(t)
            positive_dt = positive_dt[positive_dt > 0]
            frame_interval = float(np.median(positive_dt)) if len(positive_dt) else 0.0
            events.append({
                "dataset": seq["dataset"], "episode_uuid": seq["episode_uuid"],
                "environment_id": seq["environment_id"], "split": seq["split"],
                "side": seq["side"], "event": run_no,
                "reason": reason_text, "reason_group": reason_group,
                "stage": str(seq["stage"][start]), "n_frames": len(event),
                "start_s": float(t[start]), "end_s": float(t[end]),
                "flagged_span_s": float(max(t[end] - t[start] + frame_interval, frame_interval)),
                "recovery_s": float(t[next_good] - t[end]) if next_good is not None else np.nan,
            })
    detail = pd.DataFrame(events)
    detail.to_csv(output / "low_quality_events.csv", index=False)
    if not detail.empty:
        detail.groupby(["split", "dataset", "reason_group"], as_index=False).agg(
            events=("event", "size"), mean_flagged_span_s=("flagged_span_s", "mean"),
            median_recovery_s=("recovery_s", "median")).to_csv(
                output / "low_quality_summary.csv", index=False)
    return detail


def record_audit(sequences: list[dict], output: Path) -> pd.DataFrame:
    """逐手记录时间、坐标、质量标签和语义阶段覆盖，供样本保留说明使用。"""
    rows = []
    for seq in sequences:
        t = np.asarray(seq["time"], float)
        p = np.asarray(seq["position"], float)
        dt = np.diff(t)
        rows.append({
            "dataset": seq["dataset"], "episode_uuid": seq["episode_uuid"],
            "environment_id": seq["environment_id"], "split": seq["split"],
            "scene_id": seq.get("scene_id", ""), "operator_id": seq.get("operator_id", ""),
            "side": seq["side"], "frames": len(t),
            "duplicate_timestamps": int(np.sum(dt == 0)),
            "nonincreasing_timestamps": int(np.sum(dt <= 0)),
            "missing_timestamps": int(np.sum(~np.isfinite(t))),
            "valid_coordinate_fraction": float(np.isfinite(p).all(axis=1).mean()),
            "quality_annotation_coverage": float(np.mean(seq["quality_known"])),
            "bad_flag_rate": float(np.mean(seq["bad"][seq["quality_known"]])) if np.any(seq["quality_known"]) else np.nan,
            "semantic_stage_coverage": float(np.mean(np.asarray(seq["stage"]) != "unlabeled")),
            "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "record_audit.csv", index=False)
    return audit


def audit_metadata_overlap(sequences: list[dict], output: Path) -> pd.DataFrame:
    """检查场景和操作者标识是否跨越官方划分；缺失元数据时明确报告。"""
    records = pd.DataFrame([{
        "episode_uuid": seq["episode_uuid"], "split": seq["split"],
        "scene_id": seq.get("scene_id", ""), "operator_id": seq.get("operator_id", ""),
    } for seq in sequences]).drop_duplicates("episode_uuid")
    checks = []
    for field in ("scene_id", "operator_id"):
        available = records[records[field].astype(str).str.strip() != ""]
        if available.empty:
            checks.append({"field": field, "available_records": 0,
                           "overlap_identifiers": 0, "status": "not_available_in_read_metadata"})
            continue
        counts = available.groupby(field).split.nunique()
        overlap = int((counts > 1).sum())
        checks.append({"field": field, "available_records": int(len(available)),
                       "overlap_identifiers": overlap,
                       "status": "overlap_found" if overlap else "no_overlap_found"})
    result = pd.DataFrame(checks)
    result.to_csv(output / "split_overlap_audit.csv", index=False)
    return result


def save_figures(output: Path) -> list[str]:
    """把已生成的汇总表画成论文可用的英文图，所有图保存为 PNG。"""
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 图1：逐环境汇总的一步状态误差，展示模型和简单基线的差异。
    state = pd.read_csv(output / "q1_state_metrics.csv")
    state_mean = state.groupby(["split", "method"], as_index=False).mean_error.mean()
    methods = ["last_value", "constant_velocity", "cv_kalman"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(2)
    width = 0.24
    for offset, method in enumerate(methods):
        vals = []
        for split_name in ("validation", "test"):
            row = state_mean[(state_mean.split == split_name) & (state_mean.method == method)]
            vals.append(float(row.mean_error.iloc[0]) if len(row) else np.nan)
        ax.bar(x + (offset - 1) * width, vals, width, label=method.replace("_", " "))
    ax.set_xticks(x, ["Validation", "Test"])
    ax.set_ylabel("Mean one-step position error (record coordinate units)")
    ax.set_title("Problem 1: One-step state prediction")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "q1_state_error.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(str(path.name))

    # 图2：报告坏帧分类的区分能力，并与多数类准确率基线并列。
    quality = pd.read_csv(output / "q1_quality_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    x = np.arange(len(quality))
    width = 0.24
    axes[0].bar(x - width, quality.bad_precision, width, label="Bad-frame precision")
    axes[0].bar(x, quality.bad_recall, width, label="Bad-frame recall")
    axes[0].bar(x + width, quality.balanced_accuracy, width, label="Balanced accuracy")
    axes[0].set_xticks(x, quality.split.str.title())
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Quality model performance")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x - width/2, quality.accuracy, width, label="Logistic model")
    axes[1].bar(x + width/2, quality.majority_accuracy, width, label="Majority baseline")
    axes[1].set_xticks(x, quality.split.str.title())
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Raw accuracy under class imbalance")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Problem 1: Observation reliability")
    fig.tight_layout()
    path = figure_dir / "q1_quality_model.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(str(path.name))

    # 图3：按验证/测试误差展示语义阶段影响，只画卡尔曼模型样本较多的阶段。
    stage = pd.read_csv(output / "q1_state_by_stage.csv")
    stage = stage[(stage.method == "cv_kalman") & (stage.stage != "unlabeled")]
    top_stage = (stage.groupby("stage").n_points.sum().nlargest(8).index
                 if not stage.empty else [])
    stage = stage[stage.stage.isin(top_stage)]
    if not stage.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for split_name, part in stage.groupby("split"):
            part = part.groupby("stage", as_index=False).mean_error.mean()
            ax.plot(part.stage, part.mean_error, marker="o", label=split_name.title())
        ax.set_ylabel("Mean one-step error (record coordinate units)")
        ax.set_title("Problem 1: State prediction by semantic segment")
        ax.tick_params(axis="x", labelrotation=30)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = figure_dir / "q1_error_by_stage.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        saved.append(str(path.name))

    # 图4：不同遮挡时长下各方法误差，图表数据与 q2_summary.csv 一致。
    q2 = pd.read_csv(output / "q2_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method, part in q2.groupby("method"):
        part = part.sort_values("duration_s")
        ax.plot(part.duration_s, part.mae, marker="o", label=method.replace("_", " "))
    ax.set_xlabel("Masked duration (s)")
    ax.set_ylabel("Mean distance error (record coordinate units)")
    ax.set_title("Problem 2: Prediction error by masked duration")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "q2_error_by_duration.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(str(path.name))

    # 图5：复核资源与区间可靠性并列呈现，避免只追求覆盖率而忽略区间宽度。
    model_rows = q2[q2.method.isin(["adaptive_request", "fixed_interval"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for method, part in model_rows.groupby("method"):
        part = part.sort_values("duration_s")
        axes[0].plot(part.duration_s, part.requests, marker="o", label=method.replace("_", " "))
        axes[1].plot(part.duration_s, part.waiting_s, marker="o", label=method.replace("_", " "))
        axes[2].plot(part.duration_s, part.coverage_90, marker="o", label=method.replace("_", " "))
    axes[0].set_ylabel("Requests per window")
    axes[0].set_title("Observation use")
    axes[1].set_ylabel("Waiting time (s)")
    axes[1].set_title("Waiting cost")
    axes[2].axhline(NOMINAL_COVERAGE, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Empirical coverage")
    axes[2].set_title("Nominal 90% intervals")
    for ax in axes:
        ax.set_xlabel("Masked duration (s)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Problem 2: Observation and uncertainty trade-offs")
    fig.tight_layout()
    path = figure_dir / "q2_observation_tradeoff.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(str(path.name))

    # 图4：真实质量标记的持续时长；不对标记区间内的位置计算真值误差。
    events_path = output / "low_quality_events.csv"
    if events_path.exists():
        events = pd.read_csv(events_path)
        if not events.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for reason, part in events.groupby("reason_group"):
                ax.hist(part.flagged_span_s, bins=20, alpha=0.5, label=reason)
            ax.set_xlabel("Flagged duration (s)")
            ax.set_ylabel("Number of intervals")
            ax.set_title("Natural low-quality intervals in supplied records")
            ax.legend(frameon=False)
            fig.tight_layout()
            path = figure_dir / "low_quality_duration.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            saved.append(str(path.name))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="只读取C题提供的第一、二问原始记录")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Restocking_Problem_and_Data_EN 数据目录")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    split = load_split(args.data_root)
    sequences = list(attach_split(load_sequences(args.data_root), split))
    audit = record_audit(sequences, args.output)
    overlap_audit = audit_metadata_overlap(sequences, args.output)
    quality_table, q1_quality_test, quality_threshold = fit_quality_model(sequences, args.output)

    accel_var, measurement_var, parameter_report = estimate_motion_parameters(sequences)
    state_detail = one_step_state_replay(sequences, accel_var, measurement_var, args.output)
    interval_factor, error_tolerance = validation_calibration(state_detail)
    q2_detail, replay_seconds = run_mask_replay(
        sequences, accel_var, measurement_var, interval_factor, error_tolerance, args.output)
    low_quality_events = audit_quality_runs(sequences, args.output)
    figure_names = save_figures(args.output)

    env_counts = split.groupby("split").environment_id.nunique().to_dict()
    record_counts = split.groupby("split").episode_uuid.nunique().to_dict()
    report = {
        "unique_records": int(split.episode_uuid.nunique()),
        "hand_sequences": int(len(sequences)),
        "hand_frames": int(audit.frames.sum()),
        "record_counts_by_split": record_counts,
        "environment_counts_by_split": env_counts,
        "scene_operator_overlap_audit": overlap_audit.to_dict(orient="records"),
        "quality_threshold_from_validation": quality_threshold,
        "q1_quality_test": q1_quality_test,
        "motion_parameters_from_train": parameter_report,
        "q2_interval_factor_from_validation": interval_factor,
        "q2_error_tolerance_from_validation": error_tolerance,
        "q2_nominal_coverage": NOMINAL_COVERAGE,
        "q2_return_delay_s": RETURN_DELAY_S,
        "q2_request_budget": REQUEST_BUDGET,
        "q2_cost_per_request": REQUEST_COST,
        "q2_mask_durations_s": list(MASK_DURATIONS),
        "q2_windows": int(q2_detail.drop_duplicates(
            ["episode_uuid", "side", "run", "duration_s", "window"]).shape[0]),
        "q2_replay_seconds": replay_seconds,
        "low_quality_events": int(len(low_quality_events)),
        "figures": figure_names,
        "notes": [
            "Coordinates are the coordinate-wise median of the 21 recorded hand keypoints.",
            "A hand state is not interpreted as a product position.",
            "Held-out records are qualified references, not independent physical ground truth.",
            "Artificial masking does not represent actual robot performance.",
        ],
    }
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
