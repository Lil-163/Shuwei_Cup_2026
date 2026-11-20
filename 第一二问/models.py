"""Transparent models for Q1/Q2: quality score, CV Kalman filter, and replay policies."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def quality_features(position: np.ndarray, dt: np.ndarray, flag_bad: np.ndarray | None = None):
    """Frame-level features. Missingness/continuity/implausible jumps are observable cues."""
    p = np.asarray(position, float)
    dt = np.asarray(dt, float)
    bad = np.zeros(len(p)) if flag_bad is None else np.asarray(flag_bad, float)
    valid = np.isfinite(p).all(axis=1)
    speed = np.zeros(len(p))
    if len(p) > 1:
        delta = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(dt[1:], 1e-6)
        speed[1:] = delta
    completeness = np.isfinite(p).mean(axis=1)
    return np.column_stack([completeness, np.log1p(np.maximum(dt, 0)),
                            np.log1p(speed), bad, valid.astype(float)])

class ReliabilityModel:
    """Logistic baseline estimating whether an observation is qualified."""
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=0))
    def fit(self, x, y):
        self.model.fit(np.asarray(x), np.asarray(y).astype(int)); return self
    def predict_good_probability(self, x):
        return self.model.predict_proba(np.asarray(x))[:, 1]

def cv_matrices(dt: float, accel_var: float, measurement_var: float):
    I = np.eye(3); Z = np.zeros((3, 3))
    F = np.block([[I, dt*I], [Z, I]])
    G = np.vstack([0.5*dt*dt*I, dt*I])
    Q = accel_var * (G @ G.T)
    H = np.hstack([I, Z])
    R = max(measurement_var, 1e-8) * I
    return F, Q, H, R

@dataclass
class CVKalman:
    state: np.ndarray
    covariance: np.ndarray
    accel_var: float = 1.0
    measurement_var: float = 1e-3
    def predict(self, dt: float):
        F,Q,_,_ = cv_matrices(dt, self.accel_var, self.measurement_var)
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
        return self.state[:3].copy(), self.covariance[:3,:3].copy()
    def update(self, observation: np.ndarray, dt: float = 0.0, noise_scale: float = 1.0):
        _,_,H,R = cv_matrices(max(dt, 1e-6), self.accel_var, self.measurement_var*noise_scale)
        innovation = np.asarray(observation)-H@self.state
        S = H@self.covariance@H.T+R
        K = np.linalg.solve(S, H@self.covariance).T
        self.state = self.state + K@innovation
        I=np.eye(6)
        # Joseph form is numerically stable and preserves positive semidefiniteness.
        A=I-K@H
        self.covariance=A@self.covariance@A.T+K@R@K.T

def init_filter(p0, p1, dt, accel_var=1.0, measurement_var=1e-3):
    vel=(np.asarray(p1)-np.asarray(p0))/max(dt,1e-6)
    x=np.r_[p1,vel]
    P=np.diag([measurement_var]*3+[10*measurement_var]*3)
    return CVKalman(x,P,accel_var,measurement_var)

def hold_last(last, horizon):
    return np.repeat(np.asarray(last)[None,:], horizon, axis=0)

def constant_velocity(p0, p1, dt, steps, step_dt):
    v=(np.asarray(p1)-np.asarray(p0))/max(dt,1e-6)
    return np.asarray(p1)[None,:]+np.arange(1,steps+1)[:,None]*step_dt*v

def replay_with_requests(reference, timestamps, valid, policy="adaptive", tolerance=0.05,
                         delay_s=0.0, request_budget=3, fixed_interval=5, accel_var=0.5,
                         measurement_var=1e-4):
    """Causal replay: mask all interior observations; delayed queries return only an
    earlier valid sample already present in the record. Returns predictions and accounting.
    `reference` is used only by evaluator; no future sample is exposed to predictor.
    """
    p=np.asarray(reference,float); t=np.asarray(timestamps,float); ok=np.asarray(valid,bool)
    n=len(p); pred=np.full_like(p,np.nan); sigma=np.full(n,np.nan)
    if n<3: return pred, {"requests":0,"waiting_s":0.0}
    k=2; filt=init_filter(p[0],p[1],t[1]-t[0],accel_var,measurement_var)
    pending=[]; requests=0; waiting=0.; last_request=-10**9
    for i in range(2,n):
        dt=max(t[i]-t[i-1],1e-6)
        estimate,cov=filt.predict(dt)
        pred[i]=estimate; sigma[i]=np.sqrt(np.trace(cov)/3)
        # Responses arrive only now and can only use actual, prior qualified samples.
        ready=[q for q in pending if q[0] <= t[i]]
        pending=[q for q in pending if q[0] > t[i]]
        for _,sample_idx in ready:
            if sample_idx < i and ok[sample_idx]:
                filt.update(p[sample_idx],dt=max(t[i]-t[sample_idx],1e-6))
        if requests < request_budget and not pending:
            should=(policy=="adaptive" and sigma[i]>tolerance) or \
                   (policy=="fixed" and i-last_request>=fixed_interval)
            if should:
                # Query the latest available real observation from the held-out past only.
                candidates=np.flatnonzero(ok[:i])
                if len(candidates):
                    j=int(candidates[-1]); pending.append((t[i]+delay_s,j))
                    requests+=1; waiting+=delay_s; last_request=i
    return pred,{"requests":requests,"waiting_s":waiting,"uncertainty":sigma}
