"""Read only the contest's supplied archives and build per-hand real-record sequences."""
from __future__ import annotations
import io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def _stage_at(times, segments):
    """Attach each recorded timestamp to its source semantic interval."""
    labels=np.full(len(times),"unlabeled",dtype=object)
    for start,end,label in segments:
        if not np.isfinite(start) or not np.isfinite(end) or end<=start:
            continue
        mask=(times>=start)&(times<end)
        labels[mask]=str(label) if pd.notna(label) else "unlabeled"
    return labels


def _semantic_segments(zf, path):
    if path not in zf.namelist():
        return []
    table=pd.read_csv(io.BytesIO(zf.read(path)))
    start_col="segment.start_time" if "segment.start_time" in table else "start_time"
    end_col="segment.end_time" if "segment.end_time" in table else "end_time"
    label_col="segment.subtask_description" if "segment.subtask_description" in table else "description"
    if not {start_col,end_col,label_col}.issubset(table.columns):
        return []
    return list(zip(pd.to_numeric(table[start_col],errors="coerce"),
                    pd.to_numeric(table[end_col],errors="coerce"),table[label_col]))


def _metadata_id(fields, words):
    """尽量读取原始元数据中的场景/操作者标识；缺少时保留空值，不猜测。"""
    for key, value in fields.items():
        low=str(key).lower()
        if any(word in low for word in words) and pd.notna(value):
            text=str(value).strip()
            if text and text.lower() not in ("nan","none"):
                return text
    return ""

def hand_keypoints(points):
    """Return the 21 recorded 3-D keypoints in a consistent numeric layout."""
    arr=np.asarray(points,dtype=object)
    if arr.shape==(21,3):
        try:
            a=arr.astype(float)
        except (TypeError,ValueError):
            a=np.stack([np.asarray(x,dtype=float).ravel() for x in arr])
    else:
        a=np.stack([np.asarray(x,dtype=float).ravel() for x in arr])
    if a.shape!=(21,3):
        raise ValueError(f"Expected 21x3 recorded keypoints, got {a.shape}")
    return a


def robust_hand_point(points):
    """Median of the 21 recorded keypoints; no interpolation or fabricated values.

    The same 21x3 logical layout arrives in two physical shapes depending on the
    archive: EgoDemo cells are (21,) object arrays whose elements are 3-vectors,
    while EgoPro/EgoStandard rows are plain 21x3 numeric lists. Both are read
    as recorded, then reduced with the coordinate-wise median.
    """
    a=hand_keypoints(points)
    return np.nanmedian(a,axis=0)

def load_split(data_root):
    path=Path(data_root)/"Restocking_Tasks_and_Constraints"/"record_split.csv"
    split=pd.read_csv(path)
    needed={"episode_uuid","environment_id","split"}
    if not needed.issubset(split.columns): raise ValueError(f"Missing split fields: {needed-set(split.columns)}")
    if split.groupby("environment_id").split.nunique().max()!=1:
        raise ValueError("Official split leaks an environment across train/validation/test")
    if split.episode_uuid.duplicated().any():
        raise ValueError("Official split contains duplicate episode_uuid rows")
    return split

def _ego_demo(zf):
    for name in zf.namelist():
        if not name.endswith("/data/chunk-000/file-000.parquet"): continue
        prefix=name.rsplit("/data/",1)[0]
        data=pd.read_parquet(io.BytesIO(zf.read(name)))
        meta_name=prefix+"/meta/episodes/chunk-000/file-000.parquet"
        meta=pd.read_parquet(io.BytesIO(zf.read(meta_name)))
        rec=meta.iloc[0]
        episode=str(rec["episode_uuid"]); env=str(rec["environment_id"])
        meta_fields=rec.to_dict()
        bad=json.loads(zf.read(prefix+"/bad_frame_ratio.json").decode("utf-8"))
        details={int(x["frame"]):x for x in bad.get("hand_bad_frame_details",[])}
        annotation_path=prefix+"/annotation.json"
        annotation=json.loads(zf.read(annotation_path).decode("utf-8")) if annotation_path in zf.namelist() else {}
        segments=[(x.get("start_time"),x.get("end_time"),x.get("description","unlabeled"))
                  for x in annotation.get("subtask_annotations",[])]
        for side in ("left","right"):
            col=f"observation.state.hand_{side}_world"
            cloud=np.stack([hand_keypoints(v) for v in data[col].to_numpy()])
            pts=np.nanmedian(cloud,axis=1)
            frame=data["frame_index"].to_numpy(dtype=int)
            reasons=np.array([details.get(int(k),{}).get(f"{side}_bad_reason") or "" for k in frame],dtype=object)
            flags=reasons!=""
            times=data["timestamp"].to_numpy(float)
            yield {"dataset":"EgoDemo","episode_uuid":episode,"environment_id":env,
                   "side":side,"time":times,"position":pts,"keypoints":cloud,
                   "bad":flags,"bad_reason":reasons,"quality_known":np.ones(len(frame),dtype=bool),
                   "stage":_stage_at(times,segments),"source":name,
                   "scene_id":_metadata_id(meta_fields,("scene_id","scene")),
                   "operator_id":_metadata_id(meta_fields,("operator_id","operator"))}

def _mcap_tables(zf, dataset):
    prefix=dataset+"/"
    sessions=sorted({n[len(prefix):].split("/",1)[0] for n in zf.namelist()
                     if n.startswith(prefix) and "/tables/pose_left_hand.csv" in n})
    for session in sessions:
        pfx=f"{prefix}{session}/tables/"
        meta_raw=pd.read_csv(io.BytesIO(zf.read(f"{prefix}{session}/session_metadata.csv")),encoding="utf-8-sig")
        # session_metadata.csv is long-format: one row per (field, value) pair.
        fields={str(k):v for k,v in zip(meta_raw["field"],meta_raw["value"])}
        episode_raw=fields.get("task_info.episode_uuid")
        env_raw=fields.get("task_info.environment_id")
        if episode_raw is None or not isinstance(episode_raw,str) or episode_raw.strip() in ("","nan"):
            raise ValueError(f"No task_info.episode_uuid in session_metadata for {dataset}/{session}")
        episode=episode_raw.strip()
        env=str(env_raw).strip() if env_raw is not None and str(env_raw).strip() not in ("","nan") else ""
        scene_id=_metadata_id(fields,("scene_id","scene"))
        operator_id=_metadata_id(fields,("operator_id","operator"))
        flag_path=pfx+"annotation_bad_frame_pose_hand.csv"
        flags=pd.read_csv(io.BytesIO(zf.read(flag_path)))
        bad_col="is_bad" if "is_bad" in flags else None
        reason_cols=[c for c in flags.columns if c.startswith("problem_type[")]
        segments=_semantic_segments(zf,pfx+"annotation_semantic_segments.csv")
        for side in ("left","right"):
            path=pfx+f"pose_{side}_hand.csv"
            if path not in zf.namelist(): continue
            df=pd.read_csv(io.BytesIO(zf.read(path)))
            clouds=[]
            for _,row in df.iterrows():
                xyz=[]
                for j in range(21):
                    xyz.append([row[f"transforms[{j}].pos.{axis}"] for axis in "xyz"])
                clouds.append(hand_keypoints(xyz))
            cloud=np.asarray(clouds,float)
            pos=np.nanmedian(cloud,axis=1)
            # Same recorded message index is the key; no nearest-time fabricated labels.
            bad=np.zeros(len(df),dtype=bool)
            reasons=np.full(len(df),"",dtype=object)
            quality_known=np.zeros(len(df),dtype=bool)
            if bad_col and "_message_index" in flags and "_message_index" in df:
                flag_index=flags.set_index("_message_index")
                matched=df["_message_index"].isin(flag_index.index).to_numpy()
                quality_known=matched
                f=flag_index[bad_col]
                bad=df["_message_index"].map(f).fillna(False).astype(bool).to_numpy()
                if reason_cols:
                    reason_map={}
                    for _,flag_row in flag_index.iterrows():
                        parts=[str(flag_row[c]) for c in reason_cols if pd.notna(flag_row[c]) and str(flag_row[c]).strip()]
                        if parts:
                            reason_map[flag_row.name]="|".join(parts)
                    reasons=np.array([reason_map.get(k,"") for k in df["_message_index"]],dtype=object)
            times=df["_offset_from_first_log_s"].to_numpy(float)
            yield {"dataset":dataset,"episode_uuid":episode,"environment_id":env,
                   "side":side,"time":times,
                   "position":pos,"keypoints":cloud,"bad":bad,"bad_reason":reasons,"quality_known":quality_known,
                   "stage":_stage_at(times,segments),"source":path,
                   "scene_id":scene_id,"operator_id":operator_id}

def load_sequences(data_root):
    raw=Path(data_root)/"Raw_Data"
    for dataset in ("EgoDemo","EgoPro","EgoStandard"):
        archive=raw/f"{dataset}.zip"
        if not archive.exists(): raise FileNotFoundError(archive)
        with zipfile.ZipFile(archive) as zf:
            seqs=list(_ego_demo(zf) if dataset=="EgoDemo" else _mcap_tables(zf,dataset))
        yield from seqs

def attach_split(sequences, split):
    ids=split.set_index("episode_uuid")
    for s in sequences:
        if s["episode_uuid"] not in ids.index:
            raise ValueError(f"No official train/validation/test assignment for {s['episode_uuid']} ({s['dataset']})")
        meta=ids.loc[s["episode_uuid"]]
        if s["environment_id"] and s["environment_id"]!=meta.environment_id:
            raise ValueError(f"Environment mismatch for {s['episode_uuid']}")
        s["environment_id"]=meta.environment_id
        s["split"]=meta.split
        yield s
