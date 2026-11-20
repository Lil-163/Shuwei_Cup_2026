"""Read only the contest's supplied archives and build per-hand real-record sequences."""
from __future__ import annotations
import io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def robust_hand_point(points):
    """Median of the 21 recorded keypoints; no interpolation or fabricated values."""
    a=np.asarray(points,dtype=float)
    if a.shape!=(21,3):
        raise ValueError(f"Expected 21x3 recorded keypoints, got {a.shape}")
    return np.nanmedian(a,axis=0)

def load_split(data_root):
    path=Path(data_root)/"Restocking_Tasks_and_Constraints"/"record_split.csv"
    split=pd.read_csv(path)
    needed={"episode_uuid","environment_id","split"}
    if not needed.issubset(split.columns): raise ValueError(f"Missing split fields: {needed-set(split.columns)}")
    if split.groupby("environment_id").split.nunique().max()!=1:
        raise ValueError("Official split leaks an environment across train/validation/test")
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
        bad=json.loads(zf.read(prefix+"/bad_frame_ratio.json").decode("utf-8"))
        details={int(x["frame"]):x for x in bad.get("hand_bad_frame_details",[])}
        for side in ("left","right"):
            col=f"observation.state.hand_{side}_world"
            pts=np.stack([robust_hand_point(v) for v in data[col].to_numpy()])
            frame=data["frame_index"].to_numpy(dtype=int)
            flags=np.array([details.get(int(k),{}).get(f"{side}_bad_reason") is not None for k in frame])
            yield {"dataset":"EgoDemo","episode_uuid":episode,"environment_id":env,
                   "side":side,"time":data["timestamp"].to_numpy(float),"position":pts,
                   "bad":flags,"source":name}

def _mcap_tables(zf, dataset):
    prefix=dataset+"/"
    sessions=sorted({n[len(prefix):].split("/",1)[0] for n in zf.namelist()
                     if n.startswith(prefix) and "/tables/pose_left_hand.csv" in n})
    for session in sessions:
        pfx=f"{prefix}{session}/tables/"
        meta=pd.read_csv(io.BytesIO(zf.read(f"{prefix}{session}/session_metadata.csv")),encoding="utf-8-sig")
        meta=meta.iloc[0].to_dict()
        episode=str(meta.get("episode_uuid",session)); env=str(meta.get("environment_id",""))
        flag_path=pfx+"annotation_bad_frame_pose_hand.csv"
        flags=pd.read_csv(io.BytesIO(zf.read(flag_path)))
        bad_col="is_bad" if "is_bad" in flags else None
        for side in ("left","right"):
            path=pfx+f"pose_{side}_hand.csv"
            if path not in zf.namelist(): continue
            df=pd.read_csv(io.BytesIO(zf.read(path)))
            pts=[]
            for _,row in df.iterrows():
                xyz=[]
                for j in range(21):
                    xyz.append([row[f"transforms[{j}].pos.{axis}"] for axis in "xyz"])
                pts.append(robust_hand_point(xyz))
            pos=np.asarray(pts,float)
            # Same recorded message index is the key; no nearest-time fabricated labels.
            bad=np.zeros(len(df),dtype=bool)
            if bad_col and "_message_index" in flags and "_message_index" in df:
                f=flags.set_index("_message_index")[bad_col]
                bad=df["_message_index"].map(f).fillna(False).astype(bool).to_numpy()
            yield {"dataset":dataset,"episode_uuid":episode,"environment_id":env,
                   "side":side,"time":df["_offset_from_first_log_s"].to_numpy(float),
                   "position":pos,"bad":bad,"source":path}

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
