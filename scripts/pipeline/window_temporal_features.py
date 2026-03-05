import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


# Minimal, general-purpose sliding window aggregator for 2D/3D temporal features.
# It assumes input CSV has per-frame features, grouped by columns like emotion/actor/filename.
# If a time column is missing, it derives time from frame_idx / fps.


GROUP_CANDIDATES = ["dataset", "actor", "filename", "emotion"]
BASE_FEATURES = {"avg_velocity", "avg_acceleration"}
VEL_SUFFIX = "_vel"
ACC_SUFFIX = "_accel"


def detect_groups(df: pd.DataFrame) -> List[str]:
    return [c for c in GROUP_CANDIDATES if c in df.columns]


def detect_features(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in BASE_FEATURES:
            cols.append(c)
        elif c.endswith(VEL_SUFFIX) or c.endswith(ACC_SUFFIX):
            cols.append(c)
    return cols


def ensure_time(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    if "time_sec" in df.columns:
        return df
    if "frame_idx" not in df.columns:
        raise ValueError("time_sec not found and frame_idx missing; cannot derive time")
    df = df.copy()
    df["time_sec"] = df["frame_idx"] / float(fps)
    return df


def high_ratio(values: np.ndarray, thr: float) -> float:
    if values.size == 0:
        return np.nan
    return float(np.mean(values >= thr))


def high_stretch(values: np.ndarray, thr: float) -> float:
    # longest consecutive run above thr
    longest = curr = 0
    for v in values:
        if v >= thr:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 0
    return float(longest)


def aggregate_window(df: pd.DataFrame, features: List[str], thresholds: Dict[str, float]):
    stats = {}
    for f in features:
        arr = df[f].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            stats[f + "_mean"] = np.nan
            stats[f + "_max"] = np.nan
            stats[f + "_p95"] = np.nan
            stats[f + "_std"] = np.nan
            stats[f + "_high_ratio"] = np.nan
            stats[f + "_high_stretch"] = np.nan
            continue
        stats[f + "_mean"] = float(np.mean(arr))
        stats[f + "_max"] = float(np.max(arr))
        stats[f + "_p95"] = float(np.percentile(arr, 95))
        stats[f + "_std"] = float(np.std(arr))
        thr = thresholds.get(f, np.percentile(arr, 90))
        stats[f + "_high_ratio"] = high_ratio(arr, thr)
        stats[f + "_high_stretch"] = high_stretch(arr, thr)
    return stats


def process(df: pd.DataFrame, window_sec: float, hop_sec: float, fps: float, min_count: int) -> pd.DataFrame:
    df = ensure_time(df, fps)
    groups = detect_groups(df)
    features = detect_features(df)
    if not features:
        raise ValueError("No feature columns detected (avg_velocity/_vel/_accel)")

    df = df.copy()
    df = df.sort_values(groups + ["time_sec"]) if groups else df.sort_values("time_sec")

    # precompute thresholds per group based on full series (per feature)
    thresholds = {}
    if groups:
        for key, g in df.groupby(groups):
            for f in features:
                arr = g[f].to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                thresholds[(key, f)] = float(np.percentile(arr, 90))
    else:
        for f in features:
            arr = df[f].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            thresholds[(None, f)] = float(np.percentile(arr, 90))

    rows = []
    if groups:
        grouped = df.groupby(groups)
    else:
        grouped = [(None, df)]

    for key, g in grouped:
        if g.empty:
            continue
        t_min = g["time_sec"].min()
        t_max = g["time_sec"].max()
        start = t_min
        while start <= t_max:
            end = start + window_sec
            win = g[(g["time_sec"] >= start) & (g["time_sec"] < end)]
            if len(win) >= min_count:
                base = {}
                if groups:
                    if isinstance(key, tuple):
                        for col, val in zip(groups, key):
                            base[col] = val
                    else:
                        base[groups[0]] = key
                base.update({
                    "t_start": float(start),
                    "t_end": float(end),
                    "num_frames": int(len(win)),
                })
                thr_key = lambda f: thresholds.get((key, f), thresholds.get((None, f), np.nan))
                per_feats = aggregate_window(win, features, {f: thr_key(f) for f in features})
                base.update(per_feats)
                rows.append(base)
            start += hop_sec

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="per-frame CSV (2D or 3D)")
    ap.add_argument("--output", required=True, help="output CSV for windowed features")
    ap.add_argument("--window_sec", type=float, default=1.0)
    ap.add_argument("--hop_sec", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=10.0, help="used only if time_sec is absent; CAER=10fps, BVH≈60fps")
    ap.add_argument("--min_count", type=int, default=1, help="min frames per window to keep")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out_df = process(df, args.window_sec, args.hop_sec, args.fps, args.min_count)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[OK] saved {len(out_df)} windows -> {args.output}")


if __name__ == "__main__":
    main()
