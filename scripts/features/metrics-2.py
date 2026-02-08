#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py — EXACT feature extraction matching analyze_pose_emotion (v1–v5)

This script reproduces the per-image feature computation exactly as in your
analyze_pose_emotion*.py code family:

- CONF_TH = 0.30
- MIN_VALID_KP = 10
- valid_kp = count(conf >= CONF_TH) over 17 COCO keypoints
- pose normalization:
    * requires left/right hip + left/right shoulder visible
    * pelvis = mean(left_hip, right_hip)
    * shoulder_mid = mean(left_shoulder, right_shoulder)
    * scale = ||shoulder_mid - pelvis||
    * norm_xy = (xy - pelvis) / scale
- safe_point uses CONF_TH
- angles are in DEGREES
- contraction = -mean distance of selected upper-body points to torso_center
  (torso_center = mean(ls, rs, lh, rh) in normalized coords)
- shoulder_width = ||ls - rs|| in normalized coords
- arm_span_norm = |lw.x - rw.x| / shoulder_width
- head_dx/head_dy = nose - shoulder_mid (normalized coords)
- bbox_w/bbox_h taken from "box" field when present

Outputs:
- per_sample_metrics.csv (exact columns used by v1 script)
- summary_counts.csv
- summary_missingness.csv

Input format:
ROOT/
  emotion_1/alphapose-results.json
  emotion_2/alphapose-results.json
  ...

Run:
python metrics.py --root /path/to/alphapose_outputs_by_label --outdir /path/to/out --json_name alphapose-results.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

CONF_TH = 0.30
MIN_VALID_KP = 10

KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

FEATURE_COLUMNS = [
    "emotion","image_id","pose_score","valid_kp","bbox_w","bbox_h",
    "shoulder_width",
    "left_hand_height","right_hand_height",
    "arm_span_norm",
    "left_elbow_angle","right_elbow_angle",
    "left_knee_angle","right_knee_angle",
    "contraction",
    "hand_height_asym","elbow_asym",
    "head_dx","head_dy",
]

def parse_kp(kps: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    AlphaPose COCO keypoints: flat list [x,y,score]*17
    Returns:
      xy: (17,2)
      cf: (17,)
    """
    a = np.array(kps, dtype=float).reshape(-1, 3)
    xy = a[:, :2]
    cf = a[:, 2]
    return xy, cf


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees. a,b,c are 2D points."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-8 or nbc < 1e-8:
        return np.nan
    cosv = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def safe_point(xy: np.ndarray, cf: np.ndarray, name: str) -> Optional[np.ndarray]:
    i = KP[name]
    if cf[i] >= CONF_TH and np.isfinite(xy[i]).all():
        return xy[i]
    return None


def normalize_pose(xy: np.ndarray, cf: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    EXACT normalization (matches analyze_pose_emotion2.py):
    - requires hips + shoulders present
    - pelvis as origin
    - torso length (shoulder_mid - pelvis) as scale
    """
    lh = safe_point(xy, cf, "left_hip")
    rh = safe_point(xy, cf, "right_hip")
    ls = safe_point(xy, cf, "left_shoulder")
    rs = safe_point(xy, cf, "right_shoulder")

    if lh is None or rh is None or ls is None or rs is None:
        return None, None

    pelvis = (lh + rh) / 2.0
    shoulder_mid = (ls + rs) / 2.0
    scale = np.linalg.norm(shoulder_mid - pelvis)

    if (not np.isfinite(scale)) or scale < 1e-6:
        return None, None

    norm_xy = (xy - pelvis[None, :]) / scale
    info = {"scale": float(scale)}
    return norm_xy, info


def contraction_index(norm_xy: np.ndarray, cf: np.ndarray) -> float:
    """
    EXACT definition (matches analyze_pose_emotion2.py):
    More positive = more contracted (we define as negative mean distance).
    """
    use = ["nose","left_shoulder","right_shoulder","left_elbow","right_elbow",
           "left_wrist","right_wrist","left_hip","right_hip"]
    pts = []
    for n in use:
        i = KP[n]
        if cf[i] >= CONF_TH and np.isfinite(norm_xy[i]).all():
            pts.append(norm_xy[i])
    if len(pts) < 5:
        return np.nan

    pts = np.stack(pts, axis=0)
    ls, rs = norm_xy[KP["left_shoulder"]], norm_xy[KP["right_shoulder"]]
    lh, rh = norm_xy[KP["left_hip"]], norm_xy[KP["right_hip"]]
    torso_center = (ls + rs + lh + rh) / 4.0
    d = np.linalg.norm(pts - torso_center[None, :], axis=1)
    return float(-np.mean(d))


def extract_one(item: Dict[str, Any], emotion: str) -> Optional[Dict[str, Any]]:
    image_id = item.get("image_id", item.get("imgname", item.get("image", item.get("file_name", item.get("filename")))))
    kps = item.get("keypoints", None)
    box = item.get("box", [np.nan]*4)
    score = item.get("score", np.nan)

    if kps is None or image_id is None:
        return None

    xy, cf = parse_kp(kps)

    valid = int(np.sum(cf >= CONF_TH))
    if valid < MIN_VALID_KP:
        return None

    norm_xy, info = normalize_pose(xy, cf)
    if norm_xy is None:
        return None

    # grab points in normalized coords
    ls = safe_point(norm_xy, cf, "left_shoulder")
    rs = safe_point(norm_xy, cf, "right_shoulder")
    le = safe_point(norm_xy, cf, "left_elbow")
    re = safe_point(norm_xy, cf, "right_elbow")
    lw = safe_point(norm_xy, cf, "left_wrist")
    rw = safe_point(norm_xy, cf, "right_wrist")

    lh = safe_point(norm_xy, cf, "left_hip")
    rh = safe_point(norm_xy, cf, "right_hip")
    lk = safe_point(norm_xy, cf, "left_knee")
    rk = safe_point(norm_xy, cf, "right_knee")
    la = safe_point(norm_xy, cf, "left_ankle")
    ra = safe_point(norm_xy, cf, "right_ankle")
    nose = safe_point(norm_xy, cf, "nose")

    shoulder_width = np.linalg.norm(ls - rs) if (ls is not None and rs is not None) else np.nan

    left_hand_height = (lw[1] - ls[1]) if (lw is not None and ls is not None) else np.nan
    right_hand_height = (rw[1] - rs[1]) if (rw is not None and rs is not None) else np.nan

    arm_span = abs(lw[0] - rw[0]) if (lw is not None and rw is not None) else np.nan
    arm_span_norm = arm_span / shoulder_width if (np.isfinite(arm_span) and np.isfinite(shoulder_width) and shoulder_width > 1e-6) else np.nan

    left_elbow_angle = angle(ls, le, lw) if (ls is not None and le is not None and lw is not None) else np.nan
    right_elbow_angle = angle(rs, re, rw) if (rs is not None and re is not None and rw is not None) else np.nan

    left_knee_angle = angle(lh, lk, la) if (lh is not None and lk is not None and la is not None) else np.nan
    right_knee_angle = angle(rh, rk, ra) if (rh is not None and rk is not None and ra is not None) else np.nan

    contract = contraction_index(norm_xy, cf)

    hand_height_asym = abs(left_hand_height - right_hand_height) if (np.isfinite(left_hand_height) and np.isfinite(right_hand_height)) else np.nan
    elbow_asym = abs(left_elbow_angle - right_elbow_angle) if (np.isfinite(left_elbow_angle) and np.isfinite(right_elbow_angle)) else np.nan

    if nose is not None and ls is not None and rs is not None:
        shoulder_mid = (ls + rs) / 2.0
        head_dx = float(nose[0] - shoulder_mid[0])
        head_dy = float(nose[1] - shoulder_mid[1])
    else:
        head_dx, head_dy = np.nan, np.nan

    try:
        bx, by, bw, bh = [float(x) for x in (box if len(box) == 4 else [np.nan]*4)]
    except Exception:
        bw, bh = np.nan, np.nan

    rec = {
        "emotion": emotion,
        "image_id": str(image_id),
        "pose_score": float(score) if score is not None else np.nan,
        "valid_kp": valid,
        "bbox_w": bw,
        "bbox_h": bh,
        "shoulder_width": float(shoulder_width) if np.isfinite(shoulder_width) else np.nan,
        "left_hand_height": float(left_hand_height) if np.isfinite(left_hand_height) else np.nan,
        "right_hand_height": float(right_hand_height) if np.isfinite(right_hand_height) else np.nan,
        "arm_span_norm": float(arm_span_norm) if np.isfinite(arm_span_norm) else np.nan,
        "left_elbow_angle": float(left_elbow_angle) if np.isfinite(left_elbow_angle) else np.nan,
        "right_elbow_angle": float(right_elbow_angle) if np.isfinite(right_elbow_angle) else np.nan,
        "left_knee_angle": float(left_knee_angle) if np.isfinite(left_knee_angle) else np.nan,
        "right_knee_angle": float(right_knee_angle) if np.isfinite(right_knee_angle) else np.nan,
        "contraction": float(contract) if np.isfinite(contract) else np.nan,
        "hand_height_asym": float(hand_height_asym) if np.isfinite(hand_height_asym) else np.nan,
        "elbow_asym": float(elbow_asym) if np.isfinite(elbow_asym) else np.nan,
        "head_dx": float(head_dx) if np.isfinite(head_dx) else np.nan,
        "head_dy": float(head_dy) if np.isfinite(head_dy) else np.nan,
    }
    return rec


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory with per-emotion subfolders.")
    ap.add_argument("--json_name", default="alphapose-results.json", help="AlphaPose JSON filename inside each emotion folder.")
    ap.add_argument("--outdir", required=True, help="Output directory for CSV.")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    emotion_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    for ed in emotion_dirs:
        emotion = ed.name
        jf = ed / args.json_name
        if not jf.exists():
            continue
        try:
            items = load_json(jf)
        except Exception:
            continue
        for item in items:
            rec = extract_one(item, emotion)
            if rec is not None:
                records.append(rec)

    df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    out_csv = outdir / "per_sample_metrics.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # missingness summary
    miss_rows = []
    n = len(df)
    for c in df.columns:
        if c in ("emotion","image_id"):
            continue
        miss_n = int(df[c].isna().sum())
        miss_rows.append({"metric": c, "missing_n": miss_n, "missing_rate": miss_n / n if n else np.nan})
    pd.DataFrame(miss_rows).sort_values("missing_rate", ascending=False).to_csv(outdir / "summary_missingness.csv", index=False)

    # counts
    counts = {
        "n_samples": n,
        "n_emotions": int(df["emotion"].nunique()) if n else 0,
        "mean_valid_kp": float(df["valid_kp"].mean()) if n else np.nan,
    }
    pd.DataFrame([counts]).to_csv(outdir / "summary_counts.csv", index=False)

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {outdir / 'summary_missingness.csv'}")
    print(f"[OK] wrote {outdir / 'summary_counts.csv'}")


if __name__ == "__main__":
    main()
