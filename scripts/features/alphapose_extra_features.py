#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extra features computed from AlphaPose JSON keypoints.
This does NOT modify existing scripts; it generates a separate CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

CONF_TH = 0.30

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

EXTRA_COLUMNS = [
    "emotion", "image_id",
    "torso_tilt_angle",
    "shoulder_slope",
    "hip_slope",
    "shoulder_hip_slope_diff",
    "leg_spread_norm",
    "hand_to_head_norm",
    "hand_to_hip_norm",
    "head_to_hip_height",
    "upper_lower_ratio",
]


def parse_kp(kps: Any) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array(kps, dtype=float).reshape(-1, 3)
    xy = a[:, :2]
    cf = a[:, 2]
    return xy, cf


def safe_point(xy: np.ndarray, cf: np.ndarray, name: str) -> Optional[np.ndarray]:
    i = KP[name]
    if cf[i] >= CONF_TH and np.isfinite(xy[i]).all():
        return xy[i]
    return None


def normalize_pose(xy: np.ndarray, cf: np.ndarray) -> Optional[np.ndarray]:
    lh = safe_point(xy, cf, "left_hip")
    rh = safe_point(xy, cf, "right_hip")
    ls = safe_point(xy, cf, "left_shoulder")
    rs = safe_point(xy, cf, "right_shoulder")
    if lh is None or rh is None or ls is None or rs is None:
        return None
    pelvis = (lh + rh) / 2.0
    shoulder_mid = (ls + rs) / 2.0
    scale = np.linalg.norm(shoulder_mid - pelvis)
    if (not np.isfinite(scale)) or scale < 1e-6:
        return None
    return (xy - pelvis[None, :]) / scale


def compute_extra(norm_xy: np.ndarray, cf: np.ndarray) -> Dict[str, float]:
    ls = safe_point(norm_xy, cf, "left_shoulder")
    rs = safe_point(norm_xy, cf, "right_shoulder")
    lh = safe_point(norm_xy, cf, "left_hip")
    rh = safe_point(norm_xy, cf, "right_hip")
    la = safe_point(norm_xy, cf, "left_ankle")
    ra = safe_point(norm_xy, cf, "right_ankle")
    lw = safe_point(norm_xy, cf, "left_wrist")
    rw = safe_point(norm_xy, cf, "right_wrist")
    nose = safe_point(norm_xy, cf, "nose")

    out: Dict[str, float] = {}

    # torso tilt relative to vertical (y axis)
    if ls is not None and rs is not None and lh is not None and rh is not None:
        shoulder_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0
        torso_vec = shoulder_mid - hip_mid
        torso_tilt = np.degrees(np.arctan2(torso_vec[0], torso_vec[1]))
        out["torso_tilt_angle"] = float(torso_tilt)

        # slopes
        out["shoulder_slope"] = float((rs[1] - ls[1]) / (rs[0] - ls[0] + 1e-8))
        out["hip_slope"] = float((rh[1] - lh[1]) / (rh[0] - lh[0] + 1e-8))
        out["shoulder_hip_slope_diff"] = float(abs(out["shoulder_slope"] - out["hip_slope"]))

        out["head_to_hip_height"] = float(nose[1] - hip_mid[1]) if nose is not None else np.nan
        upper = np.linalg.norm(nose - shoulder_mid) if nose is not None else np.nan
        lower = np.linalg.norm(shoulder_mid - hip_mid)
        out["upper_lower_ratio"] = float(upper / lower) if (np.isfinite(upper) and lower > 1e-6) else np.nan
    else:
        out["torso_tilt_angle"] = np.nan
        out["shoulder_slope"] = np.nan
        out["hip_slope"] = np.nan
        out["shoulder_hip_slope_diff"] = np.nan
        out["head_to_hip_height"] = np.nan
        out["upper_lower_ratio"] = np.nan

    # leg spread normalized by shoulder width
    if la is not None and ra is not None and ls is not None and rs is not None:
        shoulder_width = np.linalg.norm(ls - rs)
        leg_spread = np.linalg.norm(la - ra)
        out["leg_spread_norm"] = float(leg_spread / shoulder_width) if shoulder_width > 1e-6 else np.nan
    else:
        out["leg_spread_norm"] = np.nan

    # hand to head distance normalized
    if nose is not None and ls is not None and rs is not None:
        shoulder_width = np.linalg.norm(ls - rs)
        d1 = np.linalg.norm(lw - nose) if lw is not None else np.nan
        d2 = np.linalg.norm(rw - nose) if rw is not None else np.nan
        d = np.nanmean([d1, d2]) if np.isfinite(d1) or np.isfinite(d2) else np.nan
        out["hand_to_head_norm"] = float(d / shoulder_width) if (np.isfinite(d) and shoulder_width > 1e-6) else np.nan
    else:
        out["hand_to_head_norm"] = np.nan

    # hand to hip distance normalized
    if lh is not None and rh is not None and ls is not None and rs is not None:
        shoulder_width = np.linalg.norm(ls - rs)
        d1 = np.linalg.norm(lw - lh) if lw is not None else np.nan
        d2 = np.linalg.norm(rw - rh) if rw is not None else np.nan
        d = np.nanmean([d1, d2]) if np.isfinite(d1) or np.isfinite(d2) else np.nan
        out["hand_to_hip_norm"] = float(d / shoulder_width) if (np.isfinite(d) and shoulder_width > 1e-6) else np.nan
    else:
        out["hand_to_hip_norm"] = np.nan

    return out


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory with per-emotion subfolders.")
    ap.add_argument("--json_name", default="alphapose-results.json", help="AlphaPose JSON filename.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    for ed in sorted([d for d in root.iterdir() if d.is_dir()]):
        emotion = ed.name
        jf = ed / args.json_name
        if not jf.exists():
            continue
        try:
            items = load_json(jf)
        except Exception:
            continue

        for item in items:
            image_id = item.get("image_id", item.get("imgname", item.get("image", item.get("file_name", item.get("filename")))))
            kps = item.get("keypoints", None)
            if kps is None or image_id is None:
                continue

            xy, cf = parse_kp(kps)
            norm_xy = normalize_pose(xy, cf)
            if norm_xy is None:
                continue

            extra = compute_extra(norm_xy, cf)
            rec = {"emotion": emotion, "image_id": str(image_id), **extra}
            records.append(rec)

    df = pd.DataFrame(records, columns=EXTRA_COLUMNS)
    out_csv = outdir / "extra_features.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {out_csv}")
    print(f"Samples: {len(df)}")


if __name__ == "__main__":
    main()
