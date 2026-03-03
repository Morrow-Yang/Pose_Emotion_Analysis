#!/usr/bin/env python3
"""
Merge geometry features (analysis_v4-style) with temporal motion features.
Outputs two CSVs in --out_dir:
- pose_features_v4_geom.csv: geometry-only features
- pose_features_v4_with_temporal.csv: geometry joined with temporal features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def parse_kp(arr_51: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array(arr_51, dtype=float)
    if a.size % 3 != 0:
        raise ValueError("keypoints size not divisible by 3")
    a = a.reshape(-1, 3)
    if a.shape[0] < 17:
        pad = np.full((17 - a.shape[0], 3), np.nan)
        a = np.vstack([a, pad])
    xy = a[:17, :2]
    cf = a[:17, 2]
    return xy, cf


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
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


def normalize_pose(xy: np.ndarray, cf: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    lh = safe_point(xy, cf, "left_hip")
    rh = safe_point(xy, cf, "right_hip")
    ls = safe_point(xy, cf, "left_shoulder")
    rs = safe_point(xy, cf, "right_shoulder")
    if lh is None or rh is None or ls is None or rs is None:
        return None, None
    pelvis = (lh + rh) / 2.0
    shoulder_mid = (ls + rs) / 2.0
    scale = np.linalg.norm(shoulder_mid - pelvis)
    if not np.isfinite(scale) or scale < 1e-6:
        return None, None
    norm_xy = (xy - pelvis[None, :]) / scale
    return norm_xy, {"scale": float(scale)}


def contraction_index(norm_xy: np.ndarray, cf: np.ndarray) -> float:
    use = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    ]
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


def build_geometry(root: Path, json_name: str) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for ed in sorted([d for d in root.iterdir() if d.is_dir()]):
        emotion = ed.name
        jf = ed / json_name
        if not jf.exists():
            continue
        try:
            data = json.load(jf.open("r", encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            kps = item.get("keypoints")
            if kps is None:
                continue
            try:
                xy, cf = parse_kp(kps)
            except Exception:
                continue
            valid = int(np.sum(cf >= CONF_TH))
            if valid < MIN_VALID_KP:
                continue
            norm_xy, _info = normalize_pose(xy, cf)
            if norm_xy is None:
                continue
            ls = safe_point(norm_xy, cf, "left_shoulder")
            rs = safe_point(norm_xy, cf, "right_shoulder")
            lw = safe_point(norm_xy, cf, "left_wrist")
            rw = safe_point(norm_xy, cf, "right_wrist")
            le = safe_point(norm_xy, cf, "left_elbow")
            re = safe_point(norm_xy, cf, "right_elbow")
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
            arm_span_norm = arm_span / shoulder_width if (
                np.isfinite(arm_span) and np.isfinite(shoulder_width) and shoulder_width > 1e-6
            ) else np.nan
            left_elbow_angle = angle(ls, le, lw) if (ls is not None and le is not None and lw is not None) else np.nan
            right_elbow_angle = angle(rs, re, rw) if (rs is not None and re is not None and rw is not None) else np.nan
            left_knee_angle = angle(lh, lk, la) if (lh is not None and lk is not None and la is not None) else np.nan
            right_knee_angle = angle(rh, rk, ra) if (rh is not None and rk is not None and ra is not None) else np.nan
            contract = contraction_index(norm_xy, cf)
            hand_height_asym = (
                abs(left_hand_height - right_hand_height)
                if (np.isfinite(left_hand_height) and np.isfinite(right_hand_height))
                else np.nan
            )
            elbow_asym = (
                abs(left_elbow_angle - right_elbow_angle)
                if (np.isfinite(left_elbow_angle) and np.isfinite(right_elbow_angle))
                else np.nan
            )
            if nose is not None and ls is not None and rs is not None:
                shoulder_mid = (ls + rs) / 2.0
                head_dx = float(nose[0] - shoulder_mid[0])
                head_dy = float(nose[1] - shoulder_mid[1])
            else:
                head_dx, head_dy = np.nan, np.nan

            box = item.get("box", [np.nan] * 4)
            bx, by, bw, bh = [float(x) for x in (box if len(box) == 4 else [np.nan] * 4)]
            image_id = item.get("image_id", item.get("imgname", item.get("image", item.get("file_name"))))
            rec: Dict[str, float] = {
                "emotion": emotion,
                "image_id": str(image_id) if image_id is not None else "",
                "pose_score": float(item.get("score", np.nan)) if item.get("score", None) is not None else np.nan,
                "valid_kp": valid,
                "bbox_w": bw,
                "bbox_h": bh,
                "shoulder_width": shoulder_width,
                "left_hand_height": left_hand_height,
                "right_hand_height": right_hand_height,
                "arm_span_norm": arm_span_norm,
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "left_knee_angle": left_knee_angle,
                "right_knee_angle": right_knee_angle,
                "contraction": contract,
                "hand_height_asym": hand_height_asym,
                "elbow_asym": elbow_asym,
                "head_dx": head_dx,
                "head_dy": head_dy,
            }
            flat = norm_xy.reshape(-1)
            rec.update({f"kp_{i:02d}": float(flat[i]) for i in range(flat.shape[0])})
            records.append(rec)
    return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root with per-emotion folders")
    ap.add_argument("--json_name", default="alphapose-results.json")
    ap.add_argument("--temporal_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geom_df = build_geometry(root, args.json_name)
    geom_csv = out_dir / "pose_features_v4_geom.csv"
    geom_df.to_csv(geom_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] geometry features: {len(geom_df)} rows -> {geom_csv}")

    temporal_df = pd.read_csv(args.temporal_csv)
    merge_cols = [c for c in temporal_df.columns if c not in {"dataset"}]
    merged = geom_df.merge(temporal_df[merge_cols], on=["emotion", "image_id"], how="left")
    merged_csv = out_dir / "pose_features_v4_with_temporal.csv"
    merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] merged features: {len(merged)} rows -> {merged_csv}")


if __name__ == "__main__":
    main()
