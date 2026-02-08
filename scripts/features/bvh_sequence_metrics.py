#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract per-sequence pose features from BVH (3D) and summarize by mean/std.
Outputs:
  - per_sequence_metrics_mean.csv (feature means; compatible feature names)
  - per_sequence_metrics_mean_std.csv (means + stds)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

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

JOINT_MAP = {
    "left_shoulder": "LeftShoulder",
    "right_shoulder": "RightShoulder",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "nose": "Head",
}


@dataclass
class BvhNode:
    name: str
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    channels: List[str] = field(default_factory=list)
    children: List["BvhNode"] = field(default_factory=list)
    parent: Optional["BvhNode"] = None


def rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "X":
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)
    if axis == "Y":
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=float)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def parse_bvh(path: Path) -> Tuple[BvhNode, List[BvhNode], np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    root: Optional[BvhNode] = None
    stack: List[BvhNode] = []
    nodes_in_order: List[BvhNode] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTION"):
            break
        if line.startswith("ROOT ") or line.startswith("JOINT "):
            name = line.split()[1]
            node = BvhNode(name=name)
            if stack:
                node.parent = stack[-1]
                stack[-1].children.append(node)
            else:
                root = node
            stack.append(node)
            nodes_in_order.append(node)
        elif line.startswith("End Site"):
            node = BvhNode(name=(stack[-1].name + "_EndSite"))
            node.parent = stack[-1]
            stack[-1].children.append(node)
            stack.append(node)
        elif line.startswith("OFFSET "):
            parts = line.split()
            stack[-1].offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        elif line.startswith("CHANNELS "):
            parts = line.split()
            count = int(parts[1])
            stack[-1].channels = parts[2:2 + count]
        elif line == "}":
            stack.pop()
        i += 1

    if root is None:
        raise ValueError(f"No ROOT found in {path}")

    # parse motion
    while i < len(lines) and not lines[i].strip().startswith("Frames:"):
        i += 1
    if i >= len(lines):
        raise ValueError(f"No Frames section in {path}")
    frames = int(lines[i].strip().split(":")[1])
    i += 1
    while i < len(lines) and not lines[i].strip().startswith("Frame Time:"):
        i += 1
    i += 1

    total_channels = sum(len(n.channels) for n in nodes_in_order)
    motion = []
    for j in range(frames):
        if i + j >= len(lines):
            break
        row = [float(x) for x in lines[i + j].strip().split()]
        if len(row) != total_channels:
            continue
        motion.append(row)

    motion_arr = np.array(motion, dtype=float)
    return root, nodes_in_order, motion_arr


def build_channel_index(nodes_in_order: List[BvhNode]) -> List[Tuple[BvhNode, str]]:
    channel_index = []
    for n in nodes_in_order:
        for ch in n.channels:
            channel_index.append((n, ch))
    return channel_index


def compute_global_positions(nodes_in_order: List[BvhNode], channel_index: List[Tuple[BvhNode, str]], frame: np.ndarray) -> Dict[str, np.ndarray]:
    # collect channel values per node
    node_channels: Dict[str, Dict[str, float]] = {n.name: {} for n in nodes_in_order}
    for (n, ch), val in zip(channel_index, frame):
        node_channels[n.name][ch] = float(val)

    global_mats: Dict[str, np.ndarray] = {}
    positions: Dict[str, np.ndarray] = {}

    for n in nodes_in_order:
        local = translation_matrix(n.offset[0], n.offset[1], n.offset[2])
        for ch in n.channels:
            v = node_channels[n.name].get(ch, 0.0)
            if ch.endswith("position"):
                axis = ch[0]
                if axis == "X":
                    local = local @ translation_matrix(v, 0.0, 0.0)
                elif axis == "Y":
                    local = local @ translation_matrix(0.0, v, 0.0)
                else:
                    local = local @ translation_matrix(0.0, 0.0, v)
            elif ch.endswith("rotation"):
                local = local @ rotation_matrix(ch[0], v)

        if n.parent is None:
            global_mats[n.name] = local
        else:
            global_mats[n.name] = global_mats[n.parent.name] @ local

        pos = global_mats[n.name] @ np.array([0.0, 0.0, 0.0, 1.0])
        positions[n.name] = pos[:3]

    return positions


def angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-8 or nbc < 1e-8:
        return np.nan
    cosv = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def compute_features(norm_pos: Dict[str, np.ndarray]) -> Dict[str, float]:
    def get(name: str) -> Optional[np.ndarray]:
        jn = JOINT_MAP.get(name)
        if jn is None:
            return None
        return norm_pos.get(jn)

    ls = get("left_shoulder")
    rs = get("right_shoulder")
    le = get("left_elbow")
    re = get("right_elbow")
    lw = get("left_wrist")
    rw = get("right_wrist")
    lh = get("left_hip")
    rh = get("right_hip")
    lk = get("left_knee")
    rk = get("right_knee")
    la = get("left_ankle")
    ra = get("right_ankle")
    nose = get("nose")

    shoulder_width = np.linalg.norm(ls - rs) if (ls is not None and rs is not None) else np.nan
    left_hand_height = (lw[1] - ls[1]) if (lw is not None and ls is not None) else np.nan
    right_hand_height = (rw[1] - rs[1]) if (rw is not None and rs is not None) else np.nan

    arm_span = np.linalg.norm(lw - rw) if (lw is not None and rw is not None) else np.nan
    arm_span_norm = arm_span / shoulder_width if (np.isfinite(arm_span) and np.isfinite(shoulder_width) and shoulder_width > 1e-6) else np.nan

    left_elbow_angle = angle3(ls, le, lw) if (ls is not None and le is not None and lw is not None) else np.nan
    right_elbow_angle = angle3(rs, re, rw) if (rs is not None and re is not None and rw is not None) else np.nan

    left_knee_angle = angle3(lh, lk, la) if (lh is not None and lk is not None and la is not None) else np.nan
    right_knee_angle = angle3(rh, rk, ra) if (rh is not None and rk is not None and ra is not None) else np.nan

    use_names = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist", "left_hip", "right_hip"]
    pts = [get(n) for n in use_names if get(n) is not None]
    if len(pts) >= 5 and ls is not None and rs is not None and lh is not None and rh is not None:
        pts_arr = np.stack(pts, axis=0)
        torso_center = (ls + rs + lh + rh) / 4.0
        d = np.linalg.norm(pts_arr - torso_center[None, :], axis=1)
        contraction = float(-np.mean(d))
    else:
        contraction = np.nan

    hand_height_asym = abs(left_hand_height - right_hand_height) if (np.isfinite(left_hand_height) and np.isfinite(right_hand_height)) else np.nan
    elbow_asym = abs(left_elbow_angle - right_elbow_angle) if (np.isfinite(left_elbow_angle) and np.isfinite(right_elbow_angle)) else np.nan

    if nose is not None and ls is not None and rs is not None:
        shoulder_mid = (ls + rs) / 2.0
        head_dx = float(nose[0] - shoulder_mid[0])
        head_dy = float(nose[1] - shoulder_mid[1])
    else:
        head_dx, head_dy = np.nan, np.nan

    return {
        "shoulder_width": float(shoulder_width) if np.isfinite(shoulder_width) else np.nan,
        "left_hand_height": float(left_hand_height) if np.isfinite(left_hand_height) else np.nan,
        "right_hand_height": float(right_hand_height) if np.isfinite(right_hand_height) else np.nan,
        "arm_span_norm": float(arm_span_norm) if np.isfinite(arm_span_norm) else np.nan,
        "left_elbow_angle": float(left_elbow_angle) if np.isfinite(left_elbow_angle) else np.nan,
        "right_elbow_angle": float(right_elbow_angle) if np.isfinite(right_elbow_angle) else np.nan,
        "left_knee_angle": float(left_knee_angle) if np.isfinite(left_knee_angle) else np.nan,
        "right_knee_angle": float(right_knee_angle) if np.isfinite(right_knee_angle) else np.nan,
        "contraction": float(contraction) if np.isfinite(contraction) else np.nan,
        "hand_height_asym": float(hand_height_asym) if np.isfinite(hand_height_asym) else np.nan,
        "elbow_asym": float(elbow_asym) if np.isfinite(elbow_asym) else np.nan,
        "head_dx": float(head_dx) if np.isfinite(head_dx) else np.nan,
        "head_dy": float(head_dy) if np.isfinite(head_dy) else np.nan,
    }


def normalize_positions(positions: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    lh = positions.get(JOINT_MAP["left_hip"])
    rh = positions.get(JOINT_MAP["right_hip"])
    ls = positions.get(JOINT_MAP["left_shoulder"])
    rs = positions.get(JOINT_MAP["right_shoulder"])
    if lh is None or rh is None or ls is None or rs is None:
        return None
    pelvis = (lh + rh) / 2.0
    shoulder_mid = (ls + rs) / 2.0
    scale = np.linalg.norm(shoulder_mid - pelvis)
    if (not np.isfinite(scale)) or scale < 1e-6:
        return None
    norm = {k: (v - pelvis) / scale for k, v in positions.items()}
    return norm


def summarize_sequence(path: Path) -> Tuple[pd.DataFrame, int]:
    _, nodes_in_order, motion = parse_bvh(path)
    channel_index = build_channel_index(nodes_in_order)

    rows = []
    for frame in motion:
        positions = compute_global_positions(nodes_in_order, channel_index, frame)
        norm_pos = normalize_positions(positions)
        if norm_pos is None:
            continue
        feats = compute_features(norm_pos)
        rows.append(feats)

    if not rows:
        return pd.DataFrame(), 0
    df = pd.DataFrame(rows)
    return df, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_root", required=True, help="Root containing BVH subfolders (F01, M01, etc.)")
    ap.add_argument("--file_info", required=True, help="file-info.csv path")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    bvh_root = Path(args.bvh_root)
    file_info = Path(args.file_info)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    info_df = pd.read_csv(file_info)
    info_df["filename"] = info_df["filename"].astype(str)
    info_map = info_df.set_index("filename").to_dict(orient="index")

    mean_rows = []
    mean_std_rows = []

    bvh_files = sorted(bvh_root.glob("**/*.bvh"))
    for bf in bvh_files:
        stem = bf.stem
        meta = info_map.get(stem)
        if meta is None:
            continue

        seq_df, used_frames = summarize_sequence(bf)
        if seq_df.empty:
            continue

        means = seq_df.mean(numeric_only=True)
        stds = seq_df.std(numeric_only=True)

        base = {
            "emotion": meta.get("emotion"),
            "image_id": stem,
            "actor_id": meta.get("actor_ID"),
            "actor_gender": meta.get("actor_gender"),
            "scenario_id": meta.get("scenario_ID"),
            "version": meta.get("version"),
            "n_frames": int(used_frames),
            "pose_score": np.nan,
            "valid_kp": 17,
            "bbox_w": np.nan,
            "bbox_h": np.nan,
        }

        mean_row = base.copy()
        for f in FEATURE_COLUMNS:
            if f in ("emotion", "image_id", "pose_score", "valid_kp", "bbox_w", "bbox_h"):
                continue
            mean_row[f] = float(means.get(f, np.nan))
        mean_rows.append(mean_row)

        mean_std_row = base.copy()
        for f in FEATURE_COLUMNS:
            if f in ("emotion", "image_id", "pose_score", "valid_kp", "bbox_w", "bbox_h"):
                continue
            mean_std_row[f] = float(means.get(f, np.nan))
            mean_std_row[f + "_std"] = float(stds.get(f, np.nan))
        mean_std_rows.append(mean_std_row)

    mean_df = pd.DataFrame(mean_rows)
    mean_std_df = pd.DataFrame(mean_std_rows)

    out_mean = outdir / "per_sequence_metrics_mean.csv"
    out_mean_std = outdir / "per_sequence_metrics_mean_std.csv"
    mean_df.to_csv(out_mean, index=False, encoding="utf-8-sig")
    mean_std_df.to_csv(out_mean_std, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote {out_mean}")
    print(f"[OK] wrote {out_mean_std}")
    print(f"Sequences: {len(mean_df)}")


if __name__ == "__main__":
    main()
