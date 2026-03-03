"""
BVH Geometric & Temporal Analysis  –  v2
==========================================
Fixes from v1:
  - Geometric features now aggregated over ALL frames per file (mean/std/range),
    not just frame-0 (which was often a neutral T-pose).
  - Added 3D-only features that have no 2D equivalent:
      * spine_bend_deg     – lumbar flexion (Hips→Spine→Spine2)
      * lateral_lean_deg   – lateral trunk tilt (signed L/R)
      * head_forward_deg   – head forward-lean relative to neck
      * pelvis_height_norm – absolute Y position of Hips / spine-length
      * foot_spread_norm   – distance between feet / shoulder-width
      * hand_depth_diff    – difference in Z-depth between hands (3D space)
      * wrist_z_asym       – signed Z difference of wrists
      * knee_bend_asym     – |left_knee_angle - right_knee_angle|
      * body_extent        – bounding-box diagonal of all joints (expressivity)
  - RF split is actor-stratified (GroupShuffleSplit) to prevent leakage.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from utils_bvh_parser import BVHParser

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
DATASET_ROOT      = Path("data/raw/kinematic-dataset-of-actors-expressing-emotions-2.1.0")
TEMPORAL_FEAT_CSV = Path("outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv")
OUT_DIR           = Path("outputs/analysis/geom_bvh_v2")


def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex B (degrees)."""
    ba, bc = a - b, c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1, 1))))


def normalize_coords(raw: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
    """Center on Hips, scale by Head-Hips spine length."""
    hip = raw.get("Hips")
    head = raw.get("Head")
    if hip is None or head is None:
        return {}, np.nan
    spine = np.linalg.norm(head - hip)
    if spine < 1e-8:
        return {}, np.nan
    return {k: (v - hip) / spine for k, v in raw.items()}, spine


def contraction_index_3d(norm_coords: Dict[str, np.ndarray]) -> float:
    """Negative mean distance to torso center; more positive = more contracted."""
    use = ["Head", "LeftShoulder", "RightShoulder", "LeftForeArm", "RightForeArm",
           "LeftHand", "RightHand", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"]
    pts = []
    for n in use:
        if n in norm_coords:
            pts.append(norm_coords[n])
    if len(pts) < 5:
        return np.nan
    pts = np.stack(pts, axis=0)
    if {"LeftShoulder", "RightShoulder", "LeftUpLeg", "RightUpLeg"}.issubset(norm_coords.keys()):
        torso_center = (norm_coords["LeftShoulder"] + norm_coords["RightShoulder"] +
                        norm_coords["LeftUpLeg"] + norm_coords["RightUpLeg"]) / 4.0
    else:
        torso_center = pts.mean(axis=0)
    d = np.linalg.norm(pts - torso_center[None, :], axis=1)
    return float(-np.mean(d))


def extract_frame_features(raw: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Extract all geometry features (2D-like + 3D-only) from a single frame."""
    norm, spine = normalize_coords(raw)
    if not norm:
        return {}

    f: Dict[str, float] = {}

    ls  = norm.get("LeftShoulder")
    rs  = norm.get("RightShoulder")
    lh  = norm.get("LeftHand")
    rh  = norm.get("RightHand")
    lf  = norm.get("LeftForeArm")
    rf  = norm.get("RightForeArm")
    lu  = norm.get("LeftUpLeg")
    ru  = norm.get("RightUpLeg")
    ll  = norm.get("LeftLeg")
    rl  = norm.get("RightLeg")
    lft = norm.get("LeftFoot")
    rft = norm.get("RightFoot")
    head = norm.get("Head")
    neck = norm.get("Neck")
    sp1  = norm.get("Spine")
    sp2  = norm.get("Spine2")
    sp3  = norm.get("Spine3")

    # ── 2D-equivalent features ─────────────────────────────────────────────────
    sw = float(np.linalg.norm(ls - rs)) if (ls is not None and rs is not None) else np.nan
    f["shoulder_width"] = sw

    f["left_hand_height"]  = float(lh[1] - ls[1]) if (lh is not None and ls is not None) else np.nan
    f["right_hand_height"] = float(rh[1] - rs[1]) if (rh is not None and rs is not None) else np.nan

    if lh is not None and rh is not None and np.isfinite(sw) and sw > 1e-6:
        f["arm_span_norm"] = float(abs(lh[0] - rh[0]) / sw)
    else:
        f["arm_span_norm"] = np.nan

    f["left_elbow_angle"]  = angle_3d(ls  if ls  is not None else np.nan,
                                       lf  if lf  is not None else np.nan,
                                       lh  if lh  is not None else np.nan)
    f["right_elbow_angle"] = angle_3d(rs  if rs  is not None else np.nan,
                                       rf  if rf  is not None else np.nan,
                                       rh  if rh  is not None else np.nan)
    f["left_knee_angle"]   = angle_3d(lu  if lu  is not None else np.nan,
                                       ll  if ll  is not None else np.nan,
                                       lft if lft is not None else np.nan)
    f["right_knee_angle"]  = angle_3d(ru  if ru  is not None else np.nan,
                                       rl  if rl  is not None else np.nan,
                                       rft if rft is not None else np.nan)

    f["contraction"] = contraction_index_3d(norm)

    f["hand_height_asym"] = (abs(f["left_hand_height"] - f["right_hand_height"])
                              if np.isfinite(f["left_hand_height"]) and np.isfinite(f["right_hand_height"])
                              else np.nan)
    f["elbow_asym"] = (abs(f["left_elbow_angle"] - f["right_elbow_angle"])
                       if np.isfinite(f["left_elbow_angle"]) and np.isfinite(f["right_elbow_angle"])
                       else np.nan)

    if ls is not None and rs is not None and head is not None:
        smid = (ls + rs) / 2.0
        hdiff = head - smid
        f["head_dx"], f["head_dy"], f["head_dz"] = float(hdiff[0]), float(hdiff[1]), float(hdiff[2])
    else:
        f["head_dx"] = f["head_dy"] = f["head_dz"] = np.nan

    if head is not None:
        vec = head / (np.linalg.norm(head) + 1e-9)
        f["trunk_tilt_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(vec, [0.0, 1.0, 0.0]), -1, 1))))
    else:
        f["trunk_tilt_deg"] = np.nan

    # ── 3D-only features ───────────────────────────────────────────────────────
    # 1. Spine bend: angle at Spine node (Hips(=0)→Spine→Spine2)
    hips_n = norm.get("Hips")  # = [0,0,0] after normalisation
    f["spine_bend_deg"] = angle_3d(
        hips_n if hips_n is not None else np.nan,
        sp1    if sp1   is not None else np.nan,
        sp2    if sp2   is not None else np.nan)

    # 2. Lateral lean: signed angle of trunk projected onto coronal (XY) plane
    if head is not None:
        vec = head / (np.linalg.norm(head) + 1e-9)
        f["lateral_lean_deg"] = float(np.degrees(np.arctan2(vec[0], vec[1])))
    else:
        f["lateral_lean_deg"] = np.nan

    # 3. Head forward lean: angle at Neck (Spine3→Neck→Head)
    f["head_forward_deg"] = angle_3d(
        sp3  if sp3  is not None else np.nan,
        neck if neck is not None else np.nan,
        head if head is not None else np.nan)

    # 4. Pelvis height: raw Hips Y / spine length (absolute elevation, e.g. sitting vs standing)
    hip_raw = raw.get("Hips")
    f["pelvis_height_norm"] = float(hip_raw[1] / spine) if (hip_raw is not None and np.isfinite(spine)) else np.nan

    # 5. Foot spread: L-R foot distance normalised by shoulder width
    if lft is not None and rft is not None and np.isfinite(sw) and sw > 1e-6:
        f["foot_spread_norm"] = float(np.linalg.norm(lft - rft) / sw)
    else:
        f["foot_spread_norm"] = np.nan

    # 6. Hand depth difference: |Z_Lwrist - Z_Rwrist|  (forward/backward reach asymmetry)
    if lh is not None and rh is not None:
        f["hand_depth_diff"] = float(abs(lh[2] - rh[2]))
        f["wrist_z_asym"]    = float(lh[2] - rh[2])
    else:
        f["hand_depth_diff"] = f["wrist_z_asym"] = np.nan

    # 7. Knee bend asymmetry
    f["knee_bend_asym"] = (abs(f["left_knee_angle"] - f["right_knee_angle"])
                           if np.isfinite(f["left_knee_angle"]) and np.isfinite(f["right_knee_angle"])
                           else np.nan)

    # 8. Total body extent: bounding-box diagonal of all normalised joints
    all_pts = np.stack(list(norm.values()))
    mn, mx = all_pts.min(0), all_pts.max(0)
    f["body_extent"] = float(np.linalg.norm(mx - mn))

    return f


def aggregate_file_features(bvh_path: str, subsample: int = 30) -> Dict[str, float]:
    """Extract per-frame geometry and aggregate to file-level stats (mean/std/range)."""
    parser = BVHParser(bvh_path)
    n_frames = len(parser.frames)
    indices = list(range(0, n_frames, subsample)) if subsample > 0 else list(range(n_frames))

    records: List[Dict[str, float]] = []
    for i in indices:
        coords = parser.get_joint_world_coords(i)
        feats = extract_frame_features(coords)
        if feats:
            records.append(feats)

    if not records:
        return {}

    df_frames = pd.DataFrame(records)
    agg: Dict[str, float] = {}
    for col in df_frames.columns:
        vals = df_frames[col].dropna()
        if len(vals) == 0:
            agg[f"{col}_mean"] = agg[f"{col}_std"] = agg[f"{col}_range"] = np.nan
        else:
            agg[f"{col}_mean"]  = float(vals.mean())
            agg[f"{col}_std"]   = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            agg[f"{col}_range"] = float(vals.max() - vals.min())
    return agg


def load_temporal_features(csv_path: Path) -> pd.DataFrame:
    """Load and aggregate per-frame velocity CSV to file-level stats."""
    if not csv_path.exists():
        print(f"[WARN] Temporal CSV not found: {csv_path}")
        return pd.DataFrame()
    vel_cols = ["avg_velocity","head_vel","l_shoulder_vel","r_shoulder_vel",
                "l_elbow_vel","r_elbow_vel","l_wrist_vel","r_wrist_vel"]
    tdf = pd.read_csv(csv_path, usecols=["filename","emotion"] + vel_cols)
    agg = tdf.groupby(["filename","emotion"], as_index=False).agg(
        {c: ["mean","std","max"] for c in vel_cols})
    agg.columns = (["filename","emotion"] +
                   [f"{c}_{s}" for c in vel_cols for s in ("mean","std","max")])
    return agg


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_csv = OUT_DIR / "bvh_geom_features.csv"

    info_df = pd.read_csv(DATASET_ROOT / "file-info.csv")
    bvh_dir = DATASET_ROOT / "BVH"

    # ── Step 1: extract / load geometry features (all frames, subsample=30) ───
    if feat_csv.exists():
        print(f"[+] Load existing geometry features: {feat_csv}")
        df = pd.read_csv(feat_csv)
    else:
        rows: List[dict] = []
        print("[+] Extracting 3D geometry features (all frames, subsample every 30) ...")
        for _, row in tqdm(info_df.iterrows(), total=len(info_df)):
            fname   = row["filename"]
            emotion = row["emotion"]
            actor   = row["actor_ID"]
            bvh_path = bvh_dir / actor / f"{fname}.bvh"
            if not bvh_path.exists():
                continue
            try:
                feats = aggregate_file_features(str(bvh_path), subsample=30)
                if not feats:
                    continue
                feats["emotion"]  = emotion
                feats["filename"] = fname
                feats["actor"]    = actor
                rows.append(feats)
            except Exception as e:
                print(f"[WARN] skip {fname}: {e}")
        df = pd.DataFrame(rows)
        df.to_csv(feat_csv, index=False, encoding="utf-8-sig")
        print(f"[+] Saved {len(df)} file-level rows -> {feat_csv}")

    # ── Step 2: merge temporal velocity stats ─────────────────────────────────
    tdf = load_temporal_features(TEMPORAL_FEAT_CSV)
    if not tdf.empty:
        df = df.merge(tdf, on=["filename","emotion"], how="left")
        print(f"[+] Merged temporal features; df shape: {df.shape}")

    # ── Step 3: define feature sets ───────────────────────────────────────────
    GEOM_2D_LIKE = [c for c in df.columns
                    if any(c.startswith(p) for p in [
                        "shoulder_width","left_hand_height","right_hand_height",
                        "arm_span_norm","left_elbow_angle","right_elbow_angle",
                        "left_knee_angle","right_knee_angle","contraction",
                        "hand_height_asym","elbow_asym","head_dx","head_dy",
                        "head_dz","trunk_tilt_deg"])
                    and c not in ("emotion","filename","actor")]

    GEOM_3D_ONLY = [c for c in df.columns
                    if any(c.startswith(p) for p in [
                        "spine_bend_deg","lateral_lean_deg","head_forward_deg",
                        "pelvis_height_norm","foot_spread_norm","hand_depth_diff",
                        "wrist_z_asym","knee_bend_asym","body_extent"])
                    and c not in ("emotion","filename","actor")]

    VEL_FEATS = [c for c in df.columns
                 if any(c.startswith(p) for p in [
                     "avg_velocity","head_vel","l_shoulder_vel","r_shoulder_vel",
                     "l_elbow_vel","r_elbow_vel","l_wrist_vel","r_wrist_vel"])
                 and c not in ("emotion","filename","actor")]

    ALL_FEATS = GEOM_2D_LIKE + GEOM_3D_ONLY + VEL_FEATS
    print(f"[+] Features – 2D-like: {len(GEOM_2D_LIKE)}, 3D-only: {len(GEOM_3D_ONLY)}, "
          f"velocity: {len(VEL_FEATS)}, total: {len(ALL_FEATS)}")

    df_clean = df.dropna(subset=ALL_FEATS, how="all").copy()
    for c in ALL_FEATS:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    print(f"[+] Clean samples: {len(df_clean)}")

    # ── Step 4: Kruskal-Wallis per feature ────────────────────────────────────
    kw_rows = []
    for feat in ALL_FEATS:
        if feat not in df_clean.columns:
            continue
        groups = [g[feat].values for _, g in df_clean.groupby("emotion")]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        try:
            H, p = kruskal(*groups)
            kw_rows.append({"feature": feat, "H": H, "p": p,
                             "category": ("2D-like" if feat in GEOM_2D_LIKE
                                          else "3D-only" if feat in GEOM_3D_ONLY
                                          else "velocity")})
        except Exception:
            pass
    kw_df = pd.DataFrame(kw_rows).sort_values("H", ascending=False)
    kw_df.to_csv(OUT_DIR / "kruskal_results.csv", index=False)
    print("[+] Saved kruskal_results.csv")
    print("\n=== Top-20 features by KW H ===")
    print(kw_df.head(20).to_string(index=False))

    # ── Step 5: PCA (standardised) ────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    X_all = df_clean[ALL_FEATS].values
    X_scaled = StandardScaler().fit_transform(X_all)

    pca = PCA(n_components=2, random_state=42)
    pc  = pca.fit_transform(X_scaled)
    pd.DataFrame({"pc1": pc[:,0], "pc2": pc[:,1],
                  "emotion": df_clean["emotion"].values}).to_csv(
        OUT_DIR / "pca_2d.csv", index=False)
    print(f"[+] PCA explained_var: {pca.explained_variance_ratio_}")

    # ── Step 6: t-SNE ─────────────────────────────────────────────────────────
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    ts   = tsne.fit_transform(X_scaled)
    pd.DataFrame({"tsne1": ts[:,0], "tsne2": ts[:,1],
                  "emotion": df_clean["emotion"].values}).to_csv(
        OUT_DIR / "tsne_2d.csv", index=False)
    print("[+] Saved tsne_2d.csv")

    # ── Step 7: RF with actor-level group split ────────────────────────────────
    from sklearn.model_selection import GroupShuffleSplit
    y      = df_clean["emotion"].values
    groups = df_clean["actor"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_scaled, y, groups))
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"[+] RF split – train actors: {len(set(groups[train_idx]))}, "
          f"test actors: {len(set(groups[test_idx]))}, test N={len(y_te)}")

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    rpt    = classification_report(y_te, y_pred, output_dict=True)
    fi     = sorted(zip(ALL_FEATS, clf.feature_importances_),
                    key=lambda x: x[1], reverse=True)

    with open(OUT_DIR / "rf_report.json", "w", encoding="utf-8") as fh:
        json.dump({"accuracy": acc, "report": rpt, "feature_importance": fi},
                  fh, ensure_ascii=False, indent=2)

    print(f"\n=== RF (actor-stratified) accuracy={acc:.3f} ===")
    for emo in ["Angry","Disgust","Fearful","Happy","Neutral","Sad","Surprise"]:
        if emo in rpt:
            r = rpt[emo]
            print(f"  {emo:<10} P={r['precision']:.2f} R={r['recall']:.2f} "
                  f"F1={r['f1-score']:.2f}  N={int(r['support'])}")
    print(f"  macro-avg F1={rpt['macro avg']['f1-score']:.3f}")
    print("Top-10 features:", [f[0] for f in fi[:10]])
    print(f"\n[✓] All outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
