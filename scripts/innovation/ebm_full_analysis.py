"""
EBM (Emotional Body Motion) Full Analysis Pipeline
====================================================
Mirrors the 3D BVH analysis pipeline exactly:
  1. Geometric feature extraction (2D-like + 3D-only)  → per-file aggregation
  2. Temporal velocity features (per-frame joint velocities → file-level stats)
  3. Dynamic kinetic energy analysis (energy profiles, FFT, burst count)
  4. Kruskal-Wallis discriminability for all features
  5. PCA / t-SNE dimensionality reduction
  6. Random Forest classification (actor-stratified GroupShuffleSplit)
  7. Visualisation: effect sizes, RF summary, energy analysis (4 figures)

EBM dataset: 4,060 CSVs, 29 actors × 4 scenarios × 5 takes × 7 emotions
  - 150 frames per file, world coordinates (metres), 19 joints
  - Filename: {actor}_{scenario}_{take}_{emotion}.csv
  - Label mapping: 1=Angry, 2=Disgust, 3=Fearful, 4=Happy, 5=Neutral, 6=Sad, 7=Surprise

Differences from BVH:
  - No FK required (already world coordinate CSVs)
  - No Spine2 / Spine3 joints → spine_bend_deg uses Spine1, head_forward_deg uses Spine1
  - FPS assumed 30 Hz (150 frames / ~5s clips)
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from scipy.stats import kruskal
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
EBM_DIR  = Path("data/raw/Emotional Body Motion Data/Emotional Body Motion Data")
OUT_DIR  = Path("outputs/analysis/ebm_full")
FIGS_DIR = Path("docs/figs_ebm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── label mapping (aligned with BVH) ──────────────────────────────────────────
EMO_MAP = {1: "Angry", 2: "Disgust", 3: "Fearful", 4: "Happy",
           5: "Neutral", 6: "Sad", 7: "Surprise"}
EMOTIONS = ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprise"]
EMO_COLS = {
    "Angry": "#e74c3c", "Disgust": "#8e44ad", "Fearful": "#3498db",
    "Happy": "#f39c12", "Neutral": "#95a5a6", "Sad": "#2980b9",
    "Surprise": "#27ae60",
}

# 19 joints available in EBM
JOINTS = ["Hips", "Spine", "Spine1", "Neck", "Head",
          "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
          "RightShoulder", "RightArm", "RightForeArm", "RightHand",
          "RightUpLeg", "RightLeg", "RightFoot",
          "LeftUpLeg", "LeftLeg", "LeftFoot"]

FPS = 30.0  # assumed frame rate

# ── geometry helpers ───────────────────────────────────────────────────────────

def angle_3d(a, b, c):
    """Angle at vertex B (degrees)."""
    if a is None or b is None or c is None:
        return np.nan
    ba, bc = a - b, c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1, 1))))


def normalize_coords(raw: Dict[str, np.ndarray]):
    """Center on Hips, scale by Head-Hips spine length."""
    hip = raw.get("Hips")
    head = raw.get("Head")
    if hip is None or head is None:
        return {}, np.nan
    spine = np.linalg.norm(head - hip)
    if spine < 1e-8:
        return {}, np.nan
    return {k: (v - hip) / spine for k, v in raw.items()}, spine


def contraction_index_3d(norm_coords):
    use = ["Head", "LeftShoulder", "RightShoulder", "LeftForeArm", "RightForeArm",
           "LeftHand", "RightHand", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"]
    pts = [norm_coords[n] for n in use if n in norm_coords]
    if len(pts) < 5:
        return np.nan
    pts = np.stack(pts)
    torso_keys = {"LeftShoulder", "RightShoulder", "LeftUpLeg", "RightUpLeg"}
    if torso_keys.issubset(norm_coords.keys()):
        tc = sum(norm_coords[k] for k in torso_keys) / 4.0
    else:
        tc = pts.mean(axis=0)
    return float(-np.mean(np.linalg.norm(pts - tc[None, :], axis=1)))


def extract_frame_features(raw: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Extract all geometry features from a single EBM frame (world coords)."""
    norm, spine = normalize_coords(raw)
    if not norm:
        return {}

    f = {}
    ls  = norm.get("LeftShoulder");  rs  = norm.get("RightShoulder")
    lh  = norm.get("LeftHand");      rh  = norm.get("RightHand")
    lf  = norm.get("LeftForeArm");   rf  = norm.get("RightForeArm")
    la  = norm.get("LeftArm");       ra  = norm.get("RightArm")
    lu  = norm.get("LeftUpLeg");     ru  = norm.get("RightUpLeg")
    ll  = norm.get("LeftLeg");       rl  = norm.get("RightLeg")
    lft = norm.get("LeftFoot");      rft = norm.get("RightFoot")
    head = norm.get("Head");         neck = norm.get("Neck")
    sp1  = norm.get("Spine");        sp2  = norm.get("Spine1")  # EBM has Spine1 not Spine2

    # ── 2D-equivalent features ─────────────────────────────────────────────────
    sw = float(np.linalg.norm(ls - rs)) if (ls is not None and rs is not None) else np.nan
    f["shoulder_width"] = sw

    f["left_hand_height"]  = float(lh[1] - ls[1]) if (lh is not None and ls is not None) else np.nan
    f["right_hand_height"] = float(rh[1] - rs[1]) if (rh is not None and rs is not None) else np.nan

    if lh is not None and rh is not None and np.isfinite(sw) and sw > 1e-6:
        f["arm_span_norm"] = float(abs(lh[0] - rh[0]) / sw)
    else:
        f["arm_span_norm"] = np.nan

    # Elbow angles: LeftShoulder→LeftForeArm→LeftHand  (using Arm joint as shoulder-side)
    f["left_elbow_angle"]  = angle_3d(la, lf, lh)
    f["right_elbow_angle"] = angle_3d(ra, rf, rh)
    f["left_knee_angle"]   = angle_3d(lu, ll, lft)
    f["right_knee_angle"]  = angle_3d(ru, rl, rft)

    f["contraction"] = contraction_index_3d(norm)

    f["hand_height_asym"] = (abs(f["left_hand_height"] - f["right_hand_height"])
                              if np.isfinite(f.get("left_hand_height", np.nan))
                              and np.isfinite(f.get("right_hand_height", np.nan))
                              else np.nan)
    f["elbow_asym"] = (abs(f["left_elbow_angle"] - f["right_elbow_angle"])
                       if np.isfinite(f.get("left_elbow_angle", np.nan))
                       and np.isfinite(f.get("right_elbow_angle", np.nan))
                       else np.nan)

    if ls is not None and rs is not None and head is not None:
        smid = (ls + rs) / 2.0
        hdiff = head - smid
        f["head_dx"], f["head_dy"], f["head_dz"] = float(hdiff[0]), float(hdiff[1]), float(hdiff[2])
    else:
        f["head_dx"] = f["head_dy"] = f["head_dz"] = np.nan

    if head is not None:
        vec = head / (np.linalg.norm(head) + 1e-9)
        f["trunk_tilt_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(vec, [0, 1, 0]), -1, 1))))
    else:
        f["trunk_tilt_deg"] = np.nan

    # ── 3D-only features ───────────────────────────────────────────────────────
    hips_n = norm.get("Hips")
    # spine_bend_deg: Hips→Spine→Spine1 (EBM has no Spine2, use Spine1)
    f["spine_bend_deg"] = angle_3d(hips_n, sp1, sp2)

    if head is not None:
        vec = head / (np.linalg.norm(head) + 1e-9)
        f["lateral_lean_deg"] = float(np.degrees(np.arctan2(vec[0], vec[1])))
    else:
        f["lateral_lean_deg"] = np.nan

    # head_forward_deg: Spine1→Neck→Head (EBM has no Spine3, use Spine1)
    f["head_forward_deg"] = angle_3d(sp2, neck, head)

    hip_raw = raw.get("Hips")
    f["pelvis_height_norm"] = float(hip_raw[1] / spine) if (hip_raw is not None and np.isfinite(spine)) else np.nan

    if lft is not None and rft is not None and np.isfinite(sw) and sw > 1e-6:
        f["foot_spread_norm"] = float(np.linalg.norm(lft - rft) / sw)
    else:
        f["foot_spread_norm"] = np.nan

    if lh is not None and rh is not None:
        f["hand_depth_diff"] = float(abs(lh[2] - rh[2]))
        f["wrist_z_asym"]    = float(lh[2] - rh[2])
    else:
        f["hand_depth_diff"] = f["wrist_z_asym"] = np.nan

    f["knee_bend_asym"] = (abs(f["left_knee_angle"] - f["right_knee_angle"])
                           if np.isfinite(f.get("left_knee_angle", np.nan))
                           and np.isfinite(f.get("right_knee_angle", np.nan))
                           else np.nan)

    all_pts = np.stack(list(norm.values()))
    mn, mx = all_pts.min(0), all_pts.max(0)
    f["body_extent"] = float(np.linalg.norm(mx - mn))

    return f


# ── file I/O helpers ──────────────────────────────────────────────────────────

def parse_filename(fname: str):
    """Parse {actor}_{scenario}_{take}_{emotion}.csv → actor, emotion label."""
    parts = Path(fname).stem.split("_")
    if len(parts) != 4:
        return None, None, None
    actor = int(parts[0])
    emo_id = int(parts[3])
    emo = EMO_MAP.get(emo_id, None)
    return actor, emo, Path(fname).stem


def read_frame_coords(row: pd.Series) -> Dict[str, np.ndarray]:
    """Convert one row of an EBM CSV to joint-name → xyz dict."""
    coords = {}
    for j in JOINTS:
        x = row.get(f"{j}.x", np.nan)
        y = row.get(f"{j}.y", np.nan)
        z = row.get(f"{j}.z", np.nan)
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            coords[j] = np.array([x, y, z], dtype=np.float64)
    return coords


def aggregate_file_geom(csv_path: Path, subsample: int = 5) -> Dict[str, float]:
    """Read CSV, extract per-frame geometry → file-level mean/std/range."""
    df = pd.read_csv(csv_path)
    n = len(df)
    indices = list(range(0, n, subsample)) if subsample > 0 else list(range(n))

    records = []
    for i in indices:
        raw = read_frame_coords(df.iloc[i])
        feats = extract_frame_features(raw)
        if feats:
            records.append(feats)
    if not records:
        return {}

    df_f = pd.DataFrame(records)
    agg = {}
    for col in df_f.columns:
        vals = df_f[col].dropna()
        if len(vals) == 0:
            agg[f"{col}_mean"] = agg[f"{col}_std"] = agg[f"{col}_range"] = np.nan
        else:
            agg[f"{col}_mean"]  = float(vals.mean())
            agg[f"{col}_std"]   = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            agg[f"{col}_range"] = float(vals.max() - vals.min())
    return agg


# ── velocity / energy helpers ────────────────────────────────────────────────

def compute_velocity_and_energy(csv_path: Path, fps: float = 30.0):
    """Compute per-frame joint velocities and kinetic energy from world coords."""
    df = pd.read_csv(csv_path)
    n = len(df)
    if n < 2:
        return pd.DataFrame()

    # compute velocity for selected joints
    vel_joints = {
        "head_vel": "Head",
        "l_shoulder_vel": "LeftArm",
        "r_shoulder_vel": "RightArm",
        "l_elbow_vel": "LeftForeArm",
        "r_elbow_vel": "RightForeArm",
        "l_wrist_vel": "LeftHand",
        "r_wrist_vel": "RightHand",
    }
    vel_data = []
    for i in range(n):
        row_cur = df.iloc[i]
        if i == 0:
            # first frame: velocity = 0
            vel_row = {k: 0.0 for k in vel_joints}
            vel_row["avg_velocity"] = 0.0
        else:
            row_prev = df.iloc[i - 1]
            vels = {}
            for vname, jname in vel_joints.items():
                cur  = np.array([row_cur[f"{jname}.x"], row_cur[f"{jname}.y"], row_cur[f"{jname}.z"]])
                prev = np.array([row_prev[f"{jname}.x"], row_prev[f"{jname}.y"], row_prev[f"{jname}.z"]])
                vels[vname] = float(np.linalg.norm(cur - prev) * fps)
            vel_row = vels
            vel_row["avg_velocity"] = float(np.mean(list(vels.values())))
        vel_row["frame"] = i
        vel_data.append(vel_row)
    return pd.DataFrame(vel_data)


def aggregate_velocity_stats(vel_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate per-frame velocities to file-level mean/std/max."""
    vcols = ["avg_velocity", "head_vel", "l_shoulder_vel", "r_shoulder_vel",
             "l_elbow_vel", "r_elbow_vel", "l_wrist_vel", "r_wrist_vel"]
    agg = {}
    for c in vcols:
        if c in vel_df.columns:
            vals = vel_df[c].dropna()
            if len(vals) > 0:
                agg[f"{c}_mean"] = float(vals.mean())
                agg[f"{c}_std"]  = float(vals.std())
                agg[f"{c}_max"]  = float(vals.max())
            else:
                agg[f"{c}_mean"] = agg[f"{c}_std"] = agg[f"{c}_max"] = np.nan
    return agg


def compute_energy_features(vel_df: pd.DataFrame, fps: float = 30.0) -> Dict[str, float]:
    """Compute kinetic energy features from per-frame velocities."""
    vcols = ["avg_velocity", "head_vel", "l_shoulder_vel", "r_shoulder_vel",
             "l_elbow_vel", "r_elbow_vel", "l_wrist_vel", "r_wrist_vel"]
    arm_cols = ["l_shoulder_vel", "r_shoulder_vel", "l_elbow_vel", "r_elbow_vel",
                "l_wrist_vel", "r_wrist_vel"]
    head_cols = ["head_vel"]

    avail = [c for c in vcols if c in vel_df.columns]
    if not avail:
        return {}

    E = vel_df[avail].pow(2).sum(axis=1).values
    if len(E) < 5:
        return {}

    # arms share
    E_total = vel_df[avail].pow(2).sum(axis=1).clip(lower=1e-12)
    arm_avail = [c for c in arm_cols if c in vel_df.columns]
    head_avail = [c for c in head_cols if c in vel_df.columns]
    arms_share = vel_df[arm_avail].pow(2).sum(axis=1) / E_total if arm_avail else pd.Series(0, index=vel_df.index)
    head_share = vel_df[head_avail].pow(2).sum(axis=1) / E_total if head_avail else pd.Series(0, index=vel_df.index)

    # burst count
    thr = np.median(E) * 1.5
    peaks, _ = find_peaks(E, height=thr, distance=5)

    # dominant freq
    dom_freq = np.nan
    if len(E) >= 8:
        e_centered = E - E.mean()
        freqs = rfftfreq(len(e_centered), d=1.0 / fps)
        mags  = np.abs(rfft(e_centered))
        mags[0] = 0
        dom_freq = float(freqs[np.argmax(mags)])

    return {
        "E_mean": float(np.mean(E)),
        "E_std":  float(np.std(E)),
        "E_max":  float(np.max(E)),
        "E_min":  float(np.min(E)),
        "E_range": float(np.ptp(E)),
        "E_cv":    float(np.std(E) / (np.mean(E) + 1e-12)),
        "E_skew":  float(pd.Series(E).skew()),
        "burst_count": len(peaks),
        "dom_freq_hz": dom_freq,
        "arms_share_mean": float(arms_share.mean()),
        "head_share_mean": float(head_share.mean()),
    }


# ── main pipeline ─────────────────────────────────────────────────────────────

def extract_all_features():
    """Phase 1: Extract geometry + velocity + energy features for all files."""
    feat_csv = OUT_DIR / "ebm_all_features.csv"
    if feat_csv.exists():
        print(f"[+] Loading existing features: {feat_csv}")
        return pd.read_csv(feat_csv)

    csv_files = sorted(EBM_DIR.glob("*.csv"))
    print(f"[+] Found {len(csv_files)} EBM CSV files")

    rows = []
    for fp in tqdm(csv_files, desc="EBM feature extraction"):
        actor, emo, stem = parse_filename(fp.name)
        if emo is None:
            continue

        try:
            # geometry
            geom = aggregate_file_geom(fp, subsample=5)
            if not geom:
                continue

            # velocity
            vel_df = compute_velocity_and_energy(fp, fps=FPS)
            vel_stats = aggregate_velocity_stats(vel_df) if not vel_df.empty else {}

            # energy
            energy_feats = compute_energy_features(vel_df, fps=FPS) if not vel_df.empty else {}

            rec = {"filename": stem, "emotion": emo, "actor": actor}
            rec.update(geom)
            rec.update(vel_stats)
            rec.update(energy_feats)
            rows.append(rec)
        except Exception as e:
            print(f"[WARN] skip {fp.name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(feat_csv, index=False, encoding="utf-8-sig")
    print(f"[+] Saved {len(df)} rows → {feat_csv}")
    return df


def define_feature_sets(df):
    """Define 2D-like, 3D-only, velocity, and energy feature sets."""
    meta = {"emotion", "filename", "actor"}

    GEOM_2D_LIKE = [c for c in df.columns
                    if any(c.startswith(p) for p in [
                        "shoulder_width", "left_hand_height", "right_hand_height",
                        "arm_span_norm", "left_elbow_angle", "right_elbow_angle",
                        "left_knee_angle", "right_knee_angle", "contraction",
                        "hand_height_asym", "elbow_asym", "head_dx", "head_dy",
                        "head_dz", "trunk_tilt_deg"])
                    and c not in meta]

    GEOM_3D_ONLY = [c for c in df.columns
                    if any(c.startswith(p) for p in [
                        "spine_bend_deg", "lateral_lean_deg", "head_forward_deg",
                        "pelvis_height_norm", "foot_spread_norm", "hand_depth_diff",
                        "wrist_z_asym", "knee_bend_asym", "body_extent"])
                    and c not in meta]

    VEL_FEATS = [c for c in df.columns
                 if any(c.startswith(p) for p in [
                     "avg_velocity", "head_vel", "l_shoulder_vel", "r_shoulder_vel",
                     "l_elbow_vel", "r_elbow_vel", "l_wrist_vel", "r_wrist_vel"])
                 and c not in meta]

    ENERGY_FEATS = [c for c in df.columns
                    if any(c.startswith(p) for p in [
                        "E_mean", "E_std", "E_max", "E_min", "E_range", "E_cv",
                        "E_skew", "burst_count", "dom_freq_hz",
                        "arms_share_mean", "head_share_mean"])
                    and c not in meta]

    return GEOM_2D_LIKE, GEOM_3D_ONLY, VEL_FEATS, ENERGY_FEATS


def run_kruskal_wallis(df, all_feats, geom_2d, geom_3d, vel_feats, energy_feats):
    """Kruskal-Wallis test per feature."""
    kw_rows = []
    for feat in all_feats:
        if feat not in df.columns:
            continue
        groups = [g[feat].values for _, g in df.groupby("emotion")]
        groups = [g[g == g] for g in groups]  # remove NaN
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        try:
            H, p = kruskal(*groups)
            N, k = sum(len(g) for g in groups), len(groups)
            eps2 = max(0, (H - k + 1) / (N - k))
            cat = ("2D-like" if feat in geom_2d
                   else "3D-only" if feat in geom_3d
                   else "velocity" if feat in vel_feats
                   else "energy")
            kw_rows.append({"feature": feat, "H": H, "p": p, "eps2": eps2, "category": cat})
        except Exception:
            pass
    kw_df = pd.DataFrame(kw_rows).sort_values("H", ascending=False)
    kw_df.to_csv(OUT_DIR / "kruskal_results.csv", index=False)
    return kw_df


def run_dim_reduction(X_scaled, emotions, out_dir):
    """PCA + t-SNE and save results."""
    pca = PCA(n_components=2, random_state=42)
    pc  = pca.fit_transform(X_scaled)
    pd.DataFrame({"pc1": pc[:, 0], "pc2": pc[:, 1], "emotion": emotions}).to_csv(
        out_dir / "pca_2d.csv", index=False)
    print(f"[+] PCA explained_var: {pca.explained_variance_ratio_}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    ts   = tsne.fit_transform(X_scaled)
    pd.DataFrame({"tsne1": ts[:, 0], "tsne2": ts[:, 1], "emotion": emotions}).to_csv(
        out_dir / "tsne_2d.csv", index=False)
    print("[+] Saved PCA + t-SNE")
    return pc, ts


def run_rf_classification(X_scaled, y, groups, all_feats):
    """Random Forest with actor-stratified GroupShuffleSplit."""
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
    fi     = sorted(zip(all_feats, clf.feature_importances_),
                    key=lambda x: x[1], reverse=True)
    cm     = confusion_matrix(y_te, y_pred, labels=EMOTIONS)

    result = {"accuracy": acc, "report": rpt,
              "feature_importance": fi, "confusion_matrix": cm.tolist(),
              "y_te": y_te.tolist(), "y_pred": y_pred.tolist()}

    with open(OUT_DIR / "rf_report.json", "w", encoding="utf-8") as fh:
        json.dump({"accuracy": acc, "report": rpt, "feature_importance": fi},
                  fh, ensure_ascii=False, indent=2)

    print(f"\n=== RF (actor-stratified) accuracy={acc:.3f} ===")
    for emo in EMOTIONS:
        if emo in rpt:
            r = rpt[emo]
            print(f"  {emo:<10} P={r['precision']:.2f} R={r['recall']:.2f} "
                  f"F1={r['f1-score']:.2f}  N={int(r['support'])}")
    print(f"  macro-avg F1={rpt['macro avg']['f1-score']:.3f}")
    print(f"  Top-10 features: {[f[0] for f in fi[:10]]}")
    return result


# ── visualisation functions ───────────────────────────────────────────────────

def plot_effect_sizes(kw_df: pd.DataFrame, df: pd.DataFrame, all_feats):
    """3-panel effect size figure (matches BVH version)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8),
                             gridspec_kw={"width_ratios": [1, 1, 1.2], "wspace": 0.38})
    fig.suptitle("EBM – Feature Discriminability (Kruskal-Wallis Effect Sizes)",
                 fontsize=13, fontweight="bold", y=0.98)

    # Panel A: lollipop chart (top-20 by ε²)
    ax = axes[0]
    top = kw_df.nlargest(20, "eps2").sort_values("eps2")
    cat_colors = {"2D-like": "#3498db", "3D-only": "#e74c3c", "velocity": "#27ae60", "energy": "#f39c12"}
    colors = [cat_colors.get(c, "#999") for c in top["category"]]
    ax.barh(range(len(top)), top["eps2"], color=colors, height=0.6, edgecolor="white")
    ax.set_yticks(range(len(top)))

    def shorten(s):
        return (s.replace("_mean", "μ").replace("_std", "σ").replace("_range", "Δ")
                 .replace("_max", "⬆").replace("left_", "L.").replace("right_", "R.")
                 .replace("_norm", "")).strip()
    ax.set_yticklabels([shorten(f) for f in top["feature"]], fontsize=7)
    ax.set_xlabel("ε² (effect size)", fontsize=9)
    ax.set_title("(A) Top-20 features by ε²", fontsize=10, fontweight="bold")
    ax.axvline(0.01, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax.axvline(0.06, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax.axvline(0.14, color="gray", ls="--", lw=0.6, alpha=0.5)
    patches = [mpatches.Patch(color=c, label=l) for l, c in cat_colors.items()]
    ax.legend(handles=patches, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: violin plot by category
    ax = axes[1]
    for cat, col in cat_colors.items():
        vals = kw_df[kw_df.category == cat]["eps2"].values
        if len(vals) == 0:
            continue
        parts = ax.violinplot([vals], positions=[list(cat_colors.keys()).index(cat)],
                              widths=0.6, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(col); body.set_alpha(0.6)
        parts["cmedians"].set_color("white")
    ax.set_xticks(range(len(cat_colors)))
    ax.set_xticklabels(cat_colors.keys(), fontsize=8, rotation=20)
    ax.set_ylabel("ε²", fontsize=9)
    ax.set_title("(B) Effect size distribution by category", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C: heatmap of large/medium effects
    ax = axes[2]
    large = kw_df[kw_df.eps2 >= 0.06].nlargest(15, "eps2")
    if len(large) > 0:
        feat_list = large["feature"].tolist()
        mat = np.zeros((len(feat_list), len(EMOTIONS)))
        for i, feat in enumerate(feat_list):
            if feat not in df.columns:
                continue
            for j, emo in enumerate(EMOTIONS):
                vals = df[df.emotion == emo][feat].dropna()
                mat[i, j] = vals.median() if len(vals) > 0 else np.nan
        # z-score rows
        for i in range(mat.shape[0]):
            row_std = np.nanstd(mat[i])
            if row_std > 1e-8:
                mat[i] = (mat[i] - np.nanmean(mat[i])) / row_std

        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(EMOTIONS)))
        ax.set_xticklabels(EMOTIONS, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(feat_list)))
        ax.set_yticklabels([shorten(f) for f in feat_list], fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_title("(C) Per-emotion median (z-scored)", fontsize=10, fontweight="bold")

    fig.savefig(FIGS_DIR / "ebm_effect_sizes.png", dpi=150, bbox_inches="tight")
    print(f"[✓] Saved → {FIGS_DIR}/ebm_effect_sizes.png")
    plt.close(fig)


def plot_rf_summary(rf_result, kw_df):
    """RF summary figure (confusion matrix + P/R/F1 + feature importance)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [1.1, 1, 1], "wspace": 0.35})
    fig.suptitle(f"EBM – RF Classification Summary (Acc={rf_result['accuracy']:.1%})",
                 fontsize=13, fontweight="bold", y=1.0)

    # Panel A: confusion matrix
    ax = axes[0]
    cm = np.array(rf_result["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            txt_col = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=7, color=txt_col)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("(A) Confusion Matrix", fontsize=10, fontweight="bold")

    # Panel B: per-emotion F1
    ax = axes[1]
    rpt = rf_result["report"]
    f1s = [rpt.get(e, {}).get("f1-score", 0) for e in EMOTIONS]
    bars = ax.barh(range(len(EMOTIONS)), f1s,
                   color=[EMO_COLS[e] for e in EMOTIONS], edgecolor="white")
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS, fontsize=8)
    ax.set_xlabel("F1 Score", fontsize=9)
    ax.set_xlim(0, 1)
    for i, v in enumerate(f1s):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_title("(B) Per-emotion F1", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C: top-15 feature importance
    ax = axes[2]
    fi = rf_result["feature_importance"][:15]
    names = [f[0] for f in fi][::-1]
    imps  = [f[1] for f in fi][::-1]

    def shorten(s):
        return (s.replace("_mean", "μ").replace("_std", "σ").replace("_range", "Δ")
                 .replace("_max", "⬆").replace("left_", "L.").replace("right_", "R.").replace("_norm", ""))

    ax.barh(range(len(names)), imps, color="#2980b9", edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([shorten(n) for n in names], fontsize=7)
    ax.set_xlabel("Importance", fontsize=9)
    ax.set_title("(C) Top-15 RF features", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIGS_DIR / "ebm_rf_summary.png", dpi=150, bbox_inches="tight")
    print(f"[✓] Saved → {FIGS_DIR}/ebm_rf_summary.png")
    plt.close(fig)


def plot_energy_analysis(df):
    """4-panel energy analysis figure (matches BVH version)."""
    energy_feats = ["E_mean", "E_std", "E_max", "E_range", "E_cv", "E_skew",
                    "burst_count", "dom_freq_hz", "arms_share_mean", "head_share_mean"]
    avail_e = [c for c in energy_feats if c in df.columns]
    if not avail_e:
        print("[WARN] No energy features available for energy plot")
        return

    N, k = len(df), 7
    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.35})
    fig.suptitle("EBM – Dynamic Kinetic Energy Analysis", fontsize=13, fontweight="bold", y=0.98)

    # Panel A: E_mean violin per emotion
    ax = axes[0, 0]
    data_v = [df[df.emotion == e]["E_mean"].dropna().values for e in EMOTIONS]
    parts  = ax.violinplot(data_v, positions=range(len(EMOTIONS)), widths=0.6,
                           showmedians=True, showextrema=True)
    for body, emo in zip(parts["bodies"], EMOTIONS):
        body.set_facecolor(EMO_COLS[emo]); body.set_alpha(0.65)
    parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(1.5)
    for key in ["cmins", "cmaxes", "cbars"]:
        if key in parts:
            parts[key].set_color("#888"); parts[key].set_linewidth(0.7)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Kinetic Energy", fontsize=9)
    ax.set_title("(A) Mean energy distribution per emotion", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: burst_count boxplot
    ax = axes[0, 1]
    data_b = [df[df.emotion == e]["burst_count"].dropna().values for e in EMOTIONS]
    bp = ax.boxplot(data_b, patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", linewidth=1.5))
    for patch, emo in zip(bp["boxes"], EMOTIONS):
        patch.set_facecolor(EMO_COLS[emo]); patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(EMOTIONS) + 1))
    ax.set_xticklabels(EMOTIONS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Burst count", fontsize=9)
    ax.set_title("(B) Energy burst count per emotion", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C: dominant frequency boxplot
    ax = axes[1, 0]
    data_f = [df[df.emotion == e]["dom_freq_hz"].dropna().values for e in EMOTIONS]
    bp = ax.boxplot(data_f, patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", linewidth=1.5))
    for patch, emo in zip(bp["boxes"], EMOTIONS):
        patch.set_facecolor(EMO_COLS[emo]); patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(EMOTIONS) + 1))
    ax.set_xticklabels(EMOTIONS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Dominant frequency (Hz)", fontsize=9)
    ax.set_title("(C) Energy oscillation frequency", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel D: KW H for energy features
    ax = axes[1, 1]
    kw_rows = []
    for feat in avail_e:
        groups = [df[df.emotion == e][feat].dropna().values for e in EMOTIONS]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        eps2 = max(0, (H - k + 1) / (N - k))
        kw_rows.append({"feature": feat, "H": H, "eps2": eps2})
    kw_e = pd.DataFrame(kw_rows).sort_values("H", ascending=True)
    colors_bar = ["#e74c3c" if e >= 0.14 else "#f39c12" if e >= 0.06 else "#3498db"
                  for e in kw_e["eps2"]]
    ax.barh(range(len(kw_e)), kw_e["H"], color=colors_bar, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(kw_e)))
    ax.set_yticklabels(kw_e["feature"], fontsize=8)
    for i, (h, e2) in enumerate(zip(kw_e["H"], kw_e["eps2"])):
        ax.text(h + 0.5, i, f"ε²={e2:.3f}", va="center", fontsize=7)
    ax.set_xlabel("Kruskal-Wallis H", fontsize=9)
    ax.set_title("(D) KW discriminability of energy features", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    patches = [mpatches.Patch(color="#e74c3c", label="large (≥0.14)"),
               mpatches.Patch(color="#f39c12", label="medium (≥0.06)"),
               mpatches.Patch(color="#3498db", label="small (<0.06)")]
    ax.legend(handles=patches, fontsize=7, loc="lower right")

    fig.savefig(FIGS_DIR / "ebm_energy_analysis.png", dpi=150, bbox_inches="tight")
    print(f"[✓] Saved → {FIGS_DIR}/ebm_energy_analysis.png")
    plt.close(fig)


def plot_dim_reduction(pc, ts, emotions):
    """PCA + t-SNE scatter plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("EBM – Dimensionality Reduction", fontsize=13, fontweight="bold")

    for ax, coords, title in [(axes[0], pc, "(A) PCA"), (axes[1], ts, "(B) t-SNE")]:
        for emo in EMOTIONS:
            mask = emotions == emo
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=EMO_COLS[emo], label=emo, s=8, alpha=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, markerscale=2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIGS_DIR / "ebm_dim_reduction.png", dpi=150, bbox_inches="tight")
    print(f"[✓] Saved → {FIGS_DIR}/ebm_dim_reduction.png")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EBM Full Analysis Pipeline")
    print("=" * 60)

    # Phase 1: feature extraction
    df = extract_all_features()
    print(f"[+] Total files: {len(df)}, Emotions: {df['emotion'].value_counts().to_dict()}")

    # Phase 2: define feature sets
    GEOM_2D, GEOM_3D, VEL, ENERGY = define_feature_sets(df)
    ALL_FEATS = GEOM_2D + GEOM_3D + VEL + ENERGY
    print(f"[+] Features – 2D-like: {len(GEOM_2D)}, 3D-only: {len(GEOM_3D)}, "
          f"velocity: {len(VEL)}, energy: {len(ENERGY)}, total: {len(ALL_FEATS)}")

    # Clean
    df_clean = df.dropna(subset=ALL_FEATS, how="all").copy()
    for c in ALL_FEATS:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    print(f"[+] Clean samples: {len(df_clean)}")

    # Phase 3: Kruskal-Wallis
    print("\n[+] Running Kruskal-Wallis tests ...")
    kw_df = run_kruskal_wallis(df_clean, ALL_FEATS, GEOM_2D, GEOM_3D, VEL, ENERGY)
    print(f"\n=== Top-20 features by KW H ===")
    print(kw_df.head(20).to_string(index=False))

    n_large  = (kw_df["eps2"] >= 0.14).sum()
    n_medium = ((kw_df["eps2"] >= 0.06) & (kw_df["eps2"] < 0.14)).sum()
    n_small  = (kw_df["eps2"] < 0.06).sum()
    print(f"\nEffect size summary: {n_large} large, {n_medium} medium, {n_small} small / negligible")

    # Phase 4: Scaling + dim reduction
    X_all = df_clean[ALL_FEATS].values
    X_scaled = StandardScaler().fit_transform(X_all)
    emo_arr = df_clean["emotion"].values

    print("\n[+] Running PCA + t-SNE ...")
    pc, ts = run_dim_reduction(X_scaled, emo_arr, OUT_DIR)

    # Phase 5: RF classification
    print("\n[+] Running Random Forest classification ...")
    groups = df_clean["actor"].values
    rf_result = run_rf_classification(X_scaled, emo_arr, groups, ALL_FEATS)

    # Phase 6: Visualisations
    print("\n[+] Generating visualisations ...")
    plot_effect_sizes(kw_df, df_clean, ALL_FEATS)
    plot_rf_summary(rf_result, kw_df)
    plot_energy_analysis(df_clean)
    plot_dim_reduction(pc, ts, emo_arr)

    # Summary
    print("\n" + "=" * 60)
    print("EBM ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Files processed   : {len(df_clean)}")
    print(f"  Feature dimensions : {len(ALL_FEATS)}")
    print(f"  RF accuracy        : {rf_result['accuracy']:.1%}")
    print(f"  RF macro-F1        : {rf_result['report']['macro avg']['f1-score']:.3f}")
    print(f"  KW large effects   : {n_large}")
    print(f"\nOutputs:")
    print(f"  {OUT_DIR}/ebm_all_features.csv")
    print(f"  {OUT_DIR}/kruskal_results.csv")
    print(f"  {OUT_DIR}/pca_2d.csv, tsne_2d.csv")
    print(f"  {OUT_DIR}/rf_report.json")
    print(f"  {FIGS_DIR}/ebm_effect_sizes.png")
    print(f"  {FIGS_DIR}/ebm_rf_summary.png")
    print(f"  {FIGS_DIR}/ebm_energy_analysis.png")
    print(f"  {FIGS_DIR}/ebm_dim_reduction.png")


if __name__ == "__main__":
    main()
