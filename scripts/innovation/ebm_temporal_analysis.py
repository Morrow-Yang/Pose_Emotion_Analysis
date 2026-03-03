"""
EBM Temporal Dynamics Analysis
================================
Extends the EBM pipeline with time-domain features that capture HOW motion
evolves across the 5-second clip, not just its average level.

Feature groups extracted (per file):
  A. Phase features     – 5 × 1-second segments, E & velocity per phase (10 features)
  B. Phase ratios       – each phase's energy share of the total (5 features)
  C. Shape descriptors  – peak time, rise time, sustain ratio, slope,
                          autocorrelation, jerk, energy CV (7 features)
  Total new features: 22 temporal features

Outputs:
  outputs/analysis/ebm_temporal/
    ebm_temporal_features.csv   – 4060 × (22+3) rows
    kruskal_temporal.csv        – KW results for temporal features
    temporal_curves.npz         – mean/std energy & velocity curves per emotion

  docs/figs_ebm/
    ebm_temporal_curves.png     – mean energy & velocity time curves (7 emotions)
    ebm_phase_heatmap.png       – phase energy heatmap (7 emos × 5 phases)
    ebm_temporal_effects.png    – KW effect sizes for temporal features
    ebm_temporal_rf_delta.png   – RF accuracy with vs without temporal features
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
EBM_DIR  = Path("data/raw/Emotional Body Motion Data/Emotional Body Motion Data")
OUT_DIR  = Path("outputs/analysis/ebm_temporal")
FIGS_DIR = Path("docs/figs_ebm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
EMO_MAP = {1: "Angry", 2: "Disgust", 3: "Fearful", 4: "Happy",
           5: "Neutral", 6: "Sad", 7: "Surprise"}
EMOTIONS = ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprise"]
EMO_COLS = {
    "Angry":    "#e74c3c",
    "Disgust":  "#8e44ad",
    "Fearful":  "#3498db",
    "Happy":    "#f39c12",
    "Neutral":  "#95a5a6",
    "Sad":      "#2980b9",
    "Surprise": "#27ae60",
}

# Velocity joints (matching ebm_full_analysis.py exactly)
VEL_JOINTS = {
    "head_vel":       "Head",
    "l_shoulder_vel": "LeftArm",
    "r_shoulder_vel": "RightArm",
    "l_elbow_vel":    "LeftForeArm",
    "r_elbow_vel":    "RightForeArm",
    "l_wrist_vel":    "LeftHand",
    "r_wrist_vel":    "RightHand",
}

FPS        = 30.0
N_FRAMES   = 150        # fixed clip length (frames 0-149)
N_VEL      = N_FRAMES - 1  # 149 velocity frames (frames 1-149)
N_PHASES   = 5
PHASE_LEN  = N_VEL // N_PHASES          # 29 frames per phase
N_INTERP   = 100        # resample length for average curves

COHEN_LARGE  = 0.14
COHEN_MEDIUM = 0.06
COHEN_SMALL  = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-file feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_velocities(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return dict: vel_name → 1-D array of length (N-1) velocities (m/s)."""
    n = len(df)
    result = {}
    for vname, jname in VEL_JOINTS.items():
        cols = [f"{jname}.x", f"{jname}.y", f"{jname}.z"]
        if not all(c in df.columns for c in cols):
            continue
        xyz = df[cols].values.astype(float)  # (N, 3)
        diff = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * FPS  # (N-1,)
        result[vname] = diff
    # avg_velocity across all joints
    if result:
        result["avg_velocity"] = np.stack(list(result.values())).mean(axis=0)
    return result


def compute_energy_series(vel_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Kinetic energy  E(t) = Σ_j v_j(t)²  for each velocity frame."""
    if not vel_dict:
        return np.array([])
    # exclude avg_velocity from the sum to avoid double-counting
    joint_vels = [v for k, v in vel_dict.items() if k != "avg_velocity"]
    if not joint_vels:
        return np.array([])
    stacked = np.stack(joint_vels, axis=1)   # (N-1, n_joints)
    return (stacked ** 2).sum(axis=1)         # (N-1,)


def extract_temporal_features(csv_path: Path) -> Optional[Dict]:
    """Extract all 22 temporal features from a single EBM CSV file."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if len(df) < 10:
        return None

    vel_dict = compute_velocities(df)
    if not vel_dict or "avg_velocity" not in vel_dict:
        return None

    E   = compute_energy_series(vel_dict)   # (149,)
    vel = vel_dict["avg_velocity"]           # (149,)

    if len(E) < N_PHASES:
        return None

    feat: Dict = {}

    # ── A. Phase features ────────────────────────────────────────────────────
    # Split 149 frames into 5 phases of ~29-30 frames each
    phase_boundaries = np.array_split(np.arange(len(E)), N_PHASES)
    for i, idx in enumerate(phase_boundaries, start=1):
        feat[f"E_phase{i}"] = float(np.mean(E[idx]))
        feat[f"vel_phase{i}"] = float(np.mean(vel[idx]))

    # ── B. Phase energy ratios ───────────────────────────────────────────────
    E_total = np.sum(E) + 1e-12
    for i, idx in enumerate(phase_boundaries, start=1):
        feat[f"E_ratio_phase{i}"] = float(np.sum(E[idx]) / E_total)

    # ── C. Temporal shape descriptors ───────────────────────────────────────
    T = len(E)

    # C1. peak_time_norm: when does peak energy occur? (0=start, 1=end)
    feat["peak_time_norm"] = float(np.argmax(E) / (T - 1))

    # C2. energy_front_ratio: energy in first half vs total
    half = T // 2
    feat["energy_front_ratio"] = float(np.sum(E[:half]) / E_total)

    # C3. rise_time_norm: first frame to reach 80% of peak
    peak_val = np.max(E)
    above_80 = np.where(E >= 0.8 * peak_val)[0]
    feat["rise_time_norm"] = float(above_80[0] / (T - 1)) if len(above_80) > 0 else 1.0

    # C4. sustain_ratio: fraction of frames above 50% of peak
    feat["sustain_ratio"] = float(np.mean(E >= 0.5 * peak_val))

    # C5. energy_slope: linear trend coefficient (normalised by mean energy)
    t_norm = np.linspace(0, 1, T)
    slope = np.polyfit(t_norm, E, 1)[0]
    feat["energy_slope"] = float(slope / (np.mean(E) + 1e-12))

    # C6. energy_autocorr_lag1: smoothness of energy series
    E_z = E - E.mean()
    var = np.dot(E_z, E_z)
    if var > 1e-12:
        feat["energy_autocorr_lag1"] = float(np.dot(E_z[:-1], E_z[1:]) / var)
    else:
        feat["energy_autocorr_lag1"] = 0.0

    # C7. jerk_mean: mean |dv/dt| — measures abruptness of velocity changes
    jerk = np.abs(np.diff(vel)) * FPS
    feat["jerk_mean"] = float(np.mean(jerk))

    return feat


def parse_filename(fname: str):
    parts = Path(fname).stem.split("_")
    if len(parts) != 4:
        return None, None
    actor = int(parts[0])
    emo   = EMO_MAP.get(int(parts[3]), None)
    return actor, emo


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build temporal curve bank (mean ± std E(t) per emotion)
# ─────────────────────────────────────────────────────────────────────────────

def build_curve_bank(files: List[Path]) -> Dict[str, np.ndarray]:
    """
    For each emotion, return:
      curves[emo]['E_mean']   – (N_INTERP,) mean energy curve
      curves[emo]['E_std']    – (N_INTERP,) std  energy curve
      curves[emo]['vel_mean'] – (N_INTERP,) mean velocity curve
      curves[emo]['vel_std']  – (N_INTERP,) std  velocity curve
    """
    # bank[emo] = list of arrays of length N_VEL
    e_bank   = {e: [] for e in EMOTIONS}
    vel_bank = {e: [] for e in EMOTIONS}

    for fp in tqdm(files, desc="Building curve bank", ncols=80, leave=False):
        actor, emo = parse_filename(fp.name)
        if emo is None:
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        vel_dict = compute_velocities(df)
        if not vel_dict or "avg_velocity" not in vel_dict:
            continue
        E   = compute_energy_series(vel_dict)
        vel = vel_dict["avg_velocity"]
        if len(E) < 10:
            continue
        # interpolate to N_INTERP points
        x_orig  = np.linspace(0, 1, len(E))
        x_new   = np.linspace(0, 1, N_INTERP)
        e_interp   = np.interp(x_new, x_orig, E)
        vel_interp = np.interp(x_new, x_orig, vel)
        e_bank[emo].append(e_interp)
        vel_bank[emo].append(vel_interp)

    curves = {}
    for emo in EMOTIONS:
        if not e_bank[emo]:
            continue
        ea  = np.stack(e_bank[emo])    # (n_clips, N_INTERP)
        va  = np.stack(vel_bank[emo])
        curves[emo] = {
            "E_mean":   ea.mean(0),
            "E_std":    ea.std(0),
            "vel_mean": va.mean(0),
            "vel_std":  va.std(0),
            "n":        len(ea),
        }
    return curves


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kruskal-Wallis for temporal features
# ─────────────────────────────────────────────────────────────────────────────

def run_kruskal_temporal(feat_df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in feat_df.columns if c not in ("actor", "emotion")]
    rows = []
    groups = [feat_df[feat_df.emotion == e] for e in EMOTIONS]
    N = len(feat_df)
    k = len(EMOTIONS)
    for col in feat_cols:
        data = [g[col].dropna().values for g in groups]
        data = [d for d in data if len(d) > 0]
        if len(data) < 2:
            continue
        try:
            H, p = kruskal(*data)
        except Exception:
            continue
        eps2 = max(0.0, (H - k + 1) / (N - k))
        rows.append({"feature": col, "H": H, "p": p, "eps2": eps2})
    kw = pd.DataFrame(rows).sort_values("eps2", ascending=False).reset_index(drop=True)
    return kw


# ─────────────────────────────────────────────────────────────────────────────
# 4. RF comparison: baseline vs. baseline + temporal features
# ─────────────────────────────────────────────────────────────────────────────

def rf_comparison(feat_df: pd.DataFrame, base_feat_csv: Path) -> Dict:
    """Compare RF accuracy with and without temporal features."""
    print("[RF] Loading baseline features …")
    base = pd.read_csv(base_feat_csv)
    # merge on filename / actor + emotion + scenario + take
    # use actor + emotion as merge key after aligning filenames
    temp_feat_cols = [c for c in feat_df.columns if c not in ("actor", "emotion", "filename")]
    merged = base.merge(
        feat_df[["filename"] + temp_feat_cols],
        on="filename", how="inner", suffixes=("", "_t")
    )
    print(f"[RF] Merged: {len(merged)} rows (base={len(base)}, temporal={len(feat_df)})")

    actors  = merged["actor"].values.astype(int)
    labels  = merged["emotion"].values

    # pick feature columns (no NaN cols)
    base_cols = [c for c in base.columns
                 if c not in ("actor", "scenario", "emotion", "take", "filename")
                 and merged[c].notna().mean() > 0.5]
    temp_cols = [c for c in temp_feat_cols
                 if merged[c].notna().mean() > 0.5]

    results = {}
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    for tag, cols in [("baseline", base_cols),
                      ("baseline+temporal", base_cols + temp_cols)]:
        X = merged[cols].fillna(merged[cols].median())
        accs, f1s = [], []
        for tr, te in gss.split(X, labels, groups=actors):
            clf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                         random_state=42, class_weight="balanced")
            clf.fit(X.iloc[tr], labels[tr])
            pred = clf.predict(X.iloc[te])
            accs.append(accuracy_score(labels[te], pred))
            f1s.append(f1_score(labels[te], pred, average="macro"))
        results[tag] = {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                        "f1":  float(np.mean(f1s)),  "f1_std":  float(np.std(f1s))}
        print(f"  {tag:25s}  acc={results[tag]['acc']:.3f}±{results[tag]['acc_std']:.3f}"
              f"  F1={results[tag]['f1']:.3f}±{results[tag]['f1_std']:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_curves(curves: Dict, save_path: Path):
    """Figure 1: Mean energy & velocity time curves per emotion with CI band."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.linspace(0, 5, N_INTERP)   # 0-5 seconds

    for emo in EMOTIONS:
        if emo not in curves:
            continue
        c  = curves[emo]
        col = EMO_COLS[emo]
        n   = c["n"]
        se  = c["E_std"] / np.sqrt(n)

        axes[0].plot(x, c["E_mean"], color=col, lw=2, label=emo)
        axes[0].fill_between(x,
                              c["E_mean"] - se,
                              c["E_mean"] + se,
                              color=col, alpha=0.12)

        axes[1].plot(x, c["vel_mean"], color=col, lw=2, label=emo)
        axes[1].fill_between(x,
                              c["vel_mean"] - c["vel_std"] / np.sqrt(n),
                              c["vel_mean"] + c["vel_std"] / np.sqrt(n),
                              color=col, alpha=0.12)

    # Phase dividers (every 1 second)
    for ax, title, ylabel in [
        (axes[0], "Mean Kinetic Energy  E(t) = Σ v²", "Energy (m²/s²)"),
        (axes[1], "Mean Average Joint Velocity",       "Velocity (m/s)")
    ]:
        for t_phase in [1, 2, 3, 4]:
            ax.axvline(t_phase, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, 5)

    # Phase labels drawn at top of each panel using axes-fraction y coords
    from matplotlib.transforms import blended_transform_factory
    for ax in axes:
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for t_label, t_x in [("Ph.1", 0.5), ("Ph.2", 1.5), ("Ph.3", 2.5),
                              ("Ph.4", 3.5), ("Ph.5", 4.5)]:
            ax.text(t_x, 0.97, t_label, ha="center", va="top",
                    fontsize=7, color="gray", transform=trans)

    handles, labels_l = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_l, loc="lower center", ncol=7,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("EBM Temporal Dynamics: Energy & Velocity Curves by Emotion",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig] Saved: {save_path}")


def plot_phase_heatmap(feat_df: pd.DataFrame, save_path: Path):
    """Figure 2: Phase energy-ratio heatmap (7 emotions × 5 phases)."""
    ratio_cols = [f"E_ratio_phase{i}" for i in range(1, N_PHASES + 1)]
    matrix = np.zeros((len(EMOTIONS), N_PHASES))
    for r, emo in enumerate(EMOTIONS):
        sub = feat_df[feat_df.emotion == emo]
        for c, col in enumerate(ratio_cols):
            matrix[r, c] = sub[col].median()

    # Normalise each row so colours show within-emotion phase distribution
    matrix_norm = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.05})

    # left: absolute median energy per phase
    abs_matrix = np.zeros((len(EMOTIONS), N_PHASES))
    for r, emo in enumerate(EMOTIONS):
        sub = feat_df[feat_df.emotion == emo]
        for c in range(N_PHASES):
            abs_matrix[r, c] = sub[f"E_phase{c+1}"].median()

    for ax, data, title, fmt, cmap in [
        (axes[0], abs_matrix,   "Absolute: Median Energy per Phase",
         ".2f", "YlOrRd"),
        (axes[1], matrix_norm * 100, "Relative: Phase Energy Share (%)",
         ".1f", "Blues"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(N_PHASES))
        ax.set_xticklabels([f"Phase {i}\n({i-1}-{i}s)" for i in range(1, N_PHASES+1)],
                           fontsize=9)
        ax.set_yticks(range(len(EMOTIONS)))
        ax.set_yticklabels(EMOTIONS, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        for r in range(len(EMOTIONS)):
            for c in range(N_PHASES):
                ax.text(c, r, format(data[r, c], fmt),
                        ha="center", va="center", fontsize=8,
                        color="white" if data[r, c] > data.max() * 0.65 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("EBM Phase Energy Distribution  (5 × 1-second phases)",
                 fontsize=12, fontweight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig] Saved: {save_path}")


def plot_temporal_effects(kw: pd.DataFrame, save_path: Path):
    """Figure 3: KW effect size (ε²) bar chart for all temporal features."""
    kw_sorted = kw.sort_values("eps2", ascending=True)

    # colour bars by effect level
    def bar_color(eps2):
        if eps2 >= COHEN_LARGE:
            return "#e74c3c"
        elif eps2 >= COHEN_MEDIUM:
            return "#f39c12"
        elif eps2 >= COHEN_SMALL:
            return "#3498db"
        return "#bdc3c7"

    colours = [bar_color(v) for v in kw_sorted["eps2"]]
    y_pos   = np.arange(len(kw_sorted))

    fig, ax = plt.subplots(figsize=(9, max(5, len(kw_sorted) * 0.38)))
    bars = ax.barh(y_pos, kw_sorted["eps2"], color=colours, edgecolor="white",
                   height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(kw_sorted["feature"], fontsize=9)
    ax.set_xlabel("Effect size  ε²  (Kruskal-Wallis)", fontsize=11)
    ax.set_title("Temporal Feature Discriminability (7 Emotions)",
                 fontsize=12, fontweight="bold")

    # threshold lines
    for thr, label, ls in [(COHEN_LARGE,  "Large (0.14)",  "-"),
                            (COHEN_MEDIUM, "Medium (0.06)", "--"),
                            (COHEN_SMALL,  "Small (0.01)",  ":")]:
        ax.axvline(thr, color="gray", ls=ls, lw=1.2, alpha=0.7, label=label)

    # value annotations
    for bar, val in zip(bars, kw_sorted["eps2"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    # legend patches for colours
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#e74c3c", label="Large  ε²≥0.14"),
        Patch(color="#f39c12", label="Medium 0.06–0.14"),
        Patch(color="#3498db", label="Small  0.01–0.06"),
        Patch(color="#bdc3c7", label="Negligible <0.01"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, kw_sorted["eps2"].max() * 1.15 + 0.005)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig] Saved: {save_path}")


def plot_rf_delta(rf_results: Dict, save_path: Path):
    """Figure 4: Accuracy / F1 comparison before and after adding temporal features."""
    tags = ["baseline", "baseline+temporal"]
    labels_bar = ["Baseline\n(107 features)", "Baseline\n+ Temporal\n(129 features)"]
    accs = [rf_results[t]["acc"] for t in tags]
    acc_errs = [rf_results[t]["acc_std"] for t in tags]
    f1s  = [rf_results[t]["f1"]  for t in tags]
    f1_errs  = [rf_results[t]["f1_std"]  for t in tags]

    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    b1 = ax.bar(x - w/2, accs, w, yerr=acc_errs, capsize=4,
                color=["#bdc3c7", "#3498db"], label="Accuracy", alpha=0.85)
    b2 = ax.bar(x + w/2, f1s,  w, yerr=f1_errs,  capsize=4,
                color=["#ecf0f1", "#85c1e9"], label="Macro-F1", alpha=0.85,
                edgecolor="gray", linewidth=0.8)

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=10)
    ax.set_ylim(0, min(1.0, max(accs + f1s) * 1.20 + 0.05))
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("RF Classification: Effect of Adding Temporal Features",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # delta arrow
    delta = accs[1] - accs[0]
    sign  = "+" if delta >= 0 else ""
    ax.annotate(f"Acc Δ = {sign}{delta:.3f}",
                xy=(0.5, max(accs) + acc_errs[np.argmax(accs)] + 0.03),
                xycoords=("axes fraction", "data"),
                ha="center", fontsize=10, color="#e74c3c" if delta > 0 else "#7f8c8d",
                fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig] Saved: {save_path}")


def plot_shape_descriptor_boxplots(feat_df: pd.DataFrame, save_path: Path):
    """Figure 5: Boxplots of key shape descriptors per emotion."""
    shape_feats = [
        ("peak_time_norm",       "Peak Energy Time\n(0=start, 1=end)"),
        ("energy_front_ratio",   "Front Half Energy\nShare"),
        ("rise_time_norm",       "Rise Time\n(normalised)"),
        ("sustain_ratio",        "Sustain Ratio\n(fraction > 50% peak)"),
        ("energy_slope",         "Energy Slope\n(normalised)"),
        ("energy_autocorr_lag1", "Energy Autocorr\n(lag-1)"),
        ("jerk_mean",            "Mean Jerk\n(|Δvelocity|)"),
    ]
    n_feat = len(shape_feats)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for idx, (col, ylabel) in enumerate(shape_feats):
        ax = axes[idx]
        data  = [feat_df[feat_df.emotion == e][col].dropna().values for e in EMOTIONS]
        colors = [EMO_COLS[e] for e in EMOTIONS]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.5))
        for patch, col_c in zip(bp["boxes"], colors):
            patch.set_facecolor(col_c)
            patch.set_alpha(0.7)
        ax.set_xticklabels([e[:3] for e in EMOTIONS], fontsize=8)
        ax.set_title(ylabel, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    # hide the 8th unused subplot
    axes[n_feat].set_visible(False)
    axes[7].set_visible(False)

    fig.suptitle("Temporal Shape Descriptor Distributions per Emotion  (EBM)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig] Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  EBM Temporal Dynamics Analysis")
    print("=" * 65)

    # ── collect files ─────────────────────────────────────────────────────────
    files = sorted(EBM_DIR.glob("*.csv"))
    print(f"[Data] Found {len(files)} EBM files")

    # ── Phase A: extract temporal features ───────────────────────────────────
    feat_csv = OUT_DIR / "ebm_temporal_features.csv"
    if feat_csv.exists():
        print(f"[+] Loading cached temporal features: {feat_csv}")
        feat_df = pd.read_csv(feat_csv)
    else:
        records = []
        for fp in tqdm(files, desc="Extracting temporal features", ncols=80):
            actor, emo = parse_filename(fp.name)
            if emo is None:
                continue
            tf = extract_temporal_features(fp)
            if tf is None:
                continue
            tf["actor"]    = actor
            tf["emotion"]  = emo
            tf["filename"] = fp.stem
            records.append(tf)
        feat_df = pd.DataFrame(records)
        feat_df.to_csv(feat_csv, index=False)
        print(f"[+] Temporal features saved: {feat_csv}  ({feat_df.shape})")

    print(f"\n[Summary] Shape: {feat_df.shape}")
    print(f"  Emotions: {feat_df.emotion.value_counts().to_dict()}")

    # ── Phase B: average temporal curves ─────────────────────────────────────
    npz_path = OUT_DIR / "temporal_curves.npz"
    if npz_path.exists():
        print(f"[+] Loading cached temporal curves: {npz_path}")
        npz = np.load(npz_path, allow_pickle=True)
        curves = npz["curves"].item()
    else:
        print("[Curves] Building per-emotion temporal curve bank …")
        curves = build_curve_bank(files)
        np.savez(npz_path, curves=curves)
        print(f"[+] Curves saved: {npz_path}")

    # ── Phase C: Kruskal-Wallis ──────────────────────────────────────────────
    print("[KW] Running Kruskal-Wallis on temporal features …")
    kw = run_kruskal_temporal(feat_df)
    kw.to_csv(OUT_DIR / "kruskal_temporal.csv", index=False)
    print(f"[KW] Results saved. Top-10:")
    print(kw.head(10)[["feature", "H", "eps2"]].to_string(index=False))
    large  = (kw.eps2 >= COHEN_LARGE).sum()
    medium = ((kw.eps2 >= COHEN_MEDIUM) & (kw.eps2 < COHEN_LARGE)).sum()
    small  = ((kw.eps2 >= COHEN_SMALL)  & (kw.eps2 < COHEN_MEDIUM)).sum()
    print(f"\n[KW] Effect sizes: Large={large}, Medium={medium}, Small={small}")

    # ── Phase D: RF comparison ──────────────────────────────────────────────
    base_feat_csv = Path("outputs/analysis/ebm_full/ebm_all_features.csv")
    rf_results = None
    if base_feat_csv.exists():
        print("\n[RF] Running comparison (baseline vs. +temporal) …")
        rf_results = rf_comparison(feat_df, base_feat_csv)
        import json
        with open(OUT_DIR / "rf_comparison.json", "w") as f:
            json.dump(rf_results, f, indent=2)
    else:
        print(f"[RF] Skipping – baseline CSV not found: {base_feat_csv}")

    # ── Phase E: visualisations ──────────────────────────────────────────────
    print("\n[Viz] Generating figures …")

    plot_temporal_curves(
        curves,
        FIGS_DIR / "ebm_temporal_curves.png"
    )

    plot_phase_heatmap(
        feat_df,
        FIGS_DIR / "ebm_phase_heatmap.png"
    )

    plot_temporal_effects(
        kw,
        FIGS_DIR / "ebm_temporal_effects.png"
    )

    plot_shape_descriptor_boxplots(
        feat_df,
        FIGS_DIR / "ebm_temporal_boxplots.png"
    )

    if rf_results is not None:
        plot_rf_delta(
            rf_results,
            FIGS_DIR / "ebm_temporal_rf_delta.png"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DONE – EBM Temporal Dynamics Analysis")
    print("=" * 65)
    print(f"\nOutputs in  : {OUT_DIR}")
    print(f"Figures in  : {FIGS_DIR}")
    print(f"\nFeature summary:")
    print(f"  Phase features  : {N_PHASES} × 3 = {N_PHASES*3} (E, vel, ratio)")
    print(f"  Shape descriptors: 7")
    print(f"  Total new features: {N_PHASES*3 + 7}")
    print(f"\nKW discriminability:")
    print(f"  Large (ε²≥0.14)  : {large}")
    print(f"  Medium (0.06–0.14): {medium}")
    print(f"  Small  (0.01–0.06): {small}")
    if rf_results:
        b  = rf_results["baseline"]
        bt = rf_results["baseline+temporal"]
        print(f"\nRF accuracy:  {b['acc']:.3f} → {bt['acc']:.3f}  "
              f"(Δ = {bt['acc']-b['acc']:+.3f})")
        print(f"RF macro-F1:  {b['f1']:.3f} → {bt['f1']:.3f}  "
              f"(Δ = {bt['f1']-b['f1']:+.3f})")


if __name__ == "__main__":
    main()
