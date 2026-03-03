"""
BVH Dynamic Energy Analysis
============================
Computes per-frame kinetic energy from joint velocities and derives:
  1. Energy time profiles per emotion (median ± IQR over time-normalised sequences)
  2. Energy summary statistics per file (mean, std, max, burst count, etc.)
  3. FFT frequency analysis (dominant frequency per emotion)
  4. Body-segment energy share (arms / legs / spine-head)
  5. KW discriminability of energy features
  6. Visualisation (4-panel figure → docs/figs_3d_temporal/bvh_energy_analysis.png)

Input:  outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv
        (per-frame joint velocities already computed)
Output: outputs/analysis/energy_bvh/   + figures
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kruskal
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
TEMPORAL_CSV = Path("outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv")
OUT_DIR      = Path("outputs/analysis/energy_bvh")
FIGS_DIR     = Path("docs/figs_3d_temporal")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprise"]
EMO_COLS = {
    "Angry"  : "#e74c3c", "Disgust": "#8e44ad", "Fearful": "#3498db",
    "Happy"  : "#f39c12", "Neutral": "#95a5a6", "Sad"    : "#2980b9",
    "Surprise": "#27ae60",
}

# Joint velocity columns available in the CSV
# Energy = sum of squared velocities across joints
VEL_COLS = ["avg_velocity", "head_vel", "l_shoulder_vel", "r_shoulder_vel",
            "l_elbow_vel", "r_elbow_vel", "l_wrist_vel", "r_wrist_vel"]

# Body segments (indices into VEL_COLS for segment attribution)
SEG_ARMS   = ["l_shoulder_vel", "r_shoulder_vel", "l_elbow_vel", "r_elbow_vel",
              "l_wrist_vel", "r_wrist_vel"]
SEG_HEAD   = ["head_vel"]
SEG_GLOBAL = ["avg_velocity"]   # catch-all / hip proxy

# ── helpers ───────────────────────────────────────────────────────────────────

def kinetic_energy(row: pd.Series) -> float:
    """Proxy kinetic energy: sum of squared joint velocities."""
    vals = row[VEL_COLS].dropna().values
    return float(np.sum(vals ** 2)) if len(vals) > 0 else np.nan


def normalise_series(s: np.ndarray, n_bins: int = 100) -> np.ndarray:
    """Time-normalise a 1-D array to fixed length via linear interpolation."""
    x_old = np.linspace(0, 1, len(s))
    x_new = np.linspace(0, 1, n_bins)
    return np.interp(x_new, x_old, s)


def dominant_freq(energy: np.ndarray, fps: float = 125.0) -> float:
    """Return the dominant frequency (Hz) in the energy time series."""
    if len(energy) < 8:
        return np.nan
    e = energy - energy.mean()
    freqs = rfftfreq(len(e), d=1.0 / fps)
    mags  = np.abs(rfft(e))
    # ignore DC (0 Hz)
    mags[0] = 0
    return float(freqs[np.argmax(mags)])


def burst_count(energy: np.ndarray, threshold_factor: float = 1.5) -> int:
    """Count energy bursts above threshold_factor × median."""
    thr = np.median(energy) * threshold_factor
    peaks, _ = find_peaks(energy, height=thr, distance=5)
    return len(peaks)


# ── main pipeline ─────────────────────────────────────────────────────────────

def load_and_compute():
    print("[+] Loading temporal features …")
    df = pd.read_csv(TEMPORAL_CSV)
    # fill missing velocities with 0 (joint not tracked in that frame)
    for c in VEL_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # kinetic energy per frame
    df["kinetic_E"] = df.apply(kinetic_energy, axis=1)

    # body-segment energy share (arms vs head)
    arm_cols  = [c for c in SEG_ARMS   if c in df.columns]
    head_cols = [c for c in SEG_HEAD   if c in df.columns]
    df["E_arms"]  = df[arm_cols ].pow(2).sum(axis=1)
    df["E_head"]  = df[head_cols].pow(2).sum(axis=1)
    df["E_total"] = df[VEL_COLS ].pow(2).sum(axis=1).clip(lower=1e-12)
    df["arms_share"] = df["E_arms"]  / df["E_total"]
    df["head_share"] = df["E_head"]  / df["E_total"]

    print(f"[+] Frame-level rows: {len(df)}  |  Cols: {df.columns.tolist()[:8]} …")
    return df


def file_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-file energy statistics and FFT features."""
    rows = []
    for (fname, emo), grp in df.groupby(["filename", "emotion"]):
        grp = grp.sort_values("frame")
        E   = grp["kinetic_E"].values
        if len(E) < 5:
            continue
        row = {
            "filename"         : fname,
            "emotion"          : emo,
            "E_mean"           : float(np.mean(E)),
            "E_std"            : float(np.std(E)),
            "E_max"            : float(np.max(E)),
            "E_min"            : float(np.min(E)),
            "E_range"          : float(np.ptp(E)),
            "E_cv"             : float(np.std(E) / (np.mean(E) + 1e-12)),  # coeff of variation
            "E_skew"           : float(pd.Series(E).skew()),
            "burst_count"      : burst_count(E, threshold_factor=1.5),
            "dom_freq_hz"      : dominant_freq(E, fps=125.0),
            "arms_share_mean"  : float(grp["arms_share"].mean()),
            "head_share_mean"  : float(grp["head_share"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compute_median_profiles(df: pd.DataFrame, n_bins: int = 100) -> dict:
    """Per emotion: median energy profile (time-normalised to n_bins)."""
    profiles = {}
    for emo in EMOTIONS:
        sub = df[df["emotion"] == emo]
        all_curves = []
        for _, grp in sub.groupby("filename"):
            E = grp.sort_values("frame")["kinetic_E"].values
            if len(E) >= 5:
                all_curves.append(normalise_series(E, n_bins))
        if all_curves:
            mat            = np.stack(all_curves)
            profiles[emo]  = {
                "median": np.median(mat, axis=0),
                "q25"   : np.percentile(mat, 25, axis=0),
                "q75"   : np.percentile(mat, 75, axis=0),
            }
    return profiles


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_all(df_frame: pd.DataFrame, df_file: pd.DataFrame, profiles: dict):
    N, k = len(df_file), 7

    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.35})
    fig.suptitle("3D BVH – Dynamic Kinetic Energy Analysis",
                 fontsize=13, fontweight="bold", y=0.98)

    # ── A: median energy profiles ─────────────────────────────────────────────
    ax = axes[0, 0]
    t  = np.linspace(0, 100, 100)
    for emo in EMOTIONS:
        if emo not in profiles:
            continue
        p   = profiles[emo]
        col = EMO_COLS[emo]
        ax.plot(t, p["median"], color=col, linewidth=1.8, label=emo)
        ax.fill_between(t, p["q25"], p["q75"], color=col, alpha=0.12)
    ax.set_xlabel("Normalised time (%)", fontsize=9)
    ax.set_ylabel("Kinetic Energy (a.u.)", fontsize=9)
    ax.set_title("(A) Median energy profiles (IQR shaded)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    # ── B: E_mean violin per emotion ──────────────────────────────────────────
    ax = axes[0, 1]
    data_v = [df_file[df_file.emotion == e]["E_mean"].dropna().values for e in EMOTIONS]
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
    ax.set_ylabel("Mean kinetic energy (a.u.)", fontsize=9)
    ax.set_title("(B) Mean energy distribution per emotion", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # ── C: dominant frequency boxplot ─────────────────────────────────────────
    ax = axes[1, 0]
    data_f = [df_file[df_file.emotion == e]["dom_freq_hz"].dropna().values for e in EMOTIONS]
    bp = ax.boxplot(data_f, patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, emo in zip(bp["boxes"], EMOTIONS):
        patch.set_facecolor(EMO_COLS[emo]); patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(EMOTIONS)+1))
    ax.set_xticklabels(EMOTIONS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Dominant frequency (Hz)", fontsize=9)
    ax.set_title("(C) Energy oscillation frequency per emotion", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # ── D: KW H for energy features ───────────────────────────────────────────
    ax = axes[1, 1]
    energy_feats = ["E_mean", "E_std", "E_max", "E_range", "E_cv", "E_skew",
                    "burst_count", "dom_freq_hz", "arms_share_mean", "head_share_mean"]
    kw_rows = []
    for feat in energy_feats:
        if feat not in df_file.columns:
            continue
        groups = [df_file[df_file.emotion == e][feat].dropna().values for e in EMOTIONS]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        eps2 = max(0, (H - k + 1) / (N - k))
        kw_rows.append({"feature": feat, "H": H, "eps2": eps2})
    kw_df = pd.DataFrame(kw_rows).sort_values("H", ascending=True)
    colors_bar = ["#e74c3c" if e >= 0.14 else "#f39c12" if e >= 0.06 else "#3498db"
                  for e in kw_df["eps2"]]
    ax.barh(range(len(kw_df)), kw_df["H"], color=colors_bar, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(kw_df)))
    ax.set_yticklabels(kw_df["feature"], fontsize=8)
    for i, (h, e) in enumerate(zip(kw_df["H"], kw_df["eps2"])):
        ax.text(h + 1, i, f"ε²={e:.3f}", va="center", fontsize=7)
    ax.set_xlabel("Kruskal-Wallis H", fontsize=9)
    ax.set_title("(D) KW discriminability of energy features", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    # legend for effect size colour
    patches = [mpatches.Patch(color="#e74c3c", label="large (≥0.14)"),
               mpatches.Patch(color="#f39c12", label="medium (≥0.06)"),
               mpatches.Patch(color="#3498db", label="small (<0.06)")]
    ax.legend(handles=patches, fontsize=7, loc="lower right")

    fig.savefig(FIGS_DIR / "bvh_energy_analysis.png", dpi=150, bbox_inches="tight")
    print(f"[✓] Saved → {FIGS_DIR}/bvh_energy_analysis.png")
    plt.close(fig)
    return kw_df


def main():
    df_frame = load_and_compute()
    print("[+] Computing file-level energy statistics …")
    df_file  = file_level_features(df_frame)
    df_file.to_csv(OUT_DIR / "bvh_energy_features.csv", index=False)
    print(f"[+] Saved {len(df_file)} file-level rows → {OUT_DIR}/bvh_energy_features.csv")

    print("[+] Computing median energy profiles (time-normalised) …")
    profiles = compute_median_profiles(df_frame, n_bins=100)

    print("[+] Generating visualisation …")
    kw_df = plot_all(df_frame, df_file, profiles)

    # print summary
    print("\n" + "="*60)
    print("DYNAMIC ENERGY SUMMARY (per emotion, median)")
    print("="*60)
    for emo in EMOTIONS:
        sub = df_file[df_file.emotion == emo]
        print(f"  {emo:<10}  E_mean={sub.E_mean.median():.4f}  "
              f"E_cv={sub.E_cv.median():.3f}  "
              f"dom_freq={sub.dom_freq_hz.median():.2f}Hz  "
              f"bursts={sub.burst_count.median():.1f}  "
              f"arms%={sub.arms_share_mean.median()*100:.1f}")
    print("\nKW results:")
    print(kw_df.sort_values("H", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
