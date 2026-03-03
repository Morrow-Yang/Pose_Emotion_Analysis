"""
Visualize arm_span_norm and left_elbow_angle distributions per emotion.
Outputs: docs/figs_stats/
  - arm_span_ecdf.png
  - elbow_ecdf.png
  - arm_span_ridge.png
  - elbow_ridge.png
  - median_heatmap.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import gaussian_kde

# ── config ──────────────────────────────────────────────────────────────────
CSV   = "outputs/analysis/analysis/v4/pose_features_v4.csv"
OUT   = Path("docs/figs_stats")
OUT.mkdir(parents=True, exist_ok=True)

EMO_ORDER = ["Angry", "Disgust", "Happy", "Neutral", "Surprise", "Sad", "Fear"]
# ordered roughly high→low arm_span median
PALETTE = {
    "Angry":    "#e63946",
    "Disgust":  "#f4a261",
    "Happy":    "#2a9d8f",
    "Neutral":  "#457b9d",
    "Surprise": "#a8dadc",
    "Sad":      "#6d6875",
    "Fear":     "#b5838d",
}

df = pd.read_csv(CSV)

# ── helper: clip to [1%, 99%] ────────────────────────────────────────────────
def clip_feat(series):
    lo, hi = series.quantile(0.01), series.quantile(0.99)
    return series.clip(lo, hi)

# ════════════════════════════════════════════════════════════════════════════
# 1. ECDF plots
# ════════════════════════════════════════════════════════════════════════════
def plot_ecdf(feat, xlabel, fname, clip=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for emo in EMO_ORDER:
        vals = df.loc[df["emotion"] == emo, feat].dropna()
        if clip:
            vals = clip_feat(vals)
        vals_sorted = np.sort(vals)
        ecdf = np.arange(1, len(vals_sorted)+1) / len(vals_sorted)
        ax.plot(vals_sorted, ecdf, color=PALETTE[emo], linewidth=1.8, label=emo, alpha=0.9)
        # median tick
        med = np.median(vals_sorted)
        ax.axvline(med, color=PALETTE[emo], linewidth=0.7, linestyle="--", alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Cumulative proportion", fontsize=11)
    ax.set_title(f"ECDF — {xlabel} by emotion", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    print(f"[saved] {OUT/fname}")

plot_ecdf("arm_span_norm",   "arm_span_norm (wrist-dist / shoulder-width)", "arm_span_ecdf.png")
plot_ecdf("left_elbow_angle","left elbow angle (°)",                         "elbow_ecdf.png")

# ════════════════════════════════════════════════════════════════════════════
# 2. Ridge / joy plots
# ════════════════════════════════════════════════════════════════════════════
def plot_ridge(feat, xlabel, fname, clip=True, bw=0.15):
    fig, axes = plt.subplots(len(EMO_ORDER), 1,
                             figsize=(7, len(EMO_ORDER)*0.95),
                             sharex=True)
    for i, emo in enumerate(EMO_ORDER):
        ax = axes[i]
        vals = df.loc[df["emotion"] == emo, feat].dropna()
        if clip:
            vals = clip_feat(vals)
        color = PALETTE[emo]
        # KDE
        lo, hi = vals.quantile(0.005), vals.quantile(0.995)
        xs = np.linspace(lo, hi, 300)
        try:
            kde = gaussian_kde(vals, bw_method=bw)
            ys  = kde(xs)
        except Exception:
            ys  = np.zeros_like(xs)
        ax.fill_between(xs, ys, alpha=0.55, color=color)
        ax.plot(xs, ys, color=color, linewidth=1.5)
        # median line
        med = vals.median()
        ax.axvline(med, color="black", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.set_yticks([])
        ax.set_ylabel(emo, rotation=0, labelpad=55, va="center", fontsize=9)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    axes[-1].set_xlabel(xlabel, fontsize=11)
    fig.suptitle(f"Distribution — {xlabel} by emotion\n(dashed = median)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT/fname}")

plot_ridge("arm_span_norm",   "arm_span_norm", "arm_span_ridge.png", bw=0.2)
plot_ridge("left_elbow_angle","left elbow angle (°)", "elbow_ridge.png", bw=0.03)

# ════════════════════════════════════════════════════════════════════════════
# 3. Median heatmap (both features, emotion as rows)
# ════════════════════════════════════════════════════════════════════════════
feats_for_heatmap = {
    "arm_span_norm":    "arm_span_norm",
    "left_elbow_angle": "L elbow angle (°)",
}

rows = {}
for feat, label in feats_for_heatmap.items():
    rows[label] = df.groupby("emotion")[feat].median()

heat_df = pd.DataFrame(rows).loc[EMO_ORDER]

# z-score each column for color comparability
heat_z = (heat_df - heat_df.mean()) / heat_df.std()

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(heat_z.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)

ax.set_yticks(range(len(EMO_ORDER)))
ax.set_yticklabels(EMO_ORDER, fontsize=10)
ax.set_xticks(range(len(heat_df.columns)))
ax.set_xticklabels(heat_df.columns, fontsize=10)

# annotate raw medians
for i, emo in enumerate(EMO_ORDER):
    for j, col in enumerate(heat_df.columns):
        val = heat_df.loc[emo, col]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=9, color="black")

plt.colorbar(im, ax=ax, label="z-score across emotions")
ax.set_title("Median per emotion (z-scored for colour)", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "median_heatmap.png", dpi=150)
plt.close(fig)
print(f"[saved] {OUT/'median_heatmap.png'}")

print("\nAll done →", OUT)
