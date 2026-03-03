"""
Effect size analysis for 3D BVH temporal features.
- KW: epsilon-squared (ε²) = H / (N-1)  [0–1; small≥0.01, medium≥0.06, large≥0.14]
- Pairwise MW: rank-biserial correlation r_rb = 1 - 2U/(n1*n2)  [0–1; small≥0.1, medium≥0.3, large≥0.5]
Produces 4 figures in docs/figs_3d_temporal/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations

DATA_PATH = Path("outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv")
OUT_DIR   = Path("docs/figs_3d_temporal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMO_ORDER = ["Happy", "Angry", "Fearful", "Disgust", "Surprise", "Neutral", "Sad"]
PALETTE   = {
    "Happy":"#F4C430","Angry":"#E84040","Fearful":"#8A2BE2",
    "Disgust":"#228B22","Surprise":"#FF7F00","Neutral":"#999999","Sad":"#4169E1",
}

JOINT_LABELS = {
    "head_vel":"Head", "l_shoulder_vel":"L-Shoulder", "r_shoulder_vel":"R-Shoulder",
    "l_elbow_vel":"L-Elbow", "r_elbow_vel":"R-Elbow",
    "l_wrist_vel":"L-Wrist", "r_wrist_vel":"R-Wrist", "avg_velocity":"Average",
}

print("[+] Loading data …")
df = pd.read_csv(DATA_PATH)
df["emotion"] = pd.Categorical(df["emotion"], categories=EMO_ORDER, ordered=True)
df = df.dropna(subset=["avg_velocity"])
N_total = len(df)
k = df["emotion"].nunique()
print(f"    N={N_total:,}  k={k}")

# ──────────────────────────────────────────────────────────────────────────────
# 1. KW epsilon-squared per joint
# ──────────────────────────────────────────────────────────────────────────────
print("[1/4] KW ε² per joint …")

kw_rows = []
for col, label in JOINT_LABELS.items():
    sub = df.dropna(subset=[col])
    groups = [g[col].values for _, g in sub.groupby("emotion", observed=True)]
    H, p = kruskal(*groups)
    N = sum(len(g) for g in groups)
    eps2 = H / (N - 1)           # epsilon-squared
    eta2 = (H - k + 1) / (N - k) # eta-squared (alternative)
    kw_rows.append({"feature": label, "H": H, "N": N,
                    "eps2": eps2, "eta2": max(eta2, 0)})

kw_df = pd.DataFrame(kw_rows).sort_values("eps2", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# left: H statistic
axes[0].barh(kw_df["feature"], kw_df["H"], color="#4C72B0", edgecolor="white")
axes[0].set_xlabel("Kruskal-Wallis H", fontsize=11)
axes[0].set_title("H Statistic\n(significance proxy)", fontsize=11)
axes[0].grid(axis="x", alpha=0.3)
for bar, row in zip(axes[0].patches, kw_df.itertuples()):
    axes[0].text(bar.get_width()+1000, bar.get_y()+bar.get_height()/2,
                 f"{row.H:.0f}", va="center", fontsize=8)

# right: epsilon-squared
bar_colors = ["#c0392b" if v >= 0.14 else "#e67e22" if v >= 0.06 else "#2ecc71"
              for v in kw_df["eps2"]]
axes[1].barh(kw_df["feature"], kw_df["eps2"], color=bar_colors, edgecolor="white")
axes[1].axvline(0.01, color="gray", linestyle=":", linewidth=1, label="small (0.01)")
axes[1].axvline(0.06, color="orange", linestyle="--", linewidth=1, label="medium (0.06)")
axes[1].axvline(0.14, color="red", linestyle="-", linewidth=1, label="large (0.14)")
axes[1].set_xlabel("ε² (epsilon-squared)", fontsize=11)
axes[1].set_title("Effect Size ε²\n(practical importance)", fontsize=11)
axes[1].legend(fontsize=8, loc="lower right")
axes[1].grid(axis="x", alpha=0.3)
for bar, row in zip(axes[1].patches, kw_df.itertuples()):
    axes[1].text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                 f"{row.eps2:.4f}", va="center", fontsize=8)

fig.suptitle("3D BVH Velocity: Statistical Significance vs Effect Size\n"
             f"(N={N_total:,} frames, k=7 emotions)", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "3d_effect_kw.png", dpi=180)
plt.close(fig)
print("    ε² values:")
for _, r in kw_df.sort_values("eps2", ascending=False).iterrows():
    tag = "LARGE" if r.eps2>=0.14 else "medium" if r.eps2>=0.06 else "small"
    print(f"      {r.feature:<14} ε²={r.eps2:.5f}  [{tag}]")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Pairwise rank-biserial r for avg_velocity
# ──────────────────────────────────────────────────────────────────────────────
print("[2/4] Pairwise rank-biserial r (avg_velocity) …")

N_SAMPLE = 30_000
rng = np.random.default_rng(42)
sampled = {}
for e in EMO_ORDER:
    vals = df[df["emotion"]==e]["avg_velocity"].values
    sampled[e] = rng.choice(vals, min(N_SAMPLE, len(vals)), replace=False)

pairs = list(combinations(EMO_ORDER, 2))
raw_p, raw_r = [], []
for e1, e2 in pairs:
    u, p = mannwhitneyu(sampled[e1], sampled[e2], alternative="two-sided")
    n1, n2 = len(sampled[e1]), len(sampled[e2])
    r_rb = 1 - 2*u/(n1*n2)          # rank-biserial, range [−1, 1]
    raw_p.append(p)
    raw_r.append(r_rb)

_, p_fdr, _, _ = multipletests(raw_p, method="fdr_bh")

# build symmetric matrices for r and −log10(p)
r_mat = pd.DataFrame(np.zeros((7,7)), index=EMO_ORDER, columns=EMO_ORDER)
p_mat = pd.DataFrame(np.zeros((7,7)), index=EMO_ORDER, columns=EMO_ORDER)
for (e1,e2), r, pv in zip(pairs, raw_r, p_fdr):
    r_mat.loc[e1,e2] = r_mat.loc[e2,e1] = r
    log_p = -np.log10(max(pv, 1e-300))
    p_mat.loc[e1,e2] = p_mat.loc[e2,e1] = log_p

mask = np.eye(7, dtype=bool)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# left: rank-biserial r
sns.heatmap(r_mat.abs(), annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=0.6, mask=mask, linewidths=0.5, linecolor="white",
            cbar_kws={"label": "|rank-biserial r|"}, ax=axes[0])
axes[0].set_title("Effect Size: |Rank-Biserial r|\n(0.1=small, 0.3=medium, 0.5=large)", fontsize=10)

# right: -log10(p_FDR)
sns.heatmap(p_mat, annot=True, fmt=".1f", cmap="Blues",
            mask=mask, linewidths=0.5, linecolor="white",
            cbar_kws={"label": "-log₁₀(p_FDR)"}, ax=axes[1])
axes[1].set_title("Statistical Significance: -log₁₀(p_FDR)\n(>1.3 = p<0.05)", fontsize=10)

fig.suptitle("Pairwise Comparison of avg_velocity (subsampled 30k/emotion)", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "3d_effect_pairwise.png", dpi=180)
plt.close(fig)

print("    rank-biserial r (sorted by |r|):")
pair_summary = sorted(zip(pairs, raw_r, p_fdr), key=lambda x: abs(x[1]), reverse=True)
for (e1,e2), r, pv in pair_summary:
    tag = "LARGE" if abs(r)>=0.5 else "medium" if abs(r)>=0.3 else "small"
    print(f"      {e1:<10} vs {e2:<10} r={r:+.3f}  p_FDR={pv:.2e}  [{tag}]")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Distribution overlap visualisation (per-emotion violin + median markers)
#    to intuitively show "do the distributions actually separate?"
# ──────────────────────────────────────────────────────────────────────────────
print("[3/4] Overlap violin …")
p99 = df["avg_velocity"].quantile(0.99)
df_plot = df[df["avg_velocity"] <= p99].copy()

fig, ax = plt.subplots(figsize=(9, 5))
parts = ax.violinplot(
    [df_plot[df_plot["emotion"]==e]["avg_velocity"].values for e in EMO_ORDER],
    positions=range(len(EMO_ORDER)), widths=0.7, showmedians=True, showextrema=False
)
for i, (pc, e) in enumerate(zip(parts["bodies"], EMO_ORDER)):
    pc.set_facecolor(PALETTE[e])
    pc.set_alpha(0.75)
parts["cmedians"].set_color("black")
parts["cmedians"].set_linewidth(2)

# annotate medians
for i, e in enumerate(EMO_ORDER):
    med = df_plot[df_plot["emotion"]==e]["avg_velocity"].median()
    ax.text(i, med+0.0003, f"{med:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(range(len(EMO_ORDER)))
ax.set_xticklabels(EMO_ORDER, fontsize=10)
ax.set_ylabel("Normalised Velocity (clipped p99)", fontsize=11)
ax.set_title("Distribution Overlap: avg_velocity per Emotion\n"
             "(wide overlap explains modest ε² despite p≈0)", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# add ε² annotation box
eps2_avg = kw_df[kw_df["feature"]=="Average"]["eps2"].values[0]
ax.text(0.98, 0.97, f"KW ε² = {eps2_avg:.4f}\n(small effect)",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
        fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "3d_overlap_violin.png", dpi=180)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Common Language Effect Size (CL): P(Happy > Sad), P(Happy > Angry) etc.
#    Most intuitive for non-statisticians
# ──────────────────────────────────────────────────────────────────────────────
print("[4/4] Common Language Effect Size (CLES) …")

focus_pairs = [
    ("Happy", "Sad"), ("Happy", "Neutral"), ("Happy", "Disgust"),
    ("Angry", "Sad"), ("Angry", "Neutral"), ("Fearful", "Sad"),
    ("Angry", "Fearful"), ("Disgust", "Surprise"), ("Neutral", "Sad"),
]

cles_rows = []
for e1, e2 in focus_pairs:
    v1 = sampled[e1]
    v2 = sampled[e2]
    u, _ = mannwhitneyu(v1, v2, alternative="greater")
    cles = u / (len(v1)*len(v2))     # = P(X1 > X2)
    cles_rows.append({"pair": f"{e1}\nvs {e2}", "e1": e1, "e2": e2, "CLES": cles})

cles_df = pd.DataFrame(cles_rows).sort_values("CLES", ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = [PALETTE[r.e1] for r in cles_df.itertuples()]
bars = ax.bar(cles_df["pair"], cles_df["CLES"], color=bar_colors,
              edgecolor="white", linewidth=0.8)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance (0.5)")
ax.axhline(0.56, color="orange", linestyle=":", linewidth=1, label="small (0.56)")
ax.axhline(0.64, color="red", linestyle=":", linewidth=1, label="medium (0.64)")
for bar, val in zip(bars, cles_df["CLES"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("P(row emotion velocity > column emotion)", fontsize=10)
ax.set_title("Common Language Effect Size (CLES)\nProbability that one emotion has higher velocity than another", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "3d_cles.png", dpi=180)
plt.close(fig)

print("\n    CLES values:")
for r in cles_df.itertuples():
    tag = "LARGE" if r.CLES>=0.71 else "medium" if r.CLES>=0.64 else "small" if r.CLES>=0.56 else "negligible"
    print(f"      P({r.e1:8s} > {r.e2:8s}) = {r.CLES:.3f}  [{tag}]")

print(f"\n[✓] 4 figures saved to {OUT_DIR}/")
