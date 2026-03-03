"""
3D BVH Temporal Feature Visualization & Statistical Analysis
Produces 6 figures in docs/figs_3d_temporal/
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

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv")
OUT_DIR   = Path("docs/figs_3d_temporal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── palette & order ────────────────────────────────────────────────────────────
EMO_ORDER = ["Happy", "Angry", "Fearful", "Disgust", "Surprise", "Neutral", "Sad"]
PALETTE   = {
    "Happy":   "#F4C430",
    "Angry":   "#E84040",
    "Fearful": "#8A2BE2",
    "Disgust": "#228B22",
    "Surprise":"#FF7F00",
    "Neutral": "#999999",
    "Sad":     "#4169E1",
}

JOINT_LABELS = {
    "head_vel":        "Head",
    "l_shoulder_vel":  "L-Shoulder",
    "r_shoulder_vel":  "R-Shoulder",
    "l_elbow_vel":     "L-Elbow",
    "r_elbow_vel":     "R-Elbow",
    "l_wrist_vel":     "L-Wrist",
    "r_wrist_vel":     "R-Wrist",
    "avg_velocity":    "Average",
}

print("[+] Loading data …")
df = pd.read_csv(DATA_PATH)
df["emotion"] = pd.Categorical(df["emotion"], categories=EMO_ORDER, ordered=True)
df = df.dropna(subset=["avg_velocity"])

# cap extreme outliers at p99 per emotion for display
p99 = df["avg_velocity"].quantile(0.99)
df_plot = df[df["avg_velocity"] <= p99].copy()

colors = [PALETTE[e] for e in EMO_ORDER]

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 – Mean ± SEM bar chart (sorted by mean)
# ══════════════════════════════════════════════════════════════════════════════
print("[1/6] Bar chart mean ± SEM …")
agg = (df.groupby("emotion", observed=True)["avg_velocity"]
         .agg(mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)))
         .reset_index()
         .sort_values("mean", ascending=False))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(agg["emotion"], agg["mean"],
              yerr=agg["sem"], capsize=4,
              color=[PALETTE[e] for e in agg["emotion"]],
              edgecolor="white", linewidth=0.8, error_kw={"elinewidth":1.2})
ax.set_xlabel("Emotion", fontsize=12)
ax.set_ylabel("Mean Normalised Velocity", fontsize=12)
ax.set_title("3D BVH – Per-Emotion Average Velocity (mean ± SEM)", fontsize=13)
ax.axhline(df["avg_velocity"].mean(), color="black", linestyle="--",
           linewidth=1, label=f"Grand mean={df['avg_velocity'].mean():.4f}")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "3d_vel_bar.png", dpi=180)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 – ECDF of avg_velocity per emotion
# ══════════════════════════════════════════════════════════════════════════════
print("[2/6] ECDF …")
fig, ax = plt.subplots(figsize=(8, 5))
for emo in EMO_ORDER:
    sub = df_plot[df_plot["emotion"] == emo]["avg_velocity"].sort_values().values
    ecdf = np.arange(1, len(sub)+1) / len(sub)
    ax.plot(sub, ecdf, color=PALETTE[emo], linewidth=1.8, label=emo)
ax.set_xlabel("Normalised Velocity (clipped at p99)", fontsize=12)
ax.set_ylabel("Cumulative Proportion", fontsize=12)
ax.set_title("3D BVH – ECDF of Average Velocity by Emotion", fontsize=13)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "3d_vel_ecdf.png", dpi=180)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 – Violin plot
# ══════════════════════════════════════════════════════════════════════════════
print("[3/6] Violin plot …")
fig, ax = plt.subplots(figsize=(9, 5))
sns.violinplot(
    data=df_plot, x="emotion", y="avg_velocity",
    order=EMO_ORDER, palette=PALETTE,
    inner="box", cut=0, linewidth=0.8, ax=ax
)
ax.set_xlabel("Emotion", fontsize=12)
ax.set_ylabel("Normalised Velocity (clipped p99)", fontsize=12)
ax.set_title("3D BVH – Velocity Distribution by Emotion (Violin)", fontsize=13)
# annotate medians
for i, emo in enumerate(EMO_ORDER):
    med = df_plot[df_plot["emotion"] == emo]["avg_velocity"].median()
    ax.text(i, med + 0.0005, f"{med:.4f}", ha="center", va="bottom",
            fontsize=7.5, color="black", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "3d_vel_violin.png", dpi=180)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 – Body-part heatmap  (emotion × joint, mean velocity)
# ══════════════════════════════════════════════════════════════════════════════
print("[4/6] Body-part heatmap …")
joint_cols = ["head_vel","l_shoulder_vel","r_shoulder_vel",
              "l_elbow_vel","r_elbow_vel","l_wrist_vel","r_wrist_vel","avg_velocity"]
hmap_data = (df.groupby("emotion", observed=True)[joint_cols]
               .mean()
               .loc[EMO_ORDER]
               .rename(columns=JOINT_LABELS))

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(
    hmap_data, annot=True, fmt=".4f", cmap="YlOrRd",
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Mean Normalised Velocity"},
    ax=ax
)
ax.set_title("3D BVH – Mean Velocity per Emotion × Body Part", fontsize=13)
ax.set_xlabel("Body Part", fontsize=11)
ax.set_ylabel("Emotion", fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "3d_bodypart_heatmap.png", dpi=180)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 – Pairwise Mann-Whitney U + FDR correction heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("[5/6] Pairwise MW-U + FDR heatmap …")
emos = EMO_ORDER
pairs = list(combinations(emos, 2))

# sub-sample for speed (max 50k frames per emotion)
N_SAMPLE = 50_000
rng = np.random.default_rng(42)
sampled = []
for e in emos:
    sub = df[df["emotion"] == e]["avg_velocity"].values
    if len(sub) > N_SAMPLE:
        sub = rng.choice(sub, N_SAMPLE, replace=False)
    sampled.append((e, sub))
sampled = dict(sampled)

raw_p = []
for e1, e2 in pairs:
    _, p = mannwhitneyu(sampled[e1], sampled[e2], alternative="two-sided")
    raw_p.append(p)

reject, p_fdr, _, _ = multipletests(raw_p, method="fdr_bh")

# build symmetric -log10(p_fdr) matrix
pmat = pd.DataFrame(np.zeros((len(emos), len(emos))), index=emos, columns=emos)
for (e1, e2), pv in zip(pairs, p_fdr):
    val = -np.log10(max(pv, 1e-300))
    pmat.loc[e1, e2] = val
    pmat.loc[e2, e1] = val

fig, ax = plt.subplots(figsize=(7, 6))
mask = np.eye(len(emos), dtype=bool)
sns.heatmap(
    pmat, annot=True, fmt=".1f", cmap="Blues",
    mask=mask, linewidths=0.5, linecolor="white",
    cbar_kws={"label": "-log₁₀(p_FDR)"},
    ax=ax
)
ax.set_title("3D BVH – Pairwise Mann-Whitney U\n-log₁₀(p) after FDR-BH correction", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "3d_pairwise_mwu.png", dpi=180)
plt.close(fig)

# print significant pairs
print(f"  Significant pairs (p_FDR < 0.05): {sum(reject)} / {len(pairs)}")
for (e1, e2), pv, sig in zip(pairs, p_fdr, reject):
    if sig:
        print(f"    {e1} vs {e2}: p_FDR={pv:.3e}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 – KW H-statistic summary across all features
# ══════════════════════════════════════════════════════════════════════════════
print("[6/6] KW summary bar …")
all_feats = list(JOINT_LABELS.keys())
kw_rows = []
for feat in all_feats:
    sub_df = df.dropna(subset=[feat])
    groups = [g[feat].values for _, g in sub_df.groupby("emotion", observed=True)]
    H, p = kruskal(*groups)
    kw_rows.append({"feature": JOINT_LABELS[feat], "H": H, "p": p})
kw_df = pd.DataFrame(kw_rows).sort_values("H", ascending=True)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(kw_df["feature"], kw_df["H"],
               color="#4C72B0", edgecolor="white", linewidth=0.8)
for bar, row in zip(bars, kw_df.itertuples()):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"H={row.H:.0f}", va="center", fontsize=8)
ax.set_xlabel("Kruskal-Wallis H Statistic", fontsize=11)
ax.set_title("3D BVH – KW Test per Velocity Feature\n(all p < 10⁻⁸⁰)", fontsize=12)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "3d_kw_summary.png", dpi=180)
plt.close(fig)

# ── print KW table ────────────────────────────────────────────────────────────
print("\n=== KW Results (per-joint velocity) ===")
for r in kw_df.sort_values("H", ascending=False).itertuples():
    print(f"  {r.feature:<14} H={r.H:7.1f}  p={r.p:.2e}")

print(f"\n[✓] All figures saved to {OUT_DIR}/")
