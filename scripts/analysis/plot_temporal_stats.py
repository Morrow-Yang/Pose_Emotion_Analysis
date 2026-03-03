"""
Visualize temporal (velocity/acceleration) features per emotion.
Outputs → docs/figs_temporal/
  vel_ecdf.png         – ECDF of avg_velocity (log-x)
  vel_ridge.png        – Ridge / joy plot of avg_velocity
  vel_p95_bar.png      – 95th-percentile velocity bar (tail energy)
  bodypart_vel_heatmap.png – median velocity per body-part × emotion
  accel_ecdf.png       – ECDF of avg_acceleration (log-x)
  kw_summary.png       – KW significance summary bar
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kruskal
from pathlib import Path

# ── config ───────────────────────────────────────────────────────────────────
CSV = "outputs/analysis/temporal/caer_filtered/temporal_motion_features.csv"
OUT = Path("docs/figs_temporal")
OUT.mkdir(parents=True, exist_ok=True)

# ordered by avg_velocity median (high → low)
EMO_ORDER = ["Surprise", "Fear", "Happy", "Neutral", "Sad", "Anger", "Disgust"]
PALETTE = {
    "Anger":    "#e63946",
    "Disgust":  "#f4a261",
    "Fear":     "#b5838d",
    "Happy":    "#2a9d8f",
    "Neutral":  "#457b9d",
    "Sad":      "#6d6875",
    "Surprise": "#a8dadc",
}

df = pd.read_csv(CSV)

# ── 1. ECDF  avg_velocity  (log x) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for emo in EMO_ORDER:
    vals = df.loc[df["emotion"] == emo, "avg_velocity"].dropna()
    vals = vals[vals > 0]
    vals_s = np.sort(vals)
    ecdf = np.arange(1, len(vals_s)+1) / len(vals_s)
    ax.plot(vals_s, ecdf, color=PALETTE[emo], lw=1.8, label=emo, alpha=0.9)
    ax.axvline(np.median(vals_s), color=PALETTE[emo], lw=0.7, ls="--", alpha=0.45)

ax.set_xscale("log")
ax.set_xlabel("avg_velocity (log scale, normalised units)", fontsize=10)
ax.set_ylabel("Cumulative proportion", fontsize=10)
ax.set_title("ECDF — avg_velocity per emotion\n(KW p=1.95e-18, dashed=median)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="lower right")
ax.grid(axis="x", alpha=0.25)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(OUT/"vel_ecdf.png", dpi=150); plt.close(fig)
print("[saved] vel_ecdf.png")

# ── 2. Ridge plot  avg_velocity ─────────────────────────────────────────────
fig, axes = plt.subplots(len(EMO_ORDER), 1,
                          figsize=(7, len(EMO_ORDER)*1.0), sharex=True)
xs = np.linspace(0, 80, 400)          # focus on main body (clip display)
for i, emo in enumerate(EMO_ORDER):
    ax = axes[i]
    vals = df.loc[df["emotion"] == emo, "avg_velocity"].dropna()
    vals_c = vals.clip(0, vals.quantile(0.99))
    try:
        kde = gaussian_kde(vals_c, bw_method=0.25)
        ys  = kde(xs)
    except Exception:
        ys  = np.zeros_like(xs)
    ax.fill_between(xs, ys, alpha=0.55, color=PALETTE[emo])
    ax.plot(xs, ys, color=PALETTE[emo], lw=1.5)
    ax.axvline(np.median(vals_c), color="black", lw=1.1, ls="--", alpha=0.75)
    ax.set_yticks([])
    ax.set_ylabel(emo, rotation=0, labelpad=60, va="center", fontsize=9)
    ax.spines[["top","right","left"]].set_visible(False)

axes[-1].set_xlabel("avg_velocity (normalised units, display clipped at p99)",
                     fontsize=10)
fig.suptitle("Distribution — avg_velocity by emotion  (dashed = median)",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT/"vel_ridge.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[saved] vel_ridge.png")

# ── 3. 95th-percentile velocity bar  (tail energy) ──────────────────────────
p95 = {emo: df.loc[df["emotion"]==emo,"avg_velocity"].dropna().quantile(0.95)
       for emo in EMO_ORDER}
med = {emo: df.loc[df["emotion"]==emo,"avg_velocity"].dropna().median()
       for emo in EMO_ORDER}

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(EMO_ORDER))
bars = ax.bar(x, [p95[e] for e in EMO_ORDER],
              color=[PALETTE[e] for e in EMO_ORDER], alpha=0.85, label="95th pct")
ax.scatter(x, [med[e] for e in EMO_ORDER],
           color="black", zorder=5, s=50, label="median")
for bar, val in zip(bars, [p95[e] for e in EMO_ORDER]):
    ax.text(bar.get_x()+bar.get_width()/2, val+2, f"{val:.0f}",
            ha="center", va="bottom", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(EMO_ORDER, fontsize=9)
ax.set_ylabel("avg_velocity (normalised units)", fontsize=10)
ax.set_title("95th-percentile velocity (tail energy) per emotion\n"
             "● median shown for reference", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(OUT/"vel_p95_bar.png", dpi=150); plt.close(fig)
print("[saved] vel_p95_bar.png")

# ── 4. Body-part velocity heatmap ───────────────────────────────────────────
parts = {
    "nose":       "nose_vel",
    "L shoulder": "l_shoulder_vel",
    "R shoulder": "r_shoulder_vel",
    "L elbow":    "l_elbow_vel",
    "R elbow":    "r_elbow_vel",
}
heat_data = {}
for label, col in parts.items():
    heat_data[label] = df.groupby("emotion")[col].median()

heat_df = pd.DataFrame(heat_data).loc[EMO_ORDER]
heat_z  = (heat_df - heat_df.mean()) / heat_df.std()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                          gridspec_kw={"width_ratios":[1, 1]})

# left: raw median
im0 = axes[0].imshow(heat_df.values, aspect="auto", cmap="YlOrRd")
axes[0].set_yticks(range(len(EMO_ORDER))); axes[0].set_yticklabels(EMO_ORDER, fontsize=9)
axes[0].set_xticks(range(len(parts)));     axes[0].set_xticklabels(list(parts), fontsize=8, rotation=20, ha="right")
for i in range(len(EMO_ORDER)):
    for j, col in enumerate(parts):
        axes[0].text(j, i, f"{heat_df.iloc[i,j]:.1f}", ha="center", va="center", fontsize=7.5)
plt.colorbar(im0, ax=axes[0]).set_label("median velocity")
axes[0].set_title("Median velocity per body part", fontsize=10, fontweight="bold")

# right: z-score
im1 = axes[1].imshow(heat_z.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
axes[1].set_yticks(range(len(EMO_ORDER))); axes[1].set_yticklabels(EMO_ORDER, fontsize=9)
axes[1].set_xticks(range(len(parts)));     axes[1].set_xticklabels(list(parts), fontsize=8, rotation=20, ha="right")
for i in range(len(EMO_ORDER)):
    for j in range(len(parts)):
        axes[1].text(j, i, f"{heat_z.iloc[i,j]:.1f}", ha="center", va="center", fontsize=7.5)
plt.colorbar(im1, ax=axes[1]).set_label("z-score")
axes[1].set_title("Z-scored median velocity (green=high)", fontsize=10, fontweight="bold")

fig.suptitle("Body-part velocity heatmap per emotion", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT/"bodypart_vel_heatmap.png", dpi=150); plt.close(fig)
print("[saved] bodypart_vel_heatmap.png")

# ── 5. avg_acceleration ECDF (log x) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for emo in EMO_ORDER:
    vals = df.loc[df["emotion"]==emo, "avg_acceleration"].dropna()
    vals = vals[vals > 0]
    vals_s = np.sort(vals)
    ecdf = np.arange(1, len(vals_s)+1)/len(vals_s)
    ax.plot(vals_s, ecdf, color=PALETTE[emo], lw=1.8, label=emo, alpha=0.9)
    ax.axvline(np.median(vals_s), color=PALETTE[emo], lw=0.7, ls="--", alpha=0.45)

ax.set_xscale("log")
ax.set_xlabel("avg_acceleration (log scale, normalised units)", fontsize=10)
ax.set_ylabel("Cumulative proportion", fontsize=10)
ax.set_title("ECDF — avg_acceleration per emotion\n(KW p=1.68e-26, dashed=median)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="lower right")
ax.grid(axis="x", alpha=0.25)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(OUT/"accel_ecdf.png", dpi=150); plt.close(fig)
print("[saved] accel_ecdf.png")

# ── 6. KW significance summary ──────────────────────────────────────────────
kw_results = []
check_cols = [
    ("avg_velocity",    "avg velocity"),
    ("avg_acceleration","avg acceleration"),
    ("nose_vel",        "nose vel"),
    ("l_shoulder_vel",  "L shoulder vel"),
    ("r_shoulder_vel",  "R shoulder vel"),
    ("l_elbow_vel",     "L elbow vel"),
    ("r_elbow_vel",     "R elbow vel"),
    ("l_shoulder_accel","L shoulder accel"),
    ("r_shoulder_accel","R shoulder accel"),
    ("l_elbow_accel",   "L elbow accel"),
    ("nose_accel",      "nose accel"),
]
for col, label in check_cols:
    sub = df[["emotion", col]].dropna()
    if sub.shape[0] < 50: continue
    groups = [g[col].values for _,g in sub.groupby("emotion")]
    try:
        kw, kp = kruskal(*groups)
        kw_results.append((label, kw, kp))
    except: pass

kw_results.sort(key=lambda x: -x[1])
fig, ax = plt.subplots(figsize=(7, 4.5))
labels_kw = [r[0] for r in kw_results]
kw_stats   = [r[1] for r in kw_results]
colors_kw  = ["#e63946" if r[2]<0.001 else "#f4a261" if r[2]<0.05 else "#adb5bd"
              for r in kw_results]
bars = ax.barh(labels_kw, kw_stats, color=colors_kw, alpha=0.85)
ax.axvline(0, color="black", lw=0.5)
for bar, (_, kw, kp) in zip(bars, kw_results):
    ax.text(kw+0.5, bar.get_y()+bar.get_height()/2,
            f"p={kp:.1e}", va="center", fontsize=7.5)
ax.set_xlabel("Kruskal–Wallis statistic", fontsize=10)
ax.set_title("Temporal feature discriminability (KW stat)\n"
             "red=p<0.001, orange=p<0.05, grey=n.s.", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(OUT/"kw_summary.png", dpi=150); plt.close(fig)
print("[saved] kw_summary.png")

print("\nAll done →", OUT)
