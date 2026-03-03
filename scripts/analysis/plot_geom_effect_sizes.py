"""
Geometric Feature Effect Sizes: ε² per feature + pairwise rank-biserial r
==========================================================================
Panels:
  A: Lollipop chart – ε² for ALL 96 features, sorted within category
  B: Violin – ε² distribution by category with Cohen thresholds
  C: Heatmap – pairwise rank-biserial r for top-12 most discriminative features
     (one value per emotion pair)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import mannwhitneyu
from pathlib import Path

OUT_DIR  = Path("outputs/analysis/geom_bvh_v2")
FIGS_DIR = Path("docs/figs_3d_temporal")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprise"]
N_TOTAL  = 1402
K        = 7

CAT_COLS = {
    "velocity": "#e74c3c",
    "3D-only" : "#2ecc71",
    "2D-like" : "#3498db",
}
# Cohen's f² thresholds for η²/ε²: small=0.01, medium=0.06, large=0.14
THRESHOLDS = [(0.01, "small"),  (0.06, "medium"), (0.14, "large")]

# ─────────────────────────────────────────────────────────────────────────────

def load() -> tuple:
    kw = pd.read_csv(OUT_DIR / "kruskal_results.csv")
    kw["eps2"] = ((kw["H"] - K + 1) / (N_TOTAL - K)).clip(lower=0)
    feat_df = pd.read_csv(OUT_DIR / "bvh_geom_features.csv")
    return kw, feat_df


def panel_lollipop(ax, kw: pd.DataFrame):
    """Panel A: ε² per feature, one row per feature, grouped by category."""
    # order: velocity (by eps2 desc), 2D-like, 3D-only
    order = ["velocity", "2D-like", "3D-only"]
    chunks = []
    for cat in order:
        sub = kw[kw.category == cat].sort_values("eps2", ascending=True)
        chunks.append(sub)
    plot_df = pd.concat(chunks, ignore_index=True)

    # short label
    def shorten(name):
        name = name.replace("_mean","_μ").replace("_std","_σ").replace("_range","_Δ")
        name = name.replace("avg_velocity","vel_avg").replace("_vel_","_")
        name = name.replace("r_wrist","rW").replace("l_wrist","lW")
        name = name.replace("r_elbow","rE").replace("l_elbow","lE")
        name = name.replace("r_shoulder","rS").replace("l_shoulder","lS")
        name = name.replace("head_forward_deg","head_fwd")
        name = name.replace("pelvis_height_norm","pelvis_h")
        name = name.replace("lateral_lean_deg","lat_lean")
        name = name.replace("foot_spread_norm","foot_spr")
        name = name.replace("left_elbow_angle","Lelbow")
        name = name.replace("right_elbow_angle","Relbow")
        name = name.replace("left_hand_height","Lhand_h")
        name = name.replace("right_hand_height","Rhand_h")
        name = name.replace("left_knee_angle","Lknee")
        name = name.replace("right_knee_angle","Rknee")
        name = name.replace("contraction","contract")
        name = name.replace("elbow_asym","elb_asym")
        name = name.replace("hand_height_asym","hand_asym")
        name = name.replace("shoulder_width","shoul_w")
        name = name.replace("trunk_tilt_deg","trunk_tlt")
        name = name.replace("head_forward_deg","head_fwd")
        name = name.replace("wrist_z_asym","wz_asym")
        name = name.replace("hand_depth_diff","hdepth_Δ")
        name = name.replace("knee_bend_asym","kbend_asym")
        name = name.replace("spine_bend_deg","spine_b")
        name = name.replace("body_extent","body_ext")
        name = name.replace("arm_span_norm","arm_span")
        return name

    labels = [shorten(n) for n in plot_df["feature"]]
    colors = [CAT_COLS[c] for c in plot_df["category"]]
    y      = np.arange(len(plot_df))
    eps    = plot_df["eps2"].values

    # stem lines
    ax.hlines(y, 0, eps, colors=colors, linewidth=0.8, alpha=0.6)
    ax.scatter(eps, y, c=colors, s=18, zorder=3)

    # threshold vlines
    threshold_styles = [(0.01, ":", 0.5), (0.06, "--", 0.6), (0.14, "-.", 0.7)]
    for val, ls, al in threshold_styles:
        ax.axvline(val, color="gray", linestyle=ls, linewidth=0.9, alpha=al)

    ax.annotate("small\n0.01", xy=(0.01, len(plot_df)-1),
                xytext=(0.012, len(plot_df)*0.92), fontsize=5.5, color="gray",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    ax.annotate("medium\n0.06", xy=(0.06, len(plot_df)*0.7),
                xytext=(0.065, len(plot_df)*0.78), fontsize=5.5, color="gray",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    ax.annotate("large\n0.14", xy=(0.14, len(plot_df)*0.4),
                xytext=(0.148, len(plot_df)*0.48), fontsize=5.5, color="gray",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel("Effect size ε²  (KW η²-equivalent)", fontsize=8)
    ax.set_title("(A) Per-feature ε²: all 96 features", fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # category separator lines + labels
    pos = {"velocity": 0, "2D-like": 0, "3D-only": 0}
    idx = 0
    for cat in order:
        n = (kw.category == cat).sum()
        if idx > 0:
            ax.axhline(idx - 0.5, color="#cccccc", linewidth=0.8)
        # midpoint label
        mid = idx + n / 2 - 0.5
        ax.text(-0.005, mid, cat, ha="right", va="center",
                fontsize=6.5, color=CAT_COLS[cat], fontweight="bold",
                transform=ax.get_yaxis_transform())
        idx += n

    handles = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLS.items()]
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    ax.set_xlim(-0.005, max(eps) * 1.08)


def panel_violin(ax, kw: pd.DataFrame):
    """Panel B: violin of ε² distribution per category."""
    data   = [kw[kw.category == c]["eps2"].values for c in ["velocity", "2D-like", "3D-only"]]
    colors = [CAT_COLS[c] for c in ["velocity", "2D-like", "3D-only"]]
    labels = ["velocity\n(n=24)", "2D-like\n(n=45)", "3D-only\n(n=27)"]

    parts = ax.violinplot(data, positions=[1, 2, 3], widths=0.6,
                          showmedians=True, showextrema=True)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.65)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.5)
    for key in ["cmaxis", "cmins", "cmaxes", "cbars"]:
        if key in parts:
            parts[key].set_color("#888888")
            parts[key].set_linewidth(0.8)

    for val, ls, lbl in [(0.01, ":", "small"), (0.06, "--", "medium"), (0.14, "-.", "large")]:
        ax.axhline(val, color="gray", linestyle=ls, linewidth=0.9, alpha=0.7,
                   label=f"ε²={val} ({lbl})")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ε²", fontsize=8)
    ax.set_title("(B) ε² distribution by category", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def panel_pairwise_heatmap(ax, feat_df: pd.DataFrame, kw: pd.DataFrame, top_n: int = 12):
    """Panel C: rank-biserial r heatmap for top-N features × emotion pairs."""
    # only use features that actually exist in feat_df (velocity cols not stored there)
    available = set(feat_df.columns)
    kw_avail = kw[kw["feature"].isin(available)]
    top_feats = kw_avail.nlargest(top_n, "eps2")["feature"].tolist()

    pairs = []
    for i, e1 in enumerate(EMOTIONS):
        for e2 in EMOTIONS[i+1:]:
            pairs.append((e1, e2))

    mat = np.full((len(top_feats), len(pairs)), np.nan)
    pair_labels = [f"{a[:3]}\nvs\n{b[:3]}" for a, b in pairs]

    for fi, feat in enumerate(top_feats):
        for pi, (e1, e2) in enumerate(pairs):
            g1 = feat_df[feat_df.emotion == e1][feat].dropna().values
            g2 = feat_df[feat_df.emotion == e2][feat].dropna().values
            if len(g1) < 2 or len(g2) < 2:
                continue
            u, _ = mannwhitneyu(g1, g2, alternative="two-sided")
            r = 1 - 2 * u / (len(g1) * len(g2))  # rank-biserial r
            mat[fi, pi] = r

    def shorten_feat(name):
        name = name.replace("_mean","_μ").replace("_std","_σ").replace("_range","_Δ")
        name = name.replace("avg_velocity","vel_avg")
        name = name.replace("r_wrist_vel","rW_vel").replace("l_wrist_vel","lW_vel")
        name = name.replace("r_elbow_vel","rE_vel").replace("l_elbow_vel","lE_vel")
        name = name.replace("r_shoulder_vel","rS_vel").replace("l_shoulder_vel","lS_vel")
        name = name.replace("left_elbow_angle","Lelbow").replace("right_elbow_angle","Relbow")
        name = name.replace("left_hand_height","Lhand_h")
        name = name.replace("pelvis_height_norm","pelvis_h")
        name = name.replace("foot_spread_norm","foot_spr")
        name = name.replace("contraction","contract")
        return name

    feat_labels = [shorten_feat(f) for f in top_feats]

    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
    cmap = plt.cm.RdBu_r
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_labels, fontsize=5.5, rotation=0, ha="center")
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(feat_labels, fontsize=6.5)
    ax.set_title(f"(C) Pairwise rank-biserial r  (top-{top_n} features by ε²)", 
                 fontsize=9, fontweight="bold")

    # annotate cells
    for i in range(len(top_feats)):
        for j in range(len(pairs)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=4.5, color="white" if abs(v) > 0.35 else "black")

    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="rank-biserial r")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    kw, feat_df = load()

    fig = plt.figure(figsize=(22, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.42, wspace=0.35,
                            left=0.13, right=0.97, top=0.93, bottom=0.06)

    ax_a = fig.add_subplot(gs[:, 0])   # tall left: lollipop
    ax_b = fig.add_subplot(gs[0, 1])   # top-right: violin
    ax_c = fig.add_subplot(gs[1, 1])   # bottom-right: heatmap

    fig.suptitle("3D BVH Geometric Features – Effect Size Analysis",
                 fontsize=13, fontweight="bold")

    panel_lollipop(ax_a, kw)
    panel_violin(ax_b, kw)
    panel_pairwise_heatmap(ax_c, feat_df, kw, top_n=12)

    fig.savefig(FIGS_DIR / "3d_geom_effect_sizes.png", dpi=150, bbox_inches="tight")
    print(f"[✓] Saved → {FIGS_DIR}/3d_geom_effect_sizes.png")
    plt.close(fig)

    # ── print summary table ──────────────────────────────────────────────────
    print("\n=== Cohen's convention for ε² ===")
    print("  negligible < 0.01  ≤ small < 0.06  ≤ medium < 0.14  ≤ large")
    for cat in ["velocity", "2D-like", "3D-only"]:
        sub = kw[kw.category == cat]
        cnt = sub.shape[0]
        n_large  = (sub.eps2 >= 0.14).sum()
        n_medium = ((sub.eps2 >= 0.06) & (sub.eps2 < 0.14)).sum()
        n_small  = ((sub.eps2 >= 0.01) & (sub.eps2 < 0.06)).sum()
        n_neg    = (sub.eps2 < 0.01).sum()
        print(f"  {cat:<10} (n={cnt:2d})  "
              f"large={n_large}  medium={n_medium}  small={n_small}  negligible={n_neg}  "
              f"median_ε²={sub.eps2.median():.4f}")


if __name__ == "__main__":
    main()
