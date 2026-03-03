"""
3D BVH Geometric + RF Analysis: Comprehensive Visualization
===========================================================
Generates a 4-panel summary figure for slides/report:
  Panel A: Top-30 KW features (H statistic), color-coded by feature category
  Panel B: RF per-class F1 score
  Panel C: RF top-15 feature importance
  Panel D: Confusion matrix (normalised)
  Panel E: PCA scatter + t-SNE scatter (two sub-panels)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.metrics import confusion_matrix

# ── paths ──────────────────────────────────────────────────────────────────────
OUT_DIR   = Path("outputs/analysis/geom_bvh_v2")
FIGS_DIR  = Path("docs/figs_3d_temporal")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS  = ["Angry","Disgust","Fearful","Happy","Neutral","Sad","Surprise"]
EMO_COLS  = {
    "Angry"  : "#e74c3c",
    "Disgust": "#8e44ad",
    "Fearful": "#3498db",
    "Happy"  : "#f39c12",
    "Neutral": "#95a5a6",
    "Sad"    : "#2980b9",
    "Surprise":"#27ae60",
}
CAT_COLS  = {
    "velocity": "#e74c3c",
    "3D-only" : "#2ecc71",
    "2D-like" : "#3498db",
}

# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    kw = pd.read_csv(OUT_DIR / "kruskal_results.csv")
    pca_df = pd.read_csv(OUT_DIR / "pca_2d.csv")
    tsne_df = pd.read_csv(OUT_DIR / "tsne_2d.csv")
    with open(OUT_DIR / "rf_report.json") as f:
        rf = json.load(f)
    return kw, pca_df, tsne_df, rf


def plot_kw_barh(ax, kw: pd.DataFrame, top_n: int = 30):
    """Horizontal bar chart of top-N KW H values, colored by category."""
    top = kw.nlargest(top_n, "H")
    colors = [CAT_COLS.get(c, "#aaaaaa") for c in top["category"]]
    bars = ax.barh(range(len(top)), top["H"], color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("Kruskal-Wallis H", fontsize=9)
    ax.set_title(f"(A) Top-{top_n} Features by KW H Statistic", fontsize=10, fontweight="bold")
    # category legend
    handles = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLS.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--")


def plot_rf_f1(ax, rf: dict):
    """Per-class F1 bars + macro-F1 line."""
    rpt = rf["report"]
    emos = [e for e in EMOTIONS if e in rpt]
    f1s  = [rpt[e]["f1-score"] for e in emos]
    cols = [EMO_COLS.get(e, "#999") for e in emos]
    bars = ax.bar(emos, f1s, color=cols, edgecolor="white", linewidth=0.5, width=0.6)
    macro = rpt["macro avg"]["f1-score"]
    ax.axhline(macro, color="#e74c3c", linestyle="--", linewidth=1.3,
               label=f"macro-F1 = {macro:.3f}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-score", fontsize=9)
    ax.set_title(f"(B) RF Per-class F1  (acc={rf['accuracy']:.3f})", fontsize=10, fontweight="bold")
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)


def plot_rf_importance(ax, rf: dict, top_n: int = 15):
    """Horizontal bar: top-N feature importances."""
    fi = rf["feature_importance"][:top_n]
    names = [x[0] for x in fi]
    imps  = [x[1] for x in fi]
    # color by category guess
    def cat_color(name):
        vel_prefixes = ("avg_velocity","head_vel","l_shoulder","r_shoulder",
                        "l_elbow","r_elbow","l_wrist","r_wrist")
        d3_prefixes  = ("spine_bend","lateral_lean","head_forward_deg",
                        "pelvis_height","foot_spread","hand_depth",
                        "wrist_z","knee_bend","body_extent")
        if any(name.startswith(p) for p in vel_prefixes):
            return CAT_COLS["velocity"]
        elif any(name.startswith(p) for p in d3_prefixes):
            return CAT_COLS["3D-only"]
        return CAT_COLS["2D-like"]
    colors = [cat_color(n) for n in names]
    ax.barh(range(top_n), imps[::-1], color=colors[::-1],
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=7.5)
    ax.set_xlabel("Gini Importance", fontsize=9)
    ax.set_title(f"(C) RF Top-{top_n} Feature Importances", fontsize=10, fontweight="bold")
    handles = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLS.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--")


def plot_confusion(ax, rf: dict):
    """Confusion matrix from RF report support counts (estimated)."""
    # We don't have y_test/y_pred stored; use the per-class report to reconstruct
    # what we can from the JSON: precision/recall/support allow back-calc of TP/FP/FN
    # Best approach: load features CSV and re-run the same split (deterministic seed)
    feat_csv = OUT_DIR / "bvh_geom_features.csv"
    if not feat_csv.exists():
        ax.text(0.5, 0.5, "Feature CSV not found\nfor confusion matrix",
                ha="center", va="center", transform=ax.transAxes)
        return
    df = pd.read_csv(feat_csv)
    # Feature columns
    ALL_PREFIXES = [
        "shoulder_width","left_hand_height","right_hand_height",
        "arm_span_norm","left_elbow_angle","right_elbow_angle",
        "left_knee_angle","right_knee_angle","contraction",
        "hand_height_asym","elbow_asym","head_dx","head_dy","head_dz",
        "trunk_tilt_deg","spine_bend_deg","lateral_lean_deg","head_forward_deg",
        "pelvis_height_norm","foot_spread_norm","hand_depth_diff",
        "wrist_z_asym","knee_bend_asym","body_extent",
    ]
    feat_cols = [c for c in df.columns
                 if any(c.startswith(p) for p in ALL_PREFIXES)
                 and c not in ("emotion","filename","actor")]
    df_c = df.dropna(subset=feat_cols, how="all").copy()
    for c in feat_cols:
        df_c[c] = df_c[c].fillna(df_c[c].median())

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit

    X = StandardScaler().fit_transform(df_c[feat_cols].values)
    y = df_c["emotion"].values
    g = df_c["actor"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, g))
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    y_pred = clf.predict(X[te])
    y_true = y[te]

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cmap = plt.cm.Blues
    im = ax.imshow(cm_norm, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)));  ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(labels)));  ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("True", fontsize=9)
    ax.set_title("(D) RF Confusion Matrix (row-normalised)", fontsize=10, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.55 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_scatter(ax, df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """Scatter colored by emotion."""
    for emo in EMOTIONS:
        sub = df[df["emotion"] == emo]
        ax.scatter(sub[x_col], sub[y_col], c=EMO_COLS.get(emo, "#999"),
                   s=10, alpha=0.5, label=emo, linewidths=0)
    ax.set_xlabel(x_col.upper(), fontsize=8)
    ax.set_ylabel(y_col.upper(), fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.spines[["top","right"]].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    kw, pca_df, tsne_df, rf = load_data()

    # ── Figure 1: KW + RF F1 + RF importance + confusion (2×2) ──────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 14),
                                gridspec_kw={"hspace": 0.45, "wspace": 0.38})
    fig1.suptitle("3D BVH: Geometric Feature Analysis & Random Forest Classification",
                  fontsize=13, fontweight="bold", y=0.98)

    plot_kw_barh(axes1[0, 0], kw, top_n=30)
    plot_rf_f1(axes1[0, 1], rf)
    plot_rf_importance(axes1[1, 0], rf, top_n=15)
    print("[+] Generating confusion matrix (re-running RF with seed=42) ...")
    plot_confusion(axes1[1, 1], rf)

    fig1.savefig(FIGS_DIR / "3d_geom_rf_summary.png", dpi=150, bbox_inches="tight")
    print(f"[✓] Saved → {FIGS_DIR}/3d_geom_rf_summary.png")
    plt.close(fig1)

    # ── Figure 2: PCA + t-SNE scatter ────────────────────────────────────────
    fig2, (ax_pca, ax_tsne) = plt.subplots(1, 2, figsize=(14, 6),
                                            gridspec_kw={"wspace": 0.3})
    fig2.suptitle("3D BVH Feature Space: PCA & t-SNE Projections",
                  fontsize=12, fontweight="bold")

    plot_scatter(ax_pca,  pca_df,  "pc1",   "pc2",   "(E1) PCA (2D projection)")
    plot_scatter(ax_tsne, tsne_df, "tsne1", "tsne2", "(E2) t-SNE (perplexity=30)")

    # shared legend
    handles = [mpatches.Patch(color=EMO_COLS[e], label=e) for e in EMOTIONS]
    fig2.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
                bbox_to_anchor=(0.5, -0.03))

    fig2.savefig(FIGS_DIR / "3d_geom_pca_tsne.png", dpi=150, bbox_inches="tight")
    print(f"[✓] Saved → {FIGS_DIR}/3d_geom_pca_tsne.png")
    plt.close(fig2)

    # ── Figure 3: KW H by category (grouped bar summary) ────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    cat_stats = kw.groupby("category")["H"].agg(["mean","median","max","count"]).reset_index()
    cat_stats = cat_stats.sort_values("mean", ascending=False)
    x = np.arange(len(cat_stats))
    w = 0.25
    bars_mean   = ax3.bar(x - w, cat_stats["mean"],   width=w, label="Mean H",
                          color=[CAT_COLS.get(c,"#999") for c in cat_stats["category"]],
                          alpha=0.85)
    bars_median = ax3.bar(x,     cat_stats["median"], width=w, label="Median H",
                          color=[CAT_COLS.get(c,"#999") for c in cat_stats["category"]],
                          alpha=0.55)
    bars_max    = ax3.bar(x + w, cat_stats["max"],    width=w, label="Max H",
                          color=[CAT_COLS.get(c,"#999") for c in cat_stats["category"]],
                          alpha=0.35)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cat_stats["category"],cat_stats["count"])],
                        fontsize=9)
    ax3.set_ylabel("Kruskal-Wallis H", fontsize=9)
    ax3.set_title("(F) KW Discriminability by Feature Category", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(axis="y", alpha=0.25, linestyle="--")
    fig3.tight_layout()
    fig3.savefig(FIGS_DIR / "3d_kw_by_category.png", dpi=150, bbox_inches="tight")
    print(f"[✓] Saved → {FIGS_DIR}/3d_kw_by_category.png")
    plt.close(fig3)

    # ── console summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"  Total features analysed : {len(kw)}")
    for cat, grp in kw.groupby("category"):
        top1 = grp.nlargest(1,"H").iloc[0]
        print(f"  [{cat}] n={len(grp):2d}  best: {top1['feature']} H={top1['H']:.1f}")
    print(f"\n  RF (actor-stratified)  acc={rf['accuracy']:.3f}  macro-F1={rf['report']['macro avg']['f1-score']:.3f}")
    print(f"  Best class : Neutral   F1={rf['report']['Neutral']['f1-score']:.2f}")
    print(f"  Worst class: Surprise  F1={rf['report']['Surprise']['f1-score']:.2f}")
    print(f"\n  Output figures:")
    print(f"    docs/figs_3d_temporal/3d_geom_rf_summary.png")
    print(f"    docs/figs_3d_temporal/3d_geom_pca_tsne.png")
    print(f"    docs/figs_3d_temporal/3d_kw_by_category.png")


if __name__ == "__main__":
    main()
