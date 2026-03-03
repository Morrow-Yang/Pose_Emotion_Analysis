#!/usr/bin/env python3
"""
Plot geometry-only figures from pose_features_v4.csv.
Outputs to docs/figs_geom.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Helper: robust clipping to reduce extreme outliers for plotting
def clip_series(s: pd.Series, lo=0.01, hi=0.99):
    q1, q2 = s.quantile([lo, hi])
    return s.clip(q1, q2)

ROOT = Path(__file__).resolve().parents[2]
GEOM_CSV = ROOT / "outputs/analysis/analysis/v4/pose_features_v4.csv"
OUT_DIR = ROOT / "docs/figs_geom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8")


def boxplot_feature(df: pd.DataFrame, feature: str, fname: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    df_plot = df.copy()
    df_plot[feature] = clip_series(df_plot[feature])
    df_plot.boxplot(column=feature, by="emotion", ax=ax, rot=20, grid=False, showcaps=False)
    ax.set_title(title)
    ax.set_xlabel("emotion")
    ax.set_ylabel(feature)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=200)
    plt.close(fig)


def scatter_pairs(df: pd.DataFrame, x: str, y: str, fname: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    emotions = sorted(df["emotion"].unique())
    cmap = plt.cm.tab10
    for i, emo in enumerate(emotions):
        sub = df[df["emotion"] == emo].copy()
        sub[x] = clip_series(sub[x])
        sub[y] = clip_series(sub[y])
        ax.scatter(sub[x], sub[y], s=8, color=cmap(i % 10), alpha=0.35, label=emo)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(markerscale=1, fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=200)
    plt.close(fig)


def pca_geom(df: pd.DataFrame):
    geom_cols = [
        "arm_span_norm",
        "left_hand_height",
        "right_hand_height",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
        "contraction",
        "hand_height_asym",
        "elbow_asym",
        "head_dx",
        "head_dy",
    ]
    dfp = df[["emotion"] + geom_cols].dropna()
    if dfp.empty:
        return
    X = dfp[geom_cols].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    comp = PCA(n_components=2, random_state=42).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 4))
    emotions = sorted(dfp["emotion"].unique())
    cmap = plt.cm.Set2
    for i, emo in enumerate(emotions):
        mask = dfp["emotion"] == emo
        ax.scatter(comp[mask, 0], comp[mask, 1], s=8, color=cmap(i % len(emotions)), alpha=0.5, label=emo)
    ax.set_xlabel("PC1 (geom)")
    ax.set_ylabel("PC2 (geom)")
    ax.set_title("PCA on geometry features")
    ax.legend(markerscale=1, fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_geom.png", dpi=200)
    plt.close(fig)


def main():
    assert GEOM_CSV.exists(), f"missing {GEOM_CSV}"
    df = pd.read_csv(GEOM_CSV)

    boxplot_feature(df, "arm_span_norm", "arm_span_norm_box.png", "Arm span (norm) by emotion")
    boxplot_feature(df, "hand_height_asym", "hand_height_asym_box.png", "Hand height asymmetry by emotion")
    scatter_pairs(df.dropna(subset=["arm_span_norm", "contraction"]), "arm_span_norm", "contraction", "armspan_vs_contraction.png", "Arm span vs contraction")
    pca_geom(df)
    print(f"Saved geometry-only figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
