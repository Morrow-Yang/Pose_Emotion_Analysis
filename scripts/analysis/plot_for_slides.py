#!/usr/bin/env python3
"""
Generate slide-ready figures for CAER analysis (geometry + temporal).
Figures saved under docs/figs.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]  # repo root
TEMPORAL_CSV = ROOT / "outputs/analysis/temporal/caer_filtered/temporal_motion_features.csv"
MERGED_CSV = ROOT / "outputs/analysis/analysis/caer_v4_filtered/pose_features_v4_with_temporal.csv"
OUT_DIR = ROOT / "docs/figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8")


def velocity_box(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    df.boxplot(column="avg_velocity", by="emotion", ax=ax, rot=20)
    ax.set_title("Avg velocity by emotion")
    ax.set_xlabel("emotion")
    ax.set_ylabel("avg_velocity (norm)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "velocity_box.png", dpi=200)
    plt.close(fig)


def high_motion_bar(df: pd.DataFrame):
    thr = df["avg_velocity"].quantile(0.90)
    ratios = df.groupby("emotion")["avg_velocity"].apply(lambda s: (s >= thr).mean()).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ratios.plot(kind="bar", ax=ax, color="#2b83ba")
    ax.set_title(f"High-motion ratio (avg_velocity >= P90={thr:.3f})")
    ax.set_ylabel("ratio")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "high_motion_ratio.png", dpi=200)
    plt.close(fig)


def geom_vel_scatter(df: pd.DataFrame):
    cols = ["arm_span_norm", "avg_velocity", "emotion"]
    dfp = df[cols].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    emotions = sorted(dfp["emotion"].unique())
    cmap = plt.cm.tab10
    for i, emo in enumerate(emotions):
        sub = dfp[dfp["emotion"] == emo]
        ax.scatter(sub["arm_span_norm"], sub["avg_velocity"], s=12, color=cmap(i % 10), alpha=0.7, label=emo)
    ax.set_xlabel("arm_span_norm")
    ax.set_ylabel("avg_velocity")
    ax.set_title("Geometry vs motion")
    ax.legend(markerscale=1, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "geom_vel_scatter.png", dpi=200)
    plt.close(fig)


def pca_emotion(df: pd.DataFrame):
    feat_cols = [
        "arm_span_norm",
        "left_hand_height",
        "right_hand_height",
        "avg_velocity",
        "l_wrist_vel",
        "r_wrist_vel",
        "l_wrist_accel",
        "r_wrist_accel",
        "avg_acceleration",
    ]
    dfp = df[["emotion"] + feat_cols].dropna()
    if dfp.empty:
        return
    X = dfp[feat_cols].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    comp = PCA(n_components=2, random_state=42).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 4))
    emotions = sorted(dfp["emotion"].unique())
    cmap = plt.cm.tab10
    for i, emo in enumerate(emotions):
        mask = dfp["emotion"] == emo
        ax.scatter(comp[mask, 0], comp[mask, 1], s=12, color=cmap(i % 10), alpha=0.7, label=emo)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA on geom + motion features")
    ax.legend(markerscale=1, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_emotion.png", dpi=200)
    plt.close(fig)


def flow_timeline():
    steps = [
        "帧抽取 (10fps)",
        "AlphaPose 2D",
        "Top1 过滤",
        "几何特征",
        "时序特征",
        "滑窗统计",
        "几何+时序融合",
    ]
    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.axis("off")
    y = 0.5
    for i, step in enumerate(steps):
        x = i * 1.3
        ax.add_patch(plt.Rectangle((x, y - 0.25), 1.1, 0.5, edgecolor="#2b83ba", facecolor="#a6cee3"))
        ax.text(x + 0.55, y, step, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.arrow(x + 1.1, y, 0.2, 0, head_width=0.05, head_length=0.06, fc="gray", ec="gray")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "analysis_flow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    assert TEMPORAL_CSV.exists(), f"missing {TEMPORAL_CSV}"
    assert MERGED_CSV.exists(), f"missing {MERGED_CSV}"
    tdf = pd.read_csv(TEMPORAL_CSV)
    mdf = pd.read_csv(MERGED_CSV)
    velocity_box(tdf)
    high_motion_bar(tdf)
    geom_vel_scatter(mdf)
    pca_emotion(mdf)
    flow_timeline()
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
