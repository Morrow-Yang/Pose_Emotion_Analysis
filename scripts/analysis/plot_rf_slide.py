"""
Single-slide RF Summary Figure
===============================
One tight figure (~16×9 aspect) for PPT:
  Left column:
    - Confusion matrix (normalised, Blues)
  Right column (top to bottom):
    - Per-class F1 bar
    - Feature importance (top-12)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

OUT_DIR  = Path("outputs/analysis/geom_bvh_v2")
FIGS_DIR = Path("docs/figs_3d_temporal")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprise"]
EMO_SHORT = {"Angry":"Ang","Disgust":"Dis","Fearful":"Fea","Happy":"Hap",
             "Neutral":"Neu","Sad":"Sad","Surprise":"Sur"}
EMO_COLS  = {"Angry":"#e74c3c","Disgust":"#8e44ad","Fearful":"#3498db",
             "Happy":"#f39c12","Neutral":"#95a5a6","Sad":"#2980b9","Surprise":"#27ae60"}
CAT_COLS  = {"velocity":"#e74c3c","3D-only":"#2ecc71","2D-like":"#3498db"}

def shorten_feat(name):
    name = name.replace("_mean","_μ").replace("_std","_σ").replace("_range","_Δ")
    subs = [("avg_velocity","vel_avg"),
            ("r_wrist_vel","rWrist"),("l_wrist_vel","lWrist"),
            ("r_elbow_vel","rElbow"),("l_elbow_vel","lElbow"),
            ("r_shoulder_vel","rShoul"),("l_shoulder_vel","lShoul"),
            ("left_elbow_angle","L_elbow"),("right_elbow_angle","R_elbow"),
            ("left_hand_height","L_hand_h"),("right_hand_height","R_hand_h"),
            ("head_forward_deg","head_fwd"),
            ("pelvis_height_norm","pelvis_h"),
            ("elbow_asym","elb_asym"),
            ("hand_height_asym","hand_asym"),
            ("left_knee_angle","L_knee"),("right_knee_angle","R_knee"),
            ("contraction","contract"),]
    for old, new in subs:
        name = name.replace(old, new)
    return name

def cat_of(name):
    vel = ("avg_velocity","head_vel","l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist")
    d3  = ("spine_bend","lateral_lean","head_forward_deg","pelvis_height","foot_spread",
           "hand_depth","wrist_z","knee_bend_asym","body_extent")
    if any(name.startswith(p) for p in vel): return "velocity"
    if any(name.startswith(p) for p in d3):  return "3D-only"
    return "2D-like"


def rebuild_rf():
    """Re-run RF with same seed to get y_true / y_pred for confusion matrix."""
    df = pd.read_csv(OUT_DIR / "bvh_geom_features.csv")
    ALL_PREFIXES = [
        "shoulder_width","left_hand_height","right_hand_height","arm_span_norm",
        "left_elbow_angle","right_elbow_angle","left_knee_angle","right_knee_angle",
        "contraction","hand_height_asym","elbow_asym","head_dx","head_dy","head_dz",
        "trunk_tilt_deg","spine_bend_deg","lateral_lean_deg","head_forward_deg",
        "pelvis_height_norm","foot_spread_norm","hand_depth_diff","wrist_z_asym",
        "knee_bend_asym","body_extent",
    ]
    feat_cols = [c for c in df.columns
                 if any(c.startswith(p) for p in ALL_PREFIXES)
                 and c not in ("emotion","filename","actor")]
    df_c = df.dropna(subset=feat_cols, how="all").copy()
    for c in feat_cols:
        df_c[c] = df_c[c].fillna(df_c[c].median())
    X = StandardScaler().fit_transform(df_c[feat_cols].values)
    y = df_c["emotion"].values
    g = df_c["actor"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, g))
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    y_pred = clf.predict(X[te])
    return y[te], y_pred


def main():
    with open(OUT_DIR / "rf_report.json") as f:
        rf = json.load(f)
    rpt = rf["report"]
    acc = rf["accuracy"]
    macro_f1 = rpt["macro avg"]["f1-score"]

    print("[+] Rebuilding RF for confusion matrix ...")
    y_true, y_pred = rebuild_rf()
    labels = [e for e in EMOTIONS if e in set(y_true)]

    # ── layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8.5))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.05, right=0.97, top=0.90, bottom=0.10,
                           wspace=0.32, hspace=0.48)
    ax_cm  = fig.add_subplot(gs[:, 0])   # full left: confusion matrix
    ax_f1  = fig.add_subplot(gs[0, 1])   # top-right: F1
    ax_imp = fig.add_subplot(gs[1, 1])   # bottom-right: importance

    fig.suptitle(
        f"Random Forest Classification  |  3D BVH Features  |  "
        f"Accuracy = {acc:.3f}   Macro-F1 = {macro_f1:.3f}   "
        f"(actor-stratified, 5-fold hold-out, n_test={len(y_true)})",
        fontsize=11, fontweight="bold", y=0.97)

    # ── A: confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    short_labels = [EMO_SHORT[e] for e in labels]
    ax_cm.set_xticks(range(len(labels))); ax_cm.set_xticklabels(short_labels, fontsize=9)
    ax_cm.set_yticks(range(len(labels))); ax_cm.set_yticklabels(labels, fontsize=9)
    ax_cm.set_xlabel("Predicted", fontsize=9); ax_cm.set_ylabel("True", fontsize=9)
    ax_cm.set_title("(A) Confusion Matrix (row-normalised)", fontsize=10, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            cnt = cm[i, j]
            ax_cm.text(j, i, f"{v:.2f}\n({cnt})",
                       ha="center", va="center", fontsize=7.5,
                       color="white" if v > 0.52 else "black")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.03)

    # ── B: per-class F1 ───────────────────────────────────────────────────────
    emos    = [e for e in EMOTIONS if e in rpt]
    f1s     = [rpt[e]["f1-score"] for e in emos]
    prec    = [rpt[e]["precision"] for e in emos]
    rec     = [rpt[e]["recall"] for e in emos]
    support = [int(rpt[e]["support"]) for e in emos]
    x       = np.arange(len(emos))
    w       = 0.26

    bars_p = ax_f1.bar(x - w, prec, width=w, label="Precision",
                       color=[EMO_COLS[e] for e in emos], alpha=0.55, edgecolor="white")
    bars_r = ax_f1.bar(x,     rec,  width=w, label="Recall",
                       color=[EMO_COLS[e] for e in emos], alpha=0.80, edgecolor="white")
    bars_f = ax_f1.bar(x + w, f1s,  width=w, label="F1",
                       color=[EMO_COLS[e] for e in emos], alpha=1.00, edgecolor="white")
    ax_f1.axhline(macro_f1, color="#e74c3c", linestyle="--", linewidth=1.3,
                  label=f"macro-F1={macro_f1:.3f}")

    for bar, v in zip(bars_f, f1s):
        ax_f1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=6.5)

    ax_f1.set_ylim(0, 1.08)
    ax_f1.set_xticks(x); ax_f1.set_xticklabels(emos, rotation=30, ha="right", fontsize=7.5)
    ax_f1.set_ylabel("Score", fontsize=8)
    ax_f1.set_title("(B) Per-class Precision / Recall / F1", fontsize=10, fontweight="bold")
    ax_f1.legend(fontsize=7, ncol=2, loc="lower right")
    ax_f1.spines[["top","right"]].set_visible(False)

    # N labels below x-axis
    for i, (bar, n) in enumerate(zip(bars_r, support)):
        ax_f1.text(x[i], -0.065, f"n={n}", ha="center", va="top",
                   fontsize=6, color="#666666", transform=ax_f1.get_xaxis_transform())

    # ── C: feature importance (top-12) ───────────────────────────────────────
    fi_all = rf["feature_importance"]
    top12  = fi_all[:12]
    names  = [shorten_feat(x[0]) for x in top12]
    imps   = [x[1] for x in top12]
    cats   = [cat_of(x[0]) for x in top12]
    colors = [CAT_COLS[c] for c in cats]

    y12 = np.arange(12)
    ax_imp.barh(y12, imps[::-1], color=colors[::-1], edgecolor="white", linewidth=0.4, height=0.7)
    ax_imp.set_yticks(y12); ax_imp.set_yticklabels(names[::-1], fontsize=7)
    ax_imp.set_xlabel("Gini Importance", fontsize=8)
    ax_imp.set_title("(C) Top-12 Feature Importances", fontsize=10, fontweight="bold")
    ax_imp.spines[["top","right"]].set_visible(False)
    ax_imp.grid(axis="x", alpha=0.25, linestyle="--")

    handles = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLS.items()]
    ax_imp.legend(handles=handles, fontsize=7, loc="lower right")

    # ── save ─────────────────────────────────────────────────────────────────
    fig.savefig(FIGS_DIR / "rf_slide.png", dpi=150, bbox_inches="tight")
    print(f"[✓] Saved → {FIGS_DIR}/rf_slide.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
