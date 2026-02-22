#说明：这个代码是对之前5万张情绪图跑alphapose之后的json文件进行分析，
# 计算下面参数指标并对计算得到的指标
#   1.绘制箱型图和显著性分析、
#   2.计算效应量得到不同情绪之间最显著的轴特征
#   3.此外还进行聚类分析，每张图像的17个姿态点作为一个样本，得到表达情绪的几个底层姿态模型
# rec = {
#                 "emotion": emotion,
#                 "image_id": image_id,
#                 "pose_score": float(score) if score is not None else np.nan,
#                 "valid_kp": valid,
#                 "bbox_w": bw,
#                 "bbox_h": bh,
#                 "shoulder_width": shoulder_width,
#                 "left_hand_height": left_hand_height,
#                 "right_hand_height": right_hand_height,
#                 "arm_span_norm": arm_span_norm,
#                 "left_elbow_angle": left_elbow_angle,
#                 "right_elbow_angle": right_elbow_angle,
#                 "left_knee_angle": left_knee_angle,
#                 "right_knee_angle": right_knee_angle,
#                 "contraction": contract,
#                 "hand_height_asym": hand_height_asym,
#                 "elbow_asym": elbow_asym,
#                 "head_dx": head_dx,
#                 "head_dy": head_dy,
#             }

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


# -----------------------------
# CONFIG (CLI)
# -----------------------------
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing per-emotion subfolders.")
    ap.add_argument("--json_name", default="alphapose-results.json", help="AlphaPose result filename in each emotion folder.")
    ap.add_argument("--out_dir", required=False, default=None, help="Output directory. If not set, uses <root>/pose_analysis_v1_cli.")
    return ap.parse_args()

_args = parse_args()
ROOT = Path(_args.root)
JSON_NAME = _args.json_name
OUT_DIR = Path(_args.out_dir) if _args.out_dir else (ROOT / "pose_analysis_v1_cli")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_TH = 0.30
MIN_VALID_KP = 10

# clustering
CLUSTER_K_LIST = [3, 4, 5, 6, 7, 8]  # will pick best by silhouette
RANDOM_STATE = 42

# If umap-learn is installed, we'll use it; otherwise PCA 2D
USE_UMAP_IF_AVAILABLE = True


# -----------------------------
# COCO keypoint indices
# -----------------------------
KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
]


# -----------------------------
# Helpers
# -----------------------------
def parse_kp(arr_51):
    """Return (xy: [17,2], conf: [17])"""
    a = np.array(arr_51, dtype=float).reshape(17, 3)
    xy = a[:, :2]
    cf = a[:, 2]
    return xy, cf


def angle(a, b, c):
    """Angle ABC in degrees. a,b,c are 2D points."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-8 or nbc < 1e-8:
        return np.nan
    cosv = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def safe_point(xy, cf, name):
    i = KP[name]
    if cf[i] >= CONF_TH and np.isfinite(xy[i]).all():
        return xy[i]
    return None


def normalize_pose(xy, cf):
    """Normalize using pelvis as origin and torso length as scale.
       Returns norm_xy [17,2], plus info dict; returns (None,None) if cannot normalize.
    """
    lh = safe_point(xy, cf, "left_hip")
    rh = safe_point(xy, cf, "right_hip")
    ls = safe_point(xy, cf, "left_shoulder")
    rs = safe_point(xy, cf, "right_shoulder")

    if lh is None or rh is None or ls is None or rs is None:
        return None, None

    pelvis = (lh + rh) / 2.0
    shoulder_mid = (ls + rs) / 2.0
    scale = np.linalg.norm(shoulder_mid - pelvis)

    if not np.isfinite(scale) or scale < 1e-6:
        return None, None

    norm_xy = (xy - pelvis[None, :]) / scale
    info = {"scale": float(scale)}
    return norm_xy, info


def contraction_index(norm_xy, cf):
    """More positive = more contracted (we define as negative mean distance)."""
    # Use upper-body-ish points if present
    use = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
           "left_wrist", "right_wrist", "left_hip", "right_hip"]
    pts = []
    for n in use:
        i = KP[n]
        if cf[i] >= CONF_TH and np.isfinite(norm_xy[i]).all():
            pts.append(norm_xy[i])
    if len(pts) < 5:
        return np.nan

    pts = np.stack(pts, axis=0)
    # torso center = mean of shoulders and hips
    ls, rs = norm_xy[KP["left_shoulder"]], norm_xy[KP["right_shoulder"]]
    lh, rh = norm_xy[KP["left_hip"]], norm_xy[KP["right_hip"]]
    torso_center = (ls + rs + lh + rh) / 4.0
    d = np.linalg.norm(pts - torso_center[None, :], axis=1)
    return float(-np.mean(d))


def cliffs_delta(x, y):
    """Cliff's delta: [-1,1]. Robust effect size for ordinal/non-normal data."""
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # O(n*m) could be heavy; use rank-based approximation for speed
    # Exact via broadcasting is OK for small; for large, do rank method:
    combined = np.concatenate([x, y])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    rx = ranks[: len(x)]
    ry = ranks[len(x) :]
    # Use Mann–Whitney U relationship
    # U = sum(rx) - n_x(n_x+1)/2
    nx, ny = len(x), len(y)
    U = rx.sum() - nx * (nx + 1) / 2.0
    delta = (2 * U) / (nx * ny) - 1
    return float(delta)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# Load all emotions
# -----------------------------
records = []
emotion_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])

for ed in emotion_dirs:
    emotion = ed.name
    jf = ed / JSON_NAME
    if not jf.exists():
        continue
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"[WARN] {jf} not a list. Skipped.")
        continue

    for item in data:
        try:
            image_id = item.get("image_id", None)
            kps = item.get("keypoints", None)
            box = item.get("box", [np.nan]*4)
            score = item.get("score", np.nan)
            if kps is None:
                continue

            xy, cf = parse_kp(kps)

            valid = int(np.sum(cf >= CONF_TH))
            if valid < MIN_VALID_KP:
                continue

            norm_xy, info = normalize_pose(xy, cf)
            if norm_xy is None:
                continue

            # core points
            ls = safe_point(norm_xy, cf, "left_shoulder")
            rs = safe_point(norm_xy, cf, "right_shoulder")
            lw = safe_point(norm_xy, cf, "left_wrist")
            rw = safe_point(norm_xy, cf, "right_wrist")
            le = safe_point(norm_xy, cf, "left_elbow")
            re = safe_point(norm_xy, cf, "right_elbow")
            lh = safe_point(norm_xy, cf, "left_hip")
            rh = safe_point(norm_xy, cf, "right_hip")
            lk = safe_point(norm_xy, cf, "left_knee")
            rk = safe_point(norm_xy, cf, "right_knee")
            la = safe_point(norm_xy, cf, "left_ankle")
            ra = safe_point(norm_xy, cf, "right_ankle")
            nose = safe_point(norm_xy, cf, "nose")

            # Features (interpretable)
            shoulder_width = np.linalg.norm(ls - rs) if (ls is not None and rs is not None) else np.nan

            # hand height relative to shoulder (y axis: image coords; after normalization still same sign convention)
            left_hand_height = (lw[1] - ls[1]) if (lw is not None and ls is not None) else np.nan
            right_hand_height = (rw[1] - rs[1]) if (rw is not None and rs is not None) else np.nan

            # arm span openness
            arm_span = abs(lw[0] - rw[0]) if (lw is not None and rw is not None) else np.nan
            arm_span_norm = arm_span / shoulder_width if (np.isfinite(arm_span) and np.isfinite(shoulder_width) and shoulder_width > 1e-6) else np.nan

            # elbow angles
            left_elbow_angle = angle(ls, le, lw) if (ls is not None and le is not None and lw is not None) else np.nan
            right_elbow_angle = angle(rs, re, rw) if (rs is not None and re is not None and rw is not None) else np.nan

            # knee angles
            left_knee_angle = angle(lh, lk, la) if (lh is not None and lk is not None and la is not None) else np.nan
            right_knee_angle = angle(rh, rk, ra) if (rh is not None and rk is not None and ra is not None) else np.nan

            # contraction
            contract = contraction_index(norm_xy, cf)

            # symmetry
            hand_height_asym = abs(left_hand_height - right_hand_height) if (np.isfinite(left_hand_height) and np.isfinite(right_hand_height)) else np.nan
            elbow_asym = abs(left_elbow_angle - right_elbow_angle) if (np.isfinite(left_elbow_angle) and np.isfinite(right_elbow_angle)) else np.nan

            # head position relative to shoulder mid (proxy for head up/down & lean)
            if nose is not None and ls is not None and rs is not None:
                shoulder_mid = (ls + rs) / 2.0
                head_dx = float(nose[0] - shoulder_mid[0])
                head_dy = float(nose[1] - shoulder_mid[1])
            else:
                head_dx, head_dy = np.nan, np.nan

            # bbox size proxy (helps later stratify if needed)
            bx, by, bw, bh = [float(x) for x in (box if len(box) == 4 else [np.nan]*4)]

            rec = {
                "emotion": emotion,
                "image_id": image_id,
                "pose_score": float(score) if score is not None else np.nan,
                "valid_kp": valid,
                "bbox_w": bw,
                "bbox_h": bh,
                "shoulder_width": shoulder_width,
                "left_hand_height": left_hand_height,
                "right_hand_height": right_hand_height,
                "arm_span_norm": arm_span_norm,
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "left_knee_angle": left_knee_angle,
                "right_knee_angle": right_knee_angle,
                "contraction": contract,
                "hand_height_asym": hand_height_asym,
                "elbow_asym": elbow_asym,
                "head_dx": head_dx,
                "head_dy": head_dy,
            }

            # For clustering later: save normalized coords (34 dims)
            flat = norm_xy.reshape(-1)
            rec.update({f"kp_{i:02d}": float(flat[i]) for i in range(flat.shape[0])})
            records.append(rec)

        except Exception:
            continue

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No valid samples loaded. Check paths/filters.")

# Save features
feat_csv = OUT_DIR / "pose_features.csv"
df.to_csv(feat_csv, index=False, encoding="utf-8-sig")
print("Saved features:", feat_csv)
print("Samples:", len(df), "Emotions:", df["emotion"].nunique())


# -----------------------------
# A) Explainable statistics
# -----------------------------
FEATURES = [
    "left_hand_height", "right_hand_height",
    "arm_span_norm",
    "left_elbow_angle", "right_elbow_angle",
    "contraction",
    "hand_height_asym", "elbow_asym",
    "head_dx", "head_dy",
    "left_knee_angle", "right_knee_angle",
]

# Overall Kruskal
overall_rows = []
pair_rows = []

emotions = sorted(df["emotion"].unique())

for f in FEATURES:
    groups = [df.loc[df["emotion"] == e, f].dropna().to_numpy() for e in emotions]
    # Keep only emotions with enough samples
    valid_pairs = [(e, g) for e, g in zip(emotions, groups) if len(g) >= 10]
    if len(valid_pairs) < 2:
        continue
    emos2, groups2 = zip(*valid_pairs)

    try:
        H, p = kruskal(*groups2)
    except Exception:
        continue

    overall_rows.append({
        "feature": f,
        "n_emotions": len(emos2),
        "H": float(H),
        "p": float(p),
        "median_by_emotion": "; ".join([f"{e}:{np.median(g):.4f}" for e, g in zip(emos2, groups2)]),
        "n_by_emotion": "; ".join([f"{e}:{len(g)}" for e, g in zip(emos2, groups2)]),
    })

    # Pairwise Mann–Whitney + Cliff's delta
    pvals = []
    pairs = []
    deltas = []
    for i in range(len(emos2)):
        for j in range(i + 1, len(emos2)):
            e1, e2 = emos2[i], emos2[j]
            x, y = groups2[i], groups2[j]
            # Mann-Whitney U (two-sided)
            try:
                U, p2 = mannwhitneyu(x, y, alternative="two-sided")
            except Exception:
                continue
            d = cliffs_delta(x, y)
            pairs.append((e1, e2, float(U), float(d)))
            pvals.append(float(p2))
            deltas.append(float(d))

    if pvals:
        rej, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        for (e1, e2, U, d), p_raw, p_a, rj in zip(pairs, pvals, p_adj, rej):
            pair_rows.append({
                "feature": f,
                "emotion_1": e1,
                "emotion_2": e2,
                "U": U,
                "p_raw": p_raw,
                "p_fdr": float(p_a),
                "reject_fdr_0.05": bool(rj),
                "cliffs_delta": d,
            })

overall_df = pd.DataFrame(overall_rows).sort_values("p", ascending=True)
pair_df = pd.DataFrame(pair_rows).sort_values(["feature", "p_fdr"], ascending=[True, True])

overall_path = OUT_DIR / "stats_overall.csv"
pair_path = OUT_DIR / "stats_pairwise.csv"
overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
pair_df.to_csv(pair_path, index=False, encoding="utf-8-sig")

print("Saved stats:", overall_path)
print("Saved pairwise:", pair_path)


# Plot: boxplots for key interpretable features
PLOT_FEATURES = ["contraction", "arm_span_norm", "left_hand_height", "right_hand_height",
                 "left_elbow_angle", "right_elbow_angle", "hand_height_asym", "elbow_asym"]
plot_dir = ensure_dir(OUT_DIR / "plots_explainable")

for f in PLOT_FEATURES:
    sub = df[["emotion", f]].dropna()
    if sub.empty:
        continue
    emos = sorted(sub["emotion"].unique())
    data = [sub.loc[sub["emotion"] == e, f].to_numpy() for e in emos]

    plt.figure()
    plt.boxplot(data, labels=emos, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Emotion vs {f}")
    plt.tight_layout()
    outp = plot_dir / f"box_{f}.png"
    plt.savefig(outp, dpi=200)
    plt.close()

print("Saved explainable plots to:", plot_dir)


# -----------------------------
# B) Unsupervised clustering + visualization
# -----------------------------
cluster_dir = ensure_dir(OUT_DIR / "unsupervised")

# Build matrix for clustering: use normalized coords (34 dims) + optionally a few features
kp_cols = [c for c in df.columns if c.startswith("kp_")]
X = df[kp_cols].to_numpy(dtype=float)

# Standardize
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# 2D embedding: UMAP if available else PCA
emb2 = None
used_umap = False
if USE_UMAP_IF_AVAILABLE:
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
        emb2 = reducer.fit_transform(Xz)
        used_umap = True
    except Exception:
        emb2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xz)
else:
    emb2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xz)

# Choose K by silhouette (on Xz)
best = None
for k in CLUSTER_K_LIST:
    if k >= len(df):
        continue
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(Xz)
    # silhouette needs >1 cluster and <n samples
    try:
        s = silhouette_score(Xz, labels)
    except Exception:
        continue
    if best is None or s > best["sil"]:
        best = {"k": k, "sil": s, "model": km, "labels": labels}

if best is None:
    raise RuntimeError("Failed to cluster. Try adjusting CLUSTER_K_LIST or check data.")

k_best = best["k"]
clabels = best["labels"]
df["cluster"] = clabels
print(f"Chosen K={k_best} by silhouette={best['sil']:.4f}")

# Save clustering assignments
cluster_csv = cluster_dir / "cluster_assignments.csv"
df[["emotion", "image_id", "cluster"]].to_csv(cluster_csv, index=False, encoding="utf-8-sig")
print("Saved:", cluster_csv)

# Scatter plot of embedding (colored by emotion) - we avoid specifying colors; matplotlib will auto-cycle.
plt.figure()
for e in sorted(df["emotion"].unique()):
    idx = df["emotion"] == e
    plt.scatter(emb2[idx, 0], emb2[idx, 1], s=8, label=e, alpha=0.8)
plt.legend(markerscale=2, fontsize=8)
plt.title("2D embedding colored by emotion" + (" (UMAP)" if used_umap else " (PCA)"))
plt.tight_layout()
plt.savefig(cluster_dir / "embed_by_emotion.png", dpi=200)
plt.close()

# Scatter plot by cluster
plt.figure()
for c in sorted(df["cluster"].unique()):
    idx = df["cluster"] == c
    plt.scatter(emb2[idx, 0], emb2[idx, 1], s=8, label=f"cluster {c}", alpha=0.8)
plt.legend(markerscale=2, fontsize=8)
plt.title("2D embedding colored by cluster" + (" (UMAP)" if used_umap else " (PCA)"))
plt.tight_layout()
plt.savefig(cluster_dir / "embed_by_cluster.png", dpi=200)
plt.close()

# Emotion x cluster proportion heatmap
ct = pd.crosstab(df["emotion"], df["cluster"], normalize="index")
ct_path = cluster_dir / "emotion_cluster_proportions.csv"
ct.to_csv(ct_path, encoding="utf-8-sig")
print("Saved:", ct_path)

plt.figure()
plt.imshow(ct.to_numpy(), aspect="auto")
plt.xticks(range(ct.shape[1]), [str(c) for c in ct.columns])
plt.yticks(range(ct.shape[0]), ct.index)
plt.title("Emotion x Cluster proportions")
plt.xlabel("Cluster")
plt.ylabel("Emotion")
plt.colorbar()
plt.tight_layout()
plt.savefig(cluster_dir / "heatmap_emotion_cluster.png", dpi=200)
plt.close()

# Cluster prototypes: mean normalized skeleton per cluster
proto_dir = ensure_dir(cluster_dir / "cluster_prototypes")

def plot_skeleton(norm_xy_17x2, title, outpath):
    plt.figure()
    # draw edges
    for a, b in SKELETON_EDGES:
        ia, ib = KP[a], KP[b]
        pa, pb = norm_xy_17x2[ia], norm_xy_17x2[ib]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            plt.plot([pa[0], pb[0]], [pa[1], pb[1]])
    # draw joints
    plt.scatter(norm_xy_17x2[:, 0], norm_xy_17x2[:, 1], s=12)
    plt.gca().invert_yaxis()  # image-like orientation
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    if len(sub) < 10:
        continue
    # mean pose in normalized coord space (34 dims -> 17x2)
    M = sub[kp_cols].to_numpy(dtype=float)
    mean_flat = np.nanmean(M, axis=0)
    mean_pose = mean_flat.reshape(17, 2)
    plot_skeleton(mean_pose, f"Cluster {c} prototype (mean normalized pose)", proto_dir / f"cluster_{c}_prototype.png")

print("Saved unsupervised outputs to:", cluster_dir)
print("DONE.")
