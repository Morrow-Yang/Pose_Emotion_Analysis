###说明：这个代码是基于analyzeposeemotion.py，修改了部分代码，将分别对几种参数指标分别进行聚类；对所有参数的两两组合进行聚类；自动选择最佳聚类数量（基于轮廓系数）；生成相关性热图展示参数间关系
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG (CLI)
# -----------------------------
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing per-emotion subfolders.")
    ap.add_argument("--json_name", default="alphapose-results.json", help="AlphaPose result filename in each emotion folder.")
    ap.add_argument("--out_dir", required=False, default=None, help="Output directory. If not set, uses <root>/pose_analysis_v2_cli.")
    return ap.parse_args()

_args = parse_args()
ROOT = Path(_args.root)
JSON_NAME = _args.json_name
OUT_DIR = Path(_args.out_dir) if _args.out_dir else (ROOT / "pose_analysis_v2_cli")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_TH = 0.30
MIN_VALID_KP = 10

# clustering
CLUSTER_K_LIST = [2, 3, 4, 5, 6, 7, 8]  # will pick best by silhouette
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

# Define features for clustering
FEATURES = [
    "left_hand_height", "right_hand_height",
    "arm_span_norm",
    "left_elbow_angle", "right_elbow_angle",
    "contraction",
    "hand_height_asym", "elbow_asym",
    "head_dx", "head_dy",
    "left_knee_angle", "right_knee_angle",
]

# Filter dataframe to only include rows with valid values for all features
df_clean = df.dropna(subset=FEATURES)

if df_clean.empty:
    raise RuntimeError("No samples with valid values for all features. Check data.")

# Create output directory for clustering results
clustering_out_dir = ensure_dir(OUT_DIR / "parameter_clustering")
param_dir = ensure_dir(clustering_out_dir / "single_parameter")
pair_dir = ensure_dir(clustering_out_dir / "parameter_pairs")

print(f"Starting parameter clustering analysis...")

# 1. Clustering for each parameter separately
print("Performing clustering for each parameter separately...")
for feature in FEATURES:
    print(f"Processing parameter: {feature}")
    
    # Get valid values for this feature
    valid_data = df_clean[[feature]].dropna()
    if len(valid_data) < 10:  # Need minimum samples for clustering
        print(f"Skipping {feature}: insufficient valid data ({len(valid_data)})")
        continue
    
    X = valid_data.values.reshape(-1, 1)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters
    best_result = None
    for k in CLUSTER_K_LIST:
        if k >= len(X_scaled) or k <= 1:
            continue
        
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_scaled)
        
        try:
            sil_score = silhouette_score(X_scaled, labels)
            if best_result is None or sil_score > best_result['silhouette']:
                best_result = {
                    'k': k,
                    'model': kmeans,
                    'labels': labels,
                    'silhouette': sil_score
                }
        except:
            continue
    
    if best_result is None:
        print(f"Could not find good clustering for {feature}")
        continue
    
    # Add cluster labels to original dataframe
    df_with_labels = df_clean.copy()
    df_with_labels[f'{feature}_cluster'] = best_result['labels']
    
    # Create histogram with cluster colors
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, best_result['k']))
    
    for cluster_id in range(best_result['k']):
        cluster_indices = df_with_labels[df_with_labels[f'{feature}_cluster'] == cluster_id].index
        cluster_data = valid_data.loc[cluster_indices][feature]
        if len(cluster_data) > 0:  # Only plot if there is data
            plt.hist(cluster_data, bins=30, alpha=0.7, label=f'Cluster {cluster_id}', color=colors[cluster_id])
    
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Clustering of {feature}\nBest K={best_result["k"]}, Silhouette={best_result["silhouette"]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(param_dir / f"{feature}_clustering.png", dpi=200)
    plt.close()
    
    print(f"Completed clustering for {feature}: K={best_result['k']}, Silhouette={best_result['silhouette']:.3f}")

print(f"Single parameter clustering completed. Results saved to: {param_dir}")

# 2. Clustering for pairs of parameters
print("\nPerforming clustering for parameter pairs...")

# Generate all possible pairs of features
feature_pairs = []
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        feature_pairs.append((FEATURES[i], FEATURES[j]))

for feat1, feat2 in feature_pairs:
    print(f"Processing pair: {feat1} vs {feat2}")
    
    # Get valid data for both features
    valid_data = df_clean[[feat1, feat2]].dropna()
    if len(valid_data) < 10:  # Need minimum samples for clustering
        print(f"Skipping pair {feat1}-{feat2}: insufficient valid data ({len(valid_data)})")
        continue
    
    X = valid_data.values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters
    best_result = None
    for k in CLUSTER_K_LIST:
        if k >= len(X_scaled) or k <= 1:
            continue
        
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_scaled)
        
        try:
            sil_score = silhouette_score(X_scaled, labels)
            if best_result is None or sil_score > best_result['silhouette']:
                best_result = {
                    'k': k,
                    'model': kmeans,
                    'labels': labels,
                    'silhouette': sil_score
                }
        except:
            continue
    
    if best_result is None:
        print(f"Could not find good clustering for pair {feat1}-{feat2}")
        continue
    
    # Add cluster labels to original dataframe
    df_with_labels = df_clean.copy()
    df_with_labels[f'{feat1}_{feat2}_cluster'] = best_result['labels']
    
    # Create scatter plot with cluster colors
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, best_result['k']))
    
    for cluster_id in range(best_result['k']):
        cluster_mask = df_with_labels[f'{feat1}_{feat2}_cluster'] == cluster_id
        cluster_data = df_with_labels[cluster_mask]
        plt.scatter(
            cluster_data[feat1], 
            cluster_data[feat2], 
            c=[colors[cluster_id]], 
            label=f'Cluster {cluster_id}',
            alpha=0.7,
            s=30
        )
    
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.title(f'{feat1} vs {feat2} Clustering\nBest K={best_result["k"]}, Silhouette={best_result["silhouette"]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(pair_dir / f"{feat1}_vs_{feat2}_clustering.png", dpi=200)
    plt.close()
    
    print(f"Completed clustering for {feat1}-{feat2}: K={best_result['k']}, Silhouette={best_result['silhouette']:.3f}")

print(f"Parameter pair clustering completed. Results saved to: {pair_dir}")

# Also create correlation heatmap for all features
plt.figure(figsize=(12, 10))
correlation_matrix = df_clean[FEATURES].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Heatmap of Pose Features')
plt.tight_layout()
plt.savefig(clustering_out_dir / "feature_correlation_heatmap.png", dpi=200)
plt.close()

print(f"All clustering analysis completed!")
print(f"Results saved to: {clustering_out_dir}")
print(f"- Single parameter clustering: {param_dir}")
print(f"- Parameter pair clustering: {pair_dir}")
print(f"Total samples processed: {len(df_clean)}")
print(f"Features analyzed: {len(FEATURES)}")
print(f"Feature pairs analyzed: {len(feature_pairs)}")