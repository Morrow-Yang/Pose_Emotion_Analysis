###说明：这个代码是V4版本，专注于分析11种情绪对应的姿态指标，为每个指标单独选择最优聚类数，并使用3D可视化展示结果
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# CONFIG (CLI)
# -----------------------------
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing per-emotion subfolders.")
    ap.add_argument("--json_name", default="alphapose-results.json", help="AlphaPose result filename in each emotion folder.")
    ap.add_argument("--out_dir", required=False, default=None, help="Output directory. If not set, uses <root>/pose_analysis_v4_cli.")
    return ap.parse_args()

_args = parse_args()
ROOT = Path(_args.root)
JSON_NAME = _args.json_name
OUT_DIR = Path(_args.out_dir) if _args.out_dir else (ROOT / "pose_analysis_v4_cli")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_TH = 0.30
MIN_VALID_KP = 10

# clustering
CLUSTER_K_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 尝试多种聚类数
RANDOM_STATE = 42

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

        except Exception as e:
            print(f"[ERROR] Processing item: {e}")
            continue

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No valid samples loaded. Check paths/filters.")

# Save features
feat_csv = OUT_DIR / "pose_features_v4.csv"
df.to_csv(feat_csv, index=False, encoding="utf-8-sig")
print("Saved features:", feat_csv)
print("Samples:", len(df), "Emotions:", df["emotion"].nunique())
print("Emotion labels:", df["emotion"].unique())

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

# Count samples per emotion
emotion_counts = df_clean['emotion'].value_counts()
print("Emotion distribution:")
for emotion, count in emotion_counts.items():
    print(f"  {emotion}: {count} samples")

# Create output directory for clustering results
clustering_out_dir = ensure_dir(OUT_DIR / "individual_feature_analysis")

print(f"Starting individual feature clustering analysis...")

# Individual feature clustering analysis
individual_results = {}
feature_dir = ensure_dir(clustering_out_dir / "individual_feature_clustering")

for feature_idx, feature in enumerate(FEATURES):
    print(f"\nAnalyzing feature {feature_idx+1}/{len(FEATURES)}: {feature}")
    
    # Get data for this feature
    feature_data = df_clean[[feature]].dropna()
    
    if len(feature_data) < 10:  # Need minimum samples for clustering
        print(f"  Skipping {feature}: insufficient valid data ({len(feature_data)})")
        continue
    
    X_single = feature_data.values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_single)
    
    # Find optimal number of clusters for this feature
    best_result = None
    silhouette_scores = []
    k_values = []
    
    for k in CLUSTER_K_LIST:
        if k >= len(X_scaled) or k <= 1:
            continue
        
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_scaled)
        
        try:
            sil_score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(sil_score)
            k_values.append(k)
            
            if best_result is None or sil_score > best_result['silhouette']:
                best_result = {
                    'k': k,
                    'model': kmeans,
                    'labels': labels,
                    'silhouette': sil_score
                }
                
        except Exception as e:
            print(f"  Error calculating score for K={k}: {e}")
            continue
    
    if best_result is None:
        print(f"  Could not find good clustering for {feature}")
        continue
    
    # Store results
    individual_results[feature] = {
        'best_k': best_result['k'],
        'best_silhouette': best_result['silhouette'],
        'labels': best_result['labels'],
        'silhouette_scores': silhouette_scores,
        'k_values': k_values
    }
    
    print(f"  Best clustering for {feature}: K={best_result['k']}, Silhouette={best_result['silhouette']:.3f}")
    
    # Create visualization for this feature
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot silhouette scores
    axes[0].plot(k_values, silhouette_scores, 'bo-')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title(f'Silhouette Score vs Number of Clusters\n({feature})')
    axes[0].grid(True)
    
    # Histogram with cluster colors
    colors = plt.cm.Set1(np.linspace(0, 1, best_result['k']))
    valid_indices = feature_data.index
    feature_values = feature_data[feature]
    
    for cluster_id in range(best_result['k']):
        cluster_mask = best_result['labels'] == cluster_id
        cluster_values = feature_values.iloc[np.where(cluster_mask)[0]]
        
        if len(cluster_values) > 0:
            axes[1].hist(cluster_values, bins=30, alpha=0.7, label=f'Cluster {cluster_id}', color=colors[cluster_id])
    
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Clustering of {feature}\nBest K={best_result["k"]}, Silhouette={best_result["silhouette"]:.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(feature_dir / f"{feature}_individual_clustering.png", dpi=200, bbox_inches='tight')
    plt.close()

# Summary of individual feature clustering
print(f"\nIndividual Feature Clustering Summary:")
print("-" * 50)
for feature, results in individual_results.items():
    print(f"{feature:<20} | Best K: {results['best_k']:<2} | Silhouette: {results['best_silhouette']:.3f}")

# Create summary plot
if individual_results:
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Best K values for each feature
    features_list = list(individual_results.keys())
    best_ks = [individual_results[f]['best_k'] for f in features_list]
    silhouette_scores = [individual_results[f]['best_silhouette'] for f in features_list]
    
    y_pos = np.arange(len(features_list))
    
    axes[0].barh(y_pos, best_ks, align='center')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(features_list)
    axes[0].set_xlabel('Optimal Number of Clusters')
    axes[0].set_title('Optimal Number of Clusters for Each Feature')
    axes[0].grid(True, axis='x', alpha=0.3)
    
    axes[1].barh(y_pos, silhouette_scores, align='center', color='orange')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(features_list)
    axes[1].set_xlabel('Silhouette Score')
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Silhouette Score for Optimal Clustering of Each Feature')
    axes[1].grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(clustering_out_dir / "individual_feature_summary.png", dpi=200, bbox_inches='tight')
    plt.close()

# 3D visualization for better separation
print(f"\nGenerating 3D visualizations...")

# Select top 3 features with highest silhouette scores for 3D visualization
if len(individual_results) >= 3:
    top_3_features = sorted(individual_results.keys(), 
                           key=lambda x: individual_results[x]['best_silhouette'], 
                           reverse=True)[:3]
    
    print(f"Top 3 features for 3D visualization: {top_3_features}")
    
    # Prepare 3D visualization data
    fig = plt.figure(figsize=(18, 6))
    
    for idx, feature in enumerate(top_3_features):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        
        feature_data = df_clean[[feature]].dropna()
        X_single = feature_data.values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_single)
        
        # Use the best clustering for this feature
        best_k = individual_results[feature]['best_k']
        kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_scaled)
        
        # For 3D visualization, we need to create artificial x and y coordinates
        # We'll use random noise to spread points in x-y plane
        np.random.seed(RANDOM_STATE)
        x_coords = np.random.normal(0, 0.1, len(X_scaled))
        y_coords = np.random.normal(0, 0.1, len(X_scaled))
        z_coords = X_scaled.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, best_k))
        
        for cluster_id in range(best_k):
            mask = labels == cluster_id
            ax.scatter(x_coords[mask], y_coords[mask], z_coords[mask], 
                      c=[colors[cluster_id]], label=f'Cluster {cluster_id}', 
                      alpha=0.6, s=20)
        
        ax.set_xlabel('Random X')
        ax.set_ylabel('Random Y')
        ax.set_zlabel(feature)
        ax.set_title(f'{feature}\n(Clustered K={best_k})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(clustering_out_dir / "3d_feature_visualization.png", dpi=200, bbox_inches='tight')
    plt.close()

# 3D visualization for all emotions using PCA
print(f"\nGenerating 3D PCA visualization for emotions...")
X_all = df_clean[FEATURES].values
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)

# Reduce to 3D using PCA
pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)
X_pca_3d = pca_3d.fit_transform(X_all_scaled)

# Create 3D scatter plot colored by emotion
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

unique_emotions = df_clean['emotion'].unique()
colors_emotion = plt.cm.tab20(np.linspace(0, 1, len(unique_emotions)))
emotion_to_color = {emotion: colors_emotion[i] for i, emotion in enumerate(unique_emotions)}

for emotion in unique_emotions:
    mask = df_clean['emotion'] == emotion
    ax.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2], 
              label=emotion, c=[emotion_to_color[emotion]], alpha=0.6, s=20)

ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2f})')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2f})')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2f})')
ax.set_title('3D PCA Visualization of Emotions')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(clustering_out_dir / "3d_pca_emotion_visualization.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"\nAll individual feature clustering analysis completed!")
print(f"Results saved to: {clustering_out_dir}")
print(f"- Individual feature clustering plots: {feature_dir}")
print(f"- Summary plots: {clustering_out_dir / 'individual_feature_summary.png'}")
print(f"- 3D feature visualization: {clustering_out_dir / '3d_feature_visualization.png'}")
print(f"- 3D PCA emotion visualization: {clustering_out_dir / '3d_pca_emotion_visualization.png'}")
print(f"Total samples processed: {len(df_clean)}")
print(f"Features analyzed: {len(FEATURES)}")
print(f"Number of emotions: {df_clean['emotion'].nunique()}")
print(f"Individual feature analysis completed for {len(individual_results)} features")