###说明：这个代码是V3版本，专注于分析11种情绪对应的姿态指标，通过聚类方法找出情绪和姿态之间的对应关系
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
    ap.add_argument("--out_dir", required=False, default=None, help="Output directory. If not set, uses <root>/pose_analysis_v3_cli.")
    return ap.parse_args()

_args = parse_args()
ROOT = Path(_args.root)
JSON_NAME = _args.json_name
OUT_DIR = Path(_args.out_dir) if _args.out_dir else (ROOT / "pose_analysis_v3_cli")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_TH = 0.30
MIN_VALID_KP = 10

# clustering
CLUSTER_K_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 尝试多种聚类数，包括11种情绪
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
feat_csv = OUT_DIR / "pose_features_v3.csv"
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
clustering_out_dir = ensure_dir(OUT_DIR / "emotion_based_clustering")
emotion_dir = ensure_dir(clustering_out_dir / "emotion_vs_cluster_analysis")

print(f"Starting emotion-based clustering analysis...")

# Prepare data for clustering
X = df_clean[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Store the true emotion labels
true_labels = df_clean['emotion'].values

# Find optimal number of clusters around 11 (number of emotions)
best_result = None
silhouette_scores = []
ari_scores = []
nmi_scores = []

for k in CLUSTER_K_LIST:
    if k >= len(X_scaled) or k <= 1:
        continue
    
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    predicted_labels = kmeans.fit_predict(X_scaled)
    
    try:
        sil_score = silhouette_score(X_scaled, predicted_labels)
        
        # Calculate Adjusted Rand Index and Normalized Mutual Information
        ari_score = adjusted_rand_score(true_labels, predicted_labels)
        nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
        
        silhouette_scores.append(sil_score)
        ari_scores.append(ari_score)
        nmi_scores.append(nmi_score)
        
        if best_result is None or sil_score > best_result['silhouette']:
            best_result = {
                'k': k,
                'model': kmeans,
                'labels': predicted_labels,
                'silhouette': sil_score,
                'ari': ari_score,
                'nmi': nmi_score
            }
            
        print(f"K={k}: Silhouette={sil_score:.3f}, ARI={ari_score:.3f}, NMI={nmi_score:.3f}")
        
    except Exception as e:
        print(f"Error calculating scores for K={k}: {e}")
        continue

if best_result is None:
    print("Could not find good clustering for any K value")
else:
    print(f"\nBest clustering result: K={best_result['k']}, "
          f"Silhouette={best_result['silhouette']:.3f}, "
          f"ARI={best_result['ari']:.3f}, "
          f"NMI={best_result['nmi']:.3f}")

# Visualize clustering results
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot silhouette scores
axes[0, 0].plot(CLUSTER_K_LIST[:len(silhouette_scores)], silhouette_scores, 'bo-')
axes[0, 0].set_xlabel('Number of Clusters (K)')
axes[0, 0].set_ylabel('Silhouette Score')
axes[0, 0].set_title('Silhouette Score vs Number of Clusters')
axes[0, 0].grid(True)

# Plot ARI scores
axes[0, 1].plot(CLUSTER_K_LIST[:len(ari_scores)], ari_scores, 'ro-')
axes[0, 1].set_xlabel('Number of Clusters (K)')
axes[0, 1].set_ylabel('Adjusted Rand Index')
axes[0, 1].set_title('ARI vs Number of Clusters')
axes[0, 1].grid(True)

# Plot NMI scores
axes[1, 0].plot(CLUSTER_K_LIST[:len(nmi_scores)], nmi_scores, 'go-')
axes[1, 0].set_xlabel('Number of Clusters (K)')
axes[1, 0].set_ylabel('Normalized Mutual Information')
axes[1, 0].set_title('NMI vs Number of Clusters')
axes[1, 0].grid(True)

# Best clustering visualization using PCA
if best_result is not None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=best_result['labels'], cmap='tab10', alpha=0.7)
    axes[1, 1].set_xlabel('PCA Component 1')
    axes[1, 1].set_ylabel('PCA Component 2')
    axes[1, 1].set_title(f'Best Clustering (K={best_result["k"]}) - PCA Projection')
    plt.colorbar(scatter, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(clustering_out_dir / "clustering_evaluation_metrics.png", dpi=200, bbox_inches='tight')
plt.close()

# Create emotion vs cluster heatmap
if best_result is not None:
    # Create contingency table
    contingency_table = pd.crosstab(pd.Series(true_labels, name='Emotion'), 
                                    pd.Series(best_result['labels'], name='Cluster'))
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Sample Count'})
    plt.title(f'Emotion Distribution Across Clusters (K={best_result["k"]})\n'
              f'Silhouette: {best_result["silhouette"]:.3f}, ARI: {best_result["ari"]:.3f}')
    plt.tight_layout()
    plt.savefig(clustering_out_dir / "emotion_cluster_contingency_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print cluster characteristics
    print(f"\nCluster characteristics for best K={best_result['k']}:")
    for cluster_id in range(best_result['k']):
        cluster_mask = best_result['labels'] == cluster_id
        cluster_samples = df_clean[cluster_mask]
        cluster_emotions = cluster_samples['emotion'].value_counts()
        
        print(f"\nCluster {cluster_id} ({len(cluster_samples)} samples):")
        for emotion, count in cluster_emotions.items():
            percentage = count / len(cluster_samples) * 100
            print(f"  {emotion}: {count} ({percentage:.1f}%)")

# Perform t-SNE visualization for better understanding
print("Generating t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, len(X_scaled)-1))
X_tsne = tsne.fit_transform(X_scaled)

# Create t-SNE plots colored by emotion and cluster
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# t-SNE colored by emotion
unique_emotions = np.unique(true_labels)
colors_emotion = plt.cm.tab20(np.linspace(0, 1, len(unique_emotions)))
emotion_to_color = {emotion: colors_emotion[i] for i, emotion in enumerate(unique_emotions)}

scatter_colors = [emotion_to_color[emotion] for emotion in true_labels]
axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=scatter_colors, alpha=0.7)
axes[0].set_title('t-SNE Visualization - Colored by Emotion')
axes[0].set_xlabel('t-SNE Component 1')
axes[0].set_ylabel('t-SNE Component 2')

# Add legend for emotions
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
           for color in colors_emotion]
axes[0].legend(handles, unique_emotions, title="Emotions", bbox_to_anchor=(1.05, 1), loc='upper left')

# t-SNE colored by cluster (if best result exists)
if best_result is not None:
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=best_result['labels'], cmap='tab10', alpha=0.7)
    axes[1].set_title(f't-SNE Visualization - Colored by Cluster (K={best_result["k"]})')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter2, ax=axes[1])
else:
    axes[1].text(0.5, 0.5, 'No clustering results', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
    axes[1].set_title('t-SNE Visualization - No Clustering')

plt.tight_layout()
plt.savefig(clustering_out_dir / "tsne_visualization.png", dpi=200, bbox_inches='tight')
plt.close()

# Feature importance analysis for each cluster
if best_result is not None:
    print("\nAnalyzing feature differences between clusters...")
    
    # Calculate mean feature values per cluster
    df_clean_with_clusters = df_clean.copy()
    df_clean_with_clusters['cluster'] = best_result['labels']
    
    feature_means = df_clean_with_clusters.groupby('cluster')[FEATURES].mean()
    
    # Create heatmap showing feature means by cluster
    plt.figure(figsize=(12, 8))
    sns.heatmap(feature_means.T, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Mean Feature Value'})
    plt.title(f'Mean Feature Values by Cluster (K={best_result["k"]})')
    plt.ylabel('Pose Features')
    plt.xlabel('Cluster')
    plt.tight_layout()
    plt.savefig(clustering_out_dir / "feature_means_by_cluster.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results_df = df_clean_with_clusters[['emotion', 'cluster'] + FEATURES]
    results_df.to_csv(clustering_out_dir / f"emotion_cluster_analysis_results_k{best_result['k']}.csv", 
                      index=False, encoding="utf-8-sig")
    
    result_path = clustering_out_dir / f"emotion_cluster_analysis_results_k{best_result['k']}.csv"
    print(f"\nDetailed results saved to: {result_path}")

print(f"\nAll emotion-based clustering analysis completed!")
print(f"Results saved to: {clustering_out_dir}")
print(f"- Evaluation metrics plot: {clustering_out_dir / 'clustering_evaluation_metrics.png'}")
print(f"- Emotion-cluster heatmap: {clustering_out_dir / 'emotion_cluster_contingency_heatmap.png'}")
print(f"- t-SNE visualization: {clustering_out_dir / 'tsne_visualization.png'}")
print(f"- Feature means by cluster: {clustering_out_dir / 'feature_means_by_cluster.png'}")
print(f"Total samples processed: {len(df_clean)}")
print(f"Features analyzed: {len(FEATURES)}")
print(f"Number of emotions: {df_clean['emotion'].nunique()}")
if best_result:
    print(f"Best number of clusters: {best_result['k']}")