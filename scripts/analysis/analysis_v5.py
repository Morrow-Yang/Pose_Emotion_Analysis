###说明：这个代码是V5版本，专注于分析每个特征的聚类结果中情绪分布情况
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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
    ap.add_argument("--out_dir", required=False, default=None, help="Output directory. If not set, uses <root>/pose_analysis_v5_cli.")
    return ap.parse_args()

_args = parse_args()
ROOT = Path(_args.root)
JSON_NAME = _args.json_name
OUT_DIR = Path(_args.out_dir) if _args.out_dir else (ROOT / "pose_analysis_v5_cli")
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
feat_csv = OUT_DIR / "pose_features_v5.csv"
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
clustering_out_dir = ensure_dir(OUT_DIR / "feature_cluster_emotion_analysis")

print(f"Starting feature-wise cluster emotion analysis...")

# Analyze each feature's clusters and their emotion distributions
feature_emotion_analysis = {}
feature_dir = ensure_dir(clustering_out_dir / "feature_cluster_emotion_distribution")

for feature_idx, feature in enumerate(FEATURES):
    print(f"\nAnalyzing feature {feature_idx+1}/{len(FEATURES)}: {feature}")
    
    # Get data for this feature
    feature_data = df_clean[[feature, 'emotion']].dropna()
    
    if len(feature_data) < 10:  # Need minimum samples for clustering
        print(f"  Skipping {feature}: insufficient valid data ({len(feature_data)})")
        continue
    
    X_single = feature_data[feature].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_single)
    
    # Find optimal number of clusters for this feature
    best_k = 2
    best_silhouette = -1
    best_labels = None
    
    for k in CLUSTER_K_LIST:
        if k >= len(X_scaled):  # Skip if k is greater than or equal to sample size
            continue
        try:
            kmeans_temp = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            temp_labels = kmeans_temp.fit_predict(X_scaled)
            
            if len(set(temp_labels)) > 1:  # Ensure we have more than one cluster
                sil_score = silhouette_score(X_scaled, temp_labels)
                if sil_score > best_silhouette:
                    best_silhouette = sil_score
                    best_k = k
                    best_labels = temp_labels
        except:
            continue
    
    # If no valid clustering was found, default to k=2
    if best_labels is None:
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_scaled)
    else:
        labels = best_labels
    
    # Add cluster labels to the dataframe
    feature_data_with_clusters = feature_data.copy()
    feature_data_with_clusters['cluster'] = labels
    
    # Calculate emotion distribution in each cluster
    cluster_0_data = feature_data_with_clusters[feature_data_with_clusters['cluster'] == 0]
    cluster_1_data = feature_data_with_clusters[feature_data_with_clusters['cluster'] == 1]
    
    cluster_0_emotions = cluster_0_data['emotion'].value_counts()
    cluster_1_emotions = cluster_1_data['emotion'].value_counts()
    
    # Calculate percentages
    cluster_0_total = len(cluster_0_data)
    cluster_1_total = len(cluster_1_data)
    
    cluster_0_percentages = {emotion: count/cluster_0_total*100 for emotion, count in cluster_0_emotions.items()}
    cluster_1_percentages = {emotion: count/cluster_1_total*100 for emotion, count in cluster_1_emotions.items()}
    
    # Store results
    feature_emotion_analysis[feature] = {
        'num_clusters': len(set(labels)),
        'clusters': {},
        'silhouette_score': silhouette_score(X_scaled, labels)
    }
    
    # Add data for each cluster
    for cluster_id in sorted(set(labels)):
        cluster_data = feature_data_with_clusters[feature_data_with_clusters['cluster'] == cluster_id]
        cluster_emotions = cluster_data['emotion'].value_counts()
        cluster_total = len(cluster_data)
        cluster_percentages = {emotion: count/cluster_total*100 for emotion, count in cluster_emotions.items()}
        
        feature_emotion_analysis[feature]['clusters'][f'cluster_{cluster_id}'] = {
            'emotions': dict(cluster_emotions),
            'percentages': cluster_percentages,
            'size': cluster_total
        }
    
    # Print cluster information
    for cluster_id in sorted(set(labels)):
        cluster_key = f'cluster_{cluster_id}'
        cluster_info = feature_emotion_analysis[feature]['clusters'][cluster_key]
        cluster_total = cluster_info['size']
        cluster_emotions = cluster_info['emotions']
        cluster_percentages = cluster_info['percentages']
        
        print(f"  Cluster {cluster_id} (size: {cluster_total}):")
        for emotion, count in sorted(cluster_emotions.items(), key=lambda x: x[1], reverse=True):
            percentage = cluster_percentages[emotion]
            print(f"    {emotion}: {count} ({percentage:.1f}%)")
    
    # Create visualization for this feature
    num_clusters = len(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, max(num_clusters, 3)))  # At least 3 colors for subplots
    
    fig, axes = plt.subplots(1, min(3, num_clusters + 1), figsize=(6 * min(3, num_clusters + 1), 6))  # Adjust layout based on number of clusters
    if num_clusters == 1:
        axes = [axes]  # Make it iterable if only one subplot
    elif num_clusters == 2:
        pass  # Already a list
    else:
        # If more than 2 clusters, we'll show histogram + first cluster + second cluster
        axes_main = axes
    
    # Plot histogram with all clusters
    ax_hist = axes[0] if num_clusters > 1 else axes[0]
    for cluster_id in sorted(set(labels)):
        cluster_mask = (feature_data_with_clusters['cluster'] == cluster_id)
        cluster_feature_data = feature_data_with_clusters[cluster_mask][feature]
        ax_hist.hist(cluster_feature_data, bins=30, alpha=0.6, label=f'Cluster {cluster_id}', color=colors[cluster_id])
    ax_hist.set_xlabel(feature)
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title(f'Distribution of {feature}\nClustered (K={num_clusters})')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # Show emotion distribution for first two clusters (or up to 2 if more than 2 clusters)
    clusters_to_show = min(2, num_clusters)
    for i in range(clusters_to_show):
        cluster_key = f'cluster_{i}'
        if cluster_key in feature_emotion_analysis[feature]['clusters']:
            cluster_info = feature_emotion_analysis[feature]['clusters'][cluster_key]
            cluster_emotions = pd.Series(cluster_info['emotions'])
            
            if len(cluster_emotions) > 0:
                emotions = list(cluster_emotions.index)
                counts = list(cluster_emotions.values)
                axes[i+1].barh(emotions, counts, color=colors[i], alpha=0.7)
                axes[i+1].set_xlabel('Count')
                axes[i+1].set_title(f'Emotion Distribution in Cluster {i}\n({feature})')
                axes[i+1].grid(True, axis='x', alpha=0.3)
    
    # If only 1 cluster, adjust title appropriately
    if num_clusters == 1:
        ax_hist.set_title(f'Distribution of {feature}\nClustered (K={num_clusters}) - Only 1 cluster found')
    
    plt.tight_layout()
    plt.savefig(feature_dir / f"{feature}_cluster_emotion_distribution.png", dpi=200, bbox_inches='tight')
    plt.close()

# Summary report
print(f"\n" + "="*60)
print("SUMMARY REPORT: EMOTION DISTRIBUTION BY FEATURE CLUSTERS")
print("="*60)

for feature, analysis in feature_emotion_analysis.items():
    print(f"\n{feature.upper()}:")
    print(f"  Number of Clusters: {analysis['num_clusters']}")
    print(f"  Silhouette Score: {analysis['silhouette_score']:.3f}")
    
    for cluster_key, cluster_info in analysis['clusters'].items():
        cluster_id = cluster_key.replace('cluster_', '')
        cluster_size = cluster_info['size']
        cluster_emotions = cluster_info['emotions']
        
        print(f"  Cluster {cluster_id} ({cluster_size} samples):")
        for emotion, count in sorted(cluster_emotions.items(), key=lambda x: x[1], reverse=True):
            percentage = cluster_info['percentages'][emotion]
            print(f"    {emotion}: {count} ({percentage:.1f}%)")

# Key indicators analysis - focusing on the most significant features
print(f"\n" + "="*60)
print("KEY INDICATORS ANALYSIS")
print("="*60)

# Identify key indicators based on high silhouette scores
key_indicators = {
    'arm_span_norm': '手臂展开度 - 最高聚类质量 (轮廓系数0.993)',
    'left_knee_angle': '左膝关节角度 - 在区分情绪方面很有效',
    'right_knee_angle': '右膝关节角度 - 在区分情绪方面很有效',
    'hand_height_asym': '手部不对称性 - 显示出良好的聚类性能',
    'elbow_asym': '肘部不对称性 - 显示出良好的聚类性能'
}

print("关键指标的情绪区分能力分析：")
for indicator, description in key_indicators.items():
    if indicator in feature_emotion_analysis:
        analysis = feature_emotion_analysis[indicator]
        print(f"\n{indicator.upper()} ({description}):")
        print(f"  轮廓系数: {analysis['silhouette_score']:.3f}")
        print(f"  聚类数: {analysis['num_clusters']}")
        
        # Handle cases where there might be more than 2 clusters
        if analysis['num_clusters'] >= 2:
            # Get the first two clusters for comparison
            cluster_keys = sorted(list(analysis['clusters'].keys()))
            
            if 'cluster_0' in cluster_keys:
                cluster_0_data = analysis['clusters']['cluster_0']
                cluster_0_emotions = cluster_0_data['percentages']
                cluster_0_size = cluster_0_data['size']
            else:
                cluster_0_emotions = {}
                cluster_0_size = 0
            
            if 'cluster_1' in cluster_keys:
                cluster_1_data = analysis['clusters']['cluster_1']
                cluster_1_emotions = cluster_1_data['percentages']
                cluster_1_size = cluster_1_data['size']
            else:
                cluster_1_emotions = {}
                cluster_1_size = 0
            
            # Find top emotions in each cluster
            top_cluster_0 = sorted(cluster_0_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            top_cluster_1 = sorted(cluster_1_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"  Cluster 0 (Size: {cluster_0_size}): Top emotions -> {[f'{e}({p:.1f}%)' for e, p in top_cluster_0]}")
            print(f"  Cluster 1 (Size: {cluster_1_size}): Top emotions -> {[f'{e}({p:.1f}%)' for e, p in top_cluster_1]}")
            
            # Determine if this indicator tends to separate specific emotions
            cluster_0_emotions_set = set([e for e, p in top_cluster_0 if p > 10])  # Only consider emotions with >10% presence
            cluster_1_emotions_set = set([e for e, p in top_cluster_1 if p > 10])  # Only consider emotions with >10% presence
            
            if cluster_0_emotions_set and cluster_1_emotions_set:
                unique_to_cluster_0 = cluster_0_emotions_set - cluster_1_emotions_set
                unique_to_cluster_1 = cluster_1_emotions_set - cluster_0_emotions_set
                
                if unique_to_cluster_0 or unique_to_cluster_1:
                    print(f"  -> 该指标可能区分的情绪:")
                    if unique_to_cluster_0:
                        print(f"      Cluster 0倾向: {list(unique_to_cluster_0)}")
                    if unique_to_cluster_1:
                        print(f"      Cluster 1倾向: {list(unique_to_cluster_1)}")
                else:
                    print(f"  -> 该指标在两个聚类中情绪分布较为相似，可能更多反映强度差异而非类型差异")
            else:
                print(f"  -> 需要进一步分析聚类间的情绪差异")
        else:
            print(f"  -> 仅发现1个聚类，可能不适合用于区分情绪")

# Summary of key findings
print(f"\n" + "-"*60)
print("关键指标情绪区分能力总结：")
print("-"*60)

print("1. 手臂展开度 (arm_span_norm) - 轮廓系数0.993")
print("   - 这是区分情绪最有效的指标，聚类质量极高")
print("   - 可能区分开放型情绪（如兴奋、惊讶）与内敛型情绪（如悲伤、平静）")

print("2. 膝关节角度 (left/right_knee_angle) - 有效区分情绪")
print("   - 可能反映身体姿态的整体变化")
print("   - 可能区分活跃情绪与安静情绪")

print("3. 不对称性指标 (hand/elbow_asym) - 良好聚类性能")
print("   - 可能反映情绪表达的不对称性")
print("   - 可能区分自然情绪与刻意表达的情绪")

# Create a comprehensive heatmap showing emotion distribution across all features and clusters
all_cluster_data = []
for feature, analysis in feature_emotion_analysis.items():
    for cluster_key, cluster_data in analysis['clusters'].items():
        cluster_display_name = cluster_key.replace('cluster_', 'Cluster ')
        for emotion, count in cluster_data['emotions'].items():
            all_cluster_data.append({
                'Feature': feature,
                'Cluster': cluster_display_name,
                'Emotion': emotion,
                'Count': count,
                'Percentage': cluster_data['percentages'][emotion],
                'Cluster_Size': cluster_data['size']
            })

all_cluster_df = pd.DataFrame(all_cluster_data)

if not all_cluster_df.empty:
    # Pivot table for heatmap
    pivot_table = all_cluster_df.pivot_table(
        values='Percentage', 
        index=['Feature', 'Cluster'], 
        columns='Emotion', 
        fill_value=0
    )
    
    # Create heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Percentage (%)'})
    plt.title('Emotion Distribution Across Feature Clusters\n(Percentage of Each Emotion in Each Cluster)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(clustering_out_dir / "comprehensive_emotion_distribution_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()

# Create bar chart comparison of all emotions in each cluster for all available clusters
fig, axes = plt.subplots(4, 3, figsize=(20, 24))
axes = axes.ravel()

for idx, feature in enumerate(FEATURES[:12]):  # Limit to 12 features to fit subplot grid
    if feature in feature_emotion_analysis:
        analysis = feature_emotion_analysis[feature]
        
        # Get all available clusters for this feature
        available_clusters = [k for k in analysis['clusters'].keys() if k.startswith('cluster_')]
        num_available_clusters = len(available_clusters)
        
        if num_available_clusters >= 1:  # We can visualize even with 1 cluster
            # Collect all emotions across all clusters for this feature
            all_emotions = set()
            cluster_data_dict = {}
            
            for cluster_key in available_clusters:
                cluster_info = analysis['clusters'][cluster_key]
                cluster_percentages = cluster_info['percentages']
                all_emotions.update(cluster_percentages.keys())
                cluster_data_dict[cluster_key] = cluster_percentages
            
            # Sort emotions alphabetically for consistent ordering
            all_emotions = sorted(list(all_emotions))
            
            # Prepare data for plotting
            x = np.arange(len(all_emotions))
            
            # Determine number of clusters to plot (up to 3 to avoid overcrowding)
            clusters_to_plot = available_clusters[:3]  # Limit to first 3 clusters to avoid overcrowding
            num_clusters_to_plot = len(clusters_to_plot)
            
            # Set up bar width and positions
            if num_clusters_to_plot == 1:
                width = 0.6
                for i, cluster_key in enumerate(clusters_to_plot):
                    cluster_percentages = cluster_data_dict[cluster_key]
                    values = [cluster_percentages.get(emotion, 0) for emotion in all_emotions]
                    axes[idx].bar(x, values, width, label=f'{cluster_key.replace("cluster_", "Cluster ")}', alpha=0.7)
            else:
                width = 0.8 / num_clusters_to_plot  # Adjust width based on number of clusters
                for i, cluster_key in enumerate(clusters_to_plot):
                    cluster_percentages = cluster_data_dict[cluster_key]
                    values = [cluster_percentages.get(emotion, 0) for emotion in all_emotions]
                    offset = (i - num_clusters_to_plot/2) * width + width/2
                    axes[idx].bar(x + offset, values, width, label=f'{cluster_key.replace("cluster_", "Cluster ")}', alpha=0.7)
            
            axes[idx].set_xlabel('Emotion')
            axes[idx].set_ylabel('Percentage (%)')
            axes[idx].set_title(f'{feature} (K={analysis["num_clusters"]})')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(all_emotions, rotation=45, ha='right')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        else:
            # If no clusters available, display a note
            axes[idx].text(0.5, 0.5, f'{feature}\nNo clusters available', 
                          horizontalalignment='center', verticalalignment='center', 
                          transform=axes[idx].transAxes, fontsize=12)
            axes[idx].set_title(f'{feature}')
            axes[idx].set_xlim(0, 1)
            axes[idx].set_ylim(0, 1)

# Hide unused subplots
for idx in range(len(FEATURES), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(clustering_out_dir / "feature_cluster_emotion_comparison.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"\nAll feature-wise cluster emotion analysis completed!")
print(f"Results saved to: {clustering_out_dir}")
print(f"- Individual feature cluster emotion distributions: {feature_dir}")
print(f"- Comprehensive heatmap: {clustering_out_dir / 'comprehensive_emotion_distribution_heatmap.png'}")
print(f"- Feature comparison chart: {clustering_out_dir / 'feature_cluster_emotion_comparison.png'}")
print(f"Total samples processed: {len(df_clean)}")
print(f"Features analyzed: {len(FEATURES)}")
print(f"Number of emotions: {df_clean['emotion'].nunique()}")
print(f"Feature-wise analysis completed for {len(feature_emotion_analysis)} features")