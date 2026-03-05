"""
创新：时间序列分析 - 分析CAER-S视频帧之间的姿态变化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_temporal_variance(csv_path, out_dir):
    """分析每种情绪内部的姿态变化方差（反映动作幅度）"""
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 特征列
    feature_cols = [
        'shoulder_width', 'left_hand_height', 'right_hand_height',
        'arm_span_norm', 'left_elbow_angle', 'right_elbow_angle',
        'contraction', 'hand_height_asym', 'elbow_asym'
    ]
    
    # 计算每种情绪的方差（动作幅度指标）
    variance_results = []
    for emotion in df['emotion'].unique():
        emotion_df = df[df['emotion'] == emotion]
        for feat in feature_cols:
            variance_results.append({
                'emotion': emotion,
                'feature': feat,
                'variance': emotion_df[feat].var(),
                'std': emotion_df[feat].std(),
                'range': emotion_df[feat].max() - emotion_df[feat].min()
            })
    
    var_df = pd.DataFrame(variance_results)
    var_df.to_csv(out_dir / 'temporal_variance_by_emotion.csv', index=False)
    
    # 可视化：方差热图
    pivot_var = var_df.pivot(index='emotion', columns='feature', values='variance')
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_var, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('情绪姿态方差热图（反映动作幅度）')
    plt.tight_layout()
    plt.savefig(out_dir / 'variance_heatmap.png', dpi=300)
    plt.close()
    
    print(f"✅ 时间序列方差分析完成：{out_dir}")
    return var_df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per_sample_metrics.csv路径")
    ap.add_argument("--outdir", required=True, help="输出目录")
    args = ap.parse_args()
    
    analyze_temporal_variance(args.csv, args.outdir)
