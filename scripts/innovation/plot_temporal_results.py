import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_temporal_features(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Composite Research Plot (4 Panels)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # A. Average Velocity (Intensity of movement)
    sns.barplot(ax=axes[0, 0], x='emotion', y='avg_velocity', data=df, palette='magma', hue='emotion', legend=False)
    axes[0, 0].set_title('A. Movement Velocity (Intensity)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Mean Velocity (norm)')
    
    # B. Average Acceleration (Suddenness of movement)
    sns.barplot(ax=axes[0, 1], x='emotion', y='avg_acceleration', data=df, palette='viridis', hue='emotion', legend=False)
    axes[0, 1].set_title('B. Movement Acceleration (Suddenness)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Mean Acceleration (norm)')
    
    # C. Hand Movement (Specific Gesture Speed)
    df_melted = df.melt(id_vars=['emotion'], value_vars=['l_wrist_vel', 'r_wrist_vel'], 
                        var_name='Hand', value_name='velocity')
    sns.boxplot(ax=axes[1, 0], x='emotion', y='velocity', hue='Hand', data=df_melted, showfliers=False)
    axes[1, 0].set_title('C. Hand Gesture Velocity', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim(0, 0.6)
    
    # D. Heatmap of Joint-wise Velocity
    joint_cols = [c for c in df.columns if '_vel' in c and 'l_' in c or 'r_' in c]
    heatmap_data = df.groupby('emotion')[joint_cols].mean()
    sns.heatmap(ax=axes[1, 1], data=heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
    axes[1, 1].set_title('D. Joint-wise Velocity Heatmap', fontsize=14, fontweight='bold')
    
    plt.suptitle('Temporal Motion Analysis: Quantitative Emotional Dynamics', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / 'research_temporal_summary.png', dpi=300)
    plt.close()
    
    # 2. Individual plot for Nose (Head movement)
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='emotion', y='nose_vel', data=df, inner="quart", palette='coolwarm', hue='emotion', legend=False)
    plt.title('Head Movement (Nose Velocity) across Emotions')
    plt.ylim(0, 0.5)
    plt.savefig(out_dir / 'head_movement_violin.png', dpi=300)
    plt.close()
    
    print(f"📊 Visualization saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="temporal_motion_features.csv路径")
    ap.add_argument("--outdir", required=True, help="输出目录")
    args = ap.parse_args()
    
    plot_temporal_features(args.csv, args.outdir)
