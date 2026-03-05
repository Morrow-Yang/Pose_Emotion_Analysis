import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path

def plot_cross_modal_validation(path_2d, path_3d, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_2d = pd.read_csv(path_2d)
    df_3d = pd.read_csv(path_3d)
    
    # Aggregating to emotion-level averages for validation of trends
    # Note: Filenames don't match globally, so we compare trends across emotions
    agg_2d = df_2d.groupby(['emotion'])['avg_velocity'].mean().reset_index()
    agg_3d = df_3d.groupby(['emotion'])['avg_velocity'].mean().reset_index()
    
    # Map emotion names to match (e.g. 'Fearful' in BVH may be 'Fear' in CAER-S)
    # Let's check common keys
    print(f"2D Emotions: {agg_2d['emotion'].unique()}")
    print(f"3D Emotions: {agg_3d['emotion'].unique()}")
    
    # Simple mapping
    mapping = {
        'Fearful': 'Fear',
    }
    agg_3d['emotion'] = agg_3d['emotion'].replace(mapping)
    
    # Merge on emotion
    merged = pd.merge(agg_2d, agg_3d, on=['emotion'], suffixes=('_2d', '_3d'))
    
    if merged.empty:
        print("❌ Could not merge data on emotion names.")
        return
        
    # Calculate Correlation on emotion-level summaries
    corr, p = spearmanr(merged['avg_velocity_2d'], merged['avg_velocity_3d'])
    
    # Plotting Correlation
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Color palette for emotions
    palette = sns.color_palette("husl", len(merged['emotion'].unique()))
    
    for i, row in merged.iterrows():
        plt.scatter(row['avg_velocity_3d'], row['avg_velocity_2d'], 
                    label=row['emotion'], s=200, alpha=0.8)
    
    # Add trend line
    sns.regplot(data=merged, x='avg_velocity_3d', y='avg_velocity_2d', 
                scatter=False, color='red', label=f'Spearman r={corr:.3f}')
    
    plt.title(f'Cross-Modal Category Validation: 2D vs 3D Trends\nSpearman Correlation: {corr:.3f}', fontsize=14)
    plt.xlabel('3D Ground Truth (Kinematic Dataset Avg Velocity)', fontsize=12)
    plt.ylabel('2D Estimation (CAER-S Dataset Avg Velocity)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(out_dir / 'validation_correlation.png', dpi=300)
    plt.savefig(out_dir / 'validation_correlation.pdf')
    
    # Create a comparative bar chart
    summary_2d = agg_2d.groupby('emotion')['avg_velocity'].mean()
    summary_3d = agg_3d.groupby('emotion')['avg_velocity'].mean()
    
    # Scale 3D to match 2D scale for visualization of trends
    scale_factor = summary_2d.mean() / summary_3d.mean()
    summary_3d_scaled = summary_3d * scale_factor
    
    comparison_df = pd.DataFrame({
        '2D_AlphaPose': summary_2d,
        '3D_GroundTruth_Scaled': summary_3d_scaled
    }).sort_values('2D_AlphaPose', ascending=False)
    
    comparison_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Velocity Trends: 2D vs 3D Validation', fontsize=14)
    plt.ylabel('Normalized Kinetic Energy', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / 'validation_trends_comparison.png', dpi=300)
    
    print(f"✅ Validation plots saved to {out_dir}")
    print(f"📊 Correlation coefficient: {corr:.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv2d", default="outputs/analysis/temporal/v1/temporal_motion_features.csv")
    ap.add_argument("--csv3d", default="outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv")
    ap.add_argument("--out", default="outputs/analysis/validation")
    args = ap.parse_args()
    
    plot_cross_modal_validation(args.csv2d, args.csv3d, args.out)
