import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils_bvh_parser import BVHParser

# Mapping from BVH Join names to our standard feature names
# Note: In BVH, the position of 'RightHand' is effectively the 'Wrist' in 2D AlphaPose terms
JOINT_MAP = {
    'head': 'Head',
    'l_shoulder': 'LeftArm',
    'r_shoulder': 'RightArm',
    'l_elbow': 'LeftForeArm',
    'r_elbow': 'RightForeArm',
    'l_wrist': 'LeftHand',
    'r_wrist': 'RightHand'
}

def analyze_bvh_temporal(dataset_root, out_dir):
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load file mapping
    info_csv = dataset_root / 'file-info.csv'
    if not info_csv.exists():
        print(f"❌ Error: {info_csv} not found.")
        return
    
    info_df = pd.read_csv(info_csv)
    # Filter only samples we have BVH for
    bvh_dir = dataset_root / 'BVH'
    
    all_motion_data = []
    
    print("🏃 Starting BVH Temporal Analysis...")
    
    for idx, row in tqdm(info_df.iterrows(), total=len(info_df)):
        filename = row['filename']
        emotion = row['emotion']
        actor = row['actor_ID']
        
        # Find the BVH file
        bvh_path = bvh_dir / actor / f"{filename}.bvh"
        if not bvh_path.exists():
            continue
            
        try:
            parser = BVHParser(str(bvh_path))
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue
            
        motion_records = []
        prev_coords = None
        
        for f_idx in range(len(parser.frames)):
            curr_coords = parser.get_joint_world_coords(f_idx)
            
            if prev_coords is not None:
                velocities = {}
                total_vel = 0
                count = 0
                
                for feat_name, bvh_name in JOINT_MAP.items():
                    if bvh_name in curr_coords and bvh_name in prev_coords:
                        # 3D Euclidean Distance
                        dist = np.linalg.norm(curr_coords[bvh_name] - prev_coords[bvh_name])
                        # Normalize? In 3D, units are already absolute (e.g. cm)
                        # But to compare with AlphaPose, we can normalize by a 'height' metric
                        # Let's use the distance from Hips to Head as a scale factor
                        height_norm = np.linalg.norm(curr_coords['Hips'] - curr_coords['Head'])
                        vel = dist / height_norm if height_norm > 0 else dist
                        
                        velocities[f'{feat_name}_vel'] = vel
                        total_vel += vel
                        count += 1
                
                avg_vel = total_vel / count if count > 0 else 0
                motion_records.append({
                    'filename': filename,
                    'emotion': emotion,
                    'frame': f_idx,
                    'avg_velocity': avg_vel,
                    **velocities
                })
            
            prev_coords = curr_coords
            
        all_motion_data.extend(motion_records)
        
    if not all_motion_data:
        print("No data collected!")
        return
        
    full_df = pd.DataFrame(all_motion_data)
    full_df.to_csv(out_dir / 'bvh_temporal_features.csv', index=False)
    
    # Summary
    summary = full_df.groupby('emotion').agg({
        'avg_velocity': ['mean', 'std'],
        'l_wrist_vel': 'mean',
        'r_wrist_vel': 'mean',
        'head_vel': 'mean'
    })
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.to_csv(out_dir / 'bvh_temporal_summary.csv')
    
    print(f"✅ BVH Analysis Complete. Saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/kinematic-dataset-of-actors-expressing-emotions-2.1.0")
    ap.add_argument("--out", default="outputs/analysis/temporal_3d/v1")
    args = ap.parse_args()
    
    analyze_bvh_temporal(args.root, args.out)
