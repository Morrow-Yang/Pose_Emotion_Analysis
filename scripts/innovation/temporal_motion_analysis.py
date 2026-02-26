import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def calculate_motion_features(json_path, emotion):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        return pd.DataFrame()

    # Normalize possible OpenPose-like dict format: {image: {version, people:[{pose_keypoints_2d}]}}
    if isinstance(data, dict):
        normalized = []
        for img, val in data.items():
            people = val.get('people', []) if isinstance(val, dict) else []
            for person in people:
                kp = person.get('pose_keypoints_2d', [])
                normalized.append({
                    'image_id': img,
                    'keypoints': kp,
                    'score': 1.0,
                    'box': [0,0,1,1]
                })
        data = normalized

    # Create a DataFrame from AlphaPose-style list
    df = pd.DataFrame(data)
    
    # Extract frame number from image_id (handles nested paths like clip/00001.jpg)
    def _frame_num(x):
        stem = Path(x).stem
        # take trailing digits if any, fallback to whole stem
        import re
        m = re.search(r"(\d+)$", stem)
        return int(m.group(1)) if m else int(stem) if stem.isdigit() else -1
    df['frame_idx'] = df['image_id'].apply(_frame_num)
    # drop unknown frames
    df = df[df['frame_idx'] >= 0]
    
    # AlphaPose can have multiple people per frame. 
    # For temporal tracking without a real tracker, we'll pick the 'main' person per frame.
    # We choose the one with the highest score + largest area.
    df['bbox_area'] = df['box'].apply(lambda b: b[2] * b[3])
    df = df.sort_values(['frame_idx', 'score', 'bbox_area'], ascending=[True, False, False])
    
    # Keep only the top 1 person per frame for simplicity of trajectory
    df_main = df.groupby('frame_idx').head(1).copy()
    
    # Sort by frame index
    df_main = df_main.sort_values('frame_idx')
    
    # Keypoints are in [x1, y1, c1, x2, y2, c2, ...]
    # We want to extract (x,y) for major joints
    # 5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow, 9: L-Wrist, 10: R-Wrist
    # 11: L-Hip, 12: R-Hip, 13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
    joints_idx = {
        'l_shoulder': 5, 'r_shoulder': 6, 
        'l_elbow': 7, 'r_elbow': 8, 
        'l_wrist': 9, 'r_wrist': 10,
        'nose': 0
    }
    
    motion_records = []
    
    # Convert keypoints to easy access format
    kps_list = []
    for _, row in df_main.iterrows():
        kp = np.array(row['keypoints']).reshape(-1, 3)
        kps_list.append(kp)
    
    # Compute differences between successive frames
    for i in range(1, len(kps_list)):
        curr_kps = kps_list[i]
        prev_kps = kps_list[i-1]
        
        curr_frame = df_main.iloc[i]['frame_idx']
        prev_frame = df_main.iloc[i-1]['frame_idx']
        curr_image = df_main.iloc[i]['image_id']
        
        # Only consider it a continuous sequence if frame gap is small (e.g., 1 frame)
        if curr_frame - prev_frame > 2:
            continue
            
        # Normalize by bbox height of current frame to be distance-invariant
        bbox = df_main.iloc[i]['box'] # [x, y, w, h]
        norm_factor = bbox[3] if bbox[3] > 0 else 1.0
        
        # Calculate velocity for each joint
        # Velocity v = dist(kp_t, kp_t-1) / norm
        velocities = {}
        accelerations = {}
        total_velocity = 0
        valid_joints = 0
        
        for name, idx in joints_idx.items():
            # Only if both frames have high confidence for this joint
            if curr_kps[idx, 2] > 0.3 and prev_kps[idx, 2] > 0.3:
                dist = np.linalg.norm(curr_kps[idx, :2] - prev_kps[idx, :2])
                vel = dist / norm_factor
                velocities[f'{name}_vel'] = vel
                total_velocity += vel
                valid_joints += 1
                
                # Acceleration: delta_v / delta_t
                if i > 1:
                    # Check if previous velocity exists for this joint
                    # We look at the last record in motion_records for this joint
                    prev_vel = motion_records[-1][f'{name}_vel'] if len(motion_records) > 0 else np.nan
                    if not np.isnan(prev_vel):
                        accel = (vel - prev_vel)
                        accelerations[f'{name}_accel'] = accel
                    else:
                        accelerations[f'{name}_accel'] = np.nan
                else:
                    accelerations[f'{name}_accel'] = np.nan
            else:
                velocities[f'{name}_vel'] = np.nan
                accelerations[f'{name}_accel'] = np.nan
        
        avg_vel = total_velocity / valid_joints if valid_joints > 0 else np.nan
        
        record = {
            'emotion': emotion,
            'image_id': curr_image,
            'frame_idx': curr_frame,
            'avg_velocity': avg_vel,
            **velocities,
            **accelerations
        }
        motion_records.append(record)
        
    return pd.DataFrame(motion_records)

def main(root_dir, out_dir, json_name="alphapose-results.json"):
    root = Path(root_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    all_motion_data = []
    
    # Iterate through emotion folders
    emotions = [d.name for d in root.iterdir() if d.is_dir()]
    print(f"Detected emotions: {emotions}")
    
    for emotion in emotions:
        json_path = root / emotion / json_name
        if not json_path.exists():
            continue
            
        print(f"Processing {emotion}...")
        df_emotion = calculate_motion_features(json_path, emotion)
        if not df_emotion.empty:
            all_motion_data.append(df_emotion)
            
    if not all_motion_data:
        print("No motion data found!")
        return
        
    full_df = pd.concat(all_motion_data)
    # Calculate avg_acceleration (absolute value to represent 'intensity' of change)
    accel_cols = [c for c in full_df.columns if '_accel' in c]
    full_df['avg_acceleration'] = full_df[accel_cols].abs().mean(axis=1)
    
    full_df.to_csv(out / 'temporal_motion_features.csv', index=False)
    
    # Generate Summary Stats
    summary = full_df.groupby('emotion').agg({
        'avg_velocity': ['mean', 'std'],
        'avg_acceleration': ['mean', 'std'],
        'l_wrist_vel': 'mean',
        'r_wrist_vel': 'mean',
        'nose_vel': 'mean' 
    })
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.to_csv(out / 'temporal_summary_stats.csv')
    
    print(f"✅ Motion analysis complete. Saved to {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="AlphaPose result root (contains Angry/, etc.)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--json_name", default="alphapose-results.json", help="AlphaPose json filename to read")
    args = ap.parse_args()

    main(args.root, args.out, args.json_name)
