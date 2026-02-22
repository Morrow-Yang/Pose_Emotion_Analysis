import os
import pandas as pd
import numpy as np
import glob

def analyze_motion_file(filepath):
    """
    Computes key physical metrics from a motion CSV file.
    Metrics:
    - Mean Velocity: Energy/Speed.
    - Bounding Box Volume: Spatial Expansion.
    - Head Tilt: Postural alignment (Y-axis).
    - Jerk: Motion smoothness.
    - Hand Height: Vertical position of hands relative to head.
    """
    try:
        df = pd.read_csv(filepath)
        
        # --- velocity ---
        hips = df[['Hips.x', 'Hips.y', 'Hips.z']].values
        vel = np.diff(hips, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        mean_speed = np.mean(speed)
        
        # --- volume ---
        # Get all joint columns (assuming naming convention 'JointName.axis')
        joint_cols = [c for c in df.columns if '.x' in c]
        joints = [c.split('.')[0] for c in joint_cols]
        
        all_x = df[[j+'.x' for j in joints]].values
        all_y = df[[j+'.y' for j in joints]].values
        all_z = df[[j+'.z' for j in joints]].values
        
        # Per frame volume
        min_x, max_x = np.min(all_x, axis=1), np.max(all_x, axis=1)
        min_y, max_y = np.min(all_y, axis=1), np.max(all_y, axis=1)
        min_z, max_z = np.min(all_z, axis=1), np.max(all_z, axis=1)
        
        volumes = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        mean_volume = np.mean(volumes)
        
        # --- head tilt ---
        # Vector from Spine1 (roughly chest) to Head
        if 'Spine1.x' in df.columns and 'Head.x' in df.columns:
            head = df[['Head.x', 'Head.y', 'Head.z']].values
            spine = df[['Spine1.x', 'Spine1.y', 'Spine1.z']].values
            vec = head - spine
            # Normalize
            norms = np.linalg.norm(vec, axis=1)
            # Avoid division by zero
            norms[norms == 0] = 1e-6
            vec = vec / norms[:, None]
            # Tilt is alignment with Y-axis (0, 1, 0)
            tilt = np.mean(vec[:, 1])
        else:
            tilt = 0.0
            
        # --- jerk ---
        head_pos = df[['Head.x', 'Head.y', 'Head.z']].values
        # 1st deriv = velocity
        v = np.diff(head_pos, axis=0)
        # 2nd deriv = acceleration
        a = np.diff(v, axis=0)
        # 3rd deriv = jerk
        j = np.diff(a, axis=0)
        mean_jerk = np.mean(np.linalg.norm(j, axis=1))

        # --- hand height ---
        if 'LeftHand.y' in df.columns and 'RightHand.y' in df.columns and 'Head.y' in df.columns:
             hands_y = (df['LeftHand.y'] + df['RightHand.y']) / 2
             head_y = df['Head.y']
             hand_height = np.mean(hands_y - head_y)
        else:
             hand_height = 0.0

        return {
            'speed': mean_speed, 
            'volume': mean_volume, 
            'tilt': tilt, 
            'jerk': mean_jerk,
            'hand_height': hand_height
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    data_dir = r"c:\Users\Mingt\Documents\AIemotion\data\raw\Emotional Body Motion Data\Emotional Body Motion Data"
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    results = []
    
    # Process a subset for speed, or all if feasible. 
    # Let's process representatively or all. 4000 files is okay.
    # But let's limit to 50 files per label to demonstrate quickly.
    
    label_counts = {}
    
    for f in files:
        filename = os.path.basename(f)
        try:
            # Filename format: P1_P2_P3_P4.csv where P4 is label
            parts = filename.replace('.csv', '').split('_')
            label = int(parts[3])
            
            if label not in label_counts:
                label_counts[label] = 0
            
            if label_counts[label] < 50: # Limit for quick analysis
                metrics = analyze_motion_file(f)
                if metrics:
                    metrics['label'] = label
                    results.append(metrics)
                    label_counts[label] += 1
        except:
            continue
            
    # Aggregate
    df_res = pd.DataFrame(results)
    summary = df_res.groupby('label').mean()
    
    print("Physical feature analysis by Label (1-7):")
    print(summary.to_markdown())
    
    # Save for later
    summary.to_csv('new_dataset_analysis_summary.csv')

if __name__ == "__main__":
    main()
