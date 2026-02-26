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

def _iter_bvh_files(dataset_root: Path, file_info: Path):
    """Yield (bvh_path, filename, emotion, actor) records.
    If file-info.csv exists, use it. Otherwise, infer from folder names: emotion = parent folder name, actor = parent of parent.
    """
    bvh_dir = dataset_root / 'BVH'
    if file_info.exists():
        info_df = pd.read_csv(file_info)
        for _, row in info_df.iterrows():
            filename = row['filename']
            emotion = row.get('emotion', 'unknown')
            actor = row.get('actor_ID', 'unknown')
            bvh_path = bvh_dir / actor / f"{filename}.bvh"
            yield bvh_path, filename, emotion, actor
    else:
        for bvh_path in bvh_dir.rglob("*.bvh"):
            filename = bvh_path.stem
            emotion = bvh_path.parent.name
            actor = bvh_path.parent.parent.name if bvh_path.parent.parent else "unknown"
            yield bvh_path, filename, emotion, actor


def analyze_bvh_temporal(dataset_root, out_dir, file_info="file-info.csv"):
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info_csv = dataset_root / file_info
    if not info_csv.exists():
        print(f"[WARN] {info_csv} not found, will infer emotion from folder names under BVH/<emotion>/<actor>/*.bvh")

    all_motion_data = []

    print("🏃 Starting BVH Temporal Analysis...")

    bvh_files = list(_iter_bvh_files(dataset_root, info_csv))

    for bvh_path, filename, emotion, actor in tqdm(bvh_files, total=len(bvh_files)):
        if not bvh_path.exists():
            continue

        try:
            parser = BVHParser(str(bvh_path))
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        motion_records = []
        prev_coords = None
        prev_vel = None

        for f_idx in range(len(parser.frames)):
            curr_coords = parser.get_joint_world_coords(f_idx)

            # pelvis-to-head scale for normalization
            if 'Hips' in curr_coords and 'Head' in curr_coords:
                height_norm = np.linalg.norm(curr_coords['Hips'] - curr_coords['Head'])
            else:
                height_norm = 1.0

            if prev_coords is not None:
                velocities = {}
                accelerations = {}
                total_vel = 0
                count = 0

                for feat_name, bvh_name in JOINT_MAP.items():
                    if bvh_name in curr_coords and bvh_name in prev_coords:
                        dist = np.linalg.norm(curr_coords[bvh_name] - prev_coords[bvh_name])
                        vel = dist / height_norm if height_norm > 1e-6 else dist
                        velocities[f'{feat_name}_vel'] = vel
                        total_vel += vel
                        count += 1

                        if prev_vel and f'{feat_name}_vel' in prev_vel:
                            accel = vel - prev_vel[f'{feat_name}_vel']
                            accelerations[f'{feat_name}_accel'] = accel
                        else:
                            accelerations[f'{feat_name}_accel'] = np.nan
                    else:
                        velocities[f'{feat_name}_vel'] = np.nan
                        accelerations[f'{feat_name}_accel'] = np.nan

                avg_vel = total_vel / count if count > 0 else np.nan
                accel_vals = [abs(v) for v in accelerations.values() if np.isfinite(v)]
                avg_accel = float(np.mean(accel_vals)) if accel_vals else np.nan

                motion_records.append({
                    'dataset': dataset_root.name,
                    'actor': actor,
                    'filename': filename,
                    'emotion': emotion,
                    'frame_idx': f_idx,
                    'time_sec': f_idx * parser.frame_time,
                    'avg_velocity': avg_vel,
                    'avg_acceleration': avg_accel,
                    **velocities,
                    **accelerations
                })

                prev_vel = velocities

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
        'avg_acceleration': ['mean', 'std'],
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
    ap.add_argument("--file_info", default="file-info.csv", help="CSV with filename, emotion, actor_ID; if missing, emotions are inferred from BVH/<emotion>/<actor>/*.bvh")
    args = ap.parse_args()

    analyze_bvh_temporal(args.root, args.out, args.file_info)
