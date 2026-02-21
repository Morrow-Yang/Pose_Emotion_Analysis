import numpy as np
from pathlib import Path
from utils_bvh_parser import BVHParser

def verify_bvh_parser(bvh_path):
    parser = BVHParser(bvh_path)
    # Get total frames
    total_frames = len(parser.frames)
    test_frames = min(10, total_frames)
    
    errors = []
    
    # Pre-map parent offsets for faster lookup
    joint_data_map = {j['name']: j for j in parser.joints}
    
    for f in range(test_frames):
        world_coords = parser.get_joint_world_coords(f)
        
        # To verify the parser math, we need to know what local_pos was used at this frame
        frame_data = parser.frames[f]
        data_ptr = 0
        
        for joint in parser.joints:
            local_pos = joint['offset'].copy()
            for channel in joint['channels']:
                val = frame_data[data_ptr]
                data_ptr += 1
                c_low = channel.lower()
                if c_low == 'xposition': local_pos[0] = val
                elif c_low == 'yposition': local_pos[1] = val
                elif c_low == 'zposition': local_pos[2] = val
                
            parent_name = joint['parent']
            if parent_name is not None:
                parent_pos = world_coords[parent_name]
                curr_pos = world_coords[joint['name']]
                
                # Length must match the norm of the local translation vector used
                expected_len = np.linalg.norm(local_pos)
                measured_len = np.linalg.norm(curr_pos - parent_pos)
                
                # Accuracy check
                diff = abs(measured_len - expected_len)
                if diff > 1e-4: # Tighter tolerance for pure math check
                    print(f"DEBUG: Joint {joint['name']} | Expected: {expected_len:.6f} | Measured: {measured_len:.6f} | Diff: {diff:.2e}")
                errors.append(diff)
            
    avg_error = np.mean(errors) if errors else 0
    max_error = np.max(errors) if errors else 0
    
    return avg_error, max_error

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory for BVH files to test")
    args = ap.parse_args()
    
    root = Path(args.root)
    # Find some BVH files (find first 5)
    bvh_files = list(root.rglob("*.bvh"))[:5]
    
    if not bvh_files:
        print("No BVH files found in the specified directory.")
    else:
        print(f"Testing {len(bvh_files)} BVH files for Bone Length Invariance...")
        all_avg = []
        for bf in bvh_files:
            avg_e, max_e = verify_bvh_parser(bf)
            all_avg.append(avg_e)
            print(f"File: {bf.name} | Avg Error: {avg_e:.2e} | Max Error: {max_e:.2e}")
        
        global_avg = np.mean(all_avg)
        print("\n" + "="*40)
        print(f"VERIFICATION COMPLETE")
        print(f"Global Mean Square Error (MSE): {global_avg:.2e}")
        print(f"Status: {'PASSED' if global_avg < 1e-5 else 'FAILED'}")
        print("="*40)
