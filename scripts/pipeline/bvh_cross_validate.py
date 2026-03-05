"""Cross-validate custom BVH forward kinematics against the `bvh` library.

This script compares joint world coordinates produced by our custom `BVHParser`
with those computed from the same BVH file using the `bvh` library hierarchy.
It reports per-joint Euclidean errors for a small sample of files.
"""
import numpy as np
from pathlib import Path
from bvh import Bvh
from utils_bvh_parser import BVHParser

# Joints to probe for numerical consistency
PROBE_JOINTS = [
    "Hips",
    "Head",
    "LeftHand",
    "RightHand",
    "LeftFoot",
    "RightFoot",
]


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def compute_world_coords_bvhlib(tree: Bvh, frame_idx: int) -> dict:
    """Compute world coordinates for every joint using the `bvh` tree structure."""
    coords = {}
    rotations = {}

    joint_names = tree.get_joints_names()
    parent_map = {}
    children_map = {name: [] for name in joint_names}
    for name in joint_names:
        parent_raw = tree.joint_parent(name)
        parent = None
        if parent_raw:
            tokens = getattr(parent_raw, "value", [])
            if tokens:
                parent = tokens[-1]
        parent_map[name] = parent
        if parent:
            children_map.setdefault(parent, []).append(name)

    def walk(joint_name: str, parent_pos: np.ndarray, parent_rot: np.ndarray):
        offset = np.array(tree.joint_offset(joint_name), dtype=float)
        channels = tree.joint_channels(joint_name)
        values = tree.frame_joint_channels(frame_idx, joint_name, channels)

        local_pos = offset.copy()
        local_rot = np.eye(3)

        for cname, val in zip(channels, values):
            cl = cname.lower()
            if "position" in cl:
                if cl.startswith("x"):
                    local_pos[0] = val
                elif cl.startswith("y"):
                    local_pos[1] = val
                elif cl.startswith("z"):
                    local_pos[2] = val
            elif "rotation" in cl:
                ang = np.radians(val)
                if cl.startswith("x"):
                    local_rot = local_rot @ _rot_x(ang)
                elif cl.startswith("y"):
                    local_rot = local_rot @ _rot_y(ang)
                elif cl.startswith("z"):
                    local_rot = local_rot @ _rot_z(ang)

        world_pos = local_pos if parent_pos is None else parent_pos + parent_rot @ local_pos
        world_rot = local_rot if parent_rot is None else parent_rot @ local_rot

        coords[joint_name] = world_pos
        rotations[joint_name] = world_rot

        for child in children_map.get(joint_name, []):
            walk(child, world_pos, world_rot)

    root_name = joint_names[0]
    walk(root_name, None, np.eye(3))
    return coords


def compare_single_file(bvh_path: Path, frame_idx: int = 0) -> dict:
    """Compare coordinates for a single BVH file at a specific frame."""
    parser = BVHParser(str(bvh_path))
    ours = parser.get_joint_world_coords(frame_idx)

    with open(bvh_path, "r") as f:
        tree = Bvh(f.read())
    theirs = compute_world_coords_bvhlib(tree, frame_idx)

    diffs = {}
    for joint in PROBE_JOINTS:
        if joint in ours and joint in theirs:
            diff = np.linalg.norm(ours[joint] - theirs[joint])
            diffs[joint] = diff
    return diffs


def run_sample(dataset_root: Path, sample_limit: int = 3) -> None:
    bvh_dir = dataset_root / "BVH"
    files = sorted(bvh_dir.glob("*/F*.bvh"))
    if not files:
        print(f"No BVH files found under {bvh_dir}")
        return

    print(f"Validating first {sample_limit} files under {bvh_dir} ...")
    for idx, bvh_path in enumerate(files[:sample_limit]):
        diffs = compare_single_file(bvh_path, frame_idx=0)
        max_diff = max(diffs.values()) if diffs else 0.0
        print(f"[{idx+1}] {bvh_path.name}: max joint error = {max_diff:.6f}")
        for joint, err in diffs.items():
            print(f"    {joint}: {err:.6f}")


if __name__ == "__main__":
    DATASET_ROOT = Path(r"c:\\Users\\Mingt\\Documents\\AIemotion\\data\\raw\\kinematic-dataset-of-actors-expressing-emotions-2.1.0")
    run_sample(DATASET_ROOT, sample_limit=3)
