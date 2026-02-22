import math
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import kruskal

from utils_bvh_parser import BVHParser

# velocity/temporal feature file produced by temporal_3d/v1 pipeline
TEMPORAL_FEATURE_CSV = Path("outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees using 3D vectors."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-8 or nbc < 1e-8:
        return np.nan
    cosv = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def contraction_index_3d(norm_coords: Dict[str, np.ndarray]) -> float:
    """Negative mean distance to torso center; more positive = more收缩."""
    use = ["Head", "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow",
           "LeftHand", "RightHand", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"]
    pts = []
    for n in use:
        if n in norm_coords:
            pts.append(norm_coords[n])
    if len(pts) < 5:
        return np.nan
    pts = np.stack(pts, axis=0)
    if {"LeftShoulder", "RightShoulder", "LeftUpLeg", "RightUpLeg"}.issubset(norm_coords.keys()):
        torso_center = (norm_coords["LeftShoulder"] + norm_coords["RightShoulder"] +
                        norm_coords["LeftUpLeg"] + norm_coords["RightUpLeg"]) / 4.0
    else:
        torso_center = pts.mean(axis=0)
    d = np.linalg.norm(pts - torso_center[None, :], axis=1)
    return float(-np.mean(d))


def trunk_tilt(norm_coords: Dict[str, np.ndarray]) -> float:
    """Angle between trunk (Hips->Head) and world up (0,1,0)."""
    if "Hips" not in norm_coords or "Head" not in norm_coords:
        return np.nan
    vec = norm_coords["Head"] - norm_coords["Hips"]
    nvec = np.linalg.norm(vec)
    if nvec < 1e-8:
        return np.nan
    vec = vec / nvec
    up = np.array([0.0, 1.0, 0.0])
    cosv = np.clip(np.dot(vec, up), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))  # 0=直立, 越大越倾斜


def normalize_coords(raw: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
    """Translate by Hips, scale by spine length (Head-Hips)."""
    if "Hips" not in raw or "Head" not in raw:
        return {}, np.nan
    spine = np.linalg.norm(raw["Head"] - raw["Hips"])
    if spine < 1e-8:
        return {}, np.nan
    normed = {k: (v - raw["Hips"]) / spine for k, v in raw.items()}
    return normed, spine


def extract_features(coords: Dict[str, np.ndarray]) -> Dict[str, float]:
    norm, spine = normalize_coords(coords)
    if not norm:
        return {}

    feats = {}
    # shoulder width
    if "LeftShoulder" in norm and "RightShoulder" in norm:
        feats["shoulder_width"] = float(np.linalg.norm(norm["LeftShoulder"] - norm["RightShoulder"]))
    else:
        feats["shoulder_width"] = np.nan

    # hand height relative to shoulder (Y axis up)
    for side in ["Left", "Right"]:
        shoulder = norm.get(f"{side}Shoulder")
        hand = norm.get(f"{side}Hand")
        feats[f"{side.lower()}_hand_height"] = float(hand[1] - shoulder[1]) if (shoulder is not None and hand is not None) else np.nan

    # arm span normalized by shoulder width
    lw, rw = norm.get("LeftHand"), norm.get("RightHand")
    if lw is not None and rw is not None and np.isfinite(feats["shoulder_width"]) and feats["shoulder_width"] > 1e-6:
        feats["arm_span_norm"] = float(abs(lw[0] - rw[0]) / feats["shoulder_width"])
    else:
        feats["arm_span_norm"] = np.nan

    # elbow & knee angles
    feats["left_elbow_angle"] = angle_3d(norm.get("LeftShoulder", np.nan), norm.get("LeftForeArm", np.nan), norm.get("LeftHand", np.nan))
    feats["right_elbow_angle"] = angle_3d(norm.get("RightShoulder", np.nan), norm.get("RightForeArm", np.nan), norm.get("RightHand", np.nan))
    feats["left_knee_angle"] = angle_3d(norm.get("LeftUpLeg", np.nan), norm.get("LeftLeg", np.nan), norm.get("LeftFoot", np.nan))
    feats["right_knee_angle"] = angle_3d(norm.get("RightUpLeg", np.nan), norm.get("RightLeg", np.nan), norm.get("RightFoot", np.nan))

    # contraction
    feats["contraction"] = contraction_index_3d(norm)

    # symmetry
    lh = feats["left_hand_height"]
    rh = feats["right_hand_height"]
    feats["hand_height_asym"] = abs(lh - rh) if (np.isfinite(lh) and np.isfinite(rh)) else np.nan

    le = feats["left_elbow_angle"]
    re = feats["right_elbow_angle"]
    feats["elbow_asym"] = abs(le - re) if (np.isfinite(le) and np.isfinite(re)) else np.nan

    # head relative to shoulder mid
    if "LeftShoulder" in norm and "RightShoulder" in norm and "Head" in norm:
        shoulder_mid = (norm["LeftShoulder"] + norm["RightShoulder"]) / 2.0
        head_dx, head_dy, head_dz = (norm["Head"] - shoulder_mid)
        feats["head_dx"] = float(head_dx)
        feats["head_dy"] = float(head_dy)
        feats["head_dz"] = float(head_dz)
    else:
        feats["head_dx"] = feats["head_dy"] = feats["head_dz"] = np.nan

    feats["trunk_tilt_deg"] = trunk_tilt(norm)
    feats["spine_len"] = spine
    return feats


def load_temporal_features(csv_path: Path) -> pd.DataFrame:
    """Aggregate per-sequence velocity stats so they can align with geometric features."""
    if not csv_path.exists():
        print(f"[WARN] Temporal CSV not found: {csv_path}")
        return pd.DataFrame()

    cols = [
        "avg_velocity",
        "head_vel",
        "l_shoulder_vel",
        "r_shoulder_vel",
        "l_elbow_vel",
        "r_elbow_vel",
        "l_wrist_vel",
        "r_wrist_vel",
    ]

    tdf = pd.read_csv(csv_path, usecols=["filename", "emotion", *cols])
    agg = tdf.groupby(["filename", "emotion"], as_index=False).agg({c: ["mean", "std"] for c in cols})
    agg.columns = [
        "filename",
        "emotion",
        *[f"{c}_{stat}" for c in cols for stat in ("mean", "std")],
    ]
    return agg


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def main():
    dataset_root = Path("data/raw/kinematic-dataset-of-actors-expressing-emotions-2.1.0")
    info_df = pd.read_csv(dataset_root / "file-info.csv")
    bvh_dir = dataset_root / "BVH"
    out_dir = Path("outputs/analysis/geom_bvh_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_csv = out_dir / "bvh_geom_features.csv"

    if feat_csv.exists():
        print(f"[+] Load existing features: {feat_csv}")
        df = pd.read_csv(feat_csv)
    else:
        rows: List[dict] = []
        print("[+] Computing 3D几何特征 (与2D对齐)...")
        for _, row in tqdm(info_df.iterrows(), total=len(info_df)):
            fname = row["filename"]
            emotion = row["emotion"]
            actor = row["actor_ID"]
            bvh_path = bvh_dir / actor / f"{fname}.bvh"
            if not bvh_path.exists():
                continue
            try:
                parser = BVHParser(str(bvh_path))
                coords = parser.get_joint_world_coords(0)  # 使用第0帧代表静态姿态
                feats = extract_features(coords)
                if not feats:
                    continue
                feats["emotion"] = emotion
                feats["filename"] = fname
                feats["actor"] = actor
                rows.append(feats)
            except Exception as e:
                print(f"[WARN] skip {fname}: {e}")
                continue

        df = pd.DataFrame(rows)
        df.to_csv(feat_csv, index=False, encoding="utf-8-sig")
        print("[+] Saved features:", feat_csv)

    # merge temporal velocity stats if available
    tdf = load_temporal_features(TEMPORAL_FEATURE_CSV)
    if not tdf.empty:
        df = df.merge(tdf, on=["filename", "emotion"], how="left")

    geom_features = [
        "shoulder_width", "left_hand_height", "right_hand_height", "arm_span_norm",
        "left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle",
        "contraction", "hand_height_asym", "elbow_asym", "head_dx", "head_dy", "head_dz",
        "trunk_tilt_deg"
    ]

    temp_features = []
    if not tdf.empty:
        temp_features = [
            "avg_velocity_mean", "avg_velocity_std",
            "head_vel_mean", "head_vel_std",
            "l_shoulder_vel_mean", "l_shoulder_vel_std",
            "r_shoulder_vel_mean", "r_shoulder_vel_std",
            "l_elbow_vel_mean", "l_elbow_vel_std",
            "r_elbow_vel_mean", "r_elbow_vel_std",
            "l_wrist_vel_mean", "l_wrist_vel_std",
            "r_wrist_vel_mean", "r_wrist_vel_std",
        ]

    FEATURES = geom_features + temp_features

    df_clean = df.dropna(subset=FEATURES)
    if df_clean.empty:
        print("[!] No clean samples for stats; abort stats stage.")
        return

    # 1) Kruskal-Wallis per feature
    stats_rows = []
    for feat in FEATURES:
        groups = [grp[feat].values for _, grp in df_clean.groupby("emotion")]
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        stats_rows.append({"feature": feat, "H": H, "p": p})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "kruskal_results.csv", index=False)
    print("[+] Saved stats: kruskal_results.csv")

    # 2) PCA
    X = df_clean[FEATURES].values
    pca = PCA(n_components=2, random_state=42)
    pc = pca.fit_transform(X)
    pca_df = pd.DataFrame({"pc1": pc[:, 0], "pc2": pc[:, 1], "emotion": df_clean["emotion"].values})
    pca_df.to_csv(out_dir / "pca_2d.csv", index=False)
    print("[+] Saved PCA projection: pca_2d.csv explained_var=", pca.explained_variance_ratio_)

    # 3) t-SNE (small perplexity for 1k samples)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, n_iter=500)
    ts = tsne.fit_transform(X)
    ts_df = pd.DataFrame({"tsne1": ts[:, 0], "tsne2": ts[:, 1], "emotion": df_clean["emotion"].values})
    ts_df.to_csv(out_dir / "tsne_2d.csv", index=False)
    print("[+] Saved t-SNE projection: tsne_2d.csv")

    # 4) KMeans silhouette sweep
    sil_rows = []
    for k in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sil_rows.append({"k": k, "silhouette": sil})
    pd.DataFrame(sil_rows).to_csv(out_dir / "kmeans_silhouette.csv", index=False)
    print("[+] Saved kmeans_silhouette.csv")

    # 5) RandomForest baseline (geometry only)
    y = df_clean["emotion"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rpt = classification_report(y_test, y_pred, output_dict=True)
    feat_imp = sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: x[1], reverse=True)

    with open(out_dir / "rf_report.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "report": rpt, "feature_importance": feat_imp}, f, ensure_ascii=False, indent=2)
    print(f"[+] RF accuracy={acc:.3f}; saved rf_report.json")

if __name__ == "__main__":
    main()
