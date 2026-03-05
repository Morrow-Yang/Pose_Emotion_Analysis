"""
创新：新增身体朝向特征和动态特征
学姐只用了角度/长度，您可以加入：
1. 身体朝向（躯干倾斜角）
2. 重心位置（上半身vs下半身质心）
3. 对称性指数（左右对称程度）
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def parse_kp(arr_51):
    """解析AlphaPose关键点"""
    a = np.array(arr_51, dtype=float).reshape(17, 3)
    return a[:, :2], a[:, 2]  # xy, confidence

def safe_point(xy, cf, idx, conf_th=0.30):
    """安全获取关键点"""
    if cf[idx] >= conf_th and np.isfinite(xy[idx]).all():
        return xy[idx]
    return None

def compute_new_features(xy, cf):
    """计算创新特征"""
    KP = {
        "nose": 0, "left_eye": 1, "right_eye": 2,
        "left_shoulder": 5, "right_shoulder": 6,
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16
    }
    
    features = {}
    
    # 获取关键点
    ls = safe_point(xy, cf, KP["left_shoulder"])
    rs = safe_point(xy, cf, KP["right_shoulder"])
    lh = safe_point(xy, cf, KP["left_hip"])
    rh = safe_point(xy, cf, KP["right_hip"])
    nose = safe_point(xy, cf, KP["nose"])
    
    # 特征1：躯干倾斜角（身体朝向）
    if ls is not None and rs is not None and lh is not None and rh is not None:
        shoulder_mid = (ls + rs) / 2
        hip_mid = (lh + rh) / 2
        torso_vec = shoulder_mid - hip_mid
        # 与垂直方向夹角（度）
        vertical_angle = np.degrees(np.arctan2(torso_vec[0], torso_vec[1]))
        features['torso_tilt_angle'] = float(vertical_angle)
    else:
        features['torso_tilt_angle'] = np.nan
    
    # 特征2：重心位置（归一化y坐标）
    if ls is not None and rs is not None and lh is not None and rh is not None:
        upper_center = (ls + rs) / 2
        lower_center = (lh + rh) / 2
        # 重心偏向上半身为正，下半身为负
        center_of_mass_y = upper_center[1] - lower_center[1]
        features['center_of_mass_shift'] = float(center_of_mass_y)
    else:
        features['center_of_mass_shift'] = np.nan
    
    # 特征3：肩髋对称性（左右差异）
    if ls is not None and rs is not None and lh is not None and rh is not None:
        shoulder_slope = (rs[1] - ls[1]) / (rs[0] - ls[0] + 1e-8)
        hip_slope = (rh[1] - lh[1]) / (rh[0] - lh[0] + 1e-8)
        symmetry_index = abs(shoulder_slope - hip_slope)
        features['body_symmetry'] = float(symmetry_index)
    else:
        features['body_symmetry'] = np.nan
    
    # 特征4：头部相对于躯干的位置
    if nose is not None and ls is not None and rs is not None:
        shoulder_mid = (ls + rs) / 2
        head_offset_x = abs(nose[0] - shoulder_mid[0])
        head_offset_y = nose[1] - shoulder_mid[1]
        features['head_offset_magnitude'] = float(np.sqrt(head_offset_x**2 + head_offset_y**2))
    else:
        features['head_offset_magnitude'] = np.nan
    
    return features

def extract_new_features(root_dir, json_name="alphapose-results.json"):
    """提取所有新特征"""
    root = Path(root_dir)
    records = []
    
    for emotion_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        emotion = emotion_dir.name
        json_file = emotion_dir / json_name
        
        if not json_file.exists():
            continue
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            image_id = item.get("image_id", "unknown")
            kps = item.get("keypoints", None)
            if kps is None:
                continue
            
            xy, cf = parse_kp(kps)
            new_feats = compute_new_features(xy, cf)
            
            record = {
                'emotion': emotion,
                'image_id': image_id,
                **new_feats
            }
            records.append(record)
    
    return pd.DataFrame(records)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="AlphaPose输出根目录")
    ap.add_argument("--outdir", required=True, help="输出目录")
    args = ap.parse_args()
    
    df = extract_new_features(args.root)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_csv = out_dir / "new_features.csv"
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 新特征提取完成：{out_csv}")
    print(f"   样本数：{len(df)}")
    print(f"   新特征：{[c for c in df.columns if c not in ['emotion', 'image_id']]}")
