# Pose_Emotion_Analysis

Pose-based emotion analysis with AlphaPose (2D) and BVH motion capture (3D). This repository includes feature extraction, statistical analysis (v1–v5), and innovation scripts for new features and temporal analysis.

## Project Structure
```
AIemotion/
  data/
    raw/                    # original datasets (not tracked)
      CAER-S/
      kinematic-dataset-of-actors-expressing-emotions-2.1.0/
    external/               # third-party tools (not tracked)
      AlphaPose-master/
  outputs/                  # generated outputs (not tracked)
    alphapose/
    features/
    analysis/
    innovation/
  scripts/
    analysis/               # analysis_v1-v5, run_all_v1_v5
    features/               # metrics-2.py, alphapose_extra_features.py, bvh_*.py
    innovation/             # innovation_*.py
  README.md
```

## Data Sources
- **CAER-S**: Contextualized Affect Representations from Scenes.
- **Kinematic Dataset of Actors Expressing Emotions (v2.1.0)**: BVH motion capture sequences.

> Note: Datasets are not included in this repo. Please download them from their official sources.

## AlphaPose Models
AlphaPose models and weights are not included. Use the official AlphaPose repository to download:
- YOLO detector weights
- FastPose model

## Quick Start
### 1) Extract features from AlphaPose outputs
```powershell
python scripts\features\metrics-2.py --root "outputs\alphapose\outputs\CAER-S\train" --outdir "outputs\features"
```

### 2) Run v1–v5 analysis
```powershell
python scripts\analysis\run_all_v1_v5.py --root "outputs\alphapose\outputs\CAER-S\train" --out_base "outputs\analysis"
```

### 3) Extract extra AlphaPose features
```powershell
python scripts\features\alphapose_extra_features.py --root "outputs\alphapose\outputs\CAER-S\train" --outdir "outputs\features"
```

### 4) BVH per-sequence features
```powershell
python scripts\features\bvh_sequence_metrics.py \
  --bvh_root "data\raw\kinematic-dataset-of-actors-expressing-emotions-2.1.0\BVH" \
  --file_info "data\raw\kinematic-dataset-of-actors-expressing-emotions-2.1.0\file-info.csv" \
  --outdir "outputs\features"
```

## Citation
If you use this project, please cite the original datasets and AlphaPose.

## License
This repository contains code only. Follow the original licenses for datasets and AlphaPose.
