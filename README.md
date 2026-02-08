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
