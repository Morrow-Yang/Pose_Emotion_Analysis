# Pose Emotion Analysis

Emotion recognition from **3D full-body motion capture** using handcrafted kinematic features and Random Forest classification. Two independent 3D datasets are analyzed across four feature dimensions 鈥?geometric posture, joint velocity, dynamic kinetic energy, and temporal dynamics 鈥?covering 7 emotion categories.

> This project focuses on 3D MoCap (BVH + EBM). An earlier 2D AlphaPose/CAER-S pipeline also exists in the repo as a legacy reference.

---

## Datasets

| Dataset | Files | Structure | FPS | Duration |
|---------|-------|-----------|-----|---------|
| **BVH** (*Kinematic Dataset of Actors Expressing Emotions v2.1.0*) | 1,402 `.bvh` | Joint Euler rotations 鈫?FK for world coords | 125 Hz | 6鈥?6 s (median 8.1 s) |
| **EBM** (*Emotional Body Motion Data*) | 4,060 `.csv` | World coordinates (metres), 19 joints | ~30 Hz | Fixed 150 frames = 5 s |

**7 Emotions (both datasets):** Angry 路 Disgust 路 Fearful 路 Happy 路 Neutral 路 Sad 路 Surprise

**EBM filename scheme:** `{actor}_{scenario}_{take}_{emotion_id}.csv`  
Emotion IDs: 1=Angry, 2=Disgust, 3=Fearful, 4=Happy, 5=Neutral, 6=Sad, 7=Surprise

Raw data live under `data/raw/` and are **not tracked by git** (see `.gitignore`).

---

## Key Results

| Dataset | Feature set | Features | KW Large effects | RF Accuracy |
|---------|-------------|----------|-----------------|-------------|
| BVH | Geometric | 72 | 26 (top: elbow_angle_range 蔚虏=0.208) | 鈥?|
| BVH | Kinetic energy | 11 | 4 (top: E_mean 蔚虏=0.367) | **58.3%** |
| EBM | Velocity + Energy + Geometry | 107 | 10 (top: wrist_vel_mean 蔚虏=0.211) | **49.9%** 鈮?human (48.4%) |
| EBM | + Temporal dynamics | +22 | 7 additional (top: jerk_mean 蔚虏=0.204) | no gain (redundant) |

> RF trained with GroupShuffleSplit by actor (prevents identity leakage).

**Three temporal motion archetypes (EBM):**
- **Front-loaded decay** 鈥?Fearful, Sad, Disgust: energy peaks in first second, declines sharply (front_ratio 65鈥?0%)
- **Sustained** 鈥?Angry, Happy: energy maintained throughout, high jerk (~1.2鈥?.9)
- **Flat / quiet** 鈥?Surprise, Neutral: low jerk (鈮?.85), smooth (autocorr 鈮?.95)

鈫?Full analysis: [`docs/analysis_3d_datasets.md`](docs/analysis_3d_datasets.md)

---

## Project Structure

```
AIemotion/
鈹溾攢鈹€ data/
鈹?  鈹溾攢鈹€ raw/                          # Original datasets (not tracked by git)
鈹?  鈹?  鈹溾攢鈹€ kinematic-dataset-of-actors-expressing-emotions-2.1.0/
鈹?  鈹?  鈹?  鈹溾攢鈹€ BVH/                  # BVH files organised by emotion/actor
鈹?  鈹?  鈹?  鈹斺攢鈹€ file-info.csv         # Manifest: filename, emotion, actor_ID
鈹?  鈹?  鈹斺攢鈹€ Emotional Body Motion Data/
鈹?  鈹?      鈹斺攢鈹€ Emotional Body Motion Data/  # 4060 CSV files (flat)
鈹?  鈹斺攢鈹€ external/
鈹?      鈹斺攢鈹€ AlphaPose-master/         # Legacy 2D pipeline dependency
鈹?
鈹溾攢鈹€ scripts/
鈹?  鈹溾攢鈹€ pipeline/                     # Core 3D analysis pipeline (main scripts)
鈹?  鈹?  鈹溾攢鈹€ utils_bvh_parser.py       # BVH Forward Kinematics engine
鈹?  鈹?  鈹溾攢鈹€ bvh_geometric_analysis.py # BVH: geometry + velocity + KW + RF
鈹?  鈹?  鈹溾攢鈹€ bvh_energy_analysis.py    # BVH: kinetic energy analysis
鈹?  鈹?  鈹溾攢鈹€ bvh_temporal_analysis.py  # BVH: per-frame velocity extraction
鈹?  鈹?  鈹溾攢鈹€ ebm_full_analysis.py      # EBM: 107 features + KW + PCA + RF
鈹?  鈹?  鈹斺攢鈹€ ebm_temporal_analysis.py  # EBM: 22 temporal features + 3 patterns
鈹?  鈹?
鈹?  鈹溾攢鈹€ analysis/                     # Visualisation and exploration scripts
鈹?  鈹?  鈹溾攢鈹€ plot_geom_effect_sizes.py
鈹?  鈹?  鈹溾攢鈹€ plot_geom_rf_summary.py
鈹?  鈹?  鈹溾攢鈹€ plot_rf_slide.py
鈹?  鈹?  鈹溾攢鈹€ plot_skeleton_gallery.py
鈹?  鈹?  鈹溾攢鈹€ plot_3d_temporal_stats.py
鈹?  鈹?  鈹溾攢鈹€ explore_ebm.py / explore_ebm2.py
鈹?  鈹?  鈹溾攢鈹€ check_ebm_labels.py
鈹?  鈹?  鈹溾攢鈹€ analysis_v1.py 鈥?analysis_v5.py  # Legacy 2D geometry analysis
鈹?  鈹?  鈹斺攢鈹€ run_all_v1_v5.py
鈹?  鈹?
鈹?  鈹斺攢鈹€ features/                     # Legacy 2D AlphaPose feature scripts
鈹?      鈹溾攢鈹€ filter_top1_alphapose.py
鈹?      鈹溾攢鈹€ yolo_filter_frames.py
鈹?      鈹斺攢鈹€ bvh_sequence_metrics.py
鈹?
鈹溾攢鈹€ outputs/
鈹?  鈹溾攢鈹€ analysis/
鈹?  鈹?  鈹溾攢鈹€ geom_bvh_v2/              # BVH geometric features + RF results
鈹?  鈹?  鈹溾攢鈹€ energy_bvh/              # BVH kinetic energy features
鈹?  鈹?  鈹溾攢鈹€ ebm_full/                 # EBM 107-feature matrix + RF results
鈹?  鈹?  鈹斺攢鈹€ ebm_temporal/            # EBM temporal features + curves
鈹?  鈹斺攢鈹€ experiments/
鈹?      鈹斺攢鈹€ classification_v1/        # Legacy 2D RF classification
鈹?
鈹溾攢鈹€ docs/
鈹?  鈹溾攢鈹€ analysis_3d_datasets.md       # Full analysis report (methods + results)
鈹?  鈹溾攢鈹€ figures/
鈹?  鈹?  鈹溾攢鈹€ (bvh) figs_3d_temporal/   # BVH analysis figures (18 PNGs)
鈹?  鈹?  鈹斺攢鈹€ (ebm) figs_ebm/           # EBM analysis figures (9 PNGs)
鈹?  鈹斺攢鈹€ PROJECT_COMPREHENSIVE_REPORT.md
鈹?
鈹斺攢鈹€ README.md
```

> `outputs/` and `data/raw/` are excluded from git (see `.gitignore`).  
> Model weights (`*.pt`, `*.pth`, `*.onnx`) are also excluded.

---

## Running the 3D Analysis Pipeline

### Prerequisites

**Conda environment** (`aiemotion`):
```bash
conda activate aiemotion
# Key packages: numpy鈮?.2, pandas鈮?.3, scipy鈮?.15, scikit-learn鈮?.7,
#               matplotlib鈮?.10, seaborn鈮?.13, tqdm
```

All scripts are run from the **workspace root** (`C:\Users\...\AIemotion`).  
Relative paths in scripts (e.g. `outputs/analysis/鈥, `docs/figs_3d_temporal/鈥) are resolved from there.

---

### Step 1 鈥?BVH: Extract per-frame velocity features

Parses all 1,402 BVH files via Forward Kinematics, outputs per-frame joint velocities.

```bash
python scripts/pipeline/bvh_temporal_analysis.py \
  --root "data/raw/kinematic-dataset-of-actors-expressing-emotions-2.1.0" \
  --out  "outputs/analysis/temporal_3d/v1"
```

**Output:** `outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv` (1,402 脳 ~30 per-frame rows)

---

### Step 2 鈥?BVH: Geometric + velocity features, KW, PCA, RF

Reads the temporal CSV from Step 1. Extracts 72 geometric + 24 velocity aggregated features, runs Kruskal-Wallis, PCA/t-SNE, and Random Forest (GroupShuffleSplit by actor).

```bash
python scripts/pipeline/bvh_geometric_analysis.py
```

**Outputs in `outputs/analysis/geom_bvh_v2/`:**
- `bvh_geom_features.csv` 鈥?1402 脳 96 feature matrix
- `kruskal_results.csv` 鈥?KW H-statistic + 蔚虏 for all 72 features
- `pca_2d.csv`, `tsne_2d.csv` 鈥?dimensionality reduction coordinates
- `rf_report.json` 鈥?classification report (accuracy = 58.25%)

**Figures in `docs/figs_3d_temporal/`:** effect size bar chart, PCA/t-SNE plots, RF summary

---

### Step 3 鈥?BVH: Kinetic energy analysis

Derives 11 energy-level features from the per-frame velocities (E_mean, E_cv, burst_count, dom_freq, arms_share, head_share, 鈥?.

```bash
python scripts/pipeline/bvh_energy_analysis.py
```

**Output:** `outputs/analysis/energy_bvh/bvh_energy_features.csv` (1402 脳 13)  
**Figure:** `docs/figs_3d_temporal/bvh_energy_analysis.png` (4-panel)

---

### Step 4 鈥?EBM: Full static analysis (107 features)

Reads all 4,060 CSV files from the EBM dataset. Extracts 107 features (24 velocity + 8 energy + 75 geometry), runs KW, PCA/t-SNE, and RF.

```bash
python scripts/pipeline/ebm_full_analysis.py
```

**Outputs in `outputs/analysis/ebm_full/`:**
- `ebm_all_features.csv` 鈥?4060 脳 110 matrix (features + actor/emotion labels)
- `kruskal_results.csv` 鈥?KW results (104/106 features significant, p < 0.05)
- `rf_report.json` 鈥?accuracy = 49.9%, macro-F1 = 0.497

**Figures in `docs/figs_ebm/`:** effect size chart, PCA/t-SNE (4 panels), energy analysis, RF summary

---

### Step 5 鈥?EBM: Temporal dynamics analysis (22 features)

Segments each 5-second clip into 5 equal phases; extracts per-phase energy/velocity and 7 shape descriptors (peak_time, front_ratio, energy_slope, jerk_mean, 鈥?. Runs KW and RF comparison vs. static baseline.

```bash
python scripts/pipeline/ebm_temporal_analysis.py
```

**Outputs in `outputs/analysis/ebm_temporal/`:**
- `ebm_temporal_features.csv` 鈥?4060 脳 25
- `kruskal_temporal.csv` 鈥?7 LARGE effects (jerk_mean 蔚虏=0.204)
- `rf_comparison.json` 鈥?RF with vs. without temporal features
- `temporal_curves.npz` 鈥?mean energy/velocity time curves per emotion

**Figures in `docs/figs_ebm/`:** time curves, phase heatmap, effect sizes, boxplots, RF delta

---

### Visualisation scripts (optional)

Generate additional publication figures from pre-computed outputs:

```bash
# BVH effect size figure
python scripts/analysis/plot_geom_effect_sizes.py

# BVH skeleton posture gallery (7 emotions side by side)
python scripts/analysis/plot_skeleton_gallery.py

# BVH RF summary slide figure
python scripts/analysis/plot_rf_slide.py

# EBM 3D temporal statistics overview
python scripts/analysis/plot_3d_temporal_stats.py
```

---

## Legacy: 2D AlphaPose / CAER-S Pipeline

Earlier work on the 2D CAER-S dataset is preserved in `scripts/analysis/analysis_v1鈥搗5.py` and `scripts/features/`. These scripts are **not actively maintained**. See `docs/PROJECT_COMPREHENSIVE_REPORT.md` for a summary of those results (RF accuracy ~26% on 7 classes from 2D skeleton geometry).

---

## Data Sources

| Dataset | Source |
|---------|--------|
| Kinematic Dataset of Actors Expressing Emotions v2.1.0 | [PhysioNet](https://physionet.org/content/kinematic-dataset-actors-emotions/2.1.0/) |
| Emotional Body Motion (EBM) | Contact dataset authors |
| CAER-S (legacy 2D) | [CAER GitHub](https://github.com/kaist-viclab/CAER) |

---

## Repository

```
GitHub: https://github.com/Morrow-Yang/Pose_Emotion_Analysis
Branch: main
