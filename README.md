# Pose Emotion Analysis

Emotion recognition from **3D full-body motion capture** using handcrafted kinematic features and Random Forest classification. Two independent 3D datasets are analyzed across four feature dimensions -- geometric posture, joint velocity, dynamic kinetic energy, and temporal dynamics -- covering 7 emotion categories.

> This project focuses on 3D MoCap (BVH + EBM). An earlier 2D AlphaPose/CAER-S pipeline also exists in the repo as a legacy reference.

---

## Datasets

| Dataset | Files | Structure | FPS | Duration |
|---------|-------|-----------|-----|---------|
| **BVH** (*Kinematic Dataset of Actors Expressing Emotions v2.1.0*) | 1,402 `.bvh` | Joint Euler rotations -> FK for world coords | 125 Hz | 6-16 s (median 8.1 s) |
| **EBM** (*Emotional Body Motion Data*) | 4,060 `.csv` | World coordinates (metres), 19 joints | ~30 Hz | Fixed 150 frames = 5 s |

**7 Emotions (both datasets):** Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprise

**EBM filename scheme:** `{actor}_{scenario}_{take}_{emotion_id}.csv`
Emotion IDs: 1=Angry, 2=Disgust, 3=Fearful, 4=Happy, 5=Neutral, 6=Sad, 7=Surprise

Raw data live under `data/raw/` and are **not tracked by git** (see `.gitignore`).

---

## Key Results

| Dataset | Feature set | Features | KW Large effects | RF Accuracy |
|---------|-------------|----------|-----------------|-------------|
| BVH | Geometric | 72 | 26 (top: elbow_angle_range e2=0.208) | -- |
| BVH | Kinetic energy | 11 | 4 (top: E_mean e2=0.367) | **58.3%** |
| EBM | Velocity + Energy + Geometry | 107 | 10 (top: wrist_vel_mean e2=0.211) | **49.9%** (approx. human 48.4%) |
| EBM | + Temporal dynamics | +22 | 7 additional (top: jerk_mean e2=0.204) | no gain (redundant) |

> RF trained with GroupShuffleSplit by actor (prevents identity leakage).

**Three temporal motion archetypes (EBM):**
- **Front-loaded decay** -- Fearful, Sad, Disgust: energy peaks in first second, declines sharply (front_ratio 65-70%)
- **Sustained** -- Angry, Happy: energy maintained throughout, high jerk (~1.2-1.9)
- **Flat / quiet** -- Surprise, Neutral: low jerk (<=0.85), smooth (autocorr >=0.95)

-> Full analysis: [`docs/analysis_3d_datasets.md`](docs/analysis_3d_datasets.md)

---

## Project Structure

```
AIemotion/
+-- data/
|   +-- raw/                          # Original datasets (not tracked by git)
|   |   +-- kinematic-dataset-of-actors-expressing-emotions-2.1.0/
|   |   |   +-- BVH/                  # BVH files organised by emotion/actor
|   |   |   +-- file-info.csv         # Manifest: filename, emotion, actor_ID
|   |   +-- Emotional Body Motion Data/
|   |       +-- Emotional Body Motion Data/  # 4060 CSV files (flat)
|   +-- external/
|       +-- AlphaPose-master/         # Legacy 2D pipeline dependency
|
+-- scripts/
|   +-- pipeline/                     # Core 3D analysis pipeline (main scripts)
|   |   +-- utils_bvh_parser.py       # BVH Forward Kinematics engine
|   |   +-- bvh_geometric_analysis.py # BVH: geometry + velocity + KW + RF
|   |   +-- bvh_energy_analysis.py    # BVH: kinetic energy analysis
|   |   +-- bvh_temporal_analysis.py  # BVH: per-frame velocity extraction
|   |   +-- ebm_full_analysis.py      # EBM: 107 features + KW + PCA + RF
|   |   +-- ebm_temporal_analysis.py  # EBM: 22 temporal features + 3 patterns
|   |
|   +-- analysis/                     # Visualisation and exploration scripts
|   |   +-- plot_geom_effect_sizes.py
|   |   +-- plot_geom_rf_summary.py
|   |   +-- plot_rf_slide.py
|   |   +-- plot_skeleton_gallery.py
|   |   +-- plot_3d_temporal_stats.py
|   |   +-- explore_ebm.py / explore_ebm2.py
|   |   +-- check_ebm_labels.py
|   |   +-- analysis_v1.py ... analysis_v5.py  # Legacy 2D geometry analysis
|   |   +-- run_all_v1_v5.py
|   |
|   +-- features/                     # Legacy 2D AlphaPose feature scripts
|       +-- filter_top1_alphapose.py
|       +-- yolo_filter_frames.py
|       +-- bvh_sequence_metrics.py
|
+-- outputs/
|   +-- analysis/
|   |   +-- geom_bvh_v2/              # BVH geometric features + RF results
|   |   +-- energy_bvh/               # BVH kinetic energy features
|   |   +-- ebm_full/                 # EBM 107-feature matrix + RF results
|   |   +-- ebm_temporal/             # EBM temporal features + curves
|   +-- experiments/
|       +-- classification_v1/        # Legacy 2D RF classification
|
+-- docs/
|   +-- analysis_3d_datasets.md       # Full analysis report (methods + results)
|   +-- figs_3d_temporal/             # BVH analysis figures (18 PNGs)
|   +-- figs_ebm/                     # EBM analysis figures (9 PNGs)
|   +-- PROJECT_COMPREHENSIVE_REPORT.md
|
+-- README.md
```

> `outputs/` and `data/raw/` are excluded from git (see `.gitignore`).
> Model weights (`*.pt`, `*.pth`, `*.onnx`) are also excluded.

---

## Running the 3D Analysis Pipeline

### Prerequisites

**Conda environment** (`aiemotion`):
```bash
conda activate aiemotion
# Key packages: numpy>=2.2, pandas>=2.3, scipy>=1.15, scikit-learn>=1.7,
#               matplotlib>=3.10, seaborn>=0.13, tqdm
```

All scripts are run from the **workspace root** (`AIemotion/`).
Relative paths in scripts (e.g. `outputs/analysis/...`, `docs/figs_3d_temporal/...`) are resolved from there.

---

### Step 1 -- BVH: Extract per-frame velocity features

Parses all 1,402 BVH files via Forward Kinematics, outputs per-frame joint velocities.

```bash
python scripts/pipeline/bvh_temporal_analysis.py \
  --root "data/raw/kinematic-dataset-of-actors-expressing-emotions-2.1.0" \
  --out  "outputs/analysis/temporal_3d/v1"
```

**Output:** `outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv` (1,402 x ~30 per-frame rows)

---

### Step 2 -- BVH: Geometric + velocity features, KW, PCA, RF

Reads the temporal CSV from Step 1. Extracts 72 geometric + 24 velocity aggregated features, runs Kruskal-Wallis, PCA/t-SNE, and Random Forest (GroupShuffleSplit by actor).

```bash
python scripts/pipeline/bvh_geometric_analysis.py
```

**Outputs in `outputs/analysis/geom_bvh_v2/`:**
- `bvh_geom_features.csv` -- 1402 x 96 feature matrix
- `kruskal_results.csv` -- KW H-statistic + e2 for all 72 features
- `pca_2d.csv`, `tsne_2d.csv` -- dimensionality reduction coordinates
- `rf_report.json` -- classification report (accuracy = 58.25%)

**Figures in `docs/figs_3d_temporal/`:** effect size bar chart, PCA/t-SNE plots, RF summary

---

### Step 3 -- BVH: Kinetic energy analysis

Derives 11 energy-level features from the per-frame velocities (E_mean, E_cv, burst_count, dom_freq, arms_share, head_share, etc.).

```bash
python scripts/pipeline/bvh_energy_analysis.py
```

**Output:** `outputs/analysis/energy_bvh/bvh_energy_features.csv` (1402 x 13)
**Figure:** `docs/figs_3d_temporal/bvh_energy_analysis.png` (4-panel)

---

### Step 4 -- EBM: Full static analysis (107 features)

Reads all 4,060 CSV files from the EBM dataset. Extracts 107 features (24 velocity + 8 energy + 75 geometry), runs KW, PCA/t-SNE, and RF.

```bash
python scripts/pipeline/ebm_full_analysis.py
```

**Outputs in `outputs/analysis/ebm_full/`:**
- `ebm_all_features.csv` -- 4060 x 110 matrix (features + actor/emotion labels)
- `kruskal_results.csv` -- KW results (104/106 features significant, p < 0.05)
- `rf_report.json` -- accuracy = 49.9%, macro-F1 = 0.497

**Figures in `docs/figs_ebm/`:** effect size chart, PCA/t-SNE (4 panels), energy analysis, RF summary

---

### Step 5 -- EBM: Temporal dynamics analysis (22 features)

Segments each 5-second clip into 5 equal phases; extracts per-phase energy/velocity and 7 shape descriptors (peak_time, front_ratio, energy_slope, jerk_mean, etc.). Runs KW and RF comparison vs. static baseline.

```bash
python scripts/pipeline/ebm_temporal_analysis.py
```

**Outputs in `outputs/analysis/ebm_temporal/`:**
- `ebm_temporal_features.csv` -- 4060 x 25
- `kruskal_temporal.csv` -- 7 LARGE effects (jerk_mean e2=0.204)
- `rf_comparison.json` -- RF with vs. without temporal features
- `temporal_curves.npz` -- mean energy/velocity time curves per emotion

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

Earlier work on the 2D CAER-S dataset is preserved in `scripts/analysis/analysis_v1-v5.py` and `scripts/features/`. These scripts are **not actively maintained**. See `docs/PROJECT_COMPREHENSIVE_REPORT.md` for a summary of those results (RF accuracy ~26% on 7 classes from 2D skeleton geometry).

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
```
