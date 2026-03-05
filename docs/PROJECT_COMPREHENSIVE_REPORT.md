# Project Comprehensive Report: Body Motion Emotion Recognition

**Project:** Multi-dimensional kinematic analysis of emotional body motion  
**Date:** 2026-03-05  
**Status:** Analysis complete — BVH and EBM datasets fully processed

---

## 1. Overview

This project investigates whether seven basic emotions (Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprise) can be distinguished from 3D full-body motion capture, using only handcrafted kinematic features and traditional ML (Random Forest). Two independent datasets are analyzed across four feature dimensions.

---

## 2. Analysis History

### Phase 1–3: 2D Foundation (Legacy)

Early work used **AlphaPose** on the **CAER-S** video dataset to extract 2D skeleton keypoints. 27 geometric + kinetic features were computed, yielding RF accuracy ~26% (7-class). A custom **BVH FK parser** (`utils_bvh_parser.py`) was built to validate 2D velocity estimates against 3D ground truth from the Kinematic BVH dataset, confirming that 2D energy rankings are physically grounded.

> Legacy scripts: `scripts/analysis/analysis_v1–v5.py`, `scripts/features/`

### Phase 4: BVH Geometric + Velocity Analysis

Extracted **72 geometric** and **24 velocity** aggregated features from 1,402 BVH clips using FK-derived world coordinates. Normalization by spine length removes actor height differences.

- Kruskal-Wallis: **26 large-effect features** (ε² ≥ 0.14), top = `left_elbow_angle_range` ε² = 0.208
- Random Forest: **58.25% accuracy** (GroupShuffleSplit by actor, 300 trees)
- Key insight: dynamic range features (std/range) outperform static means; the elbow joint is the dominant emotion signal carrier

### Phase 5: BVH Kinetic Energy Analysis

Derived **11 energy features** from instantaneous kinetic energy E(t) = Σ v_j(t)².

- **4 large effects**: E_mean ε² = 0.367, E_std = 0.357, E_max = 0.302, E_range = 0.302
- E_mean (ε² = 0.367) is the single strongest discriminator across all feature types
- Happy has highest E_mean (0.0012), Neutral/Sad lowest (0.0002)
- Angry has highest E_cv (2.014) — most explosive energy fluctuation

### Phase 6: EBM Full Static Analysis (107 Features)

Applied the same pipeline to **4,060 EBM clips** (29 actors, 19 joints, 5s fixed length):
- 107 features: 24 velocity + 8 energy + 75 geometry
- KW: **10 large effects**, 58 medium; 104/106 features significant (p < 0.05)
- Top: `l_wrist_vel_mean` ε² = 0.211
- RF: **49.9% accuracy**, macro-F1 = 0.497 — matching reported human accuracy (48.4%)
- Velocity/energy features dominate (7/10 large effects are velocity); geometry has zero large effects

### Phase 7: EBM Temporal Dynamics (22 Features)

Leveraging EBM's fixed 150-frame structure, segmented each clip into 5 phases and extracted time-shape descriptors:

- **7 large effects**: jerk_mean ε² = 0.204, vel_phase4 = 0.157, vel_phase5 = 0.155, …
- Later phases (3–5) are more discriminative than earlier ones (1–2)
- `jerk_mean` (ε² = 0.204) is a novel high-value feature absent from static analysis
- Adding temporal features to RF gave **no accuracy gain** (43.9% → 43.1%), due to redundancy with static energy features

**Three temporal archetypes identified:**

| Pattern | Emotions | Characteristics |
|---------|----------|----------------|
| Front-loaded decay | Fearful, Sad, Disgust | 65–70% energy in first half, slope < −1.0 |
| Sustained | Angry, Happy | Phase ratio ~1.3–1.5×, jerk 1.2–1.9 |
| Flat / quiet | Surprise, Neutral | Jerk ≤ 0.85, autocorrelation ≥ 0.95 |

---

## 3. Cross-Dataset Conclusions

1. **Wrist/arm velocity is the strongest emotion signal** — ε² ≈ 0.21 in both datasets independently
2. **Motion intensity > posture geometry** for emotion discrimination (arousal is the primary axis)
3. **RF classification reaches human-level performance** (49.9%–58.3% vs. human 48.4%)
4. **Temporal dynamics provide rich interpretive value** even when they don't improve classification numbers
5. **Surprise is a "static emotion"** — lowest energy but widest arm span and straightest elbows

---

## 4. Deliverables

| Artifact | Location |
|----------|----------|
| Full analysis report (methods + all numbers) | `docs/analysis_3d_datasets.md` |
| BVH geometric features + RF | `outputs/analysis/geom_bvh_v2/` |
| BVH energy features | `outputs/analysis/energy_bvh/` |
| EBM 107-feature matrix + RF | `outputs/analysis/ebm_full/` |
| EBM temporal features + curves | `outputs/analysis/ebm_temporal/` |
| BVH figures (18 PNGs) | `docs/figs_3d_temporal/` |
| EBM figures (9 PNGs) | `docs/figs_ebm/` |

---

*Generated for the AIemotion project. See README.md for setup and execution instructions.*
