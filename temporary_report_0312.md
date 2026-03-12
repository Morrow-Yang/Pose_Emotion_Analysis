# Pose Emotion Analysis — Session Report (2026-03-12)
# 姿态情绪分析 — 会话报告（2026年3月12日）

---

## 1. Work Completed / 已完成工作

Two independent 3D MoCap datasets have been fully analyzed across four feature dimensions with Kruskal-Wallis statistics and Random Forest classification.

已完成对两个独立三维动作捕捉数据集的完整分析，涵盖四个特征维度，并进行了 Kruskal-Wallis 统计检验和随机森林分类。

| Dataset / 数据集 | Clips / 样本数 | Features / 特征数 | RF Accuracy / RF 准确率 |
|---|---|---|---|
| BVH (geometric + velocity) | 1,402 | 96 | **58.25%** |
| EBM (velocity + energy + geometry) | 4,060 | 107 | **49.9%** ≈ human 48.4% |
| EBM + temporal dynamics | 4,060 | 129 | 43.1% (no gain / 无提升) |

All RF classifiers use GroupShuffleSplit by actor to prevent identity leakage.
所有随机森林分类器均按演员进行 GroupShuffleSplit 分组，防止身份信息泄露。

---

## 2. Quantitative Feature Summary Per Emotion / 各情绪特征量化摘要

All EBM values are medians unless noted. Effect sizes (ε²) from Kruskal-Wallis tests.
以下 EBM 数值均为中位数，效应量 ε² 来自 Kruskal-Wallis 检验。

---

### 2.1 Angry (愤怒)

**Core identity: highest energy + highest jerkiness + arm-dominated motion**
**核心特征：最高能量 + 最强急促感 + 手臂主导**

| Feature / 特征 | Value / 值 | ε² |
|---|---|---|
| `E_mean` (EBM) | **3.89** — highest of 7 / 全体最高 | 0.160 |
| `avg_velocity` | **0.517 m/s** — highest / 最高 | 0.166 |
| `arms_share_mean` | **84.0%** — highest arm energy ratio / 手臂能量占比最高 | 0.145 |
| `jerk_mean` | **1.922** — highest / 最高，= 3.4× Surprise | 0.204 |
| `E_cv` (BVH) | **2.014** — most volatile energy / 能量波动最剧烈 | — |
| `dom_freq` (BVH) | **0.18 Hz** — slowest rhythm / 节奏最慢 | — |
| Temporal slope / 能量斜率 | −0.10 — near-flat, sustained throughout / 几乎水平，全段持续 | — |
| Ph1/Ph5 ratio | **1.3×** — energy barely decays / 能量几乎不衰减 | — |

**Temporal archetype / 时间模式:** Sustained (持续维持型) — energy maintained across all 5 phases. Not smooth: high-frequency bursts alternating with brief pauses.
能量在五段中持续维持，但充满急促的加减速，并非流畅的大幅度动作。

---

### 2.2 Happy (快乐)

**Core identity: widest body openness + mid-clip reactivation**
**核心特征：最宽展体态 + 动作中段再激活**

| Feature / 特征 | Value / 值 | ε² |
|---|---|---|
| `arm_span_norm_mean` | **5.74** — widest of 7 / 全体最宽 | 0.131 |
| `elbow_angle_mean` | **113°** — most extended (after Surprise) | — |
| `E_mean` (BVH) | **highest in BVH** / BVH 中能量最高 | 0.367 |
| `jerk_mean` | 1.176 — second highest / 第二高 | 0.204 |
| Phase3 energy | **1.73** — rebounds after Phase2 dip / Phase2后回升，其他情绪无此现象 | unique |
| Ph1/Ph5 ratio | 1.5× — gradual gentle decay / 缓慢衰减 | — |

**Temporal archetype / 时间模式:** Sustained (持续维持型) — unique mid-clip energy reactivation at Phase 3. Possible cause: rhythmic laughter/clapping/jumping actions.
独特的 Phase3 能量回升（仅 Happy 具有此特征），可能对应笑声、鼓掌、跳跃等节律性快乐动作。

---

### 2.3 Fearful (恐惧)

**Core identity: earliest onset, fastest collapse, then freeze**
**核心特征：最早爆发、最快衰减、后期凝固**

| Feature / 特征 | Value / 值 | ε² |
|---|---|---|
| `peak_time_norm` | **0.17** — earliest energy peak of all 7 / 峰值最早 | — |
| `energy_front_ratio` | **70%** — highest front-loading / 前半段能量占比最高 | large |
| `rise_time_norm` | **0.14** — fastest onset / 起爆最快 | — |
| `head_share_mean` | **8.9%** — highest head energy ratio / 头部能量贡献最高 | 0.151 |
| Ph1/Ph5 ratio | **3.8×** | — |
| Phase 3–5 energy | Flat plateau: 0.70 → 0.58 → 0.59 / 后三段形成"凝固"平台 | unique |
| `jerk_mean` | 0.971 — moderate | 0.204 |

**Temporal archetype / 时间模式:** Front-loaded decay (前倾急衰型) with unique freeze plateau at Phases 3–5. Psychological interpretation: initial startle response → freezing reaction.
前段高能量应激，后段三段呈现"冻结"平台——对应恐惧的"惊跳→僵住"两阶段反应。

---

### 2.4 Sad (悲伤)

**Core identity: most contracted posture + high-frequency micro-bursts at low energy**
**核心特征：最收缩体态 + 高频微小能量起伏**

| Feature / 特征 | Value / 值 | ε² |
|---|---|---|
| `arm_span_norm_mean` | **2.93** — narrowest of 7 / 全体最窄 | 0.131 |
| `elbow_angle_mean` | **83°** — most bent elbow / 肘角最弯 | — |
| `trunk_tilt_deg` | 9.4° — second highest backward tilt / 后倾第二 | 0.074 |
| `E_mean` | 0.91 — second lowest / 第二低 | 0.160 |
| `burst_count` (BVH) | **57** — most micro-bursts / 爆发次数最多 | 0.100 |
| `jerk_mean` | 0.777 — low, smooth motion / 低，动作流畅 | 0.204 |
| Ph1/Ph5 ratio | **4.9×** — steep decay / 衰减明显 | — |

**Temporal archetype / 时间模式:** Front-loaded decay (前倾急衰型). Unlike Disgust (monotonic), Sad shows a more gradual fade. Frequent small tremors at low overall energy = "low-intensity fidgeting/trembling."
缓慢衰减（不同于 Disgust 的急速单调下降）；多次微小能量起伏可能对应轻微颤抖/抽泣动作。

---

### 2.5 Disgust (厌恶)

**Core identity: maximum backward trunk lean + steepest monotonic energy decay**
**核心特征：最大后仰躯干 + 最陡单调能量衰减**

| Feature / 特征 | Value / 值 | ε² |
|---|---|---|
| `trunk_tilt_deg_mean` | **9.5°** — highest backward lean / 后倾最大 | 0.074 |
| `head_share_mean` | **10.8%** — highest head ratio in EBM / 头部能量占比最高 | 0.151 |
| `E_mean` | 0.72 — second lowest / 第二低 | 0.160 |
| Ph1/Ph5 ratio | **5.9×** — steepest decay of all / 衰减比最大 | — |
| `energy_slope` | **−1.04** — steepest monotonic decline / 最陡单调下降 | large |
| `energy_autocorr_lag1` | **0.96** — highest smoothness / 动作最平滑 | — |
| `jerk_mean` | **0.627** — second lowest / 第二低 | 0.204 |

**Temporal archetype / 时间模式:** Front-loaded decay (前倾急衰型) — most extreme version: no plateau, steepest single-direction decline. "Recoil-and-withdraw" pattern.
最极端的前倾急衰：无平台，持续单调下滑，对应"遭遇厌恶刺激→后仰回避→逐渐宁静"的完整回避过程。

---

### 2.6 Surprise (惊讶)

**Core identity: a "static emotion" — pose carries information, not movement**
**核心特征："静态情绪"——由姿态而非运动传递情绪信息**

| Feature / 特征 | Value / 值 | ε² |
|---|---|---|
| `E_mean` | **0.68** — lowest of all 7 / 全体最低 | 0.160 |
| `elbow_angle_mean` | **139°** — most extended arms / 肘角最直 | — |
| `arm_span_norm` | 5.31 — second widest / 第二宽展 | 0.131 |
| `trunk_tilt_deg` | **4.7°** — most upright / 躯干最正 | 0.074 |
| `jerk_mean` | **0.563** — lowest of all / 全体最低 | 0.204 |
| `energy_autocorr_lag1` | **0.96** — tied highest smoothness / 平滑度最高 | — |
| Phase2–3 | Flat plateau 0.45→0.45 / 平台型 | unique |

**Temporal archetype / 时间模式:** Flat/restrained (平稳低调型) — single transition to open pose then near-static hold. Orienting/freeze response.
一次性切换至张臂挺身姿势后保持近乎静止——对应"定向反射"本能（暂停运动以全力处理新信息）。

---

### 2.7 Neutral (中性)

**Core identity: most regular rhythmic motion, no extremes**
**核心特征：最规律的节奏性动作，所有指标居中**

| Feature / 特征 | Value / 值 | ε² |
|---|---|---|
| `E_cv` (BVH) | **1.310** — lowest energy variability / 能量变异最小 | — |
| `dom_freq` (BVH) | **0.48 Hz** — highest / 节奏频率最高 | — |
| All static features | Near median — no extremes / 均接近中位数 | — |
| BVH RF recall | ≈86% — easiest to recall / 召回率最高 | — |

**Temporal archetype / 时间模式:** Flat/restrained (平稳低调型). Uniform, metronomic activity pattern with no emotional drive.
最均匀、最有节奏的基线运动，无任何情绪驱动的特殊模式。

---

## 3. Cross-Emotion Comparison / 跨情绪横向对比

### 3.1 Energy (Arousal) Dimension / 能量（唤醒）维度

```
Highest ←————————————————————————→ Lowest
Angry   Happy   Neutral  Fearful  Sad   Disgust  Surprise
(3.89)  (2.31)  (1.29)   (1.53)   (0.91) (0.72)  (0.68)
```

### 3.2 Arm Openness / 臂展宽度

```
Widest ←————————————————————————→ Narrowest
Happy   Surprise  Angry  Fearful  Disgust  Neutral  Sad
(5.74)  (5.31)   (5.17)  (4.45)   (3.23)   (3.92)  (2.93)
```

### 3.3 Jerkiness (Motion Abruptness) / 动作急促感

```
Highest ←————————————————————————→ Smoothest
Angry   Happy   Neutral  Fearful  Sad  Disgust  Surprise
(1.92)  (1.18)  (0.85)   (0.97)  (0.78) (0.63)  (0.56)
```

### 3.4 Three Temporal Archetypes / 三种时间动态原型

| Archetype / 原型 | Emotions / 情绪 | Key signature / 关键标志 |
|---|---|---|
| Front-loaded decay / 前倾急衰型 | Fearful, Sad, Disgust | Ph1/Ph5 ≥ 3.8×, slope < −1.0 |
| Sustained / 持续维持型 | Angry, Happy | Ph1/Ph5 ≤ 1.5×, jerk ≥ 1.18 |
| Flat / restrained / 平稳低调型 | Surprise, Neutral | jerk ≤ 0.85, autocorr ≥ 0.95 |

---

## 4. Current Limitations / 当前局限性

**1. Aggregate scalars, not trajectories / 聚合标量，非轨迹**
Every extracted feature collapses a full clip into one number (e.g., E_mean, elbow_angle_range). There is no representation of the actual pose at any specific frame, making direct GIF generation impossible from the current feature set.
所有特征将整段动作压缩为单一数值，无法直接用于 GIF 渲染。

**2. Most discriminative dimension is not visualizable as a pose / 最具区分力的维度无法表达为可视姿态**
Speed and energy are the strongest classifiers (ε² ≈ 0.16–0.21), but they describe how fast the body moves, not where it is at any moment.
速度/能量虽然区分力最强，但描述的是运动快慢而非任意时刻的姿态。

**3. Confusable emotion pairs / 易混情绪对**
- Sad ↔ Disgust: both low-energy, contracted, backward-tilted
- Happy ↔ Angry: both high-energy, wide-armed
- Fearful ↔ Neutral: similar motion amplitude

---

## 5. Recommended Next Steps for GIF Creation / GIF 制作建议路径

### Path A — Data-Driven (Recommended / 推荐)

Select and render real EBM clips. Steps:
筛选并渲染真实 EBM 片段。步骤：

1. Load `outputs/analysis/ebm_full/ebm_all_features.csv`
2. Compute Mahalanobis distance of each clip to its emotion centroid in the full 107-feature space
3. For each emotion, select **top-5 centroid-closest clips** (most prototypical) + **top-2 extremal clips** (most exaggerated on the defining feature)
4. Render raw EBM CSV joint coordinates as 3D skeleton animations with `matplotlib FuncAnimation` → export as GIF

This requires no new computation. Raw EBM CSV world coordinates are already available at 30 Hz.
无需新计算，直接使用已有 EBM 世界坐标数据（30 Hz）渲染。

### Path B — Feature-Driven Design / 特征驱动设计

Design controlled skeleton animations from the feature profiles above:
根据上述特征画像设计受控骨骼动画：

| Emotion | Pose design | Motion design |
|---|---|---|
| Angry | High arm position, moderate spread | Fast jerky swings, sustained 5 s |
| Happy | Widest arm spread (elbow ≈113°) | Mid-clip energy reactivation, bouncy ~1.2 Hz |
| Fearful | Arms pulled in, head forward/down | Sudden burst, then freeze for last 3 s |
| Sad | Narrowest arms (83° elbow), hunched trunk | Slow smooth decay + micro-tremors |
| Disgust | Backward trunk lean 9.5°, arms low | Fluid recoil, monotonic energy fade |
| Surprise | Arms wide (139° elbow), upright | Single fast transition → hold |
| Neutral | All values at median | Regular rhythmic oscillation ~0.48 Hz |

### Path C — Separability Enhancement / 提升可分性

For the most-confused pairs (Sad/Disgust, Happy/Angry):
针对易混情绪对：

- **Sad vs Disgust:** Exaggerate trunk_tilt direction (Sad = forward slump; Disgust = backward lean) and arm_span difference (2.93 vs 3.23 → render as 2.5 vs 4.0)
- **Happy vs Angry:** Encode cadence explicitly (Happy: smooth bouncy rhythm; Angry: staccato jerky rhythm)
- Add **head orientation** as a controlled variable (not in current feature set but a known strong cue): Fearful = head down/averted; Angry = head forward/confrontational

---

## 6. Git Status / Git 状态

| Status | Details |
|---|---|
| Branch | `main` (up to date with `origin/main`) |
| Untracked | `scripts/innovation/` (2 files: `csv_analysis.py`, `inspect_motion_data.py`) |
| Staged | `temporary_report_0312.md` (this file, newly added) |

The `scripts/innovation/` directory contains exploratory scripts that have not yet been committed. They will be added in this session's commit.
`scripts/innovation/` 包含尚未提交的探索性脚本，将在本次提交中一并加入。

---

*Generated: 2026-03-12 | Datasets: BVH (1,402) + EBM (4,060) | Analysis environment: Python, scikit-learn, scipy, matplotlib*
*生成日期：2026-03-12 | 数据集：BVH (1,402) + EBM (4,060) | 分析环境：Python, scikit-learn, scipy, matplotlib*
