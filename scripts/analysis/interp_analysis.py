"""
Emotion-level interpretability analysis
- Per-emotion median ± IQR for top features
- Which features best separate each emotion from rest
- Pairwise confusion explanation
"""
import pandas as pd, numpy as np, json
from scipy.stats import mannwhitneyu
from pathlib import Path

OUT   = Path("outputs/analysis/geom_bvh_v2")
EMOS  = ["Angry","Disgust","Fearful","Happy","Neutral","Sad","Surprise"]

feat_df = pd.read_csv(OUT / "bvh_geom_features.csv")
with open(OUT / "rf_report.json") as f:
    rf = json.load(f)

# ── top features (geometry only, exist in feat_df) ─────────────────────────
kw = pd.read_csv(OUT / "kruskal_results.csv")
N, k = 1402, 7
kw["eps2"] = ((kw["H"] - k + 1) / (N - k)).clip(lower=0)
avail = set(feat_df.columns)
kw_avail = kw[kw["feature"].isin(avail)].sort_values("eps2", ascending=False)
top_feats = kw_avail.head(20)["feature"].tolist()

# ── 1. per-emotion median (normalized) for top features ─────────────────────
print("=== Per-emotion MEDIAN for top-20 geometry features ===")
rows = {}
for feat in top_feats:
    row = {}
    for emo in EMOS:
        vals = feat_df[feat_df.emotion==emo][feat].dropna()
        row[emo] = f"{vals.median():.3f}"
    rows[feat] = row
med_df = pd.DataFrame(rows).T
med_df.index.name = "feature"
print(med_df.to_string())

# ── 2. one-vs-rest rank-biserial r: which emotion stands out on each feature ──
print("\n\n=== One-vs-rest rank-biserial r (positive = emotion HIGHER than others) ===")
for feat in top_feats[:15]:
    best_emo, best_r = "", 0
    row_vals = []
    for emo in EMOS:
        g1 = feat_df[feat_df.emotion==emo][feat].dropna().values
        g2 = feat_df[feat_df.emotion!=emo][feat].dropna().values
        if len(g1)<2 or len(g2)<2: continue
        u, _ = mannwhitneyu(g1, g2, alternative="two-sided")
        r = 1 - 2*u/(len(g1)*len(g2))
        row_vals.append((emo, r))
        if abs(r) > abs(best_r):
            best_r, best_emo = r, emo
    parts = "  ".join(f"{e[:3]}={r:+.2f}" for e,r in row_vals)
    print(f"  {feat:<35} || {parts}")
    print(f"    -> most distinct: {best_emo} (r={best_r:+.3f})")

# ── 3. confusion analysis (from RF report) ─────────────────────────────────
print("\n\n=== RF per-class analysis ===")
rpt = rf["report"]
for emo in EMOS:
    if emo not in rpt: continue
    v = rpt[emo]
    print(f"  {emo:<10}  P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1-score']:.3f}  N={int(v['support'])}")

# ── 4. velocity feature medians ─────────────────────────────────────────────
# load temporal CSV
tcsv = Path("outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv")
if tcsv.exists():
    vel_cols_base = ["avg_velocity","head_vel","l_shoulder_vel","r_shoulder_vel",
                     "l_elbow_vel","r_elbow_vel","l_wrist_vel","r_wrist_vel"]
    tdf = pd.read_csv(tcsv, usecols=["filename","emotion"]+vel_cols_base)
    # file-level mean
    file_vel = tdf.groupby(["filename","emotion"])[vel_cols_base].mean().reset_index()
    print("\n\n=== Per-emotion MEDIAN velocity (file-level mean) ===")
    for emo in EMOS:
        sub = file_vel[file_vel.emotion==emo]
        avg = sub["avg_velocity"].median()
        wrist = ((sub["l_wrist_vel"]+sub["r_wrist_vel"])/2).median()
        head  = sub["head_vel"].median()
        print(f"  {emo:<10}  avg_vel={avg:.5f}  wrist={wrist:.5f}  head={head:.5f}")

    # velocity std per file
    file_vstd = tdf.groupby(["filename","emotion"])[vel_cols_base].std().reset_index()
    print("\n=== Per-emotion MEDIAN velocity STD (within-file variability) ===")
    for emo in EMOS:
        sub = file_vstd[file_vstd.emotion==emo]
        avg = sub["avg_velocity"].median()
        print(f"  {emo:<10}  avg_vel_std={avg:.5f}")
