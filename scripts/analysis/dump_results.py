import pandas as pd, json, numpy as np

kw = pd.read_csv('outputs/analysis/geom_bvh_v2/kruskal_results.csv')
print('=== KW summary by category ===')
print(kw.groupby('category')['H'].describe().round(1))

print('\n=== Top-10 velocity ===')
print(kw[kw.category=='velocity'].nlargest(10,'H')[['feature','H','p']].to_string(index=False))

print('\n=== Top-10 3D-only ===')
print(kw[kw.category=='3D-only'].nlargest(10,'H')[['feature','H','p']].to_string(index=False))

print('\n=== Top-10 2D-like ===')
print(kw[kw.category=='2D-like'].nlargest(10,'H')[['feature','H','p']].to_string(index=False))

# EPS-squared effect sizes
N = 1402
kw['eps2'] = (kw['H'] - kw.shape[0] + 1) / (N - kw.shape[0])
kw['eps2'] = (kw['H'] - 6) / (N - 7)  # k=7 groups correct formula: (H - k + 1)/(N - k)
print('\n=== Effect sizes (eps^2 = (H-k+1)/(N-k), k=7) ===')
for cat in ['velocity','3D-only','2D-like']:
    sub = kw[kw.category==cat]
    print(f'  {cat}: mean eps2={sub.eps2.mean():.4f}, max eps2={sub.eps2.max():.4f}')

with open('outputs/analysis/geom_bvh_v2/rf_report.json') as f:
    rf = json.load(f)
print('\n=== RF per-class ===')
rpt = rf['report']
emotions = ['Angry','Disgust','Fearful','Happy','Neutral','Sad','Surprise']
for emo in emotions:
    if emo in rpt:
        v = rpt[emo]
        print(f'  {emo:<10} prec={v["precision"]:.3f} rec={v["recall"]:.3f} F1={v["f1-score"]:.3f} N={int(v["support"])}')
print(f'  macro-avg  prec={rpt["macro avg"]["precision"]:.3f} rec={rpt["macro avg"]["recall"]:.3f} F1={rpt["macro avg"]["f1-score"]:.3f}')
print(f'  weighted   F1={rpt["weighted avg"]["f1-score"]:.3f}')
print(f'  accuracy: {rf["accuracy"]:.4f}')

print('\n=== Top-20 feature importances ===')
for name, imp in rf['feature_importance'][:20]:
    print(f'  {imp:.5f}  {name}')

# Feature counts
feat_df = pd.read_csv('outputs/analysis/geom_bvh_v2/bvh_geom_features.csv')
print(f'\n=== Dataset shape: {feat_df.shape} ===')
print(feat_df['emotion'].value_counts())
print(f'Actors: {feat_df["actor"].nunique()}, Files: {len(feat_df)}')
