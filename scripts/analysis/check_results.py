import pandas as pd, json
from pathlib import Path

kw = pd.read_csv('outputs/analysis/geom_bvh_v2/kruskal_results.csv')
print('=== Top-20 KW features ===')
print(kw.head(20).to_string(index=False))

with open('outputs/analysis/geom_bvh_v2/rf_report.json') as f:
    rf = json.load(f)
acc = rf['accuracy']
print(f'\n=== RF accuracy: {acc:.3f} ===')
rpt = rf['report']
for emo in ['Angry','Disgust','Fearful','Happy','Neutral','Sad','Surprise']:
    if emo in rpt:
        r = rpt[emo]
        print(f'  {emo:<10} P={r["precision"]:.2f} R={r["recall"]:.2f} F1={r["f1-score"]:.2f} N={int(r["support"])}')
print(f'  macro-avg F1={rpt["macro avg"]["f1-score"]:.3f}')
print('\nTop-15 features:')
for name, imp in rf['feature_importance'][:15]:
    print(f'  {imp:.4f}  {name}')
