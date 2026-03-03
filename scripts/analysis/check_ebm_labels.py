import pandas as pd
from pathlib import Path

xl = pd.read_excel(
    "data/raw/Emotional Body Motion Data/Human Evaluation result.xlsx",
    sheet_name="Accuracy per motion",
)

# Verify: last part of filename = True answer
mismatch = 0
for i, row in xl.iterrows():
    fname = str(row["animation_index"])
    parts = fname.split("_")
    emo = int(parts[-1])
    if emo != row["True answer"]:
        mismatch += 1
        if mismatch < 5:
            print(f"MISMATCH: {fname} -> filename_emo={emo}, true={row['True answer']}")

print(f"Total mismatches: {mismatch} / {len(xl)}")
print()

# Accuracy per emotion label
for label in range(1, 8):
    sub = xl[xl["True answer"] == label]
    print(f"Label {label}: N={len(sub)}, mean_accuracy={sub['Accuracy'].mean():.3f}")

# Quick check: read one CSV, show joint layout
data_dir = Path("data/raw/Emotional Body Motion Data/Emotional Body Motion Data")
sample = pd.read_csv(data_dir / "1_1_1_1.csv")
print(f"\nSample CSV shape: {sample.shape}")
print(f"Columns: {sample.columns.tolist()}")
print(f"Hips.y range: {sample['Hips.y'].min():.3f} - {sample['Hips.y'].max():.3f}")
