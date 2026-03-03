import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path('data/raw/Emotional Body Motion Data/Emotional Body Motion Data')
files = sorted(data_dir.glob('*.csv'))

# Frame counts (sample 50)
frame_counts = [len(pd.read_csv(f)) for f in files[:50]]
print(f'Frame counts (first 50): min={min(frame_counts)}, max={max(frame_counts)}, mean={np.mean(frame_counts):.1f}')

# Coordinate scale
sample = pd.read_csv(files[0])
print(f'\nHips.y range:  {sample["Hips.y"].min():.3f} - {sample["Hips.y"].max():.3f}  (meters)')
print(f'Head.y range:  {sample["Head.y"].min():.3f} - {sample["Head.y"].max():.3f}')
print(f'Frame col:     {sample["Frame"].iloc[:5].tolist()}')

# Actor / label breakdown
actors = sorted(set(int(f.stem.split('_')[0]) for f in files))
print(f'\nActors: {len(actors)} -> {min(actors)} to {max(actors)}')

# Label distribution via Excel
xl = pd.read_excel('data/raw/Emotional Body Motion Data/Human Evaluation result.xlsx',
                   sheet_name='Accuracy per motion')
print(f'\nLabel distribution:')
print(xl['True answer'].value_counts().sort_index())
print(f'\nAccuracy stats: mean={xl["Accuracy"].mean():.3f}, median={xl["Accuracy"].median():.3f}')

# Velocity scale check: compute frame-diff for one file
sample2 = pd.read_csv(data_dir / '1_1_1_1.csv')
hips = sample2[['Hips.x','Hips.y','Hips.z']].values
v = np.linalg.norm(np.diff(hips, axis=0), axis=1)
print(f'\nHips velocity in 1_1_1_1.csv: mean={v.mean():.5f}, max={v.max():.5f} (units/frame)')
