import pandas as pd
from pathlib import Path

ebm = Path('data/raw/Emotional Body Motion Data')

# Read Excel
xl = pd.ExcelFile(ebm / 'Human Evaluation result.xlsx')
print('Sheets:', xl.sheet_names)
for sh in xl.sheet_names:
    df = xl.parse(sh)
    print(f'\n=== Sheet: {sh} ===')
    print(f'Shape: {df.shape}')
    print(f'Cols: {df.columns.tolist()}')
    print(df.head(15).to_string())

# Full columns of one CSV
data_dir = ebm / 'Emotional Body Motion Data'
files = sorted(data_dir.glob('*.csv'))
sample = pd.read_csv(files[0])
print(f'\n=== Full columns of {files[0].name} ===')
print(f'Shape: {sample.shape}')
print('All columns:')
print(sample.columns.tolist())

# Understand structure: actor_emotion_take_repetition
# First part: actor IDs
actors = sorted(set(f.stem.split('_')[0] for f in files))
p2 = sorted(set(f.stem.split('_')[1] for f in files))
p3 = sorted(set(f.stem.split('_')[2] for f in files))
p4 = sorted(set(f.stem.split('_')[3] for f in files))
print(f'\nPart 1 (actors?): {actors}')
print(f'Part 2: {p2}')
print(f'Part 3: {p3}')
print(f'Part 4: {p4}')
print(f'Total files: {len(files)}')
print(f'Expected if 29 actors x 4 x 5 x 7 = {29*4*5*7}')
