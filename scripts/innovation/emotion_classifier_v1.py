import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def train_emotion_classifier(data_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading features from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Preprocessing
    # Drop non-feature columns (assuming 'emotion' is the label)
    # Depending on previous scripts, we might have filename, image_id etc.
    drop_cols = ['filename', 'image_id', 'frame_idx', 'frame']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ['emotion'])
    y = df['emotion']
    
    # Handle NaNs (important for movement features where some joints might be missing)
    X = X.fillna(X.median())
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 4. Model: Random Forest (Best for interpretability of handcrafted features)
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. Evaluation
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\n--- Classification Report ---")
    print(report)
    
    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
        
    # 6. Feature Importance (The logical 'Research' part)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices][:15], y=X.columns[indices][:15])
    plt.title('Top 15 Most Discriminative Body Features')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(out_dir / 'feature_importance.png')
    
    # 7. Confusion Matrix (Where do emotions get confused?)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Emotion Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png')
    
    print(f"✅ Classifer experiments complete. Results in {out_dir}")

if __name__ == "__main__":
    # Note: Using the temporal features we generated earlier
    # You can also use the static features from v1-v5
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/analysis/temporal/v1/temporal_motion_features.csv")
    ap.add_argument("--out", default="outputs/experiments/classification_v1")
    args = ap.parse_args()
    
    train_emotion_classifier(args.csv, args.out)
