import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from tqdm import tqdm
import os

# Import your local feature extraction utility
# (Assumes utils.py and features.py are in the same folder)
from utils import build_features_from_hf_split

#Config files
DATASET_NAME = "timm/resisc45"
RESULTS_FILE = "rf_depth_estimators_experiments.csv"
HEATMAP_FILE = "experiment_heatmap.png"


def main():

    print(f"Loading dataset '{DATASET_NAME}' from Hugging Face...")
    dataset = load_dataset(DATASET_NAME)

    print("\n=== Starting Feature Extraction ===")

    print(f"Extracting features for TRAIN ({len(dataset['train'])} images)...")
    X_train, y_train = build_features_from_hf_split(dataset["train"])

    print(f"Extracting features for VALIDATION ({len(dataset['validation'])} images)...")
    X_val, y_val = build_features_from_hf_split(dataset["validation"])

    print(f"Extracting features for TEST ({len(dataset['test'])} images)...")
    X_test, y_test = build_features_from_hf_split(dataset["test"])

    print(f"\nFeature Extraction Complete!")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")


#Experiments
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 30, 50],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini']
    }

    grid = list(ParameterGrid(param_grid))
    results = []

    print(f"\n=== Starting Grid Search ({len(grid)} Experiments) ===")

    for params in tqdm(grid, desc="Training Models"):
        rf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            max_features=params['max_features'],
            criterion=params['criterion'],
            n_jobs=-1,  # Use all CPU cores
            random_state=42
        )

        # Train on Train set
        rf.fit(X_train, y_train)

        # Evaluate on Validation set
        y_pred = rf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')

        res = params.copy()
        res['Validation Accuracy'] = acc
        res['Validation Macro F1'] = f1
        results.append(res)

#Saves the results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved experiment results to {RESULTS_FILE}")

    # Print Top 5
    print("\nTop 5 Configurations:")
    print(df_results.sort_values(by='Validation Accuracy', ascending=False).head(5)[
              ['n_estimators', 'max_depth', 'Validation Accuracy']])

#Heatmap generation
    try:
        heatmap_df = df_results[df_results['max_features'] == 'sqrt']
        pivot_table = heatmap_df.pivot(index='max_depth', columns='n_estimators', values='Validation Accuracy')

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Validation Accuracy (max_features='sqrt')")
        plt.ylabel("Max Depth")
        plt.xlabel("Number of Estimators")
        plt.tight_layout()
        plt.savefig(HEATMAP_FILE)
        print(f"Saved heatmap to {HEATMAP_FILE}")
    except Exception as e:
        print(f"Could not generate heatmap: {e}")

#Retrain the best model
    print("\n=== Final Evaluation ===")
    print("Retraining Best Model on combined Train + Validation data...")

    best_params = df_results.sort_values(by='Validation Accuracy', ascending=False).iloc[0]

    # Merge Train and Val for the final robust model
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    best_rf = RandomForestClassifier(
        n_estimators=int(best_params['n_estimators']),
        max_depth=None if pd.isna(best_params['max_depth']) else int(best_params['max_depth']),
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )

    best_rf.fit(X_train_val, y_train_val)

    test_acc = accuracy_score(y_test, best_rf.predict(X_test))
    print(f" FINAL TEST ACCURACY: {test_acc:.4f}")

if __name__ == "__main__":
    main()