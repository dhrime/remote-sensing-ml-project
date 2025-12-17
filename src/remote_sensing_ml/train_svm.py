# train_svm.py
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

# Import feature extraction functions from utils.py
from .utils import build_features_from_hf_split

FEATURES_FILE = "extracted_features.npz"

def main():

    # 1. Load or Extract Features
    if os.path.exists(FEATURES_FILE):
        print(f"Loading saved features from {FEATURES_FILE}...")
        features_data = np.load(FEATURES_FILE)
        X_train = features_data['X_train']
        y_train = features_data['y_train']
        X_val = features_data['X_val']
        y_val = features_data['y_val']
        X_test = features_data['X_test']
        y_test = features_data['y_test']
        print("Features loaded successfully!")
    else:
        print("Features not found. Extracting features from dataset...")
        # Import load_dataset only when needed
        from datasets import load_dataset
        # Load Dataset
        print("Loading dataset...")
        dataset = load_dataset("timm/resisc45")
        
        # Extract Features
        print(f"Extracting features for Train ({len(dataset['train'])})...")
        X_train, y_train = build_features_from_hf_split(dataset["train"])

        print(f"Extracting features for Val ({len(dataset['validation'])})...")
        X_val, y_val = build_features_from_hf_split(dataset["validation"])

        print(f"Extracting features for Test ({len(dataset['test'])})...")
        X_test, y_test = build_features_from_hf_split(dataset["test"])
        
        # Save features for future use
        print(f"Saving features to {FEATURES_FILE}...")
        np.savez_compressed(
            FEATURES_FILE,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test
        )
        print("Features saved successfully!")

    # 2. Prepare Data: Merge Train and Val for training, Test for evaluation
    print("\nMerging Train and Val for training...")
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    print(f"Training Set Shape: {X_train_val.shape}")
    print(f"Test Set Shape: {X_test.shape}")

    # 3. Define the Pipeline with SVM
    n_features = X_train_val.shape[1]
    print(f"\nFeature dimensions: {n_features}")
    
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Will be replaced by grid search
        ('pca', PCA()),              # n_components set by grid search
        ('svm', SVC(kernel='rbf'))
    ])

    # 4. Define Grid Search Space for C, gamma, PCA components, and scaler
    # PCA component options (capped at n_features)
    pca_options = [c for c in [20, 50, 100] if c <= n_features]
    
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler()],
        'pca__n_components': pca_options,
        'svm__C': [1, 10, 100],
        'svm__gamma': ['auto', 'scale']
    }
    
    total_combinations = (len(param_grid['scaler']) * 
                         len(param_grid['pca__n_components']) * 
                         len(param_grid['svm__C']) * 
                         len(param_grid['svm__gamma']))

    print("\n=== Starting Grid Search for SVM ===")
    print("Searching over scaler, PCA components, C, and gamma...")
    print(f"Parameter grid:")
    print(f"  Scalers: MinMaxScaler, StandardScaler")
    print(f"  PCA components: {pca_options}")
    print(f"  C: {param_grid['svm__C']}")
    print(f"  gamma: {param_grid['svm__gamma']}")
    print(f"Total combinations: {total_combinations} (x3 CV folds = {total_combinations * 3} fits)")
    print("This may take a while...")

    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=2,
        scoring='accuracy'
    )

    grid_search.fit(X_train_val, y_train_val)

    # 5. Report Best Settings
    print(f"\n=== Grid Search Results ===")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

    # 6. Final Evaluation on Independent Test Set
    print("\n=== Evaluating Best Model on Test Set ===")
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    
    print(f"\n=== Final Test Results ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1:.4f}")
    print(f"\nBest Model Parameters:")
    print(f"  Scaler: {type(grid_search.best_params_['scaler']).__name__}")
    print(f"  PCA components: {grid_search.best_params_['pca__n_components']}")
    print(f"  C: {grid_search.best_params_['svm__C']}")
    print(f"  gamma: {grid_search.best_params_['svm__gamma']}")

if __name__ == "__main__":
    main()

