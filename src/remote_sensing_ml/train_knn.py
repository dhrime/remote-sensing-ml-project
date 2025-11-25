# train_knn.py
import numpy as np
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Ensure this imports the updated features.py above
from .utils import build_features_from_hf_split

def main():

    # 1. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset("timm/resisc45")
    

    # 2. Extract Features
    print(f"Extracting features for Train ({len(dataset['train'])})...")
    X_train, y_train = build_features_from_hf_split(dataset["train"])

    print(f"Extracting features for Val ({len(dataset['validation'])})...")
    X_val, y_val = build_features_from_hf_split(dataset["validation"])

    print(f"Extracting features for Test ({len(dataset['test'])})...")
    X_test, y_test = build_features_from_hf_split(dataset["test"])

    # 3. Prepare Data for Grid Search
    print("Merging Train and Val for Grid Search optimization...")
    X_dev = np.concatenate([X_train, X_val])
    y_dev = np.concatenate([y_train, y_val])

    print(f"Development Set Shape: {X_dev.shape}")

    # 4. Define the Pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('pca', PCA()),
        ('knn', KNeighborsClassifier())
    ])

    # 5. Define Grid Search Space
    param_grid = {
       
        'pca__n_components': [50, 100, 200], 
        
        'knn__n_neighbors': [9, 15, 21],
        
        'knn__metric': ['manhattan', 'euclidean'],
        
        'knn__weights': ['distance'] 
    }

    print("\n=== Starting Grid Search ===")
    print("Optimizing PCA and K... (This will take a few minutes)")
    

    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=2,
        scoring='accuracy'
    )

    grid_search.fit(X_dev, y_dev)

    # 6. Report Best Settings
    print(f"\nBest Parameters Found: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

    # 7. Final Evaluation on Independent Test Set
    print("\nEvaluating best model on Test set...")
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='macro')
    
    print("\n=== Final Test Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1: .4f}")
    
    # Detailed report per class 
    # print(classification_report(y_test, y_test_pred, target_names=label_names))

if __name__ == "__main__":
    main()