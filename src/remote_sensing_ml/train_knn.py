# train_knn.py
import numpy as np
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

# Import feature extraction functions from utils.py
from .utils import build_features_from_hf_split, extract_hog_from_split, extract_lbp_from_split, extract_color_from_split

FEATURES_FILE = "extracted_features.npz"
HOG_FEATURES_FILE = "hog_features.npz"
LBP_FEATURES_FILE = "lbp_features.npz"
COLOR_FEATURES_FILE = "color_features.npz"

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

    # 2. Extract and save individual feature types
    # Check if we need to load dataset for individual feature extraction
    need_dataset = not (os.path.exists(HOG_FEATURES_FILE) and 
                       os.path.exists(LBP_FEATURES_FILE) and 
                       os.path.exists(COLOR_FEATURES_FILE))
    
    if need_dataset:
        print("Loading dataset for individual feature extraction...")
        dataset = load_dataset("timm/resisc45")
    
    # HOG features
    if os.path.exists(HOG_FEATURES_FILE):
        print(f"Loading saved HOG features from {HOG_FEATURES_FILE}...")
        hog_data = np.load(HOG_FEATURES_FILE)
        X_train_hog, y_train_hog = hog_data['X_train'], hog_data['y_train']
        X_val_hog, y_val_hog = hog_data['X_val'], hog_data['y_val']
        X_test_hog, y_test_hog = hog_data['X_test'], hog_data['y_test']
        print("HOG features loaded successfully!")
    else:
        print("Extracting HOG features...")
        print(f"  Train ({len(dataset['train'])})...")
        X_train_hog, y_train_hog = extract_hog_from_split(dataset["train"])
        print(f"  Val ({len(dataset['validation'])})...")
        X_val_hog, y_val_hog = extract_hog_from_split(dataset["validation"])
        print(f"  Test ({len(dataset['test'])})...")
        X_test_hog, y_test_hog = extract_hog_from_split(dataset["test"])
        np.savez_compressed(HOG_FEATURES_FILE,
                           X_train=X_train_hog, y_train=y_train_hog,
                           X_val=X_val_hog, y_val=y_val_hog,
                           X_test=X_test_hog, y_test=y_test_hog)
        print(f"HOG features saved to {HOG_FEATURES_FILE}")
    
    # LBP features
    if os.path.exists(LBP_FEATURES_FILE):
        print(f"Loading saved LBP features from {LBP_FEATURES_FILE}...")
        lbp_data = np.load(LBP_FEATURES_FILE)
        X_train_lbp, y_train_lbp = lbp_data['X_train'], lbp_data['y_train']
        X_val_lbp, y_val_lbp = lbp_data['X_val'], lbp_data['y_val']
        X_test_lbp, y_test_lbp = lbp_data['X_test'], lbp_data['y_test']
        print("LBP features loaded successfully!")
    else:
        print("Extracting LBP features...")
        print(f"  Train ({len(dataset['train'])})...")
        X_train_lbp, y_train_lbp = extract_lbp_from_split(dataset["train"])
        print(f"  Val ({len(dataset['validation'])})...")
        X_val_lbp, y_val_lbp = extract_lbp_from_split(dataset["validation"])
        print(f"  Test ({len(dataset['test'])})...")
        X_test_lbp, y_test_lbp = extract_lbp_from_split(dataset["test"])
        np.savez_compressed(LBP_FEATURES_FILE,
                           X_train=X_train_lbp, y_train=y_train_lbp,
                           X_val=X_val_lbp, y_val=y_val_lbp,
                           X_test=X_test_lbp, y_test=y_test_lbp)
        print(f"LBP features saved to {LBP_FEATURES_FILE}")
    
    # Color histogram features
    if os.path.exists(COLOR_FEATURES_FILE):
        print(f"Loading saved Color features from {COLOR_FEATURES_FILE}...")
        color_data = np.load(COLOR_FEATURES_FILE)
        X_train_color, y_train_color = color_data['X_train'], color_data['y_train']
        X_val_color, y_val_color = color_data['X_val'], color_data['y_val']
        X_test_color, y_test_color = color_data['X_test'], color_data['y_test']
        print("Color features loaded successfully!")
    else:
        print("Extracting Color histogram features...")
        print(f"  Train ({len(dataset['train'])})...")
        X_train_color, y_train_color = extract_color_from_split(dataset["train"])
        print(f"  Val ({len(dataset['validation'])})...")
        X_val_color, y_val_color = extract_color_from_split(dataset["validation"])
        print(f"  Test ({len(dataset['test'])})...")
        X_test_color, y_test_color = extract_color_from_split(dataset["test"])
        np.savez_compressed(COLOR_FEATURES_FILE,
                           X_train=X_train_color, y_train=y_train_color,
                           X_val=X_val_color, y_val=y_val_color,
                           X_test=X_test_color, y_test=y_test_color)
        print(f"Color features saved to {COLOR_FEATURES_FILE}")

    # 3. Prepare Data: Merge Train and Val for training, Test for evaluation
    print("Merging Train and Val for training...")
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    print(f"Training Set Shape: {X_train_val.shape}")
    print(f"Test Set Shape: {X_test.shape}")

    # Experiment 1: Fix PCA=50, vary KNN neighbors [1,3,9,15,21]
    print("\n=== Experiment 1: Varying KNN Neighbors (PCA=20) ===")
    pca_components = 20
    neighbor_counts = [1, 3, 9, 15, 21]
    
    accuracies_exp1 = []
    f1_scores_exp1 = []
    
    for n_neighbors in neighbor_counts:
        print(f"\nTraining with n_neighbors={n_neighbors}, PCA={pca_components}...")
        
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pca', PCA(n_components=pca_components)),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean'))
        ])
        
        pipeline.fit(X_train_val, y_train_val)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        accuracies_exp1.append(acc)
        f1_scores_exp1.append(f1)
        
        print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Plot Experiment 1
    plt.figure(figsize=(10, 6))
    plt.plot(neighbor_counts, accuracies_exp1, marker='o', label='Accuracy', linewidth=2)
    plt.plot(neighbor_counts, f1_scores_exp1, marker='s', label='Macro F1', linewidth=2)
    plt.xlabel('KNN Neighbor Count', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Experiment 1: Accuracy and Macro F1 vs KNN Neighbor Count (PCA=20)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(neighbor_counts)
    plt.tight_layout()
    plt.savefig('experiment1_knn_neighbors.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: experiment1_knn_neighbors.png")
    plt.close()

    # Experiment 2: Fix KNN neighbors=9, vary PCA components [10,20,50,100,200]
    print("\n=== Experiment 2: Varying PCA Components (KNN neighbors=9) ===")
    n_neighbors = 9
    pca_component_counts = [10, 20, 50, 100, 200]
    
    accuracies_exp2 = []
    f1_scores_exp2 = []
    
    for n_components in pca_component_counts:
        print(f"\nTraining with n_neighbors={n_neighbors}, PCA={n_components}...")
        
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pca', PCA(n_components=n_components)),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean'))
        ])
        
        pipeline.fit(X_train_val, y_train_val)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        accuracies_exp2.append(acc)
        f1_scores_exp2.append(f1)
        
        print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Plot Experiment 2
    plt.figure(figsize=(10, 6))
    plt.plot(pca_component_counts, accuracies_exp2, marker='o', label='Accuracy', linewidth=2)
    plt.plot(pca_component_counts, f1_scores_exp2, marker='s', label='Macro F1', linewidth=2)
    plt.xlabel('PCA Component Count', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Experiment 2: Accuracy and Macro F1 vs PCA Component Count (KNN neighbors=9)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(pca_component_counts)
    plt.tight_layout()
    plt.savefig('experiment2_pca_components.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: experiment2_pca_components.png")
    plt.close()
    
    # Experiment 3: HOG features only, vary KNN neighbors [1,3,9,15,21] (PCA=20)
    print("\n=== Experiment 3: HOG Features Only - Varying KNN Neighbors (PCA=20) ===")
    pca_components = 20
    neighbor_counts = [1, 3, 9, 15, 21]
    
    X_train_val_hog = np.concatenate([X_train_hog, X_val_hog])
    y_train_val_hog = np.concatenate([y_train_hog, y_val_hog])
    
    # Adjust PCA components to not exceed number of features
    n_features_hog = X_train_val_hog.shape[1]
    pca_comp_hog = min(pca_components, n_features_hog)
    
    print(f"HOG feature dimensions: {n_features_hog}, Using PCA components: {pca_comp_hog}")
    
    accuracies_exp3 = []
    f1_scores_exp3 = []
    
    for n_neighbors in neighbor_counts:
        print(f"\nTraining with n_neighbors={n_neighbors}, PCA={pca_comp_hog}...")
        
        pipeline_hog = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pca', PCA(n_components=pca_comp_hog)),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean'))
        ])
        
        pipeline_hog.fit(X_train_val_hog, y_train_val_hog)
        y_pred_hog = pipeline_hog.predict(X_test_hog)
        
        acc_hog = accuracy_score(y_test_hog, y_pred_hog)
        f1_hog = f1_score(y_test_hog, y_pred_hog, average='macro', zero_division=0)
        
        accuracies_exp3.append(acc_hog)
        f1_scores_exp3.append(f1_hog)
        
        print(f"  Accuracy: {acc_hog:.4f}, Macro F1: {f1_hog:.4f}")
    
    # Plot Experiment 3
    plt.figure(figsize=(10, 6))
    plt.plot(neighbor_counts, accuracies_exp3, marker='o', label='Accuracy', linewidth=2)
    plt.plot(neighbor_counts, f1_scores_exp3, marker='s', label='Macro F1', linewidth=2)
    plt.xlabel('KNN Neighbor Count', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Experiment 3: HOG Features Only - Accuracy and Macro F1 vs KNN Neighbor Count (PCA={pca_comp_hog})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(neighbor_counts)
    plt.tight_layout()
    plt.savefig('experiment3_hog_only.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: experiment3_hog_only.png")
    plt.close()
    
    # Experiment 4: LBP features only, vary KNN neighbors [1,3,9,15,21] (PCA=20)
    print("\n=== Experiment 4: LBP Features Only - Varying KNN Neighbors (PCA=20) ===")
    
    X_train_val_lbp = np.concatenate([X_train_lbp, X_val_lbp])
    y_train_val_lbp = np.concatenate([y_train_lbp, y_val_lbp])
    
    # Adjust PCA components to not exceed number of features
    n_features_lbp = X_train_val_lbp.shape[1]
    pca_comp_lbp = min(pca_components, n_features_lbp)
    
    print(f"LBP feature dimensions: {n_features_lbp}, Using PCA components: {pca_comp_lbp}")
    
    accuracies_exp4 = []
    f1_scores_exp4 = []
    
    for n_neighbors in neighbor_counts:
        print(f"\nTraining with n_neighbors={n_neighbors}, PCA={pca_comp_lbp}...")
        
        pipeline_lbp = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pca', PCA(n_components=pca_comp_lbp)),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean'))
        ])
        
        pipeline_lbp.fit(X_train_val_lbp, y_train_val_lbp)
        y_pred_lbp = pipeline_lbp.predict(X_test_lbp)
        
        acc_lbp = accuracy_score(y_test_lbp, y_pred_lbp)
        f1_lbp = f1_score(y_test_lbp, y_pred_lbp, average='macro', zero_division=0)
        
        accuracies_exp4.append(acc_lbp)
        f1_scores_exp4.append(f1_lbp)
        
        print(f"  Accuracy: {acc_lbp:.4f}, Macro F1: {f1_lbp:.4f}")
    
    # Plot Experiment 4
    plt.figure(figsize=(10, 6))
    plt.plot(neighbor_counts, accuracies_exp4, marker='o', label='Accuracy', linewidth=2)
    plt.plot(neighbor_counts, f1_scores_exp4, marker='s', label='Macro F1', linewidth=2)
    plt.xlabel('KNN Neighbor Count', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Experiment 4: LBP Features Only - Accuracy and Macro F1 vs KNN Neighbor Count (PCA={pca_comp_lbp})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(neighbor_counts)
    plt.tight_layout()
    plt.savefig('experiment4_lbp_only.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: experiment4_lbp_only.png")
    plt.close()
    
    # Experiment 5: Color histogram features only, vary KNN neighbors [1,3,9,15,21] (PCA=20)
    print("\n=== Experiment 5: Color Histogram Features Only - Varying KNN Neighbors (PCA=20) ===")
    
    X_train_val_color = np.concatenate([X_train_color, X_val_color])
    y_train_val_color = np.concatenate([y_train_color, y_val_color])
    
    # Adjust PCA components to not exceed number of features
    n_features_color = X_train_val_color.shape[1]
    pca_comp_color = min(pca_components, n_features_color)
    
    print(f"Color feature dimensions: {n_features_color}, Using PCA components: {pca_comp_color}")
    
    accuracies_exp5 = []
    f1_scores_exp5 = []
    
    for n_neighbors in neighbor_counts:
        print(f"\nTraining with n_neighbors={n_neighbors}, PCA={pca_comp_color}...")
        
        pipeline_color = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pca', PCA(n_components=pca_comp_color)),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean'))
        ])
        
        pipeline_color.fit(X_train_val_color, y_train_val_color)
        y_pred_color = pipeline_color.predict(X_test_color)
        
        acc_color = accuracy_score(y_test_color, y_pred_color)
        f1_color = f1_score(y_test_color, y_pred_color, average='macro', zero_division=0)
        
        accuracies_exp5.append(acc_color)
        f1_scores_exp5.append(f1_color)
        
        print(f"  Accuracy: {acc_color:.4f}, Macro F1: {f1_color:.4f}")
    
    # Plot Experiment 5
    plt.figure(figsize=(10, 6))
    plt.plot(neighbor_counts, accuracies_exp5, marker='o', label='Accuracy', linewidth=2)
    plt.plot(neighbor_counts, f1_scores_exp5, marker='s', label='Macro F1', linewidth=2)
    plt.xlabel('KNN Neighbor Count', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Experiment 5: Color Histogram Features Only - Accuracy and Macro F1 vs KNN Neighbor Count (PCA={pca_comp_color})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(neighbor_counts)
    plt.tight_layout()
    plt.savefig('experiment5_color_only.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: experiment5_color_only.png")
    plt.close()
    
    # Experiment 6: All features combined, no PCA, vary KNN neighbors [1,3,9,15,21]
    print("\n=== Experiment 6: All Features Combined - Varying KNN Neighbors (No PCA) ===")
    
    # X_train_val and y_train_val are already prepared from combined features
    print(f"Using all combined features (shape: {X_train_val.shape})")
    print("No PCA - using all feature components")
    
    accuracies_exp6 = []
    f1_scores_exp6 = []
    
    for n_neighbors in neighbor_counts:
        print(f"\nTraining with n_neighbors={n_neighbors} (no PCA)...")
        
        pipeline_all = Pipeline([
            ('scaler', MinMaxScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean'))
        ])
        
        pipeline_all.fit(X_train_val, y_train_val)
        y_pred_all = pipeline_all.predict(X_test)
        
        acc_all = accuracy_score(y_test, y_pred_all)
        f1_all = f1_score(y_test, y_pred_all, average='macro', zero_division=0)
        
        accuracies_exp6.append(acc_all)
        f1_scores_exp6.append(f1_all)
        
        print(f"  Accuracy: {acc_all:.4f}, Macro F1: {f1_all:.4f}")
    
    # Plot Experiment 6
    plt.figure(figsize=(10, 6))
    plt.plot(neighbor_counts, accuracies_exp6, marker='o', label='Accuracy', linewidth=2)
    plt.plot(neighbor_counts, f1_scores_exp6, marker='s', label='Macro F1', linewidth=2)
    plt.xlabel('KNN Neighbor Count', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Experiment 6: All Features Combined - Accuracy and Macro F1 vs KNN Neighbor Count (No PCA)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(neighbor_counts)
    plt.tight_layout()
    plt.savefig('experiment6_all_features.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: experiment6_all_features.png")
    plt.close()
    
    print("\n=== Experiments Complete ===")

if __name__ == "__main__":
    main()