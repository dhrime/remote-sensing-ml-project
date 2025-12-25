# train_svm.py
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

FEATURES_FILE = "extracted_features.npz"
HOG_FEATURES_FILE = "hog_features.npz"
LBP_FEATURES_FILE = "lbp_features.npz"
COLOR_FEATURES_FILE = "color_features.npz"

# Subsample size for faster training
# 10,000 samples = good balance of speed and variability
SUBSAMPLE_SIZE = 10000

def get_subsample(X, y, size=SUBSAMPLE_SIZE):
    """Get stratified subsample for faster training."""
    if len(X) > size:
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=size, stratify=y, random_state=42
        )
        return X_sub, y_sub
    return X, y

def train_and_evaluate(X_train, y_train, X_test, y_test, pipeline):
    """Train pipeline and return accuracy and F1."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return acc, f1

def main():
    # ========== Load Features ==========
    print("Loading feature files...")
    
    if not all(os.path.exists(f) for f in [FEATURES_FILE, HOG_FEATURES_FILE, LBP_FEATURES_FILE, COLOR_FEATURES_FILE]):
        print("ERROR: Feature files not found. Run train_knn.py first to extract features.")
        return
    
    # Load all features
    data = np.load(FEATURES_FILE)
    X_train_all, y_train = data['X_train'], data['y_train']
    X_val_all, y_val = data['X_val'], data['y_val']
    X_test_all, y_test = data['X_test'], data['y_test']
    
    hog_data = np.load(HOG_FEATURES_FILE)
    X_train_hog, X_val_hog, X_test_hog = hog_data['X_train'], hog_data['X_val'], hog_data['X_test']
    
    lbp_data = np.load(LBP_FEATURES_FILE)
    X_train_lbp, X_val_lbp, X_test_lbp = lbp_data['X_train'], lbp_data['X_val'], lbp_data['X_test']
    
    color_data = np.load(COLOR_FEATURES_FILE)
    X_train_color, X_val_color, X_test_color = color_data['X_train'], color_data['X_val'], color_data['X_test']
    
    # Merge train+val
    X_train_val_all = np.concatenate([X_train_all, X_val_all])
    X_train_val_hog = np.concatenate([X_train_hog, X_val_hog])
    X_train_val_lbp = np.concatenate([X_train_lbp, X_val_lbp])
    X_train_val_color = np.concatenate([X_train_color, X_val_color])
    y_train_val = np.concatenate([y_train, y_val])
    
    print(f"All features: {X_train_val_all.shape}")
    print(f"HOG: {X_train_val_hog.shape}, LBP: {X_train_val_lbp.shape}, Color: {X_train_val_color.shape}")
    
    # Get subsamples for faster experiments
    X_sub, y_sub = get_subsample(X_train_val_all, y_train_val)
    print(f"\nUsing {len(X_sub)} samples for experiments (subsampled for speed)")
    
    # ========== Experiment 1: Varying C ==========
    # Controlled: gamma=0.001, PCA=90% variance, No Scaler
    print("\n" + "="*60)
    print("=== Experiment 1: Varying C (Regularization) ===")
    print("="*60)
    print("Controlled: gamma=0.001, PCA=90% variance, No Scaler")
    
    C_values = [1, 10, 100, 1000, 10000]
    accuracies_c = []
    f1_scores_c = []
    
    for C in C_values:
        print(f"Training with C={C}...")
        pipeline = Pipeline([
            ('pca', PCA(n_components=0.90)),
            ('svm', SVC(kernel='rbf', C=C, gamma=0.001))
        ])
        acc, f1 = train_and_evaluate(X_sub, y_sub, X_test_all, y_test, pipeline)
        accuracies_c.append(acc)
        f1_scores_c.append(f1)
        print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Plot Experiment 1
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(C_values)), accuracies_c, marker='o', label='Accuracy', linewidth=2)
    plt.plot(range(len(C_values)), f1_scores_c, marker='s', label='Macro F1', linewidth=2)
    plt.xlabel('C Value', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('SVM Experiment 1: Accuracy and Macro F1 vs C (gamma=0.001, PCA=90% var, No Scaler)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(C_values)), [str(c) for c in C_values])
    plt.tight_layout()
    plt.savefig('svm_exp1_varying_c.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: svm_exp1_varying_c.png")
    plt.close()
    
    # ========== Experiment 2: Varying Gamma ==========
    # Controlled: C=10000, PCA=90% variance, No Scaler
    print("\n" + "="*60)
    print("=== Experiment 2: Varying Gamma (Kernel Width) ===")
    print("="*60)
    print("Controlled: C=10000, PCA=90% variance, No Scaler")
    
    gamma_values = [0.0001, 0.001, 0.01, 0.1]
    accuracies_gamma = []
    f1_scores_gamma = []
    
    for gamma in gamma_values:
        print(f"Training with gamma={gamma}...")
        pipeline = Pipeline([
            ('pca', PCA(n_components=0.90)),
            ('svm', SVC(kernel='rbf', C=10000, gamma=gamma))
        ])
        acc, f1 = train_and_evaluate(X_sub, y_sub, X_test_all, y_test, pipeline)
        accuracies_gamma.append(acc)
        f1_scores_gamma.append(f1)
        print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Plot Experiment 2
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(gamma_values)), accuracies_gamma, marker='o', label='Accuracy', linewidth=2)
    plt.plot(range(len(gamma_values)), f1_scores_gamma, marker='s', label='Macro F1', linewidth=2)
    plt.xlabel('Gamma Value', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('SVM Experiment 2: Accuracy and Macro F1 vs Gamma (C=10000, PCA=90% var, No Scaler)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(gamma_values)), [str(g) for g in gamma_values])
    plt.tight_layout()
    plt.savefig('svm_exp2_varying_gamma.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: svm_exp2_varying_gamma.png")
    plt.close()
    
    # ========== Experiment 3: Varying PCA Components ==========
    # Controlled: C=10000, gamma=0.001, No Scaler
    print("\n" + "="*60)
    print("=== Experiment 3: Varying PCA Components ===")
    print("="*60)
    print("Controlled: C=10000, gamma=0.001, No Scaler")
    
    pca_values = [50, 100, 200, 300, 500, 800]
    accuracies_pca = []
    f1_scores_pca = []
    
    for pca_comp in pca_values:
        print(f"Training with PCA={pca_comp}...")
        pipeline = Pipeline([
            ('pca', PCA(n_components=pca_comp)),
            ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
        ])
        acc, f1 = train_and_evaluate(X_sub, y_sub, X_test_all, y_test, pipeline)
        accuracies_pca.append(acc)
        f1_scores_pca.append(f1)
        print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Plot Experiment 3
    plt.figure(figsize=(10, 6))
    plt.plot(pca_values, accuracies_pca, marker='o', label='Accuracy', linewidth=2)
    plt.plot(pca_values, f1_scores_pca, marker='s', label='Macro F1', linewidth=2)
    plt.xlabel('PCA Components', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('SVM Experiment 3: Accuracy and Macro F1 vs PCA Components (C=10000, gamma=0.001, No Scaler)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(pca_values)
    plt.tight_layout()
    plt.savefig('svm_exp3_varying_pca.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: svm_exp3_varying_pca.png")
    plt.close()
    
    # ========== Experiment 4: Feature Type Comparison ==========
    # Controlled: C=10000, gamma=0.001, PCA=90% variance, No Scaler
    print("\n" + "="*60)
    print("=== Experiment 4: Feature Type Comparison ===")
    print("="*60)
    print("Controlled: C=10000, gamma=0.001, PCA=90% variance, No Scaler")
    
    feature_results = {}
    
    # HOG
    print("Training with HOG features...")
    X_hog_sub, y_hog_sub = get_subsample(X_train_val_hog, y_train_val)
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.90)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_hog_sub, y_hog_sub, X_test_hog, y_test, pipeline)
    feature_results['HOG'] = {'acc': acc, 'f1': f1}
    print(f"  HOG - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # LBP (no PCA, only 18 features)
    print("Training with LBP features (no PCA, only 18 features)...")
    X_lbp_sub, y_lbp_sub = get_subsample(X_train_val_lbp, y_train_val)
    pipeline = Pipeline([
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_lbp_sub, y_lbp_sub, X_test_lbp, y_test, pipeline)
    feature_results['LBP'] = {'acc': acc, 'f1': f1}
    print(f"  LBP - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Color
    print("Training with Color features...")
    X_color_sub, y_color_sub = get_subsample(X_train_val_color, y_train_val)
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.90)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_color_sub, y_color_sub, X_test_color, y_test, pipeline)
    feature_results['Color'] = {'acc': acc, 'f1': f1}
    print(f"  Color - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # All Combined
    print("Training with All features...")
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.90)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_sub, y_sub, X_test_all, y_test, pipeline)
    feature_results['All'] = {'acc': acc, 'f1': f1}
    print(f"  All - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Plot Experiment 4
    feature_names = list(feature_results.keys())
    accs = [feature_results[f]['acc'] for f in feature_names]
    f1s = [feature_results[f]['f1'] for f in feature_names]
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, accs, width, label='Accuracy', color='steelblue', alpha=0.8)
    bars2 = plt.bar(x + width/2, f1s, width, label='Macro F1', color='coral', alpha=0.8)
    plt.xlabel('Feature Type', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('SVM Experiment 4: Feature Type Comparison (C=10000, gamma=0.001, PCA=90% var, No Scaler)', fontsize=14)
    plt.xticks(x, feature_names)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, max(max(accs), max(f1s)) + 0.1])
    plt.tight_layout()
    plt.savefig('svm_exp4_feature_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: svm_exp4_feature_comparison.png")
    plt.close()
    
    # ========== Experiment 5: Scaler Comparison ==========
    # Using controlled params: PCA=90% variance, C=10000, gamma=0.001
    print("\n" + "="*60)
    print("=== Experiment 5: Scaler Comparison ===")
    print("="*60)
    print("Controlled: PCA=90% variance, C=10000, gamma=0.001")
    
    scaler_results = {}
    
    # MinMaxScaler
    print("Training with MinMaxScaler...")
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('pca', PCA(n_components=0.90)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_sub, y_sub, X_test_all, y_test, pipeline)
    scaler_results['MinMaxScaler'] = {'acc': acc, 'f1': f1}
    print(f"  MinMaxScaler - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # StandardScaler
    print("Training with StandardScaler...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.90)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_sub, y_sub, X_test_all, y_test, pipeline)
    scaler_results['StandardScaler'] = {'acc': acc, 'f1': f1}
    print(f"  StandardScaler - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # No Scaler
    print("Training with No Scaler...")
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.90)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_sub, y_sub, X_test_all, y_test, pipeline)
    scaler_results['No Scaler'] = {'acc': acc, 'f1': f1}
    print(f"  No Scaler - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Plot Experiment 5
    scaler_names = list(scaler_results.keys())
    accs = [scaler_results[s]['acc'] for s in scaler_names]
    f1s = [scaler_results[s]['f1'] for s in scaler_names]
    
    x = np.arange(len(scaler_names))
    
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, accs, width, label='Accuracy', color='steelblue', alpha=0.8)
    bars2 = plt.bar(x + width/2, f1s, width, label='Macro F1', color='coral', alpha=0.8)
    plt.xlabel('Scaler Type', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('SVM Experiment 5: Scaler Comparison (C=10000, gamma=0.001, PCA=90% var)', fontsize=14)
    plt.xticks(x, scaler_names)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, max(max(accs), max(f1s)) + 0.1])
    plt.tight_layout()
    plt.savefig('svm_exp5_scaler_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: svm_exp5_scaler_comparison.png")
    plt.close()
    
    # ========== Experiment 6: Feature Combination Study ==========
    # Using controlled params: C=10000, gamma=0.001, PCA=90% variance, No Scaler
    print("\n" + "="*60)
    print("=== Experiment 6: Feature Combination Study ===")
    print("="*60)
    print("Testing pairwise feature combinations (C=10000, gamma=0.001, PCA=90% var, No Scaler)")
    
    combo_results = {}
    
    # Create feature combinations
    # HOG+LBP
    print("Training with HOG+LBP...")
    X_train_hog_lbp = np.concatenate([X_train_val_hog, X_train_val_lbp], axis=1)
    X_test_hog_lbp = np.concatenate([X_test_hog, X_test_lbp], axis=1)
    X_hog_lbp_sub, y_hog_lbp_sub = get_subsample(X_train_hog_lbp, y_train_val)
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.90)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_hog_lbp_sub, y_hog_lbp_sub, X_test_hog_lbp, y_test, pipeline)
    combo_results['HOG+LBP'] = {'acc': acc, 'f1': f1}
    print(f"  HOG+LBP ({X_train_hog_lbp.shape[1]} dims) - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # HOG+Color
    print("Training with HOG+Color...")
    X_train_hog_color = np.concatenate([X_train_val_hog, X_train_val_color], axis=1)
    X_test_hog_color = np.concatenate([X_test_hog, X_test_color], axis=1)
    X_hog_color_sub, y_hog_color_sub = get_subsample(X_train_hog_color, y_train_val)
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.90)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_hog_color_sub, y_hog_color_sub, X_test_hog_color, y_test, pipeline)
    combo_results['HOG+Color'] = {'acc': acc, 'f1': f1}
    print(f"  HOG+Color ({X_train_hog_color.shape[1]} dims) - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # LBP+Color
    print("Training with LBP+Color...")
    X_train_lbp_color = np.concatenate([X_train_val_lbp, X_train_val_color], axis=1)
    X_test_lbp_color = np.concatenate([X_test_lbp, X_test_color], axis=1)
    X_lbp_color_sub, y_lbp_color_sub = get_subsample(X_train_lbp_color, y_train_val)
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.90)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_lbp_color_sub, y_lbp_color_sub, X_test_lbp_color, y_test, pipeline)
    combo_results['LBP+Color'] = {'acc': acc, 'f1': f1}
    print(f"  LBP+Color ({X_train_lbp_color.shape[1]} dims) - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # All Combined
    print("Training with All Combined...")
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.90)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_sub, y_sub, X_test_all, y_test, pipeline)
    combo_results['All (HOG+LBP+Color)'] = {'acc': acc, 'f1': f1}
    print(f"  All Combined ({X_train_val_all.shape[1]} dims) - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Plot Experiment 6
    combo_names = list(combo_results.keys())
    accs = [combo_results[c]['acc'] for c in combo_names]
    f1s = [combo_results[c]['f1'] for c in combo_names]
    
    x = np.arange(len(combo_names))
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, accs, width, label='Accuracy', color='steelblue', alpha=0.8)
    bars2 = plt.bar(x + width/2, f1s, width, label='Macro F1', color='coral', alpha=0.8)
    plt.xlabel('Feature Combination', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('SVM Experiment 6: Feature Combination Study (C=10000, gamma=0.001, PCA=90% var, No Scaler)', fontsize=14)
    plt.xticks(x, combo_names, rotation=15, ha='right')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, max(max(accs), max(f1s)) + 0.1])
    plt.tight_layout()
    plt.savefig('svm_exp6_feature_combinations.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: svm_exp6_feature_combinations.png")
    plt.close()
    
    # ========== Experiment 7: Learning Curve ==========
    # Controlled: C=1000, gamma=0.001, No PCA, No Scaler
    print("\n" + "="*60)
    print("=== Experiment 7: Learning Curve (Training Size Effect) ===")
    print("="*60)
    print("Using All Combined features (C=1000, gamma=0.001, No PCA, No Scaler)")
    
    train_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    learning_accs = []
    learning_f1s = []
    sample_counts = []
    
    for frac in train_fractions:
        n_samples = int(len(X_train_val_all) * frac)
        sample_counts.append(n_samples)
        
        if frac < 1.0:
            X_frac, _, y_frac, _ = train_test_split(
                X_train_val_all, y_train_val, 
                train_size=n_samples, stratify=y_train_val, random_state=42
            )
        else:
            X_frac, y_frac = X_train_val_all, y_train_val
        
        print(f"Training with {frac*100:.0f}% data ({n_samples} samples)...")
        pipeline = Pipeline([
            ('svm', SVC(kernel='rbf', C=1000, gamma=0.001))
        ])
        acc, f1 = train_and_evaluate(X_frac, y_frac, X_test_all, y_test, pipeline)
        learning_accs.append(acc)
        learning_f1s.append(f1)
        print(f"  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Plot Experiment 7
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot([f*100 for f in train_fractions], learning_accs, 'o-', color='steelblue', 
             label='Accuracy', linewidth=2, markersize=8)
    ax1.plot([f*100 for f in train_fractions], learning_f1s, 's-', color='coral', 
             label='Macro F1', linewidth=2, markersize=8)
    ax1.set_xlabel('Training Data (%)', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('SVM Experiment 7: Learning Curve (C=1000, gamma=0.001, No PCA, No Scaler)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add sample count annotations
    for i, (frac, acc, count) in enumerate(zip(train_fractions, learning_accs, sample_counts)):
        ax1.annotate(f'{count}', (frac*100, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig('svm_exp7_learning_curve.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: svm_exp7_learning_curve.png")
    plt.close()
    
    # ========== Experiment 8: PCA Variance Retention ==========
    # Using All Combined features with controlled params: C=10000, gamma=0.001, No Scaler
    print("\n" + "="*60)
    print("=== Experiment 8: PCA Variance Retention ===")
    print("="*60)
    print("Using All Combined features (C=10000, gamma=0.001, No Scaler)")
    print("Comparing variance thresholds vs fixed components vs no PCA")
    
    variance_results = {}
    variance_thresholds = [0.80, 0.90, 0.95, 0.99]
    
    for var_thresh in variance_thresholds:
        print(f"Training with {var_thresh*100:.0f}% variance retained...")
        pipeline = Pipeline([
            ('pca', PCA(n_components=var_thresh)),  # Pass float for variance ratio
            ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
        ])
        pipeline.fit(X_sub, y_sub)
        n_components = pipeline.named_steps['pca'].n_components_
        y_pred = pipeline.predict(X_test_all)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        variance_results[f'{int(var_thresh*100)}% var'] = {'acc': acc, 'f1': f1, 'n_components': n_components}
        print(f"  {var_thresh*100:.0f}% variance → {n_components} components - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Add fixed PCA=300
    print(f"Training with fixed PCA=300...")
    pipeline = Pipeline([
        ('pca', PCA(n_components=300)),
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_sub, y_sub, X_test_all, y_test, pipeline)
    variance_results['Fixed 300'] = {'acc': acc, 'f1': f1, 'n_components': 300}
    print(f"  Fixed 300 components - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # No PCA (all components, but may be slow)
    print(f"Training with No PCA (all {X_sub.shape[1]} features)...")
    pipeline = Pipeline([
        ('svm', SVC(kernel='rbf', C=10000, gamma=0.001))
    ])
    acc, f1 = train_and_evaluate(X_sub, y_sub, X_test_all, y_test, pipeline)
    variance_results['No PCA'] = {'acc': acc, 'f1': f1, 'n_components': X_sub.shape[1]}
    print(f"  No PCA ({X_sub.shape[1]} features) - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    # Plot Experiment 8: Dual-axis chart
    var_names = list(variance_results.keys())
    accs = [variance_results[v]['acc'] for v in var_names]
    f1s = [variance_results[v]['f1'] for v in var_names]
    n_comps = [variance_results[v]['n_components'] for v in var_names]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(var_names))
    
    # Bar chart for accuracy and F1
    bars1 = ax1.bar(x - width/2, accs, width, label='Accuracy', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, f1s, width, label='Macro F1', color='coral', alpha=0.8)
    ax1.set_xlabel('PCA Configuration', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_ylim([0, max(max(accs), max(f1s)) + 0.15])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add component counts as annotations on bars
    for i, (bar, n_comp) in enumerate(zip(bars1, n_comps)):
        ax1.annotate(f'{n_comp} dims', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=9, color='darkblue')
    
    plt.title('SVM Experiment 8: PCA Variance Retention (C=10000, gamma=0.001, No Scaler)', fontsize=14)
    plt.xticks(x, var_names)
    plt.tight_layout()
    plt.savefig('svm_exp8_pca_variance.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: svm_exp8_pca_variance.png")
    plt.close()
    
    # ========== Summary ==========
    print("\n" + "="*60)
    print("=== ALL EXPERIMENTS COMPLETE ===")
    print("="*60)
    print("\nSaved plots:")
    print("  1. svm_exp1_varying_c.png")
    print("  2. svm_exp2_varying_gamma.png")
    print("  3. svm_exp3_varying_pca.png")
    print("  4. svm_exp4_feature_comparison.png")
    print("  5. svm_exp5_scaler_comparison.png")
    print("  6. svm_exp6_feature_combinations.png")
    print("  7. svm_exp7_learning_curve.png")
    print("  8. svm_exp8_pca_variance.png")

if __name__ == "__main__":
    main()
