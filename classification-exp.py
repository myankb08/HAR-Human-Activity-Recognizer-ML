import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, cross_val_score

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Convert to pandas for our implementation
X_df = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
y_series = pd.Series(y, dtype='category')

# For plotting
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('Generated Classification Dataset')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.colorbar()
plt.show()

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Class distribution:", np.bincount(y))

# Q2 a) Train-Test Split (70-30) and Performance Evaluation
print("\n" + "="*60)
print("Q2 a) Train-Test Split Analysis")
print("="*60)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.3, random_state=42, stratify=y_series
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train decision tree with both criteria
for criterion in ['information_gain', 'gini_index']:
    print(f"\n--- Results for {criterion} ---")

    # Train the model
    dt = DecisionTree(criterion=criterion, max_depth=5)
    dt.fit(X_train, y_train)

    # Make predictions
    y_pred = dt.predict(X_test)

    # Calculate metrics
    acc = accuracy(y_pred, y_test)
    print(f"Accuracy: {acc:.4f}")

    # Per-class precision and recall
    unique_classes = y_test.unique()
    for cls in unique_classes:
        prec = precision(y_pred, y_test, cls)
        rec = recall(y_pred, y_test, cls)
        print(f"Class {cls} - Precision: {prec:.4f}, Recall: {rec:.4f}")

    # Plot decision tree structure
    print(f"\nDecision Tree Structure ({criterion}):")
    dt.plot()

# Q2 b) 5-Fold Cross-Validation with Nested CV for Optimal Depth
print("\n" + "="*60)
print("Q2 b) 5-Fold Cross-Validation with Nested CV for Optimal Depth")
print("="*60)

def evaluate_depth_cv(X, y, max_depth, criterion='information_gain', cv_folds=5):
    """
    Evaluate a specific depth using cross-validation
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # Train model
        dt = DecisionTree(criterion=criterion, max_depth=max_depth)
        dt.fit(X_train_fold, y_train_fold)

        # Predict and calculate accuracy
        y_pred_fold = dt.predict(X_val_fold)
        acc = accuracy(y_pred_fold, y_val_fold)
        scores.append(acc)

    return np.mean(scores), np.std(scores)

# Test different depths
depths_to_test = range(1, 11)
results = {}

for criterion in ['information_gain', 'gini_index']:
    print(f"\n--- Nested Cross-Validation for {criterion} ---")
    results[criterion] = {}

    best_depth = 1
    best_score = 0

    for depth in depths_to_test:
        mean_score, std_score = evaluate_depth_cv(X_df, y_series, depth, criterion)
        results[criterion][depth] = (mean_score, std_score)

        print(f"Depth {depth}: {mean_score:.4f} ± {std_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth

    print(f"\nBest depth for {criterion}: {best_depth} (Score: {best_score:.4f})")

# Plot results
plt.figure(figsize=(12, 5))

for i, criterion in enumerate(['information_gain', 'gini_index']):
    plt.subplot(1, 2, i+1)

    depths = list(results[criterion].keys())
    means = [results[criterion][d][0] for d in depths]
    stds = [results[criterion][d][1] for d in depths]

    plt.errorbar(depths, means, yerr=stds, marker='o', capsize=5)
    plt.xlabel('Max Depth')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title(f'CV Performance vs Depth ({criterion})')
    plt.grid(True, alpha=0.3)
    plt.xticks(depths)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Summary")
print("="*60)
print("The nested cross-validation helps us find the optimal depth that generalizes well.")
print("This prevents overfitting that might occur if we just used the training accuracy.")
print("The error bars show the variability across different CV folds.")

