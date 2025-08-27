import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

# Reading the data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv(url)

print("Original dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

print("\nDataset info:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

# Clean the above data by removing redundant columns and rows with junk values
print("\n" + "="*60)
print("Data Cleaning")
print("="*60)

# Remove car name as it's not useful for prediction
data = data.drop('name', axis=1)

# Handle missing values in horsepower (marked as '?')
print("Unique values in horsepower column:", data['horsepower'].unique()[:10])

# Replace '?' with NaN and convert to numeric
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

print(f"Missing values in horsepower after conversion: {data['horsepower'].isnull().sum()}")

# Fill missing horsepower values with median
data['horsepower'] = data['horsepower'].fillna(data['horsepower'].median())

data['origin'] = data['origin'].astype('category').cat.codes

print("Dataset shape after cleaning:", data.shape)
print("No missing values:", data.isnull().sum().sum() == 0)

# Prepare features and target
X = data.drop('mpg', axis=1)
y = data['mpg']

print(f"\nFeatures: {list(X.columns)}")
print(f"Target: mpg (fuel efficiency)")
print(f"Target statistics:")
print(y.describe())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# a) Show usage of our decision tree on automotive efficiency problem
print("\n" + "="*60)
print("a) Our Decision Tree Performance")
print("="*60)

# Train our decision tree
our_dt = DecisionTree(criterion='information_gain', max_depth=6)
our_dt.fit(X_train, y_train)

# Make predictions
y_pred_our = our_dt.predict(X_test)

# Calculate metrics
our_rmse = rmse(y_pred_our, y_test)
our_mae = mae(y_pred_our, y_test)

print(f"Our Decision Tree Results:")
print(f"RMSE: {our_rmse:.4f}")
print(f"MAE: {our_mae:.4f}")

# Show tree structure (first few levels)
print(f"\nOur Decision Tree Structure:")
our_dt.plot()

# b) Compare with scikit-learn decision tree
print("\n" + "="*60)
print("b) Comparison with Scikit-Learn Decision Tree")
print("="*60)

# Train scikit-learn decision tree
sklearn_dt = DecisionTreeRegressor(max_depth=6, random_state=42)
sklearn_dt.fit(X_train, y_train)

# Make predictions
y_pred_sklearn = sklearn_dt.predict(X_test)

# Calculate metrics
sklearn_rmse = rmse(pd.Series(y_pred_sklearn), y_test)
sklearn_mae = mae(pd.Series(y_pred_sklearn), y_test)

print(f"Scikit-Learn Decision Tree Results:")
print(f"RMSE: {sklearn_rmse:.4f}")
print(f"MAE: {sklearn_mae:.4f}")

# Comparison
print(f"\n--- Performance Comparison ---")
print(f"{'Metric':<10} {'Our DT':<12} {'Sklearn DT':<12} {'Difference':<12}")
print("-" * 50)
print(f"{'RMSE':<10} {our_rmse:<12.4f} {sklearn_rmse:<12.4f} {abs(our_rmse - sklearn_rmse):<12.4f}")
print(f"{'MAE':<10} {our_mae:<12.4f} {sklearn_mae:<12.4f} {abs(our_mae - sklearn_mae):<12.4f}")

# Feature importance comparison
print(f"\n--- Feature Importance (Scikit-Learn) ---")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': sklearn_dt.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Predictions vs Actual
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_our, alpha=0.6, label='Our DT')
plt.scatter(y_test, y_pred_sklearn, alpha=0.6, label='Sklearn DT')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Predictions vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 3, 2)
residuals_our = y_test - y_pred_our
residuals_sklearn = y_test - y_pred_sklearn
plt.scatter(y_pred_our, residuals_our, alpha=0.6, label='Our DT')
plt.scatter(y_pred_sklearn, residuals_sklearn, alpha=0.6, label='Sklearn DT')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Feature Importance
plt.subplot(1, 3, 3)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance (Sklearn)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Analysis Summary")
print("="*60)
print("1. Both decision trees perform reasonably well on the automotive efficiency problem.")
print("2. The scikit-learn implementation may have slight differences due to:")
print("   - Different splitting criteria implementation")
print("   - Different handling of ties in feature selection")
print("   - Optimizations in the sklearn implementation")
print("3. Key predictive features appear to be weight, displacement, and model year.")
print("4. Both models show similar patterns in residuals, indicating comparable performance.")