import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 10  # Reduced for faster execution

def create_fake_data(N, P, case_type):
    """
    Function to create fake data for different cases
    case_type: 'real_real', 'real_discrete', 'discrete_real', 'discrete_discrete'
    """
    if case_type == 'real_real':
        # Real input, real output
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))

    elif case_type == 'real_discrete':
        # Real input, discrete output
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(0, 5, size=N), dtype="category")

    elif case_type == 'discrete_real':
        # Discrete input, real output
        X = pd.DataFrame({i: pd.Series(np.random.randint(0, 5, size=N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randn(N))

    elif case_type == 'discrete_discrete':
        # Discrete input, discrete output
        X = pd.DataFrame({i: pd.Series(np.random.randint(0, 5, size=N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randint(0, 5, size=N), dtype="category")

    return X, y

def measure_time_complexity(N_values, P_values, case_type, max_depth=3):
    """
    Function to calculate average time taken by fit() and predict() for different N and P
    """
    fit_times_N = []
    predict_times_N = []
    fit_times_P = []
    predict_times_P = []

    # Vary N (keeping P constant)
    P_constant = 5
    for N in N_values:
        fit_times = []
        predict_times = []

        for _ in range(num_average_time):
            # Create data
            X_train, y_train = create_fake_data(N, P_constant, case_type)
            X_test, y_test = create_fake_data(min(N//4, 100), P_constant, case_type)  # Smaller test set

            # Measure fit time
            dt = DecisionTree(criterion='information_gain', max_depth=max_depth)
            start_time = time.time()
            dt.fit(X_train, y_train)
            fit_time = time.time() - start_time
            fit_times.append(fit_time)

            # Measure predict time
            start_time = time.time()
            dt.predict(X_test)
            predict_time = time.time() - start_time
            predict_times.append(predict_time)

        fit_times_N.append((np.mean(fit_times), np.std(fit_times)))
        predict_times_N.append((np.mean(predict_times), np.std(predict_times)))

    # Vary P (keeping N constant)
    N_constant = 200
    for P in P_values:
        fit_times = []
        predict_times = []

        for _ in range(num_average_time):
            # Create data
            X_train, y_train = create_fake_data(N_constant, P, case_type)
            X_test, y_test = create_fake_data(50, P, case_type)  # Smaller test set

            # Measure fit time
            dt = DecisionTree(criterion='information_gain', max_depth=max_depth)
            start_time = time.time()
            dt.fit(X_train, y_train)
            fit_time = time.time() - start_time
            fit_times.append(fit_time)

            # Measure predict time
            start_time = time.time()
            dt.predict(X_test)
            predict_time = time.time() - start_time
            predict_times.append(predict_time)

        fit_times_P.append((np.mean(fit_times), np.std(fit_times)))
        predict_times_P.append((np.mean(predict_times), np.std(predict_times)))

    return fit_times_N, predict_times_N, fit_times_P, predict_times_P

def plot_complexity_results(N_values, P_values, results_dict):
    """
    Function to plot the timing results
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    case_types = ['real_real', 'real_discrete', 'discrete_real', 'discrete_discrete']
    case_names = ['Real→Real', 'Real→Discrete', 'Discrete→Real', 'Discrete→Discrete']

    for i, (case_type, case_name) in enumerate(zip(case_types, case_names)):
        fit_times_N, predict_times_N, fit_times_P, predict_times_P = results_dict[case_type]

        # Plot fit time vs N
        ax = axes[0, i]
        means_N = [x[0] for x in fit_times_N]
        stds_N = [x[1] for x in fit_times_N]
        ax.errorbar(N_values, means_N, yerr=stds_N, marker='o', capsize=5)
        ax.set_xlabel('Number of Samples (N)')
        ax.set_ylabel('Fit Time (seconds)')
        ax.set_title(f'{case_name}\nFit Time vs N')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Plot predict time vs P
        ax = axes[1, i]
        means_P = [x[0] for x in predict_times_P]
        stds_P = [x[1] for x in predict_times_P]
        ax.errorbar(P_values, means_P, yerr=stds_P, marker='s', capsize=5, color='orange')
        ax.set_xlabel('Number of Features (P)')
        ax.set_ylabel('Predict Time (seconds)')
        ax.set_title(f'{case_name}\nPredict Time vs P')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def analyze_theoretical_complexity():
    """
    Function to analyze theoretical time complexity
    """
    print("="*80)
    print("THEORETICAL TIME COMPLEXITY ANALYSIS")
    print("="*80)

    print("\n1. DECISION TREE CONSTRUCTION (FIT):")
    print("   - Time Complexity: O(N * P * log(N) * D)")
    print("   - N: Number of samples")
    print("   - P: Number of features")
    print("   - D: Maximum depth")
    print("   - log(N): For sorting continuous features at each split")
    print("   - At each node, we evaluate P features and N samples")

    print("\n2. PREDICTION:")
    print("   - Time Complexity: O(D) per sample")
    print("   - D: Depth of the tree (worst case = max_depth)")
    print("   - For M test samples: O(M * D)")
    print("   - Independent of number of features P")

    print("\n3. EXPECTED OBSERVATIONS:")
    print("   - Fit time should increase roughly linearly with N and P")
    print("   - Predict time should be roughly constant w.r.t. P")
    print("   - Discrete features may be faster (no sorting needed)")
    print("   - Real outputs may be slightly slower (MSE calculation)")

# Run the experiments
print("Starting Runtime Complexity Experiments...")
print("This may take a few minutes...")

# Define parameter ranges
N_values = [50, 100, 200, 500, 1000]
P_values = [2, 5, 10, 20, 30]

case_types = ['real_real', 'real_discrete', 'discrete_real', 'discrete_discrete']
results_dict = {}

for case_type in case_types:
    print(f"\nRunning experiments for {case_type}...")
    results_dict[case_type] = measure_time_complexity(N_values, P_values, case_type)

# Plot results
plot_complexity_results(N_values, P_values, results_dict)

# Analyze theoretical complexity
analyze_theoretical_complexity()

# Additional analysis
print("\n" + "="*80)
print("EXPERIMENTAL OBSERVATIONS")
print("="*80)

for case_type in case_types:
    fit_times_N, predict_times_N, fit_times_P, predict_times_P = results_dict[case_type]

    print(f"\n{case_type.upper()}:")

    # Analyze scaling with N
    fit_means_N = [x[0] for x in fit_times_N]
    ratio_N = fit_means_N[-1] / fit_means_N[0] if fit_means_N[0] > 0 else 0
    expected_ratio_N = N_values[-1] / N_values[0]  # Linear scaling

    print(f"  Fit time scaling with N: {ratio_N:.2f}x (expected ~{expected_ratio_N:.2f}x for linear)")

    # Analyze scaling with P
    fit_means_P = [x[0] for x in fit_times_P]
    ratio_P = fit_means_P[-1] / fit_means_P[0] if fit_means_P[0] > 0 else 0
    expected_ratio_P = P_values[-1] / P_values[0]  # Linear scaling

    print(f"  Fit time scaling with P: {ratio_P:.2f}x (expected ~{expected_ratio_P:.2f}x for linear)")

print("\nExperiments completed!")

