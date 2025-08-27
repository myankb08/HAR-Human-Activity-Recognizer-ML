"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


class TreeNode:
    """
    Class to represent a node in the decision tree
    """
    def __init__(self):
        self.feature = None  # Feature to split on
        self.split_value = None  # Split value for continuous features
        self.children = {}  # Dictionary of child nodes
        self.prediction = None  # Prediction for leaf nodes
        self.is_leaf = False
        self.depth = 0


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.feature_names = None

    def _should_stop_splitting(self, X: pd.DataFrame, y: pd.Series, depth: int) -> bool:
        """
        Check if we should stop splitting based on various conditions
        """
        # Stop if max depth reached
        if depth >= self.max_depth:
            return True

        # Stop if all samples have same target value
        if len(y.unique()) == 1:
            return True

        # Stop if too few samples
        if len(y) < 2:
            return True

        # Stop if no features left
        if X.shape[1] == 0:
            return True

        return False

    def _get_leaf_prediction(self, y: pd.Series) -> Any:
        """
        Get prediction for a leaf node
        """
        if check_ifreal(y):
            # For regression, return mean
            return y.mean()
        else:
            # For classification, return mode (most frequent class)
            return y.mode().iloc[0] if len(y.mode()) > 0 else y.iloc[0]

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> TreeNode:
        """
        Recursively build the decision tree
        """
        node = TreeNode()
        node.depth = depth

        # Check stopping conditions
        if self._should_stop_splitting(X, y, depth):
            node.is_leaf = True
            node.prediction = self._get_leaf_prediction(y)
            return node

        # Find best split
        features = pd.Series(X.columns)
        best_feature, best_split_value = opt_split_attribute(X, y, self.criterion, features)

        if best_feature is None:
            node.is_leaf = True
            node.prediction = self._get_leaf_prediction(y)
            return node

        node.feature = best_feature
        node.split_value = best_split_value

        # Split data and create child nodes
        splits = split_data(X, y, best_feature, best_split_value)
                
        (X_left, y_left), (X_right, y_right) = splits
        if len(y_left) > 0:
            node.children['left'] = self._build_tree(X_left, y_left, depth + 1)
        if len(y_right) > 0:
            node.children['right'] = self._build_tree(X_right, y_right, depth + 1)


        # If no valid children created, make this a leaf
        if not node.children:
            node.is_leaf = True
            node.prediction = self._get_leaf_prediction(y)

        return node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # Store feature names for later use
        self.feature_names = X.columns.tolist()

        # Handle discrete input by converting to one-hot encoding if needed
        # Check if any column is categorical
        #categorical_cols = X.select_dtypes(include=['category', 'object']).columns
        #if len(categorical_cols) > 0:
        #    X = one_hot_encoding(X)

        # Build the tree
        self.root = self._build_tree(X, y)

    def _predict_single(self, x: pd.Series, node: TreeNode) -> Any:
        """
        Predict for a single sample
        """
        if node.is_leaf:
            return node.prediction

        feature_value = x[node.feature]

        if node.split_value is not None:
            # Continuous feature
            if feature_value <= node.split_value:
                if 'left' in node.children:
                    return self._predict_single(x, node.children['left'])
            else:
                if 'right' in node.children:
                    return self._predict_single(x, node.children['right'])
        else:
            # Discrete feature
            if feature_value in node.children:
                return self._predict_single(x, node.children[feature_value])

        # If no matching child found, return the most common prediction from this node's subtree
        return node.prediction if hasattr(node, 'prediction') and node.prediction is not None else 0

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")

        # Handle discrete input by converting to one-hot encoding if needed
        #categorical_cols = X.select_dtypes(include=['category', 'object']).columns
        #if len(categorical_cols) > 0:
        #    X = one_hot_encoding(X)

        # Make predictions for each sample
        predictions = []
        for idx, row in X.iterrows():
            pred = self._predict_single(row, self.root)
            predictions.append(pred)

        return pd.Series(predictions, index=X.index)

    def _compute_positions(self, node, depth=0, x_offset=0):
        """
        Recursively compute positions of each node (x, y).
        Returns: (positions, subtree_width)
        """
        if node.is_leaf:
            return {node: (x_offset, -depth)}, 1  # width = 1

        positions = {}
        child_widths = []
        current_x = x_offset

        for child in node.children.values():
            child_pos, width = self._compute_positions(child, depth+1, current_x)
            positions.update(child_pos)
            child_widths.append((child, width))
            current_x += width

        total_width = sum(w for _, w in child_widths)

        # Place current node at center of its children
        first_child, _ = child_widths[0]
        last_child, last_w = child_widths[-1]
        x_first, _ = positions[first_child]
        x_last, _ = positions[last_child]
        node_x = (x_first + x_last) / 2
        positions[node] = (node_x, -depth)

        return positions, total_width

    def _plot_tree_graph(self, ax=None):
        if ax is None:
            ax = plt.gca()

        positions, _ = self._compute_positions(self.root)

        for node, (x, y) in positions.items():
            # Draw node box
            if node.is_leaf:
                label = f"{node.prediction:.2f}" if check_ifreal(pd.Series([node.prediction])) else f"{node.prediction}"
                ax.text(x, y, label, ha='center', va='center',
                        bbox=dict(boxstyle="round", facecolor="lightgreen"))
            else:
                if node.split_value is not None:
                    if isinstance(node.split_value, (int, float, np.number)):
                        label = f"{node.feature}\n<= {node.split_value:.2f}"
                    else:
                        label = f"{node.feature}\n= {node.split_value}"
                else:
                    label = f"{node.feature}"
                ax.text(x, y, label, ha='center', va='center',
                        bbox=dict(boxstyle="round", facecolor="lightblue"))

            # Draw edges
            for key, child in node.children.items():
                cx, cy = positions[child]
                ax.plot([x, cx], [y-0.2, cy+0.2], 'k-')
                if node.split_value is None:  # categorical edge label
                    ax.text((x+cx)/2, (y+cy)/2, str(key),
                            ha='center', va='center', fontsize=8, color="darkred")

        ax.axis('off')

    def plot(self) -> None:
        if self.root is None:
            print("Tree has not been fitted yet.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_tree_graph(ax=ax)
        plt.show()
