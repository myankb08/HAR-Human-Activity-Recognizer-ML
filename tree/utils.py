
"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    # This function can be used to convert discrete features into one hot encoded features.
    return pd.get_dummies(X, drop_first=False)
    pass

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    y_clean = y.dropna()
    if len(y_clean) == 0:
        raise ValueError("Empty Series")
    
    # Check if it's categorical dtype - definitely discrete
    if y_clean.dtype.name in ['category','object']:
        return False
    
    # Check if it's integer type with few unique values - likely discrete
    if y_clean.dtype in ['int64', 'int32', 'int16', 'int8']:
        unique_ratio = len(y_clean.unique()) / len(y_clean)
        if unique_ratio < 0.1:  # Less than 10% unique values suggests discrete
            return False

    # Otherwise, assume it's real
    return True


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    probs = Y.value_counts(normalize=True)
    if probs.empty:
        raise ValueError("Empty Series")
    # Adding a small constant to probs to avoid log(0)
    return -(probs * np.log2(probs+1e-7)).sum()
    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    probs = Y.value_counts(normalize=True)
    if probs.empty:
        raise ValueError("Empty Series")
    return 1 - (probs ** 2).sum()
    pass

def mse(Y: pd.Series) -> float:
    """
    Calculate mean squared error (variance) of a Series.
    """
    if Y.empty:
        raise ValueError("Empty Series")
    mean_y = Y.mean()
    return ((Y - mean_y) ** 2).mean()
    pass

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion == "information_gain":
        if check_ifreal(Y):
            base_metric = mse(Y)
            metric_fn = mse
        else:
            base_metric = entropy(Y)
            metric_fn = entropy
    elif criterion == "gini_index":
        if check_ifreal(Y):
            base_metric = mse(Y)
            metric_fn = mse
        else:
            base_metric = gini_index(Y)
            metric_fn = gini_index
    else:
        raise ValueError("Invalid criterion")

    total = len(Y)
    weighted_metric = 0.0

    df = pd.DataFrame({"Y": Y, "attr": attr})
    for _, subset in df.groupby("attr",observed=False)["Y"]:
        if len(subset) == 0:
            continue  # skip empty splits
        weight = len(subset) / total
        weighted_metric += weight * metric_fn(subset)

    return base_metric - weighted_metric


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    """ 
    According to wheather the features are real or discrete valued and the criterion, find the attribute 
    from the features series with the maximum information gain (entropy or varinace based on the type of output)
    or minimum gini index (discrete output).
    """
    if any(pd.api.types.is_numeric_dtype(X[f]) for f in features):
        return opt_split_real(X, y, criterion, features)
    return opt_split_discrete(X, y, criterion, features)
    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if pd.api.types.is_numeric_dtype(X[attribute]):
        return split_real(X, y, attribute, value)
    return split_discrete(X, y, attribute, value)
    pass

def opt_split_discrete(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series):
    best_attr, best_value, best_gain = None, None, -np.inf

    for attr in features:
        for val in X[attr].unique():
            # Split into two groups: val vs not val
            left_mask = X[attr] == val
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            # Calculate gain
            gain = information_gain(y, X[attr] == val, criterion)
            if gain > best_gain:
                best_gain, best_attr, best_value = gain, attr, val

    return best_attr, best_value

def split_discrete(X: pd.DataFrame, y: pd.Series, attribute, value):
    mask = X[attribute] == value
    return (X[mask], y[mask]), (X[~mask], y[~mask])


def opt_split_real(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series):
    best_attr, best_thr, best_gain = None, None, -np.inf
    for attr in features:
        values = np.sort(X[attr].unique())
        thresholds = (values[:-1] + values[1:]) / 2.0
        for thr in thresholds:
            left_mask = X[attr] <= thr
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            if criterion == "information_gain":
                if check_ifreal(y):
                    base_metric = mse(y)
                    weighted = (left_mask.sum()/len(y)) * mse(y[left_mask]) + (right_mask.sum()/len(y)) * mse(y[right_mask])
                else:
                    base_metric = entropy(y)
                    weighted = (left_mask.sum()/len(y)) * entropy(y[left_mask]) + (right_mask.sum()/len(y)) * entropy(y[right_mask])
            elif criterion == "gini_index":
                if check_ifreal(y):
                    base_metric = mse(y)
                    weighted = (left_mask.sum()/len(y)) * mse(y[left_mask]) + (right_mask.sum()/len(y)) * mse(y[right_mask])
                else:
                    base_metric = gini_index(y)
                    weighted = (left_mask.sum()/len(y)) * gini_index(y[left_mask]) + (right_mask.sum()/len(y)) * gini_index(y[right_mask])

            gain = base_metric - weighted
            if gain > best_gain:
                best_gain, best_attr, best_thr = gain, attr, thr
    return best_attr, best_thr

def split_real(X: pd.DataFrame, y: pd.Series, attribute, value):
    mask = X[attribute] <= value
    return (X[mask], y[mask]), (X[~mask], y[~mask])
