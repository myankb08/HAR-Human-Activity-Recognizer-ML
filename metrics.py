from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0, "Input series cannot be empty"

    # Calculate accuracy as the fraction of correct predictions
    correct_predictions = (y_hat == y).sum()
    total_predictions = y_hat.size
    return correct_predictions / total_predictions


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0, "Input series cannot be empty"

    # True positives: predicted as cls and actually cls
    true_positives = ((y_hat == cls) & (y == cls)).sum()

    # Predicted positives: all predictions as cls
    predicted_positives = (y_hat == cls).sum()

    # Handle division by zero
    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0, "Input series cannot be empty"

    # True positives: predicted as cls and actually cls
    true_positives = ((y_hat == cls) & (y == cls)).sum()

    # Actual positives: all actual instances of cls
    actual_positives = (y == cls).sum()

    # Handle division by zero
    if actual_positives == 0:
        return 0.0

    return true_positives / actual_positives


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0, "Input series cannot be empty"

    # Calculate mean squared error and take square root
    mse = np.mean((y_hat - y) ** 2)
    return np.sqrt(mse)


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0, "Input series cannot be empty"

    # Calculate mean absolute error
    return np.mean(np.abs(y_hat - y))
