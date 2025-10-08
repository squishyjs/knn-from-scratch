"""
Utility functions for data processing and model operations.
"""
import numpy as np
from collections import Counter


def normalize_data(X):
    """
    Normalize data to range [0, 1].

    Args:
        X: numpy array of shape (n_samples, n_features)

    Returns:
        numpy array: Normalized data
    """
    X_min = np.min(X)
    X_max = np.max(X)

    if X_max - X_min == 0:
        return X

    return (X - X_min) / (X_max - X_min)


def standardize_data(X):
    """
    Standardize data to have mean 0 and std 1.

    Args:
        X: numpy array of shape (n_samples, n_features)

    Returns:
        numpy array: Standardized data
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    return (X - mean) / std


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split data into training and testing sets.

    Args:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        test_size: float, proportion of test set
        random_state: int, random seed

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def get_most_common(items):
    """
    Get the most common item from a list.

    Args:
        items: list of items

    Returns:
        Most common item
    """
    counter = Counter(items)
    return counter.most_common(1)[0][0]
