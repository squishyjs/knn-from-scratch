"""
Distance metrics for KNN algorithm.
All functions are implemented from scratch using NumPy.
"""
import numpy as np


def euclidean_distance(x1, x2):
    """
    Calculate Euclidean distance between two vectors.

    Args:
        x1: numpy array of shape (n_features,)
        x2: numpy array of shape (n_features,)

    Returns:
        float: Euclidean distance
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    """
    Calculate Manhattan distance between two vectors.

    Args:
        x1: numpy array of shape (n_features,)
        x2: numpy array of shape (n_features,)

    Returns:
        float: Manhattan distance
    """
    return np.sum(np.abs(x1 - x2))


def cosine_distance(x1, x2):
    """
    Calculate Cosine distance between two vectors.

    Args:
        x1: numpy array of shape (n_features,)
        x2: numpy array of shape (n_features,)

    Returns:
        float: Cosine distance (1 - cosine similarity)
    """
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)

    if norm_x1 == 0 or norm_x2 == 0:
        return 1.0

    cosine_similarity = dot_product / (norm_x1 * norm_x2)
    return 1 - cosine_similarity


def minkowski_distance(x1, x2, p=3):
    """
    Calculate Minkowski distance between two vectors.

    Args:
        x1: numpy array of shape (n_features,)
        x2: numpy array of shape (n_features,)
        p: int, order of the norm

    Returns:
        float: Minkowski distance
    """
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)


# Dictionary to easily access distance functions
DISTANCE_METRICS = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'cosine': cosine_distance,
    'minkowski': minkowski_distance
}
