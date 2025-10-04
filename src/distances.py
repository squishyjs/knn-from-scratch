import numpy as np
from typing import Literal

MetricName = Literal["euclidean", "manhattan"]

def pairwise_distances(X: np.ndarray, Y: np.ndarray, metric: MetricName = "euclidean") -> np.ndarray:
    """
    Returns distance matrix D of shape (X.shape[0], Y.shape[0]).
    X: (n_samples, n_features)
    Y: (m_samples, n_features)
    """
    if metric == "euclidean":
        # (x - y)^2 = x^2 + y^2 - 2xy, all vectorized
        X_sq = np.sum(X**2, axis=1, keepdims=True)         # (n,1)
        Y_sq = np.sum(Y**2, axis=1, keepdims=True).T       # (1,m)
        D_sq = X_sq + Y_sq - 2 * X @ Y.T
        # numerical safety: negatives to zero
        np.maximum(D_sq, 0.0, out=D_sq)
        return np.sqrt(D_sq, dtype=X.dtype)
    elif metric == "manhattan":
        # broadcasted L1 distance
        return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
