import numpy as np
from typing import Optional, Literal, Tuple
from .distances import pairwise_distances, MetricName

Weighting = Literal["uniform", "distance"]

class KNNClassifier:
    def __init__(self, k: int = 3, metric: MetricName = "euclidean", weights: Weighting = "uniform", eps: float = 1e-9):
        assert k >= 1, "k must be >= 1"
        self.k = k
        self.metric = metric
        self.weights = weights
        self.eps = eps
        self._X = None
        self._y = None
        self._classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Store training set.
        X: (n_samples, n_features) float32/float64
        y: (n_samples,) int labels
        """
        X = np.asarray(X)
        y = np.asarray(y)
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
        self._X = X
        self._y = y
        self._classes = np.unique(y)
        return self

    def _vote(self, neighbor_labels: np.ndarray, neighbor_dists: Optional[np.ndarray]) -> int:
        """
        neighbor_labels: (k,)
        neighbor_dists: (k,) or None
        """
        if self.weights == "uniform" or neighbor_dists is None:
            # simple majority; break ties by smallest class id
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            max_count = counts.max()
            winners = labels[counts == max_count]
            return int(np.min(winners))
        else:
            # distance weighting: weight = 1 / (dist + eps)
            w = 1.0 / (neighbor_dists + self.eps)
            # sum weights per class
            class_weights = {}
            for lab, weight in zip(neighbor_labels, w):
                class_weights[lab] = class_weights.get(lab, 0.0) + weight
            # winner = max total weight; tie break by smallest class id
            max_w = max(class_weights.values())
            winners = [c for c, val in class_weights.items() if np.isclose(val, max_w)]
            return int(np.min(winners))

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Predict class labels for X.
        Processes in batches to control memory.
        """
        assert self._X is not None, "Call fit() first."
        X = np.asarray(X)
        n = X.shape[0]
        preds = np.empty((n,), dtype=int)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            D = pairwise_distances(X[start:end], self._X, metric=self.metric)  # (b, n_train)
            # argsort along rows, take k nearest
            idx = np.argpartition(D, kth=self.k-1, axis=1)[:, :self.k]
            # get sorted k indices for stable voting (optional)
            row_indices = np.arange(end - start)[:, None]
            k_dists = D[row_indices, idx]
            k_labels = self._y[idx]
            # Optionally fully sort the k neighbors by distance
            sort_order = np.argsort(k_dists, axis=1)
            k_dists = np.take_along_axis(k_dists, sort_order, axis=1)
            k_labels = np.take_along_axis(k_labels, sort_order, axis=1)

            # vote per row
            for i in range(end - start):
                preds[start + i] = self._vote(k_labels[i], k_dists[i] if self.weights == "distance" else None)
        return preds

    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Returns probabilities per class using vote fractions (uniform) or normalized distance weights.
        Shape: (n_samples, n_classes) aligned with self.classes_
        """
        assert self._X is not None, "Call fit() first."
        X = np.asarray(X)
        n = X.shape[0]
        classes = self._classes
        C = len(classes)
        proba = np.zeros((n, C), dtype=np.float32)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            D = pairwise_distances(X[start:end], self._X, metric=self.metric)
            idx = np.argpartition(D, kth=self.k-1, axis=1)[:, :self.k]
            row_indices = np.arange(end - start)[:, None]
            k_dists = D[row_indices, idx]
            k_labels = self._y[idx]

            if self.weights == "uniform":
                for i in range(end - start):
                    counts = {}
                    for lab in k_labels[i]:
                        counts[lab] = counts.get(lab, 0) + 1
                    total = float(self.k)
                    for lab, cnt in counts.items():
                        proba[start + i, class_to_idx[lab]] = cnt / total
            else:
                for i in range(end - start):
                    weights = 1.0 / (k_dists[i] + self.eps)
                    totals = {}
                    for lab, w in zip(k_labels[i], weights):
                        totals[lab] = totals.get(lab, 0.0) + float(w)
                    denom = sum(totals.values()) + 1e-12
                    for lab, wsum in totals.items():
                        proba[start + i, class_to_idx[lab]] = wsum / denom

        return proba

    @property
    def classes_(self) -> np.ndarray:
        return self._classes
