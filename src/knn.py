"""
K-Nearest Neighbors (KNN) Classifier implementation from scratch.
"""
import numpy as np
from collections import Counter
from src.distances import DISTANCE_METRICS


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier implemented from scratch.
    """

    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initialize KNN classifier.

        Args:
            k: int, number of neighbors to consider
            distance_metric: str, distance metric to use
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

        if distance_metric not in DISTANCE_METRICS:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        self.distance_function = DISTANCE_METRICS[distance_metric]

    def fit(self, X, y):
        """
        Fit the KNN model (store training data).

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)

        Returns:
            self
        """
        self.X_train = X
        self.y_train = y
        return self

    def _predict_single(self, x):
        """
        Predict class for a single sample.

        Args:
            x: numpy array of shape (n_features,)

        Returns:
            Predicted class label
        """
        # Calculate distances to all training samples
        distances = []
        for x_train in self.X_train:
            dist = self.distance_function(x, x_train)
            distances.append(dist)

        distances = np.array(distances)

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Return most common label
        counter = Counter(k_nearest_labels)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """
        Predict classes for multiple samples.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of predicted labels
        """
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities for multiple samples.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes) with probabilities
        """
        n_samples = X.shape[0]
        classes = np.unique(self.y_train)
        n_classes = len(classes)

        probas = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            # Calculate distances to all training samples
            distances = []
            for x_train in self.X_train:
                dist = self.distance_function(x, x_train)
                distances.append(dist)

            distances = np.array(distances)

            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]

            # Calculate probabilities
            for j, cls in enumerate(classes):
                probas[i, j] = np.sum(k_nearest_labels == cls) / self.k

        return probas

    def score(self, X, y):
        """
        Calculate accuracy score.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)

        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
