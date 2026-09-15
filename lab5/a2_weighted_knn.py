"""A2: Weighted k-Nearest Neighbors classifier."""

import numpy as np
from collections import Counter
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a1_features import calculate_distance, majority_vote_weighted


def find_k_nearest_neighbors_weighted(X_train, y_train, test_point, k, metric='euclidean'):
    """Find k nearest neighbors and return with their distances and labels."""
    distances = []
    for i, train_point in enumerate(X_train):
        features = train_point[:-1]
        label = train_point[-1]
        dist = calculate_distance(test_point, features, metric)
        distances.append((dist, label))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    return k_nearest


def predict_weighted(X_test, X_train, y_train, k=3, metric='euclidean'):
    """A2: Predict using weighted kNN (inverse distance weighting)."""
    predictions = []
    for test_point in X_test:
        k_nearest = find_k_nearest_neighbors_weighted(
            X_train, y_train, test_point, k, metric
        )
        predicted_label = majority_vote_weighted(k_nearest)
        predictions.append(predicted_label)
    return np.array(predictions)


def score_weighted(X_test, y_test, predictions):
    """A2: Calculate accuracy of weighted kNN predictions."""
    accuracy = np.mean(predictions == y_test)
    return accuracy