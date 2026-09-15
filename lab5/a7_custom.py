"""A7: Custom kNN package with Fit(), Predict(), Score() functions."""

import numpy as np
from collections import Counter


class CustomKNN:
    """Custom k-Nearest Neighbors classifier."""
    
    def __init__(self, n_neighbors=3, metric='euclidean', weighted=False):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weighted = weighted
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """A7: Fit method - train the model."""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """A7: Predict method - classify test samples."""
        predictions = []
        for test_point in X:
            # Calculate distances to all training points
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = self._calculate_distance(test_point, train_point)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance and take k nearest
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.n_neighbors]
            
            # Majority voting
            if self.weighted:
                # Weighted voting (inverse distance)
                labels = [label for (_, label) in k_nearest]
                weights = [1 / (dist + 1e-8) for dist, _ in k_nearest]
                label_counts = Counter()
                for label, weight in zip(labels, weights):
                    label_counts[label] += weight
                predicted = label_counts.most_common(1)[0][0]
            else:
                # Simple majority voting
                labels = [label for (_, label) in k_nearest]
                most_common = Counter(labels).most_common(1)[0][0]
                # Tie-breaking: choose smaller label
                tied = [label for label, count in Counter(labels).items() 
                        if count == Counter(labels).most_common(1)[0][1]]
                predicted = min(tied)
            
            predictions.append(predicted)
        
        return np.array(predictions)
    
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))
    
    def score(self, X, y):
        """A7: Score method - return mean accuracy."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy