"""A4: Train kNN classifier using sklearn."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def train_knn(X_train, y_train, k=3):
    """A4: Train kNN classifier with k=3."""
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    return neigh