"""A5 & A6: Test accuracy and predict behavior."""

import numpy as np


def evaluate_accuracy(neigh, X_test, y_test):
    """A5: Test accuracy of kNN classifier."""
    accuracy = neigh.score(X_test, y_test)
    return accuracy


def predict_behavior(neigh, X_test):
    """A6: Study prediction behavior using predict()."""
    predictions = neigh.predict(X_test)
    return predictions