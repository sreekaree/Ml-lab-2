"""A3: Train-test split using sklearn."""

import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.3, random_state=42):
    """A3: Divide dataset into train and test sets.
    
    Note: Before splitting, ensure only two classes are present.
    For multi-class problems, select any two classes.
    """
    # Check for multi-class and reduce to two classes if needed
    unique_classes = np.unique(y)
    if len(unique_classes) > 2:
        # Take the first two classes
        mask = np.isin(y, unique_classes[:2])
        X = X[mask]
        y = y[mask]
        print(f"Reduced to two classes: {unique_classes[:2]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def verify_binary_class(y):
    """Verify target has exactly two classes."""
    unique = np.unique(y)
    if len(unique) != 2:
        raise ValueError(f"Target must have exactly 2 classes, found {len(unique)}")
    return True