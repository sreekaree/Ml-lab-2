"""A6: Test the prediction behavior of the classifier for test vectors."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SKLKNN


def test_prediction_behavior():
    """A6: Use the predict() function to study the prediction behavior."""
    # Sample data preparation (using mall customers dataset format)
    # Features: Age, Annual Income, Spending Score
    # Target: Gender (0=Male, 1=Female after encoding)
    
    # For demonstration, create sample data
    np.random.seed(42)
    n_samples = 100
    X = np.random.rand(n_samples, 3) * 100  # Random features
    y = np.random.randint(0, 2, n_samples)  # Binary target
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train kNN classifier with k=3
    print("A6: Testing prediction behavior of kNN classifier")
    print("=" * 60)
    print("\nTraining kNN classifier with k=3...")
    neigh = SKLKNN(n_neighbors=3)
    neigh.fit(X_train, y_train)
    
    # A5: Test accuracy
    print(f"\nA5: Test accuracy using score(): {neigh.score(X_test, y_test):.4f}")
    
    # A6: Use predict() to study prediction behavior
    print("\nA6: Prediction behavior study using predict():")
    predictions = neigh.predict(X_test)
    
    # Show first 10 predictions vs actual
    print("\nFirst 10 predictions vs actual values:")
    for i in range(10):
        print(f"  Sample {i}: predicted={predictions[i]}, actual={y_test[i]}")
    
    # Analyze prediction distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\nPrediction distribution: {dict(zip(unique, counts))}")
    
    unique_actual, counts_actual = np.unique(y_test, return_counts=True)
    print(f"Actual distribution: {dict(zip(unique_actual, counts_actual))}")
    
    # Misclassification analysis
    misclassified = np.sum(predictions != y_test)
    print(f"\nMisclassified samples: {misclassified}/{len(y_test)}")
    print(f"Misclassification rate: {misclassified/len(y_test):.4f}")
    
    # Show some specific prediction details
    print("\nDetailed prediction analysis:")
    for i in range(min(5, len(X_test))):
        distances, indices = neigh.kneighbors(X_test[i].reshape(1, -1), return_distance=True)
        print(f"  Sample {i}:")
        print(f"    Predicted: {predictions[i]}, Actual: {y_test[i]}")
        print(f"    k-nearest neighbor indices: {indices[0]}")
        print(f"    k-nearest distances: {[f'{d:.4f}' for d in distances[0]]}")


if __name__ == "__main__":
    test_prediction_behavior()