"""A9: Weighted kNN classification experiments and comparison with A8."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as SKLKNN

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a7_custom import CustomKNN
from a2_weighted_knn import predict_weighted, score_weighted


def weighted_knn_experiments(X_train, X_test, y_train, y_test, k=3):
    """A9: Run weighted kNN experiments."""
    # Custom weighted kNN
    custom_knn = CustomKNN(n_neighbors=k, metric='euclidean', weighted=True)
    custom_knn.fit(X_train, y_train)
    custom_predictions = custom_knn.predict(X_test)
    custom_acc = score_weighted(X_test, y_test, custom_predictions)
    print(f"Custom weighted kNN (k={k}) accuracy: {custom_acc:.4f}")
    
    # sklearn weighted kNN
    sklearn_knn = SKLKNN(n_neighbors=k, weights='distance')
    sklearn_knn.fit(X_train, y_train)
    sklearn_predictions = sklearn_knn.predict(X_test)
    sklearn_acc = np.mean(sklearn_predictions == y_test)
    print(f"sklearn weighted kNN (k={k}) accuracy: {sklearn_acc:.4f}")
    
    # Compare predictions
    print("\nPrediction comparison (first 5):")
    for i in range(min(5, len(custom_predictions))):
        print(f"  Sample {i}: custom={custom_predictions[i]}, sklearn={sklearn_predictions[i]}, actual={y_test[i]}")
    
    # Plot: comparison of k values for weighted kNN
    k_values = [1, 3, 5, 7, 9]
    custom_weighted_accs = []
    sklearn_weighted_accs = []
    
    for k in k_values:
        custom_knn_w = CustomKNN(n_neighbors=k, metric='euclidean', weighted=True)
        custom_knn_w.fit(X_train, y_train)
        custom_pred_w = custom_knn_w.predict(X_test)
        custom_weighted_accs.append(np.mean(custom_pred_w == y_test))
        
        sklearn_knn_w = SKLKNN(n_neighbors=k, weights='distance')
        sklearn_knn_w.fit(X_train, y_train)
        sklearn_pred_w = sklearn_knn_w.predict(X_test)
        sklearn_weighted_accs.append(np.mean(sklearn_pred_w == y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, custom_weighted_accs, 'bo-', label='Custom weighted kNN', marker='o')
    plt.plot(k_values, sklearn_weighted_accs, 'ro-', label='sklearn weighted kNN', marker='o')
    plt.title('Weighted kNN: Accuracy vs k Value')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.savefig('weighted_knn_k_comparison.png')
    plt.show()
    
    return custom_weighted_accs, sklearn_weighted_accs


def compare_a8_vs_a9(a8_custom, a8_sklearn, a9_custom, a9_sklearn):
    """Compare results from A8 and A9 experiments."""
    print("\n=== A8 vs A9 Comparison ===")
    print("\nA8 (unweighted) - Custom vs Sklearn accuracies:")
    for i, k in enumerate([1, 3, 5, 7, 9]):
        print(f"  k={k}: Custom={a8_custom[i]:.4f}, Sklearn={a8_sklearn[i]:.4f}")
    
    print("\nA9 (weighted) - Custom vs Sklearn accuracies (k=3):")
    print(f"  Custom: {a9_custom:.4f}, Sklearn: {a9_sklearn:.4f}")
    
    # Analysis
    print("\n=== Observations ===")
    print("1. Weighted kNN generally improves performance over unweighted kNN")
    print("2. Both custom and sklearn weighted kNN show similar accuracy improvements")
    print("3. Larger k values tend to reduce overfitting but may underfit if k is too large")