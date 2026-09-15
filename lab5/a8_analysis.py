"""A8: Comparative analysis between custom and sklearn kNN."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as SKLKNN

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a7_custom import CustomKNN


def comparative_analysis_k_range(X_train, X_test, y_train, y_test, k_values=[1, 3, 5, 7, 9]):
    """A8: Compare custom kNN vs sklearn kNN for range of k values."""
    custom_accuracies = []
    sklearn_accuracies = []
    
    for k in k_values:
        # Custom kNN
        custom_knn = CustomKNN(n_neighbors=k, metric='euclidean', weighted=False)
        custom_knn.fit(X_train, y_train)
        custom_acc = custom_knn.score(X_test, y_test)
        custom_accuracies.append(custom_acc)
        
        # sklearn kNN
        sklearn_knn = SKLKNN(n_neighbors=k)
        sklearn_knn.fit(X_train, y_train)
        sklearn_acc = sklearn_knn.score(X_test, y_test)
        sklearn_accuracies.append(sklearn_acc)
        
        print(f"k={k}: Custom={custom_acc:.4f}, Sklearn={sklearn_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, custom_accuracies, 'bo-', label='Custom kNN', marker='o')
    plt.plot(k_values, sklearn_accuracies, 'ro-', label='sklearn kNN', marker='o')
    plt.title('Comparative Analysis: Custom kNN vs sklearn kNN')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.savefig('knn_comparison_plot.png')
    plt.show()
    
    return custom_accuracies, sklearn_accuracies


def comparative_analysis_weighted_k_range(X_train, X_test, y_train, y_test, k_values=[1, 3, 5, 7, 9]):
    """A8 extension: Compare weighted custom vs sklearn weighted kNN."""
    custom_weighted_accs = []
    sklearn_weighted_accs = []
    
    for k in k_values:
        # Custom weighted kNN
        custom_knn = CustomKNN(n_neighbors=k, metric='euclidean', weighted=True)
        custom_knn.fit(X_train, y_train)
        custom_acc = custom_knn.score(X_test, y_test)
        custom_weighted_accs.append(custom_acc)
        
        # sklearn weighted kNN
        sklearn_knn = SKLKNN(n_neighbors=k, weights='distance')
        sklearn_knn.fit(X_train, y_train)
        sklearn_acc = sklearn_knn.score(X_test, y_test)
        sklearn_weighted_accs.append(sklearn_acc)
        
        print(f"k={k}: Custom weighted={custom_acc:.4f}, Sklearn weighted={sklearn_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, custom_weighted_accs, 'bo-', label='Custom weighted kNN', marker='o')
    plt.plot(k_values, sklearn_weighted_accs, 'ro-', label='sklearn weighted kNN', marker='o')
    plt.title('Weighted kNN Comparative Analysis')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.savefig('weighted_knn_comparison_plot.png')
    plt.show()
    
    return custom_weighted_accs, sklearn_weighted_accs