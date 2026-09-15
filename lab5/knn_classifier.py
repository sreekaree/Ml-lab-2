import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def load_dataset(file_path):
    """Load dataset from Excel file."""
    data = pd.read_excel(file_path)
    return data


def encode_categorical(data, column_name):
    """Convert categorical column to numeric using label encoding."""
    data = data.copy()
    unique_values = data[column_name].unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    data[column_name] = data[column_name].map(mapping)
    return data, mapping


def impute_missing_values(data, strategy='mean', columns=None):
    """Impute missing values using specified strategy (mean, median, mode)."""
    data = data.copy()
    if columns is None:
        columns = [c for c in data.columns if c != 'CustomerID']
    
    for col in columns:
        if strategy == 'mean':
            fill_value = data[col].mean()
        elif strategy == 'median':
            fill_value = data[col].median()
        elif strategy == 'mode':
            fill_value = data[col].mode()[0]
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
        
        data[col] = data[col].fillna(fill_value)
    
    return data


def bubble_sort(arr):
    """Bubble sort algorithm."""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def insertion_sort(arr):
    """Insertion sort algorithm."""
    arr = arr.copy()
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def selection_sort(arr):
    """Selection sort algorithm."""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def calculate_distance(point1, point2, metric='euclidean'):
    """Calculate distance between two points using specified metric."""
    p1 = np.array(point1, dtype=float)
    p2 = np.array(point2, dtype=float)
    if metric == 'euclidean':
        return np.sqrt(np.sum((p1 - p2) ** 2))
    elif metric == 'manhattan':
        return np.sum(np.abs(p1 - p2))
    elif metric == 'minkowski':
        p = 3
        return np.sum(np.abs(p1 - p2) ** p) ** (1 / p)
    else:
        raise ValueError("Distance metric must be 'euclidean', 'manhattan', or 'minkowski'")


def find_k_nearest_neighbors_distances(train_features, test_feature, k, distance_metric='euclidean', sorting_algo='bubble'):
    """Find k nearest neighbors based on distance from test point to training points."""
    distances = []
    for i, train_feat in enumerate(train_features):
        dist = calculate_distance(test_feature, train_feat, distance_metric)
        distances.append((dist, i))  # (distance, index in training set)
    
    # Sort distances using the configured sorting algorithm
    if sorting_algo == 'bubble':
        # Use bubble sort
        sorted_dist = []
        dist_copy = distances.copy()
        n = len(dist_copy)
        for i in range(n):
            for j in range(0, n - i - 1):
                if dist_copy[j][0] > dist_copy[j + 1][0]:
                    dist_copy[j], dist_copy[j + 1] = dist_copy[j + 1], dist_copy[j]
        sorted_dist = dist_copy
    elif sorting_algo == 'insertion':
        # Use insertion sort
        sorted_dist = []
        dist_copy = distances.copy()
        for i in range(len(dist_copy)):
            key = dist_copy[i]
            j = i - 1
            while j >= 0 and dist_copy[j][0] > key[0]:
                dist_copy[j + 1] = dist_copy[j]
                j -= 1
            dist_copy[j + 1] = key
        sorted_dist = dist_copy
    elif sorting_algo == 'selection':
        # Use selection sort
        sorted_dist = []
        dist_copy = distances.copy()
        n = len(dist_copy)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if dist_copy[j][0] < dist_copy[min_idx][0]:
                    min_idx = j
            sorted_dist.append(dist_copy.pop(min_idx))
        # Add remaining
        if dist_copy:
            sorted_dist.extend(dist_copy)
    else:
        # Default: use Python's sorted
        sorted_dist = sorted(distances, key=lambda x: x[0])
    
    # Get k nearest neighbors indices
    k_neighbors = sorted_dist[:k]
    return k_neighbors


def majority_vote(neighbors):
    """Determine class label via majority voting with tie-breaking."""
    labels = [label for (_, label) in neighbors]
    count = Counter(labels)
    max_count = max(count.values())
    
    # Tie-breaking: choose the smallest label value
    tied_labels = [label for label, cnt in count.items() if cnt == max_count]
    
    if len(tied_labels) == 1:
        return tied_labels[0]
    else:
        return min(tied_labels)


def weighted_majority_vote(neighbors):
    """Determine class label via weighted majority voting with tie-breaking."""
    labels = [label for (_, label) in neighbors]
    distances = [dist for dist, _ in neighbors]
    
    # Weight = 1 / distance (add small epsilon to avoid division by zero)
    weights = [1 / (dist + 1e-8) for dist in distances]
    
    count = Counter()
    for label, weight in zip(labels, weights):
        count[label] += weight
    
    max_weight = max(count.values())
    tied_labels = [label for label, w in count.items() if w == max_weight]
    
    if len(tied_labels) == 1:
        return tied_labels[0]
    else:
        return min(tied_labels)


class KNNClassifier:
    """K-Nearest Neighbors classifier from scratch."""
    
    def __init__(self, n_neighbors=3, distance_metric='euclidean', 
                 sorting_algorithm='bubble', weighting=False):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.sorting_algorithm = sorting_algorithm
        self.weighting = weighting
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values."""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, test_point):
        """Predict class label for a single test point."""
        # Find k nearest neighbors from training set
        distances = []
        for i, train_point in enumerate(self.X_train):
            dist = calculate_distance(test_point, train_point, self.distance_metric)
            distances.append((dist, self.y_train[i]))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.n_neighbors]
        
        # Classify based on voting scheme
        if self.weighting:
            return weighted_majority_vote(k_nearest)
        else:
            return majority_vote(k_nearest)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def train_test_split_data(X, y, test_size=0.3, random_state=42):
    """Split dataset into train and test sets."""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


def main():
    """Main function to run kNN experiments."""
    # Load dataset
    file_path = r"C:\Users\TRIPU\OneDrive\Documents\ML\lab5\Lab Session Data (1).xlsx"
    data = load_dataset(file_path)
    
    # Encode categorical column (Gender)
    data, gender_map = encode_categorical(data, 'Gender')
    
    # Impute missing values
    data = impute_missing_values(data, strategy='mean')
    
    # Prepare features and target
    X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    y = data['Gender'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Unique classes in y: {np.unique(y)}")
    print(f"First 5 X samples:\n{X[:5]}")
    print(f"First 5 y samples: {y[:5]}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.3)
    
    print(f"\nTrain set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    # A4: Train kNN classifier with k=3
    print("\n=== A4: Training kNN classifier (k=3) ===")
    knn = KNNClassifier(n_neighbors=3, distance_metric='euclidean', sorting_algorithm='bubble')
    knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)
    print(f"Custom kNN (k=3) accuracy: {knn_accuracy:.4f}")
    
    # A5: Test accuracy
    print(f"\nA5: Test accuracy: {knn_accuracy:.4f}")
    
    # A6: Use predict() to study prediction behavior
    predictions = knn.predict(X_test)
    print("\nA6: Sample predictions vs actual:")
    for i in range(min(5, len(predictions))):
        print(f"  Sample {i}: predicted={predictions[i]}, actual={y_test[i]}")
    
    # A7: Implemented functions (Fit, Predict, Score already in KNNClassifier)
    print("\nA7: Functions Fit(), Predict(), Score() implemented in KNNClassifier")
    
    # A8: Comparative analysis with sklearn
    from sklearn.neighbors import KNeighborsClassifier as SKLKNN
    
    print("\n=== A8: Comparative analysis ===")
    k_values = [1, 3, 5, 7, 9]
    custom_accuracies = []
    sklearn_accuracies = []
    
    for k in k_values:
        # Custom kNN
        knn_custom = KNNClassifier(n_neighbors=k, distance_metric='euclidean', sorting_algorithm='bubble')
        knn_custom.fit(X_train, y_train)
        custom_acc = knn_custom.score(X_test, y_test)
        custom_accuracies.append(custom_acc)
        
        # sklearn kNN
        knn_sklearn = SKLKNN(n_neighbors=k)
        knn_sklearn.fit(X_train, y_train)
        sklearn_acc = knn_sklearn.score(X_test, y_test)
        sklearn_accuracies.append(sklearn_acc)
        
        print(f"k={k}: Custom={custom_acc:.4f}, Sklearn={sklearn_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, custom_accuracies, 'bo-', label='Custom kNN')
    plt.plot(k_values, sklearn_accuracies, 'ro-', label='Sklearn kNN')
    plt.title('Comparative Analysis: Custom kNN vs sklearn kNN')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('knn_comparison_plot.png')
    plt.show()
    
    # A9: Weighted kNN comparison
    print("\n=== A9: Weighted kNN comparison ===")
    k = 3
    # Custom weighted kNN
    knn_weighted = KNNClassifier(n_neighbors=k, distance_metric='euclidean', 
                                  sorting_algorithm='bubble', weighting=True)
    knn_weighted.fit(X_train, y_train)
    weighted_acc = knn_weighted.score(X_test, y_test)
    print(f"Custom weighted kNN (k={k}) accuracy: {weighted_acc:.4f}")
    
    # sklearn weighted kNN
    knn_sklearn_weighted = SKLKNN(n_neighbors=k, weights='distance')
    knn_sklearn_weighted.fit(X_train, y_train)
    sklearn_weighted_acc = knn_sklearn_weighted.score(X_test, y_test)
    print(f"Sklearn weighted kNN (k={k}) accuracy: {sklearn_weighted_acc:.4f}")
    
    # Plot weighted comparison
    plt.figure(figsize=(10, 6))
    k_values_w = [1, 3, 5, 7, 9]
    custom_weighted_accs = []
    sklearn_weighted_accs = []
    
    for k in k_values_w:
        knn_custom_w = KNNClassifier(n_neighbors=k, distance_metric='euclidean', 
                                      sorting_algorithm='bubble', weighting=True)
        knn_custom_w.fit(X_train, y_train)
        custom_weighted_accs.append(knn_custom_w.score(X_test, y_test))
        
        knn_sklearn_w = SKLKNN(n_neighbors=k, weights='distance')
        knn_sklearn_w.fit(X_train, y_train)
        sklearn_weighted_accs.append(knn_sklearn_w.score(X_test, y_test))
    
    plt.plot(k_values_w, custom_weighted_accs, 'bo-', label='Custom weighted kNN')
    plt.plot(k_values_w, sklearn_weighted_accs, 'ro-', label='Sklearn weighted kNN')
    plt.title('Weighted kNN Comparison')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('weighted_knn_comparison_plot.png')
    plt.show()
    
    print("\n=== Experiments Complete ===")


if __name__ == "__main__":
    main()