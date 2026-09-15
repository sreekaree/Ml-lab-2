import pandas as pd
import numpy as np


def study_dataset(file_path):
    """A1: Study dataset - read and analyze."""
    # Auto-detect file type
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path)
    else:
        data = pd.read_csv(file_path)  # try CSV as default
    
    print("First Five Records:\n")
    print(data.head())

    print("\nDataset Shape:")
    print(data.shape)

    print("\nColumn Names:")
    print(data.columns.tolist())

    print("\nData Types:")
    print(data.dtypes)

    print("\nMissing Values:")
    print(data.isnull().sum())

    return data


def encode_categorical(data, column_name):
    """A1: Encode categorical data using label encoding."""
    data = data.copy()
    unique_values = data[column_name].unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    data[column_name] = data[column_name].map(mapping)
    return data, mapping


def identify_feature_types(data):
    """A1: Identify feature types (nominal, interval, ratio)."""
    print("\nFeature Type Analysis\n")

    for column in data.columns:
        dtype = data[column].dtype

        if dtype == 'object':
            feature_type = "Nominal"

        elif "date" in column.lower():
            feature_type = "Interval"

        elif np.issubdtype(data[column].dtype, np.number):
            feature_type = "Ratio"

        else:
            feature_type = "Unknown"

        print(f"{column:30} --> {feature_type}")


def encode_categorical(data, column_name):
    """A1: Encode categorical data using label encoding."""
    data = data.copy()
    unique_values = data[column_name].unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    data[column_name] = data[column_name].map(mapping)
    return data, mapping


def impute_missing_values(data, strategy='mean', columns=None):
    """A1: Impute missing values with mean, median, or mode."""
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
    """A1c: Bubble sort algorithm."""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def insertion_sort(arr):
    """A1c: Insertion sort algorithm."""
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
    """A1c: Selection sort algorithm."""
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
    """A1c: Calculate distance between two points."""
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
        raise ValueError("Metric must be 'euclidean', 'manhattan', or 'minkowski'")


def find_k_nearest_neighbors(train_features, test_feature, k, distance_metric='euclidean'):
    """A1e: Identify k-nearest neighbors based on distance."""
    distances = []
    for train_feat in train_features:
        # train_feat and test_feature are feature vectors without label column
        dist = calculate_distance(test_feature, train_feat, distance_metric)
        distances.append((dist,))  # placeholder for label
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Get k nearest neighbors
    k_neighbors = distances[:k]
    return k_neighbors


def majority_vote(neighbors):
    """A1f: Majority voting with tie-breaking."""
    labels = [label for (_, label) in neighbors]
    count = Counter(labels)
    max_count = max(count.values())

    tied_labels = [label for label, cnt in count.items() if cnt == max_count]

    if len(tied_labels) == 1:
        return tied_labels[0]
    else:
        return min(tied_labels)


def majority_vote_weighted(neighbors):
    """A2: Weighted majority voting with tie-breaking."""
    labels = [label for (_, label) in neighbors]
    distances = [dist for dist, _ in neighbors]

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


def predict_single(test_point, X_train, y_train, k=3, metric='euclidean', weighted=False):
    """Helper: predict single sample."""
    distances = []
    for i, train_point in enumerate(X_train):
        features = train_point[:-1]
        label = train_point[-1]
        dist = calculate_distance(test_point, features, metric)
        distances.append((dist, label))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    if weighted:
        return majority_vote_weighted(k_nearest)
    else:
        return majority_vote(k_nearest)


from collections import Counter