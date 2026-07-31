import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate Euclidean Distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# K-Means Algorithm
def kmeans(data, k=3, max_iterations=100):

    # Initialize centroids using the first k points
    centroids = data[:k].copy()

    for iteration in range(max_iterations):

        clusters = []

        # Assign each point to the nearest centroid
        for point in data:

            distances = []

            for centroid in centroids:
                distances.append(euclidean_distance(point, centroid))

            cluster = np.argmin(distances)
            clusters.append(cluster)

        clusters = np.array(clusters)

        # Calculate new centroids
        new_centroids = []

        for i in range(k):

            cluster_points = data[clusters == i]

            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(centroids[i])

        new_centroids = np.array(new_centroids)

        # Stop if centroids do not change
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


# Read Excel File
marketing = pd.read_excel(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3\Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)

# Select required columns and remove missing values
data = marketing[["Income", "MntWines"]].dropna().values

# Apply K-Means
clusters, centroids = kmeans(data, k=3)

# Plot the clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    data[:, 0],
    data[:, 1],
    c=clusters,
    cmap="viridis",
    s=40
)

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    color="red",
    label="Centroids"
)

plt.xlabel("Income")
plt.ylabel("MntWines")
plt.title("K-Means Clustering")
plt.legend()
plt.grid(True)

plt.show()