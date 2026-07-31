import pandas as pd
import numpy as np


# Function to calculate Minkowski Distance
def minkowski_distance(point1, point2, p):
    distance = 0

    for i in range(len(point1)):
        distance += abs(point1[i] - point2[i]) ** p

    return distance ** (1 / p)


# Read the Excel file
dataset = pd.read_excel(
   r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3\Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)

# Select only numeric columns
numeric_data = dataset.select_dtypes(include=[np.number])

# Remove columns and rows with missing values
numeric_data = numeric_data.dropna(axis=1)
numeric_data = numeric_data.dropna()

print("Numeric Dataset Shape:")
print(numeric_data.shape)

# Select first two records
point1 = numeric_data.iloc[0].values
point2 = numeric_data.iloc[1].values

print("\nFeature Vector 1:")
print(point1)

print("\nFeature Vector 2:")
print(point2)

# Calculate distances
euclidean = minkowski_distance(point1, point2, 2)
manhattan = minkowski_distance(point1, point2, 1)

print("\nEuclidean Distance (p = 2):", euclidean)
print("Manhattan Distance (p = 1):", manhattan)