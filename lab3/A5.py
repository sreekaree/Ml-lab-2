import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate Minkowski Distance
def minkowski_distance(vector1, vector2, p):

    distance = 0

    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p

    return distance ** (1 / p)


# Read the Excel file
marketing = pd.read_excel(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3\Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)

# Select only numeric columns
numeric_data = marketing.select_dtypes(include=[np.number])

# Remove missing values (recommended)
numeric_data = numeric_data.dropna()

# Select first two numeric records
vector1 = numeric_data.iloc[0].values
vector2 = numeric_data.iloc[1].values

# Store p values and distances
p_values = []
distances = []

for p in range(1, 11):

    distance = minkowski_distance(vector1, vector2, p)

    p_values.append(p)
    distances.append(distance)

    print(f"p = {p}   Distance = {distance:.4f}")

# Plot the graph
plt.figure(figsize=(8, 5))

plt.plot(
    p_values,
    distances,
    marker='o'
)

plt.xlabel("Value of p")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance vs p")
plt.grid(True)

plt.show()