import pandas as pd
import numpy as np


# Function to calculate Dot Product
def dot_product(vector1, vector2):

    result = 0

    for i in range(len(vector1)):
        result += vector1[i] * vector2[i]

    return result


# Function to calculate Euclidean Norm
def euclidean_norm(vector):

    total = 0

    for value in vector:
        total += value ** 2

    return total ** 0.5


# Read the Excel file
marketing = pd.read_excel(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3\Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)

# Select only numeric columns
numeric_data = marketing.select_dtypes(include=[np.number])

# Remove rows with missing values
numeric_data = numeric_data.dropna()

# Select first two records
vector1 = numeric_data.iloc[0].values
vector2 = numeric_data.iloc[1].values

# Calculate Dot Product and Norms
dot = dot_product(vector1, vector2)

norm1 = euclidean_norm(vector1)

norm2 = euclidean_norm(vector2)

# Display Results
print("=" * 50)
print("Dot Product and Euclidean Norm")
print("=" * 50)

print(f"Dot Product                : {dot:.4f}")
print(f"Euclidean Norm of Vector 1 : {norm1:.4f}")
print(f"Euclidean Norm of Vector 2 : {norm2:.4f}")