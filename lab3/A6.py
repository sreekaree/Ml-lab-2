import pandas as pd
import numpy as np
from scipy.spatial.distance import minkowski


# Function to calculate Minkowski distance
def my_minkowski(vector1, vector2, p):

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

# Remove rows with missing values
numeric_data = numeric_data.dropna()

# Select the first two records
vector1 = numeric_data.iloc[0].values
vector2 = numeric_data.iloc[1].values

print("=" * 60)
print("Comparison of Minkowski Distances")
print("=" * 60)

# Compare distances for different values of p
for p in range(1, 6):

    my_distance = my_minkowski(
        vector1,
        vector2,
        p
    )

    scipy_distance = minkowski(
        vector1,
        vector2,
        p
    )

    print(f"\np = {p}")
    print(f"My Function     : {my_distance:.6f}")
    print(f"SciPy Function  : {scipy_distance:.6f}")