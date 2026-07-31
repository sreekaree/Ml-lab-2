import pandas as pd
import numpy as np
import math


# Function to calculate Mean
def calculate_mean(values):

    total = 0

    for value in values:
        total += value

    return total / len(values)


# Function to calculate Variance
def calculate_variance(values):

    mean = calculate_mean(values)

    total = 0

    for value in values:
        total += (value - mean) ** 2

    return total / len(values)


# Function to calculate Standard Deviation
def calculate_std(values):

    variance = calculate_variance(values)

    return math.sqrt(variance)


# Read Excel File
marketing = pd.read_excel(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3\Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)

# Select Income column and remove missing values
income = marketing["Income"].dropna().values

# Using Own Functions
print("=" * 60)
print("Using Own Functions")
print("=" * 60)

print(f"Mean               : {calculate_mean(income):.2f}")
print(f"Variance           : {calculate_variance(income):.2f}")
print(f"Standard Deviation : {calculate_std(income):.2f}")

# Using NumPy Functions
print("\n" + "=" * 60)
print("Using NumPy Functions")
print("=" * 60)

print(f"Mean               : {np.mean(income):.2f}")
print(f"Variance           : {np.var(income):.2f}")
print(f"Standard Deviation : {np.std(income):.2f}")