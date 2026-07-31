import pandas as pd
import math


# -------------------------------------------------------
# Function to calculate Mean
# -------------------------------------------------------

def calculate_mean(values):

    total = 0

    for value in values:
        total += value

    return total / len(values)


# -------------------------------------------------------
# Function to calculate Variance
# -------------------------------------------------------

def calculate_variance(values):

    mean = calculate_mean(values)

    total = 0

    for value in values:
        total += (value - mean) ** 2

    return total / len(values)


# -------------------------------------------------------
# Function to calculate Standard Deviation
# -------------------------------------------------------

def calculate_std(values):

    variance = calculate_variance(values)

    return math.sqrt(variance)


# -------------------------------------------------------
# Main Program
# -------------------------------------------------------

marketing = pd.read_excel(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3\Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)

# Select the Income column and remove missing values
income = marketing["Income"].dropna().tolist()

# Calculate statistics
mean = calculate_mean(income)
variance = calculate_variance(income)
std = calculate_std(income)

# Display results
print("=" * 50)
print("Statistics of Income Column")
print("=" * 50)

print(f"Mean               : {mean:.2f}")
print(f"Variance           : {variance:.2f}")
print(f"Standard Deviation : {std:.2f}")