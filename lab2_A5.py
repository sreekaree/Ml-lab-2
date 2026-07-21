import pandas as pd
import numpy as np

# ---------------------------------------------
# Load Excel File
# ---------------------------------------------

file_path = r"C:\Users\TRIPU\OneDrive\Documents\ML\Lab Session Data (1).xlsx"

df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# ---------------------------------------------
# Replace common binary text values with 0/1
# ---------------------------------------------

df = df.replace({
    't': 1, 'f': 0,
    'T': 1, 'F': 0,
    'yes': 1, 'no': 0,
    'Yes': 1, 'No': 0
})

# ---------------------------------------------
# Find Binary Columns
# ---------------------------------------------

binary_columns = []

for col in df.columns:

    values = set(df[col].dropna().unique())

    if values.issubset({0,1}):
        binary_columns.append(col)

print("Binary Attributes:")
print(binary_columns)

# ---------------------------------------------
# First Two Observation Vectors
# ---------------------------------------------

v1 = df.loc[0, binary_columns].astype(int).values
v2 = df.loc[1, binary_columns].astype(int).values

print("\nVector 1")
print(v1)

print("\nVector 2")
print(v2)

# ---------------------------------------------
# Calculate f11, f10, f01, f00
# ---------------------------------------------

f11 = np.sum((v1 == 1) & (v2 == 1))
f10 = np.sum((v1 == 1) & (v2 == 0))
f01 = np.sum((v1 == 0) & (v2 == 1))
f00 = np.sum((v1 == 0) & (v2 == 0))

print("\nf11 =", f11)
print("f10 =", f10)
print("f01 =", f01)
print("f00 =", f00)

# ---------------------------------------------
# Jaccard Coefficient
# ---------------------------------------------

JC = f11 / (f11 + f10 + f01)

# ---------------------------------------------
# Simple Matching Coefficient
# ---------------------------------------------

SMC = (f11 + f00) / (f11 + f10 + f01 + f00)

print("\nJaccard Coefficient =", JC)
print("Simple Matching Coefficient =", SMC)