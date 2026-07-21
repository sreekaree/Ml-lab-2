import pandas as pd
import numpy as np

# Load dataset
file_path = r"C:\Users\TRIPU\OneDrive\Documents\ML\Lab Session Data (1).xlsx"

df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

print("Missing Values Before Imputation:")
print(df.isnull().sum())

# ---------------------------------------
# Imputation
# ---------------------------------------

for col in df.columns:

    # Numeric Columns
    if pd.api.types.is_numeric_dtype(df[col]):

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        # Use Median if outliers exist
        if len(outliers) > 0:
            df[col] = df[col].fillna(df[col].median())

        # Otherwise Mean
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Categorical Columns
    else:

        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Imputation:")
print(df.isnull().sum())