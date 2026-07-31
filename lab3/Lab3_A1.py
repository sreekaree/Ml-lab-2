import pandas as pd
import os
import glob


def study_dataset(folder_path):
    # Find all Excel files in the folder
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

    if not excel_files:
        print("❌ No Excel (.xlsx) files found in:")
        print(folder_path)
        return None

    # Use the first Excel file found
    file_path = excel_files[0]

    print(f"\nUsing file:\n{file_path}\n")

    data = pd.read_excel(file_path)

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


def identify_feature_types(data):

    if data is None:
        return

    print("\nFeature Type Analysis\n")

    for column in data.columns:

        if pd.api.types.is_numeric_dtype(data[column]):
            feature_type = "Ratio"

        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            feature_type = "Interval"

        elif data[column].dtype == "object":
            feature_type = "Nominal"

        else:
            feature_type = "Unknown"

        print(f"{column:30} --> {feature_type}")


# Folder containing the Excel file
folder_path = r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3"

dataset = study_dataset(folder_path)

identify_feature_types(dataset)