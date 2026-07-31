import pandas as pd

# Function for Label Encoding
def label_encode(data, column):

    unique_values = data[column].dropna().unique()

    mapping = {value: index for index, value in enumerate(unique_values)}

    encoded_data = data.copy()
    encoded_data[column] = encoded_data[column].map(mapping)

    return encoded_data, mapping


# Function for One-Hot Encoding
def one_hot_encode(data, column):

    encoded_data = pd.get_dummies(
        data,
        columns=[column],
        dtype=int
    )

    return encoded_data


# ==========================
# Read Excel File
# ==========================

file_path = r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3"
marketing = pd.read_excel(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3\Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)    # Change if your sheet name is different


# ==========================
# Label Encoding
# ==========================

label_data, mapping = label_encode(
    marketing,
    "Education"
)

print("Label Encoding Mapping:\n")
print(mapping)

print("\nEncoded Dataset (First 5 Rows):\n")
print(label_data.head())


# ==========================
# One-Hot Encoding
# ==========================

onehot_data = one_hot_encode(
    marketing,
    "Marital_Status"
)

print("\nOne-Hot Encoded Dataset (First 5 Rows):\n")
print(onehot_data.head())