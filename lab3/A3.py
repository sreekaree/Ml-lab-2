import pandas as pd

# Function for Label Encoding
def label_encode(data, column):

    unique_values = data[column].dropna().unique()

    mapping = {}

    for index, value in enumerate(unique_values):
        mapping[value] = index

    data[column] = data[column].map(mapping)

    print("\nLabel Encoding Mapping:")
    print(mapping)

    return data


# Function for One-Hot Encoding
def one_hot_encode(data, column):

    return pd.get_dummies(
        data,
        columns=[column],
        dtype=int
    )


# Read Excel file (same folder as Python file)
marketing = pd.read_excel(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3\Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)

print("Original Shape:")
print(marketing.shape)

# Label Encoding
marketing = label_encode(
    marketing,
    "Education"
)

# One-Hot Encoding
marketing = one_hot_encode(
    marketing,
    "Marital_Status"
)

print("\nEncoded Shape:")
print(marketing.shape)

print("\nFirst 5 Rows:")
print(marketing.head())