import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
marketing = pd.read_excel(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\lab3\Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)

# Select Income column and remove missing values
income = marketing["Income"].dropna()

# Create Histogram
plt.figure(figsize=(8, 5))

plt.hist(
    income,
    bins=20,
    edgecolor="black"
)

plt.title("Histogram of Income")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()