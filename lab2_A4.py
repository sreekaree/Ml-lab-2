import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\TRIPU\OneDrive\Documents\ML\Lab Session Data (1).xlsx", sheet_name="thyroid0387_UCI")

df = df.replace('?', np.nan)

print("=== Attribute Data Types ===")
print(df.dtypes)
print()

print("=== Categorical Attributes & Encoding ===")
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"Categorical columns ({len(cat_cols)}):")
for col in cat_cols:
    print(f"  {col}: unique values = {df[col].nunique()}, values = {df[col].unique()[:10]}")
print()
print("Encoding scheme:")
print("  Ordinal (Label Encoding): sex, on thyroxine, query on thyroxine, on antithyroid medication,")
print("    sick, pregnant, thyroid surgery, I131 treatment, query hypothyroid, query hyperthyroid,")
print("    lithium, goitre, tumor, hypopituitary, psych, TSH measured, T3 measured, TT4 measured,")
print("    T4U measured, FTI measured, TBG measured")
print("  Nominal (One-Hot Encoding): referral source, Condition")
print()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("=== Numeric Variable Ranges ===")
print(df[num_cols].describe().loc[['min', 'max', 'mean', 'std']].to_string())
print()

print("=== Missing Values ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Count'] > 0]
print(missing_df.to_string())
print()

print("=== Outliers (IQR Method) ===")
for col in num_cols:
    if col == 'Record ID':
        continue
    data = df[col].dropna()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data < lower) | (data > upper)]
    print(f"  {col}: {len(outliers)} outliers (range: {lower:.2f} - {upper:.2f})")
print()

print("=== Mean and Variance of Numeric Variables ===")
for col in num_cols:
    if col == 'Record ID':
        continue
    data = df[col].dropna()
    print(f"  {col}: Mean = {data.mean():.4f}, Variance = {data.var():.4f}, Std = {data.std():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_cols = [c for c in num_cols if c != 'Record ID']
for i, col in enumerate(plot_cols[:3]):
    df[col].dropna().plot.box(ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()

plt.savefig(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\A4_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nBox plots saved successfully!")

plt.show()