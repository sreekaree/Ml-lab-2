import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel(r"C:\Users\TRIPU\OneDrive\Documents\ML\Lab Session Data (1).xlsx", sheet_name='thyroid0387_UCI')
df = df.replace('?', np.nan)

df20 = df.head(20).copy()

binary_cols = ['sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication',
               'sick', 'pregnant', 'thyroid surgery', 'I131 treatment',
               'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre',
               'tumor', 'hypopituitary', 'psych', 'TSH measured', 'T3 measured',
               'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured']

bin_df = df20[binary_cols].replace({'t': 1, 'f': 0, 'M': 1, 'F': 0})
bin_df = bin_df.fillna(0).astype(int)

full_df = df20.drop('Record ID', axis=1).copy()
for col in full_df.select_dtypes(include=['object', 'string']).columns:
    full_df[col] = full_df[col].fillna("missing").astype(str)

    le = LabelEncoder()
    full_df[col] = le.fit_transform(full_df[col])
full_df = full_df.fillna(0).astype(float)

n = 20
jc_matrix = np.zeros((n, n))
smc_matrix = np.zeros((n, n))
cos_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        v1 = bin_df.iloc[i].values
        v2 = bin_df.iloc[j].values
        f11 = np.sum((v1 == 1) & (v2 == 1))
        f00 = np.sum((v1 == 0) & (v2 == 0))
        f01 = np.sum((v1 == 0) & (v2 == 1))
        f10 = np.sum((v1 == 1) & (v2 == 0))

        jc_matrix[i][j] = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0
        smc_matrix[i][j] = (f11 + f00) / (f00 + f01 + f10 + f11)

        A = full_df.iloc[i].values
        B = full_df.iloc[j].values
        dot = np.dot(A, B)
        cos_matrix[i][j] = dot / (np.linalg.norm(A) * np.linalg.norm(B)) if (np.linalg.norm(A) * np.linalg.norm(B)) != 0 else 0

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.heatmap(jc_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0])
axes[0].set_title('Jaccard Coefficient')

sns.heatmap(smc_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Simple Matching Coefficient')

sns.heatmap(cos_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[2])
axes[2].set_title('Cosine Similarity')

plt.tight_layout()

plt.savefig(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\A7_Heatmaps.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nHeatmaps saved successfully!")
