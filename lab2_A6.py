import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\TRIPU\OneDrive\Documents\ML\Lab Session Data (1).xlsx", sheet_name='thyroid0387_UCI')
df = df.replace('?', np.nan)

v1 = df.iloc[0].drop('Record ID')
v2 = df.iloc[1].drop('Record ID')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in v1.index:
    if v1[col] is np.nan or v2[col] is np.nan:
        v1[col] = 0
        v2[col] = 0
    if isinstance(v1[col], str):
        combined = pd.Series([str(v1[col]), str(v2[col])])
        le.fit(combined)
        v1[col] = le.transform([str(v1[col])])[0]
        v2[col] = le.transform([str(v2[col])])[0]

A = np.array([float(x) for x in v1.values])
B = np.array([float(x) for x in v2.values])
A = np.nan_to_num(A, nan=0.0)
B = np.nan_to_num(B, nan=0.0)

dot_product = np.dot(A, B)
norm_A = np.linalg.norm(A)
norm_B = np.linalg.norm(B)
cosine_sim = dot_product / (norm_A * norm_B)

print("Vector A:", A)
print()
print("Vector B:", B)
print()
print(f"<A,B> (dot product) : {dot_product:.4f}")
print(f"||A|| (magnitude)   : {norm_A:.4f}")
print(f"||B|| (magnitude)   : {norm_B:.4f}")
print(f"Cosine Similarity   : {cosine_sim:.4f}")