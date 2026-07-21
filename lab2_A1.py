import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\TRIPU\OneDrive\Documents\ML\Lab Session Data (1).xlsx", sheet_name='Purchase data')
df = df[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']]

X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
y = df['Payment (Rs)']

print("Dataset:")
print(df)
print()
print("Feature Matrix (X):")
print(X)
print()
print("Target Vector (y):")
print(y)
print()

model = LinearRegression()
model.fit(X, y)

print("=== sklearn LinearRegression Results ===")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print()

y_pred = model.predict(X)
print("Predicted values:", y_pred)
print("Actual values:   ", y.values)
print()

rss = np.sum((y.values - y_pred) ** 2)
print(f"Residual Sum of Squares (RSS): {rss}")
print(f"R^2 Score: {r2_score(y, y_pred)}")
print()

X_matrix = X.values
y_vector = y.values
X_aug = np.column_stack([np.ones(len(X_matrix)), X_matrix])
theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y_vector

print("=== Normal Equation Results ===")
print(f"Intercept (theta_0): {theta[0]}")
print(f"Coefficients (theta_1, theta_2, theta_3): {theta[1:]}")
print()

y_pred_manual = X_aug @ theta
rss_manual = np.sum((y_vector - y_pred_manual) ** 2)
print(f"RSS (Normal Equation): {rss_manual}")
print()

rank = np.linalg.matrix_rank(X.values)
print(f"Rank of Feature Matrix: {rank}")
print()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
features = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']

for i, feature in enumerate(features):
    axes[i].scatter(df[feature], df['Payment (Rs)'], color='blue', label='Actual')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Payment (Rs)')
    axes[i].set_title(f'Payment vs {feature}')

plt.tight_layout()
plt.savefig(r"C:\Users\TRIPU\OneDrive\Documents\ML\A1_plots.png",
            dpi=300,
            bbox_inches="tight")
plt.show()