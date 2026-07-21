import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

thyroid = pd.read_excel(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\Lab Session Data (1).xlsx",
    sheet_name="thyroid0387_UCI"
)

thyroid.replace("?", np.nan, inplace=True)
X = thyroid.drop(columns=["Condition", "Record ID"])

y = thyroid["Condition"]
binary_columns = [
    "on thyroxine",
    "query on thyroxine",
    "on antithyroid medication",
    "sick",
    "pregnant",
    "thyroid surgery",
    "I131 treatment",
    "query hypothyroid",
    "query hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH measured",
    "T3 measured",
    "TT4 measured",
    "T4U measured",
    "FTI measured",
    "TBG measured"
]

for col in binary_columns:
    X[col] = X[col].map({"t": 1, "f": 0})
X["sex"] = X["sex"].map({
    "M": 1,
    "F": 0
})
X = pd.get_dummies(
    X,
    columns=["referral source"],
    dtype=int
)
numeric_columns = [
    "age",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI",
    "TBG"
]

for col in numeric_columns:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )
imputer = SimpleImputer(strategy="mean")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)
encoder = LabelEncoder()

y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
k_values = range(1, 12)

accuracies = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    accuracies.append(
        accuracy
    )

    print(
        "k =", k,
        " Accuracy =", accuracy
    )
plt.figure(figsize=(8,5))

plt.plot(
    k_values,
    accuracies,
    marker="o"
)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.grid(True)
plt.show()