import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


data = pd.read_csv("credit_approval_uci_ch2.csv")

x_train, x_test, y_train, y_test = train_test_split(
    data.drop(labels=["target"], axis=1),
    data["target"],
    test_size=0.3,
    random_state=0
)

imputer = SimpleImputer(strategy="most_frequent")

cols = ["A6", "A7"]

x_train[cols] = imputer.fit_transform(x_train[cols])
x_test[cols] = imputer.transform(x_test[cols])


x_train["A6"].unique()

x_train["A6"].value_counts().sort_values(
    ascending=False
).head(5)

top_5 = [x for x in x_train[
    "A6"].value_counts().sort_values(
        ascending=False).head(5).index
    ]

x_train_enc = x_train.copy()
x_test_enc = x_test.copy()

for label in top_5:
    x_train_enc[f"A6_{label}"] = np.where(
        x_train["A6"] == label, 1, 0)

    x_test_enc[f"A6_{label}"] = np.where(
        x_test["A6"] == label, 1, 0)

x_train_enc[["A6"] + [f"A6_{label}" for label in top_5]].head(10)


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    min_frequency=39,
    max_categories=5,
    sparse_output=False,
).set_output(transform="pandas")

x_train_ohe1 = encoder.fit_transform(x_train[["A6", "A7"]])
x_test_ohe1 = encoder.transform(x_test[["A6", "A7"]])


from feature_engine.encoding import OneHotEncoder

ohe_enc = OneHotEncoder(
    top_categories=5,
    variables=["A6", "A7"]
)

ohe_enc.fit(x_train)

x_train_ohe2 = ohe_enc.transform(x_train)
x_test_ohe2 = ohe_enc.transform(x_test)

print("\nTop 5 categories in A6:")
print(top_5)

print("\nManual Top-5 Encoding (A6):")
print(x_train_enc[["A6"] + [f"A6_{label}" for label in top_5]].head(10))

print("\nOneHotEncoder (sklearn) - first 10 rows:")
print(x_train_ohe1.head(10))

print("\nOneHotEncoder (feature_engine) - first 10 rows:")
print(x_train_ohe2.head(10))
