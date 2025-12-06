import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

variables = ["A2", "A3", "A8", "A11", "A14", "A15", "target"]
data = pd.read_csv("credit_approval_uci_ch1.csv", usecols=variables)

x_train, x_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.3,
    random_state=0,
)

imputer = KNNImputer(
    n_neighbors=5,
    weights="distance"
).set_output(transform="pandas")

imputer.fit(x_train)

x_train = imputer.transform(x_train)
x_test = imputer.transform(x_test)

print("\n=== Original x_train (before imputation) ===")
print(x_train.head())

print("\n=== Original x_test (before imputation) ===")
print(x_test.head())

print("\n=== x_train after KNN imputation ===")
print(imputer.transform(x_train).head())

print("\n=== x_test after KNN imputation ===")
print(imputer.transform(x_test).head())
