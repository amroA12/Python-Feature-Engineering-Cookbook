import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer
)

data = pd.read_csv("credit_approval_uci_ch1.csv")
x_train, x_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.3,
    random_state=0,
)

varnames = ["A1", "A3", "A4", "A5", "A6", "A7", "A8"]

indicators = [f"{var}_na" for var in varnames]

x_train_t = x_train.copy()
x_test_t = x_test.copy()

x_train_t[indicators] = x_train[varnames].isna().astype(int)
x_test_t[indicators] = x_test[varnames].isna().astype(int)

x_train_t.head()

imputer = AddMissingIndicator(
    variables=None,
    missing_only=True
)

imputer.fit(x_train)

x_train_t = imputer.transform(x_train)
x_test_t = imputer.transform(x_test)

pipe = Pipeline(
    [
        ("indicators", AddMissingIndicator(missing_only=True)),
        ("categorical", CategoricalImputer(imputation_method="frequent")),
        ("numerical", MeanMedianImputer()),
    ]
)

x_train_t = pipe.fit_transform(x_train)
x_test_t = pipe.transform(x_test)

numvars = x_train.select_dtypes(exclude="O").columns.to_list()
catvars = x_train.select_dtypes(include="O").columns.to_list()

pipe = ColumnTransformer(
    [
        ("num_imputer", SimpleImputer(
            strategy="mean", add_indicator=True), numvars),
        ("cat_imputer", SimpleImputer(
            strategy="most_frequent", add_indicator=True), catvars),
    ]
).set_output(transform="pandas")

x_train_t = pipe.fit_transform(x_train)
x_test_t = pipe.fit_transform(x_test)

print("\n=== First 5 rows of x_train_t after manual indicator creation ===")
print(x_train_t.head())

print("\n=== Indicators added manually (training set) ===")
print(x_train[varnames].isna().astype(int).head())

print("\n=== After AddMissingIndicator (training set) ===")
print(x_train_t.head())

print("\n=== Pipeline transformation (training set) ===")
print(x_train_t.head())

print("\n=== Numerical variables ===")
print(numvars)

print("\n=== Categorical variables ===")
print(catvars)

print("\n=== ColumnTransformer output (training set) ===")
print(x_train_t.head())

print("\n=== ColumnTransformer output (test set) ===")
print(x_test_t.head())
