import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from feature_engine.imputation import ArbitraryNumberImputer

data = pd.read_csv("credit_approval_uci_ch1.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.3,
    random_state=0,
)

X_train[["A2", "A3", "A8", "A11"]].max()

X_train_t = X_train.copy()
X_test_t = X_test.copy()

X_train_t[["A2", "A3", "A8", "A11"]] = X_train_t[[
    "A2", "A3", "A8", "A11"]].fillna(99)
X_test_t[["A2", "A3", "A8", "A11"]] = X_test_t[[
    "A2", "A3", "A8", "A11"]].fillna(99)

imputer = SimpleImputer(strategy="constant", fill_value=99)

var = ["A2", "A3", "A8", "A11"]
imputer.fit(X_train[var])

X_train_t [var] = imputer.transform(X_train[var])
X_test_t [var] = imputer.transform(X_test[var])

imputer = ArbitraryNumberImputer(
    arbitrary_number=99,
    variables=["A2", "A3", "A8", "A11"],
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)

print("X_train_t after imputation:")
print(X_train_t.head())

print("\nX_test_t after imputation:")
print(X_test_t.head())

print("\nMissing values in X_train_t:")
print(X_train_t.isnull().sum())

print("\nMissing values in X_test_t:")
print(X_test_t.isnull().sum())

print("\nBefore imputation:")
print(X_train[var].head())

print("\nAfter imputation:")
print(X_train_t[var].head())
