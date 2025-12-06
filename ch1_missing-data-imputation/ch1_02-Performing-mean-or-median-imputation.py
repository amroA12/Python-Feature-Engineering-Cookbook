import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer

data = pd.read_csv("credit_approval_uci_ch1.csv")

x_train, x_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size= 0.3,
    random_state=0,
)

numeric_vars = x_train.select_dtypes(
    exclude = "O").columns.to_list()

median_values = x_train[numeric_vars].median().to_dict()

x_train_t = x_train.fillna(value = median_values)
x_test_t = x_test.fillna(value = median_values)

imputer = SimpleImputer(strategy="median")

ct = ColumnTransformer(
    [("imputer", imputer, numeric_vars)],
    remainder="passthrough",
).set_output(transform="pandas")

ct.fit(x_train)

x_train_t = ct.transform(x_train)
x_test_t = ct.transform(x_test)

imputer = MeanMedianImputer(
    imputation_method="median",
    variables=numeric_vars,
)
imputer.fit(x_train)
imputer.imputer_dict_

x_train = imputer.transform(x_train)
x_test = imputer.transform(x_test)

print(x_train_t.head())

print("\nDictionary of median values used for imputation:")
print(imputer.imputer_dict_)

print("\nX_train after MeanMedianImputer:")
print(x_train.head())

print("\nX_test after MeanMedianImputer:")
print(x_test.head())
