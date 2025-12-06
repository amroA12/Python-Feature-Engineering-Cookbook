import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import CategoricalImputer

data = pd.read_csv("credit_approval_uci_ch1.csv")

x_train, x_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.3,
    random_state=0
)

categorical_vars = x_train.select_dtypes(
    include="O").columns.to_list()

frequent_values = x_train[
    categorical_vars].mode().iloc[0].to_dict()

x_train_t = x_train.fillna(value = frequent_values)
x_test_t = x_test.fillna(value = frequent_values)

imputation_dict = {var:
    "no_data" for var in categorical_vars}

imputer = SimpleImputer(strategy='most_frequent')

ct = ColumnTransformer(
    [("imputer", imputer, categorical_vars)],
    remainder="passthrough",
).set_output(transform="pandas")

ct.fit(x_train)

x_train_t = ct.transform(x_train)
x_test_t = ct.transform(x_test)

imputer = CategoricalImputer(
    imputation_method="frequent",
    variables=categorical_vars,
)

imputer.fit(x_train)
imputer.imputer_dict_

x_train_t = imputer.transform(x_train)
x_test_t = imputer.transform(x_test)

print(data.head())
print(data.columns)
print(imputer.imputer_dict_)
print(x_train_t.head())
