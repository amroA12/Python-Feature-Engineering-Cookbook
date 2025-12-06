import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv("credit_approval_uci_ch2.csv")
data.head()

x_train, x_test, y_train, y_test = train_test_split(
    data.drop(labels=["target"], axis=1),
    data["target"],
    test_size=0.3,
    random_state=0
)
x_train.shape, x_test.shape

x_train["A4"].unique()

dummies = pd.get_dummies(
    x_train["A4"], drop_first=True)
dummies.head()

x_train_enc = pd.get_dummies(x_train, drop_first=True)
x_test_enc = pd.get_dummies(x_test, drop_first=True)
x_train_enc.head()
x_test_enc.head()

cat_vars = x_train.select_dtypes(
    include="O").columns.to_list()
cat_vars

encoder = OneHotEncoder(drop="first",
    sparse_output= False)

ct = ColumnTransformer(
    [("encoder", encoder, cat_vars)],
    remainder="passthrough",
).set_output(transform= "pandas")

ct.fit(x_train)

ct.named_transformers_["encoder"].categories_

x_train_enc = ct.transform(x_train)
x_test_enc = ct.transform(x_test)
x_test_enc.head()
ct.get_feature_names_out()

from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer

ohe_enc = OneHotEncoder(drop_last=True)

ohe_enc.fit(x_train)
ohe_enc.variables_
ohe_enc.encoder_dict_

x_train_enc = ohe_enc.transform(x_train)
x_test_enc = ohe_enc.transform(x_test)


print("\n=== Unique values in A4 column ===")
print(x_train["A4"].unique())

print("\n=== First 5 rows of dummies for A4 ===")
print(dummies.head())

print("\n=== x_train after pandas get_dummies ===")
print(x_train_enc.head())

print("\n=== x_test after pandas get_dummies ===")
print(x_test_enc.head())

print("\n=== Categories learned by sklearn OneHotEncoder ===")
print(ct.named_transformers_["encoder"].categories_)

print("\n=== x_train after sklearn ColumnTransformer ===")
print(x_train_enc.head())

print("\n=== x_test after sklearn ColumnTransformer ===")
print(x_test_enc.head())

print("\n=== Feature names after sklearn ColumnTransformer ===")
print(ct.get_feature_names_out())

print("\n=== feature_engine encoder variables ===")
print(ohe_enc.variables_)

print("\n=== feature_engine encoder mapping (encoder_dict_) ===")
print(ohe_enc.encoder_dict_)

print("\n=== x_train after feature_engine OneHotEncoder ===")
print(x_train_enc.head())

print("\n=== x_test after feature_engine OneHotEncoder ===")
print(x_test_enc.head())

print("\n=== Shapes of encoded datasets ===")
print("x_train_enc shape:", x_train_enc.shape)
print("x_test_enc shape:", x_test_enc.shape)