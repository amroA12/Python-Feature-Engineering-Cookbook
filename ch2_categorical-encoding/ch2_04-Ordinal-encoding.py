import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("credit_approval_uci_ch2.csv")
x_train, x_test, y_train, y_test = train_test_split(
    data.drop(labels=["target"], axis=1),
    data["target"],
    test_size=0.3,   
    random_state=0
)

ordinal_mapping = {k: i for i, k in enumerate(x_train["A7"].unique(), 0)}

x_train_enc = x_train.copy()
x_test_enc = x_test.copy()
x_train_enc["A7"] = x_train_enc["A7"].map(ordinal_mapping)
x_test_enc["A7"] = x_test_enc["A7"].map(ordinal_mapping)

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

enc = OrdinalEncoder()
cat_vars = x_train.select_dtypes(include="O").columns.to_list()

ct = ColumnTransformer(
    [("encoder", enc, cat_vars)],
    remainder="passthrough",
).set_output(transform="pandas")

ct.fit(x_train)

x_train_enc = ct.transform(x_train)
x_test_enc = ct.transform(x_test)

from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OrdinalEncoder

enc = OrdinalEncoder(
    encoding_method="arbitrary",
    variables=cat_vars,
)

enc.fit(x_train, y_train)

enc = OrdinalEncoder(encoding_method='ordered')

enc.fit(x_train, y_train)

x_train_enc = enc.transform(x_train)
x_test_enc = enc.transform(x_test)

print("\n=== Training Data After Encoding ===")
print(x_train_enc.head())

print("\n=== Testing Data After Encoding ===")
print(x_test_enc.head())

print("\n=== Shapes ===")
print("x_train_enc shape:", x_train_enc.shape)
print("x_test_enc shape:", x_test_enc.shape)

print("\n=== Categories Learned by Encoder ===")
print(enc.encoder_dict_)