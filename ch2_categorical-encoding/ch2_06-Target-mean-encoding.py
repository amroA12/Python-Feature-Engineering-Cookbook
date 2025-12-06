import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("credit_approval_uci_ch2.csv")
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=["target"], axis=1),  
    data["target"],  
    test_size=0.3,  
    random_state=0,  
)

mapping = y_train.groupby(X_train["A7"]).mean().to_dict()

X_train_enc = X_train.copy()
X_test_enc = X_test.copy()

X_train_enc["A7"] = X_train_enc["A7"].map(mapping)
X_test_enc["A7"] = X_test_enc["A7"].map(mapping)

from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer

cat_vars = X_train.select_dtypes(include="O").columns.to_list()

enc = TargetEncoder(smooth="auto", random_state=9)

ct = ColumnTransformer(
    [("encoder", enc, cat_vars)],
    remainder="passthrough",
).set_output(transform="pandas")

X_train_enc = ct.fit_transform(X_train, y_train)
X_test_enc = ct.transform(X_test)

ct.named_transformers_["encoder"].encodings_

X_train_enc.head()

from feature_engine.encoding import MeanEncoder

mean_enc = MeanEncoder(
    smoothing="auto",
    variables=None,
    )

mean_enc.fit(X_train, y_train)

mean_enc.variables_

mean_enc.encoder_dict_

X_train_enc = mean_enc.transform(X_train)
X_test_enc = mean_enc.transform(X_test)


X_train_enc.head()

X_test_enc.head()

print("Mapping A7 => Mean target:")
print(mapping)

print("Manual encoded A7 - X_train_enc['A7'].head():")
print(X_train_enc["A7"].head())

print("TargetEncoder encodings:")
print(ct.named_transformers_["encoder"].encodings_)

print("X_train_enc after TargetEncoder (head):")
print(X_train_enc.head())

print("MeanEncoder variables:")
print(mean_enc.variables_)

print("MeanEncoder encoder_dict:")
print(mean_enc.encoder_dict_)

print("X_train_enc after MeanEncoder (head):")
print(X_train_enc.head())

print("X_test_enc after MeanEncoder (head):")
print(X_test_enc.head())

