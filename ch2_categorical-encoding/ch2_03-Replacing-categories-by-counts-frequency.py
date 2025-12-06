import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.encoding import CountFrequencyEncoder

data = pd.read_csv("credit_approval_uci_ch2.csv")
x_train, x_test, y_train, y_test = train_test_split(
    data.drop(labels=["target"], axis=1),
    data["target"],
    test_size=0.3,
    random_state=0
)

count = x_train["A7"].value_counts().to_dict()

x_train_enc = x_train.copy()
x_test_enc = x_test.copy()
x_train_enc["A7"] = x_train_enc["A7"].map(count)
x_test_enc["A7"] = x_test_enc["A7"].map(count)

count_enc = CountFrequencyEncoder(
    encoding_method="count", variables=None,
    missing_values='ignore',
)
count_enc.fit(x_train)
count_enc.variables_
count_enc.encoder_dict_

x_train_enc = count_enc.transform(x_train)
x_test_enc = count_enc.transform(x_test)

print("\nCount Dictionary for A7:")
print(count)

print("\nManual Count Encoding (first 10 rows):")
print(x_train_enc[["A7"]].head(10))

print("\nVariables encoded by CountFrequencyEncoder:")
print(count_enc.variables_)

print("\nEncoder Dictionary (Feature-engine):")
print(count_enc.encoder_dict_)

print("\nFeature-engine Count Encoding (first 10 rows):")
print(x_train_enc.head(10))
