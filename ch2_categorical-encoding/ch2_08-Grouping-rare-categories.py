import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from feature_engine.encoding import RareLabelEncoder

data = pd.read_csv("credit_approval_uci_ch2.csv")
x_train, x_test, y_train, y_test = train_test_split(
    data.drop(labels=["target"], axis=1),
    data["target"],
    test_size=0.3,
    random_state=0
)

freqs = x_train ["A7"].value_counts(normalize = True)

frequnet_cat = [x for x in freqs.loc[freqs > 0.05].index.values]

x_train_enc = x_train.copy()
x_test_enc = x_test.copy()

x_train_enc ["A7"] = np.where(x_train["A7"].isin(frequnet_cat), x_train["A7"], "Rare")
x_test_enc ["A7"] = np.where(x_test["A7"].isin(frequnet_cat), x_test["A7"], "Rare")

x_train ["A7"].value_counts(normalize = True)

rera_encoder = RareLabelEncoder(
    tol=0.05,
    n_categories=4
)

rera_encoder.fit(x_train)

x_train_enc = rera_encoder.transform(x_train)
x_test_enc = rera_encoder.transform(x_test)

print(" Frequencies of A7 (normalize=True)")
print(x_train["A7"].value_counts(normalize=True))

print(" Frequent categories (freq > 0.05)")
print(frequnet_cat)

print(" x_train after manual Rare Encoding (first 10 rows)")
print(x_train_enc.head(10))

print(" x_train after RareLabelEncoder (first 10 rows)")
print(x_train_enc.head(10))

print(" Shape before and after encoding")
print("Original:", x_train.shape)
print("Encoded :", x_train_enc.shape)