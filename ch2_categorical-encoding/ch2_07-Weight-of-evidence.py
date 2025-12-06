import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

data = pd.read_csv("credit_approval_uci_ch2.csv")
x_train, x_test, y_train, y_test = train_test_split(
    data.drop(labels=["target"], axis=1),
    data["target"],
    test_size=0.3,
    random_state=0
)

neg_y_train = pd.Series(
    np.where(y_train == 1, 0, 1),
    index = y_train.index
)

total_pos = y_train.sum()
total_neg = neg_y_train.sum()

pos = y_train.groupby(
    x_train["A1"]).sum() / total_pos
neg = neg_y_train.groupby(
    x_train["A1"]).sum() / total_neg

woe = np.log(pos/neg)

x_train_enc = x_train.copy()
x_test_enc = x_test.copy()
x_train_enc ["A1"]= x_train_enc ["A1"].map(woe)
x_test_enc ["A1"] = x_test_enc ["A1"].map(woe)

from feature_engine.encoding import WoEEncoder

woe_enc = WoEEncoder(variables= ["A1", "A9", "A12"])

woe_enc.fit(x_train, y_train)

x_train_enc = woe_enc.transform(x_train)
x_test_enc = woe_enc.transform(x_test)

print(" Manual WoE Values for A1")
print(woe)

print(" WoEEncoder learned parameters")
print(woe_enc.encoder_dict_)

print(" First 10 rows of x_train after WoEEncoder")
print(x_train_enc.head(10))

print(" Shapes before and after encoding")
print("Original x_train:", x_train.shape)
print("Encoded x_train:", x_train_enc.shape)

