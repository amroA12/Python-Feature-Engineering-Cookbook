import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("credit_approval_uci_ch2.csv")
x_train, x_test, y_train, y_test = train_test_split(
    data.drop(labels=["target"], axis=1),
    data["target"],
    test_size=0.3,
    random_state=0
)

y_train.groupby(x_train["A7"]).mean().sort_values()
print(y_train.groupby(x_train["A7"]).mean().sort_values())

ordered_labels = y_train.groupby(x_train["A7"]).mean().sort_values().index

ordinal_mapping = {
    k: i for i, k in enumerate(ordered_labels, 0)
}

x_train_enc = x_train.copy()
x_test_enc = x_test.copy()
x_train_enc["A7"] = x_train_enc["A7"].map(ordinal_mapping)
x_test_enc["A7"] = x_test_enc["A7"].map(ordinal_mapping)

x_train["A7"].head()
print(x_train["A7"].head())

y_train.groupby(x_train["A7"]).mean().plot()
plt.title("Relationship between A7 and the target")
plt.ylabel("Mean of target")
plt.show()

y_train.groupby(x_train_enc["A7"]).mean().plot()
plt.title("Relationship between A7 after encoding and the target")
plt.ylabel("Mean of target")
plt.show()

from feature_engine.encoding import OrdinalEncoder
from feature_engine.imputation import CategoricalImputer

ordinal_enc = OrdinalEncoder(encoding_method="ordered")

ordinal_enc.fit(x_train, y_train)
ordinal_enc.variables_
ordinal_enc.encoder_dict_

x_train_enc = ordinal_enc.transform(x_train)
x_test_enc = ordinal_enc.transform(x_test)

x_train_enc.head()
x_test_enc.head()

print("Variables encoded by feature_engine OrdinalEncoder:")
print(ordinal_enc.variables_)

print("\nEncoder dictionary (mapping for each variable):")
print(ordinal_enc.encoder_dict_)

print("\nFirst 5 rows of x_train after ordinal encoding:")
print(x_train_enc.head())

print("\nFirst 5 rows of x_test after ordinal encoding:")
print(x_test_enc.head())

