import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.imputation import DropMissingData

data = pd.read_csv("credit_approval_uci_ch1.csv")
data.head()

X_train, X_test, Y_train, Y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size= 0.30,
    random_state= 42,
)

fig, axes = plt.subplots(
    2, 1, figsize= (15, 10), squeeze=False)
X_train.isnull().mean().plot(
    kind='bar', color= 'gray', ax= axes[0, 0], title = "train")
X_test.isnull().mean().plot(
    kind='bar', color= 'black', ax= axes[1, 0], title = "test")
axes[0, 0].set_ylabel('Fraction of NAN')
axes[1, 0].set_ylabel('Fraction of NAN')
plt.show()

train_cca = X_train.dropna()
test_cca = X_test.dropna()

print(f"Total observations: {len(X_train)}")
print(f" Observations without NAN: {len(train_cca)}")

Y_train_cca = Y_train.loc[train_cca.index]
Y_test_cca = Y_test.loc[test_cca.index]

cca = DropMissingData(variables=None, missing_only=True)
cca.fit(X_train)
cca.variables_

train_cca = cca.transform(X_train)
test_cca = cca.transform(X_test)
train_c, Y_train_c = cca.transform_x_y(X_train, Y_train)
test_c, Y_test_c = cca.transform_x_y(X_test, Y_test)

