import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.preprocessing import PolynomialFeatures

set_config(transform_output="pandas")

df = pd.DataFrame(np.linspace(0, 10, 11), columns=["var"])

poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)

dft = poly.fit_transform(df)

poly.get_feature_names_out()

plt.rcParams["figure.dpi"] = 100

plt.plot(df["var"], dft)
plt.legend(dft.columns)
plt.xlabel("original variable")
plt.ylabel("new variables")
plt.show()

df["col"] = np.linspace(0, 5, 11)
df["feat"] = np.linspace(0, 5, 11)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

dft = poly.fit_transform(df)

poly.get_feature_names_out()

poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)

dft = poly.fit_transform(df)

poly.get_feature_names_out()

from sklearn.datasets import load_breast_cancer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    df, data.target, test_size=0.3, random_state=0
)

X_train.head()

features = ["mean smoothness", "mean compactness", "mean concavity"]

poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)

ct = ColumnTransformer([("poly", poly, features)])

ct.fit(X_train)

train_t = ct.transform(X_train)
test_t = ct.transform(X_test)

ct.get_feature_names_out()

test_t.head()

from feature_engine.wrappers import SklearnTransformerWrapper

poly = SklearnTransformerWrapper(
    transformer=PolynomialFeatures(
        degree=3, interaction_only=False, include_bias=False),
    variables=features,
)


train_t = poly.fit_transform(X_train)
test_t = poly.transform(X_test)

test_t.head()

