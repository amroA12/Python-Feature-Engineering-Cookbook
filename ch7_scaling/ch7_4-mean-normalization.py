import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X.drop(labels=["Latitude", "Longitude"], axis=1, inplace=True)

X.head()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
)

X_train.shape, X_test.shape

means = X_train.mean(axis=0)

ranges = X_train.max(axis=0) - X_train.min(axis=0)

X_train_scaled = (X_train - means) / ranges
X_test_scaled = (X_test - means) / ranges

scaler_mean = StandardScaler(
    with_mean=True, with_std=False).set_output(transform="pandas")

scaler_minmax = RobustScaler(
    with_centering=False, with_scaling=True, quantile_range=(0, 100)
).set_output(transform="pandas")

scaler_mean.fit(X_train)
scaler_minmax.fit(X_train)

X_train_scaled = scaler_minmax.transform(scaler_mean.transform(X_train))
X_test_scaled = scaler_minmax.transform(scaler_mean.transform(X_test))

X_test.describe()
X_test_scaled.describe()

X_test.hist(bins=20, figsize=(20, 12), layout=(2, 3))
plt.show()

X_test_scaled.hist(bins=20, figsize=(20, 12), layout=(2, 3))
plt.show()

