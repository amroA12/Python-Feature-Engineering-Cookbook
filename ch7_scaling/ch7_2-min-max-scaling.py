import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

scaler = MinMaxScaler().set_output(transform="pandas")

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler.data_max_
scaler.min_
scaler.data_range_
X_test_scaled.min()
X_test_scaled.max()

plt.rcParams.update({"font.size": 15})

X_test.hist(bins=20, figsize=(20, 12), layout=(2, 3))
plt.show()

X_test_scaled.hist(bins=20, figsize=(20, 12), layout=(2, 3))
plt.show()

