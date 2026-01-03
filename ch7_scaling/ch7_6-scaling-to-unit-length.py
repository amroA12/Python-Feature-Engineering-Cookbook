import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

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



scaler = Normalizer(norm="l1") 

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.round(np.linalg.norm(X_train, ord=1, axis=1), 1)

np.round(np.linalg.norm(X_train_scaled, ord=1, axis=1), 1)



scaler = Normalizer(norm="l2")

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.round(np.linalg.norm(X_train, ord=2, axis=1), 1)

np.round(np.linalg.norm(X_train_scaled, ord=2, axis=1), 1)

