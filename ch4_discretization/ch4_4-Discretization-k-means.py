import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from yellowbrick.cluster import KElbowVisualizer

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
)

variables = ["MedInc", "HouseAge", "AveRooms"]

k_means = KMeans(random_state=10)

for variable in variables:

    visualizer = KElbowVisualizer(
        k_means, k=(4, 12), metric="distortion", timings=False
    )

    visualizer.fit(X_train[variable].to_frame())

    visualizer.show()

k = 6

disc = KBinsDiscretizer(
    n_bins=k, 
    encode="onehot-dense",
    strategy="kmeans",
    subsample=None,
).set_output(transform="pandas")

disc.fit(X_train[variables])

train_features = disc.transform(X_train[variables])
train_features.head()

test_features = disc.transform(X_test[variables])
print(test_features.head())
