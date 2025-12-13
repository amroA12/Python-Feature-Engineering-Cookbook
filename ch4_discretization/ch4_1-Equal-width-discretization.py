import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

x, y = fetch_california_housing(return_X_y=True, as_frame=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y ,
    test_size=0.3,
    random_state=0
)

min_value = int(x_train["HouseAge"].min())
max_value = int(x_train["HouseAge"].max())

width = int((max_value - min_value) / 10)

interval_limits = [i for i in range(min_value, max_value, width)]

interval_limits [0] = -np.inf
interval_limits [-1] = np.inf

train_t = x_train.copy()
test_t = x_test.copy()

train_t["HouseAge_disc"] = pd.cut(
    x=x_train["HouseAge"],
    bins=interval_limits,
    include_lowest=True)

test_t["HouseAge_disc"] = pd.cut(
    x=x_test["HouseAge"],
    bins=interval_limits,
    include_lowest=True)

print(train_t[["HouseAge", "HouseAge_disc"]].head(5))

t1 = train_t["HouseAge_disc"].value_counts(normalize=True, sort=False)
t2 = test_t["HouseAge_disc"].value_counts(normalize=True, sort=False)

tmp = pd.concat([t1, t2], axis=1)
tmp.columns = ["train", "test"]

tmp.plot.bar(figsize=(8, 5))
plt.xticks(rotation=45)
plt.ylabel("Number of observations per bin")
plt.xlabel('Discretized HouseAge')
plt.title("HouseAge")
plt.show()

from feature_engine.discretisation import EqualWidthDiscretiser

variables = ['MedInc' , 'HouseAge', 'AveRooms']
disc = EqualWidthDiscretiser(bins=8 , variables=variables)

disc.fit(x_train)

train_t = disc.transform(x_train)
test_t = disc.transform(x_test)

plt.figure(figsize=(6, 12), constrained_layout=True)

for i in range(3):

    ax = plt.subplot(3, 1, i + 1)

    var = variables[i]

    t1 = train_t[var].value_counts(normalize=True, sort=False)
    t2 = test_t[var].value_counts(normalize=True, sort=False)

    tmp = pd.concat([t1, t2], axis=1)
    tmp.columns = ["train", "test"]

    tmp.sort_index(inplace=True)

    tmp.plot.bar(ax=ax)
    plt.xticks(rotation=0)
    plt.ylabel("Observations per bin")

    ax.set_title(var)

plt.show()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer

disc = KBinsDiscretizer(
    n_bins=8, encode="ordinal",
    strategy="uniform", subsample=None)

ct = ColumnTransformer(
    [("discretizer", disc, variables)],
    remainder="passthrough",
).set_output(transform="pandas")

ct.fit(x_train)

train_t = ct.transform(x_train)
test_t = ct.transform(x_test)

variables = ["discretizer__MedInc", "discretizer__HouseAge", "discretizer__AveRooms"]

plt.figure(figsize=(6, 12), constrained_layout=True)

for i in range(3):

    ax = plt.subplot(3, 1, i + 1)

    var = variables[i]

    t1 = train_t[var].value_counts(normalize=True, sort=False).sort_index()
    t2 = test_t[var].value_counts(normalize=True, sort=False).sort_index()

    tmp = pd.concat([t1, t2], axis=1)
    tmp.columns = ["train", "test"]

    tmp.plot.bar(ax=ax)
    plt.xticks(rotation=0)
    plt.ylabel("Observations per bin")

    ax.set_title(var)

plt.show()

