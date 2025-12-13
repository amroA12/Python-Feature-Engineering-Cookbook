import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

x, y = fetch_california_housing(return_X_y=True, as_frame=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=0,
)

x_train.hist(bins=30, figsize=(12, 12))
plt.show()

train_t = x_train.copy()
test_t = x_test.copy()

train_t["House_disc"], interval_limits = pd.qcut(
    x=x_train["HouseAge"],
    q=8,
    labels=None,
    retbins=True,
)

print(train_t[["HouseAge", "House_disc"]].head(5))

test_t["House_disc"] = pd.cut(
    x=x_test["HouseAge"], bins=interval_limits, include_lowest=True)

t1 = train_t["House_disc"].value_counts(normalize=True)
t2 = test_t["House_disc"].value_counts(normalize=True)

tmp = pd.concat([t1, t2], axis=1)
tmp.columns = ["train", "test"]
tmp.sort_index(inplace=True)

tmp.plot.bar()
plt.xticks(rotation=45)
plt.ylabel("Number of observations per bin")
plt.title("HouseAge")
plt.show()

from feature_engine.discretisation import EqualFrequencyDiscretiser

variables = ["MedInc", "HouseAge", "AveRooms"]

disc = EqualFrequencyDiscretiser(
    q=8, variables=variables, return_boundaries=True)

disc.fit(x_train)

disc.binner_dict_

train_t = disc.transform(x_train)
test_t = disc.transform(x_test)

plt.figure(figsize=(6, 12), constrained_layout=True)

for i in range(3):

    ax = plt.subplot(3, 1, i + 1)

    var = variables[i]

    t1 = train_t[var].value_counts(normalize=True)
    t2 = test_t[var].value_counts(normalize=True)

    tmp = pd.concat([t1, t2], axis=1)
    tmp.columns = ["train", "test"]

    tmp.sort_index(inplace=True)

    tmp.plot.bar(ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("Observations per bin")

    ax.set_title(var)

plt.show()

from sklearn.preprocessing import KBinsDiscretizer

disc = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="quantile")

disc.fit(x_train[variables])

disc.bin_edges_

train_t = x_train.copy()
test_t = x_test.copy()

train_t[variables] = disc.transform(x_train[variables])
test_t[variables] = disc.transform(x_test[variables])

train_t.head()

plt.figure(figsize=(6, 12), constrained_layout=True)

for i in range(3):

    ax = plt.subplot(3, 1, i + 1)

    var = variables[i]

    t1 = train_t[var].value_counts(normalize=True)
    t2 = test_t[var].value_counts(normalize=True)

    tmp = pd.concat([t1, t2], axis=1)
    tmp.columns = ["train", "test"]

    tmp.sort_index(inplace=True)

    tmp.plot.bar(ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("Observations per bin")

    ax.set_title(var)

plt.show()

