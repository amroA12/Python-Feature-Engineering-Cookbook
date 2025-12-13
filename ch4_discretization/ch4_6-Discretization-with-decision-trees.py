import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from feature_engine.discretisation import DecisionTreeDiscretiser

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X.head()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
)

variables = list(X.columns)[:-2]

disc = DecisionTreeDiscretiser(
    bin_output="boundaries",
    precision=3,
    cv=3,
    scoring="neg_mean_squared_error",
    variables=variables,
    regression=True,
    param_grid={"max_depth": [1, 2, 3], "min_samples_leaf": [10, 20, 50]},
)

disc.fit(X_train, y_train)

train_t = disc.transform(X_train)
test_t = disc.transform(X_test)
train_t[variables].head()

disc = DecisionTreeDiscretiser(
    bin_output="bin_number",
    cv=3,
    scoring="neg_mean_squared_error",
    variables=variables,
    regression=True,
    param_grid={"max_depth": [1, 2, 3], "min_samples_leaf": [10, 20, 50]},
)

train_t = disc.fit_transform(X_train, y_train)
test_t = disc.transform(X_test)

X_test["MedInc"].unique(), test_t["MedInc"].unique()

train_t[variables].head()

disc = DecisionTreeDiscretiser(
    bin_output="prediction",
    precision=1,
    cv=3,
    scoring="neg_mean_squared_error",
    variables=variables,
    regression=True,
    param_grid={"max_depth": [1, 2, 3], "min_samples_leaf": [10, 20, 50]},
)

train_t = disc.fit_transform(X_train, y_train)
test_t = disc.transform(X_test)

train_t[variables].head()

X_test["AveRooms"].nunique(), test_t["AveRooms"].nunique()

tree = disc.binner_dict_["AveRooms"].best_estimator_

fig = plt.figure(figsize=(20, 6))
plot_tree(tree, fontsize=10, proportion=True)
plt.show()

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
    plt.xticks(rotation=0)
    plt.ylabel("Observations per bin")

    ax.set_title(var)

plt.show()

