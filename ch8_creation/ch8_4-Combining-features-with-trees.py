import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from feature_engine.creation import DecisionTreeFeatures

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X.drop(labels=["Latitude", "Longitude"], axis=1, inplace=True)

X.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

for var in X_train.columns:
    pearson = np.corrcoef(X_train[var], y_train)[0, 1]
    pearson = np.round(pearson, 2)
    print(f"corr {var} vs target: {pearson}")
    
    param_grid = {"max_depth": [2, 3, 4, None]}

variables = ["AveRooms", "AveBedrms"]
dtf = DecisionTreeFeatures(
    variables=variables,
    features_to_combine=None,
    cv=5,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    regression=True,
)

dtf.fit(X_train, y_train)

dtf.input_features_
dtf.estimators_
trained_trees = dict()
for var, tree in zip(dtf.input_features_, dtf.estimators_):
    trained_trees[f"{var}"] = tree
    
train_t = dtf.transform(X_train)
test_t = dtf.transform(X_test)

tree_features = [var for var in test_t.columns if "tree" in var ]

test_t[tree_features].head()

for var in variables:
    pearson = np.corrcoef(X_test[var], y_test)[0, 1]
    pearson = np.round(pearson, 2)
    print(f"corr {var} vs target: {pearson}")

features = (('Population'), ('Population', 'AveOccup'),
            ('Population', 'AveOccup', 'HouseAge'))

dtf = DecisionTreeFeatures(
    variables=None,
    features_to_combine=features,
    cv=5,
    param_grid=param_grid,
    scoring="neg_mean_squared_error"
)

dtf.fit(X_train, y_train)

dtf.input_features_

train_t = dtf.transform(X_train)
test_t = dtf.transform(X_test)

tree_features = [var for var in test_t.columns if "tree" in var]

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate

lasso = Lasso(random_state=0, alpha=0.0001)

cv_results = cross_validate(lasso, X_train, y_train, cv=3)
mean = cv_results['test_score'].mean()
std = cv_results['test_score'].std()
print(f"Results: {mean} +/- {std}")

variables = ["AveRooms", "AveBedrms", "Population"]
train_t = train_t.drop(variables, axis=1)
cv_results = cross_validate(lasso, train_t, y_train, cv=3)
mean = cv_results['test_score'].mean()
std = cv_results['test_score'].std()
print(f"Results: {mean} +/- {std}")

