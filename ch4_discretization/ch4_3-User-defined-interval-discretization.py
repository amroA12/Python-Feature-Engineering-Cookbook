import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X.head()

X["Population"].hist(bins=30)
plt.title("Population")
plt.ylabel("Number of observations")
plt.show()

intervals = [0, 200, 500, 1000, 2000, np.inf]

labels = ["0-200", "200-500", "500-1000", "1000-2000", ">2000"]

X_t = X.copy()

X_t["Population_limits"] = pd.cut(
    X["Population"],
    bins=intervals,
    labels=None,
    include_lowest=True,
)

X_t["Population_range"] = pd.cut(
    X["Population"],
    bins=intervals,
    labels=labels,
    include_lowest=True,
)

X_t[["Population", "Population_range", "Population_limits"]].head()

X_t["Population_range"].value_counts().sort_index().plot.bar()
plt.xticks(rotation=0)
plt.ylabel("Number of observations")
plt.title("Population")
plt.show()

from feature_engine.discretisation import ArbitraryDiscretiser

intervals = {
    "Population": [0, 200, 500, 1000, 2000, np.inf],
    "MedInc": [0, 2, 4, 6, np.inf]}

discretizer = ArbitraryDiscretiser(
    binning_dict=intervals,
    return_boundaries=True,
)

X_t = discretizer.fit_transform(X)

X_t["Population"].value_counts(normalize=True).sort_index()

X_t["Population"].value_counts().sort_index().plot.bar()
plt.xticks(rotation=45)
plt.ylabel("Number of observations")
plt.title("Population")
plt.show()

X_t["MedInc"].value_counts().sort_index().plot.bar()
plt.xticks(rotation=0)
plt.ylabel("Number of observations")
plt.title("MedInc")
plt.show()

