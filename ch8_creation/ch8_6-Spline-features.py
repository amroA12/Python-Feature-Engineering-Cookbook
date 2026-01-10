import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

X = np.linspace(-1, 11, 20)
y = np.sin(X)
plt.rcParams["figure.dpi"] = 90

plt.plot(X, y)
plt.ylabel("y")
plt.xlabel("X")
plt.show()

linmod = Ridge(random_state=10)
linmod.fit(X.reshape(-1, 1), y)
pred = linmod.predict(X.reshape(-1, 1))

plt.plot(X, y)
plt.plot(X, pred)
plt.ylabel("y")
plt.xlabel("X")
plt.legend(["y", "linear"], bbox_to_anchor=(1, 1), loc="upper left")
plt.show()


spl = SplineTransformer(degree=3, n_knots=5)

X_t = spl.fit_transform(X.reshape(-1, 1))

X_df = pd.DataFrame(X_t, columns=spl.get_feature_names_out(["var"]))

X_df.head()

plt.plot(X, X_t)
plt.legend(spl.get_feature_names_out(["var"]), bbox_to_anchor=(1, 1), loc="upper left")
plt.xlabel("X")
plt.ylabel("Splines values")
plt.title("Splines")
plt.show()

linmod = Ridge(random_state=10)
linmod.fit(X_t, y)
pred = linmod.predict(X_t)

plt.plot(X, y)
plt.plot(X, pred)
plt.ylabel("y")
plt.xlabel("X")
plt.legend(["y", "splines"], bbox_to_anchor=(1, 1), loc="upper left")

from sklearn.datasets import fetch_california_housing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X.drop(["Latitude", "Longitude"], axis=1, inplace=True)

X.hist(bins=20, figsize=(10, 10))
plt.show()

linmod = Ridge(random_state=10)

cv = cross_validate(linmod, X, y)

mean_, std_ = np.mean(cv["test_score"]), np.std(cv["test_score"])

print(f"Model score: {mean_} +- {std_}")

spl = SplineTransformer(degree=3, n_knots=50)

ct = ColumnTransformer(
    [("splines", spl, ["AveRooms", "AveBedrms", "Population", "AveOccup"])],
    remainder="passthrough",
)
ct.fit(X, y)

cv = cross_validate(linmod, ct.transform(X), y)

mean_, std_ = np.mean(cv["test_score"]), np.std(cv["test_score"])

print(f"Model score: {mean_} +- {std_}")

ct.get_feature_names_out()

