import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from multiprocessing import freeze_support

def main():
    X = pd.read_csv("occupancy.csv", parse_dates=["date"])
    print(X.shape)
    X.head()

    y = pd.read_csv("occupancy_target.csv", index_col="id")["occupancy"]

    print(y.shape)

    y.head()

    def plot_timeseries(n_id):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

        X[X["id"] == n_id]["temperature"].plot(ax=axes[0, 0], title="temperature")
        X[X["id"] == n_id]["humidity"].plot(ax=axes[0, 1], title="humidity")
        X[X["id"] == n_id]["light"].plot(ax=axes[0, 2], title="light")
        X[X["id"] == n_id]["co2"].plot(ax=axes[1, 0], title="co2")
        X[X["id"] == n_id]["humidity_ratio"].plot(ax=axes[1, 1], title="humidity_ratio")

        plt.show()

    plot_timeseries(2)

    features = extract_features(X[["id", "light"]], column_id="id")
    features.head()

    [f for f in features.columns]

    len([f for f in features.columns if features[f].isnull().mean() > 0.5])

    len([f for f in features.columns if features[f].isnull().mean() == 1])

    feats = features.columns[10:15]

    features[feats].head()

    impute(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.1,
        random_state=42,
    )

    cls = LogisticRegression(random_state=10, C=0.01)
    cls.fit(X_train, y_train)

    print(classification_report(y_test, cls.predict(X_test)))

    features = extract_features(
        X,
        column_id="id",
        impute_function=impute,
        column_sort="date",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.1,
        random_state=42,
    )

    cls = LogisticRegression(random_state=10, C=0.000000000000001)
    cls.fit(X_train, y_train)

    print(classification_report(y_test, cls.predict(X_test)))

if __name__ == "__main__":
    freeze_support()
    main()
