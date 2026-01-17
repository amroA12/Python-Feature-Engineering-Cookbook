import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tsfresh import (
    extract_features,
    extract_relevant_features,
    select_features,
)
from tsfresh.utilities.dataframe_functions import impute
from multiprocessing import freeze_support

def main():
    X = pd.read_csv("occupancy.csv", parse_dates=["date"])
    y = pd.read_csv("occupancy_target.csv", index_col="id")["occupancy"]

    features = extract_features(
        X[["id", "light"]],
        column_id="id",
        impute_function=impute,
    )

    features.shape
    features.shape, y.shape

    features = select_features(features, y)

    len(features)
    features.head()

    feats = features.columns[0:5]

    features[feats].head()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.1,
        random_state=42,
    )

    cls = LogisticRegression(random_state=10, C=0.1, max_iter=1000)
    cls.fit(X_train, y_train)

    print(classification_report(y_test, cls.predict(X_test)))

    features = extract_relevant_features(
        X,
        y,
        column_id="id",
        column_sort="date",
    )

    features.shape

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

