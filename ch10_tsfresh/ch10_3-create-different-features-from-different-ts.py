import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import settings
from multiprocessing import freeze_support

def main():
    X = pd.read_csv("occupancy.csv", parse_dates=["date"])

    y = pd.read_csv("occupancy_target.csv", index_col="id")["occupancy"]

    minimal_feat = settings.MinimalFCParameters()

    minimal_feat.items()
    minimal_feat

    features = extract_features(
        X[["id", "light"]],
        column_id="id",
        default_fc_parameters=minimal_feat,
    )

    features.shape
    features.head()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.1,
        random_state=42,
    )

    cls = LogisticRegression(random_state=10, C=0.01)
    cls.fit(X_train, y_train)

    print(classification_report(y_test, cls.predict(X_test)))

    efficient_feat = settings.EfficientFCParameters()

    len(efficient_feat.items())

    comprehensive_feat = settings.ComprehensiveFCParameters()

    len(comprehensive_feat.items())

    light_feat = {
        "sum_values": None,
        "median": None,
        "standard_deviation": None,
        "quantile": [{"q": 0.2}, {"q": 0.7}],
    }

    co2_feat = {"root_mean_square": None, "number_peaks": [{"n": 1}, {"n": 2}]}
    kind_to_fc_parameters = {
        "light": light_feat,
        "co2": co2_feat,
    }

    features = extract_features(
        X[["id", "light", "co2"]],
        column_id="id",
        kind_to_fc_parameters=kind_to_fc_parameters,
    )

    features.shape
    features.columns
    features.head()

if __name__ == "__main__":
    freeze_support()
    main()

