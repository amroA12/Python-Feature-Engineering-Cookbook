import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction import settings
from multiprocessing import freeze_support

def main():
    X = pd.read_csv("occupancy.csv", parse_dates=["date"])
    y = pd.read_csv("occupancy_target.csv", index_col="id")["occupancy"]

    features = extract_relevant_features(
        X,
        y,
        column_id="id",
        column_sort="date",
    )

    features.shape

    cls = LogisticRegression(
        penalty="l1", 
        solver="liblinear",
        random_state=10,
        C=0.05,
        max_iter=1000,
    )

    selector = SelectFromModel(cls)

    selector.fit(features, y)

    features = selector.get_feature_names_out()

    kind_to_fc_parameters = settings.from_columns(selector.get_feature_names_out())

    features = extract_features(
        X,
        column_id="id",
        column_sort="date",
        kind_to_fc_parameters=kind_to_fc_parameters,
    )

    features.shape

    features.head()
    
if __name__ == "__main__":
    freeze_support()
    main()

