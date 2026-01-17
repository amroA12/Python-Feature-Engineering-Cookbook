import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tsfresh.transformers import RelevantFeatureAugmenter
from multiprocessing import freeze_support

def main():
    X = pd.read_csv("occupancy.csv", parse_dates=["date"])

    y = pd.read_csv("occupancy_target.csv", index_col="id")["occupancy"]
    tmp = pd.DataFrame(index=y.index)

    X_train, X_test, y_train, y_test = train_test_split(tmp, y, random_state=0)

    kind_to_fc_parameters = {
        "light": {
            "c3": [{"lag": 3}, {"lag": 2}, {"lag": 1}],
            "abs_energy": None,
            "sum_values": None,
            "fft_coefficient": [{"attr": "real", "coeff": 0}, {"attr": "abs", "coeff": 0}],
            "spkt_welch_density": [{"coeff": 2}, {"coeff": 5}, {"coeff": 8}],
            "agg_linear_trend": [
                {"attr": "intercept", "chunk_len": 50, "f_agg": "var"},
                {"attr": "slope", "chunk_len": 50, "f_agg": "var"},
            ],
            "change_quantiles": [
                {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.8},
                {"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.8},
            ],
        },
        "co2": {
            "fft_coefficient": [{"attr": "real", "coeff": 0}, {"attr": "abs", "coeff": 0}],
            "c3": [{"lag": 3}, {"lag": 2}, {"lag": 1}],
            "sum_values": None,
            "abs_energy": None,
            "sum_of_reoccurring_data_points": None,
            "sum_of_reoccurring_values": None,
        },
        "temperature": {"c3": [{"lag": 1}, {"lag": 2}, {"lag": 3}], "abs_energy": None},
    }

    augmenter = RelevantFeatureAugmenter(
        column_id="id",
        column_sort="date",
        kind_to_fc_parameters=kind_to_fc_parameters,
    )

    pipe = Pipeline(
        [
            ("augmenter", augmenter),
            ("classifier", LogisticRegression(random_state=10, C=0.01)),
        ]
    )
    pipe.set_params(augmenter__timeseries_container=X)

    pipe.fit(X_train, y_train)

    print(classification_report(y_test, pipe.predict(X_test)))

    pipe.fit(X_train, y_train)

if __name__ == "__main__":
    freeze_support()
    main()

