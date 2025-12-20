import pandas as pd
from feature_engine.datetime import DatetimeFeatures

rng_ = pd.date_range("2024-05-17", periods=20, freq="D")

data = pd.DataFrame({"date": rng_})

data.head()

dtfs = DatetimeFeatures(
    variables=None, 
    features_to_extract="all",
)

dft = dtfs.fit_transform(data)

vars_ = [v for v in dft.columns if "date" in v]

dft[vars_].head()

vars_

dtfs.variables_

dtfs = DatetimeFeatures(
    variables=None,  
    features_to_extract=None,
)

dft = dtfs.fit_transform(data)

vars_ = [v for v in dft.columns if "date" in v]

dft[vars_].head()

dtfs = DatetimeFeatures(
    variables=None,  
    features_to_extract=["week", "year", "day_of_month", "day_of_week"],
)

dft = dtfs.fit_transform(data)

vars_ = [v for v in dft.columns if "date" in v]

dft[vars_].head()

df = pd.DataFrame()

df["time"] = pd.concat(
    [
        pd.Series(
            pd.date_range(
                start="2024-08-01 09:00", freq="h", periods=3, tz="Europe/Berlin"
            )
        ),
        pd.Series(
            pd.date_range(
                start="2024-08-01 09:00", freq="h", periods=3, tz="US/Central"
            )
        ),
    ],
    axis=0,
)

df

dfts = DatetimeFeatures(
    features_to_extract=["day_of_week", "hour", "minute"],
    drop_original=False,
    utc=True, 
)

dft = dfts.fit_transform(df)

dft.head()
