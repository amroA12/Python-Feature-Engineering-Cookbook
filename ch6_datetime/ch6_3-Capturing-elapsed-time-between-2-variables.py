import datetime
import numpy as np
import pandas as pd

date = "2024-05-17"

rng_hr = pd.date_range(date, periods=20, freq="h")
rng_month = pd.date_range(date, periods=20, freq="ME")

df = pd.DataFrame(
    {"date1": rng_hr, "date2": rng_month})  

df.head()

df["elapsed_days"] = (df["date2"] - df["date1"]).dt.days

df.head()

df["weeks_passed"] = (df["date2"] - df["date1"]) / np.timedelta64(1, "W")

df.head()

df["diff_seconds"] = (df["date2"] - df["date1"]) / np.timedelta64(1, "s")
df["diff_minutes"] = (df["date2"] - df["date1"]) / np.timedelta64(1, "m")

df.head()

df["to_today"] = (datetime.datetime.today() - df["date1"]).dt.days

df.head()

import pandas as pd
from feature_engine.datetime import DatetimeSubtraction

date = "2024-05-17"

rng_hr = pd.date_range(date, periods=20, freq="h")
rng_month = pd.date_range(date, periods=20, freq="ME")

df = pd.DataFrame(
    {"date1": rng_hr, "date2": rng_month})  

df.head()

ds = DatetimeSubtraction(
    variables="date2",
    reference="date1",
    output_unit="D",
)

dft = ds.fit_transform(df)

dft.head()

