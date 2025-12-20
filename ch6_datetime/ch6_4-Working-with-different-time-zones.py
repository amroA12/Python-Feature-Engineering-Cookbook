import pandas as pd

df = pd.DataFrame()

df["time1"] = pd.concat(
    [
        pd.Series(
            pd.date_range(
                start="2024-06-10 09:00", freq="h", periods=3, tz="Europe/Berlin"
            )
        ),
        pd.Series(
            pd.date_range(
                start="2024-09-10 09:00", freq="h", periods=3, tz="US/Central"
            )
        ),
    ],
    axis=0,
)

df

df["time2"] = pd.concat(
    [
        pd.Series(
            pd.date_range(
                start="2024-07-01 09:00", freq="h", periods=3, tz="Europe/Berlin"
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

df["time1_utc"] = pd.to_datetime(df["time1"], utc=True)
df["time2_utc"] = pd.to_datetime(df["time2"], utc=True)

df

df["elapsed_days"] = (df["time2_utc"] - df["time1_utc"]).dt.days

df["elapsed_days"].head()

df["time1_london"] = df["time1_utc"].dt.tz_convert("Europe/London")
df["time2_berlin"] = df["time1_utc"].dt.tz_convert("Europe/Berlin")

df[["time1_london", "time2_berlin"]]

