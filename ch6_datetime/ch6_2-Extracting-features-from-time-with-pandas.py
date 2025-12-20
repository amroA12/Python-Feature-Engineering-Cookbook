import numpy as np
import pandas as pd

rng_ = pd.date_range("2024-05-17", periods=20, freq="1h15min10s")
df = pd.DataFrame({"date": rng_})
df.head()

df["hour"] = df["date"].dt.hour
df["min"] = df["date"].dt.minute
df["sec"] = df["date"].dt.second
df.head()

df[["h", "m", "s"]] = pd.DataFrame([(x.hour, x.minute, x.second) for x in df["date"]])
df.head()

df["hour"].unique()

df["is_morning"] = np.where((df["hour"] < 12) & (df["hour"] > 6), 1, 0)
df.head()

