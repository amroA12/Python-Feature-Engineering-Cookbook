import numpy as np
import pandas as pd

rng_ = pd.date_range("2024-05-17", periods=20, freq="D")

data = pd.DataFrame({"date": rng_})

data.head()

data["year"] = data["date"].dt.year
data[["date", "year"]].head()

data["quarter"] = data["date"].dt.quarter
data[["date", "quarter"]].head()


data["semester"] = np.where(data["quarter"] < 3, 1, 2)
data[["semester", "quarter"]].head()

data["month"] = data["date"].dt.month
data[["date", "month"]].head()

data["week"] = data["date"].dt.isocalendar().week
data[["date", "week"]].head()

data["day_mo"] = data["date"].dt.day
data[["date", "day_mo"]].head()

data["day_week"] = data["date"].dt.dayofweek
data[["date", "day_mo", "day_week"]].head()

data["is_weekend"] = (data["date"].dt.dayofweek > 4).astype(int)
data[["date", "day_week", "is_weekend"]].head()

