import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame([i for i in range(24)], columns=["hour"])

df.head()

df["hour_sin"] = np.sin(df["hour"] / df["hour"].max() * 2 * np.pi)

df["hour_cos"] = np.cos(df["hour"] / df["hour"].max() * 2 * np.pi)

df.head()

plt.rcParams["figure.dpi"] = 90

plt.scatter(df["hour"], df["hour_sin"])

plt.ylabel("Sine of hour")
plt.xlabel("Hour")
plt.title("Sine transformation")
plt.show()

plt.scatter(df["hour"], df["hour_cos"])

plt.ylabel("Cosine of hour")
plt.xlabel("Hour")
plt.title("Cosine transformation")
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
sp = ax.scatter(df["hour_sin"], df["hour_cos"], c=df["hour"])
ax.set(
    xlabel="sin(hour)",
    ylabel="cos(hour)",
)
_ = fig.colorbar(sp)
plt.show()

from feature_engine.creation import CyclicalFeatures

df = pd.DataFrame()
df["hour"] = pd.Series([i for i in range(24)])
df["month"] = pd.Series([i for i in range(1, 13)] * 2)
df["week"] = pd.Series([i for i in range(7)] * 4)

df.head()

df.max()

cyclic = CyclicalFeatures(
    variables=None,
    drop_original=False,
)

dft = cyclic.fit_transform(df)

dft.head()

cyclic.max_values_

