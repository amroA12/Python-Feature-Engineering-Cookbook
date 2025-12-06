import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("air_passengers.csv", parse_dates=["ds"], index_col=["ds"])

print(df.head())

ax = df.plot(mark_right=".", figsize=[10, 5], legend=None)
ax.set_title("Air passengers")
ax.set_ylabel("Name of passengers")
ax.set_xlabel("Time")

df_imputed = df.ffill()

plt.show()

ax = df_imputed.plot(linestyle="-", marker=".", figsize=[10, 5])
df_imputed[df.isnull()].plot(ax= ax, legend=None, marker=".", color = "r")
ax.set_title("Air passengers")
ax.set_ylabel("Number of passengers 1")
ax.set_xlabel("Time")

df_imputed = df.bfill()

plt.show()

ax = df_imputed.plot(linestyle="-", marker=".", figsize=[10, 5])
df_imputed[df.isnull()].plot(ax= ax, legend=None, marker=".", color = "r")
ax.set_title("Air passengers")
ax.set_ylabel("Number of passengers 2")
ax.set_xlabel("Time")

df_imputed = df.bfill()

plt.show()
