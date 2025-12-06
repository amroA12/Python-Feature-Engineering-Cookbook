import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("air_passengers.csv", parse_dates=["ds"], index_col=["ds"])

df_imputed = df.interpolate(method="linear")

ax = df_imputed.plot(linestyle ="-", marker=".", figsize=[10, 5])
df_imputed[df.isnull()].plot(ax= ax, legend=None, marker=".", color = "r")
ax.set_title("Air passengers")
ax.set_ylabel("Name of passengers")
ax.set_xlabel("Time")
plt.show()

df_imputed = df.interpolate(method="spline", order = 2)
ax = df_imputed.plot(linestyle ="-", marker=".", figsize=[10, 5])
df_imputed[df.isnull()].plot(ax= ax, legend=None, marker=".", color = "r")
ax.set_title("Air passengers")
ax.set_ylabel("Name of passengers")
ax.set_xlabel("Time")
plt.show()
