import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

data = pd.read_csv("bag_of_words (1).csv")

data.head()

scaler = MaxAbsScaler().set_output(transform="pandas")

scaler.fit(data)

data_scaled = scaler.transform(data)

scaler.max_abs_
data_scaled.head()
data.describe()
data_scaled.describe()

plt.rcParams["figure.dpi"] = 60

data.hist(bins=20, figsize=(20, 20))
plt.show()

data_scaled.hist(bins=20, figsize=(20, 20))
plt.show()

