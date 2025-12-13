import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer

data = pd.read_csv("bag_of_words.csv")
data.head()

data.hist(bins=30, figsize=(20, 20), layout=(3, 4))
plt.show()

data.nunique()

data.describe()

binarizer = Binarizer(threshold=0).set_output(transform="pandas")
data_t = binarizer.fit_transform(data)
data_t.describe()

data_t.hist(figsize=(20, 20), layout=(3, 4))
plt.show()

variables = data_t.columns.to_list()

plt.figure(figsize=(20, 20), constrained_layout=True)

for i in range(10):

    ax = plt.subplot(3, 4, i + 1)

    var = variables[i]

    t = data_t[var].value_counts(normalize=True)
    t.plot.bar(ax=ax)

    plt.xticks(rotation=0)
    plt.ylabel("Observations per bin")

    ax.set_title(var)

plt.show()

