import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

sns.set(style="darkgrid")

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X.hist(bins=30, figsize=(12, 12))
plt.show()

plt.figure(figsize=(8, 3))
sns.boxplot(data=X["MedInc"], orient="y")
plt.title("Boxplot")
plt.show()

def plot_boxplot_and_hist(data, variable):
    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.50, 0.85)}
    )

    sns.boxplot(x=data[variable], ax=ax_box)
    sns.histplot(data=data, x=variable, ax=ax_hist)

    plt.show()

plot_boxplot_and_hist(X, "MedInc")

def find_limits(df, variable, fold):
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)

    IQR = q3 - q1

    lower_limit = q1 - (IQR * fold)
    upper_limit = q3 + (IQR * fold)

    return lower_limit, upper_limit

lower_limit, upper_limit = find_limits(X, "MedInc", 1.5)

plot_boxplot_and_hist(X, "HouseAge")

lower_limit, upper_limit = find_limits(X, "HouseAge", 1.5)

plot_boxplot_and_hist(X, "Population")

lower_limit, upper_limit = find_limits(X, "Population", 1.5)