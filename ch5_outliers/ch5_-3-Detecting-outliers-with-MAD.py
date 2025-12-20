import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

def plot_boxplot_and_hist(data, variable):
    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.50, 0.85)}
    )

    sns.boxplot(x=data[variable], ax=ax_box)
    sns.histplot(data=data, x=variable, ax=ax_hist)

    plt.show()

plot_boxplot_and_hist(X, "mean smoothness")

def find_limits(df, variable, fold):
    median = df[variable].median()
    center = df[variable] - median
    MAD = center.abs().median() * 1.4826
    lower_limit = median - fold * MAD
    upper_limit = median + fold * MAD
    return lower_limit, upper_limit

lower_limit, upper_limit = find_limits(X, "mean smoothness", 3)

outliers = np.where(
    (X["mean smoothness"] > upper_limit) | 
    (X["mean smoothness"] < lower_limit),
    True,
    False,
)

outliers.sum()

def plot_boxplot_and_hist(data, variable):
    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.50, 0.85)}
    )

    sns.boxplot(x=data[variable], ax=ax_box)
    sns.histplot(data=data, x=variable, ax=ax_hist)
    
    plt.vlines(x=lower_limit, ymin=0, ymax=80, color='r')
    plt.vlines(x=upper_limit, ymin=0, ymax=80, color='r')

    plt.show()

plot_boxplot_and_hist(X, "mean smoothness")

lower_limit, upper_limit = find_limits(X, "worst texture", 3)

outliers = np.where(
    (X["worst texture"] > upper_limit) |
    (X["worst texture"] < lower_limit),
    True,
    False,
)

outliers.sum()

plot_boxplot_and_hist(X, "worst texture")

