from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from feature_engine.outliers import OutlierTrimmer

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0,
)

def find_limits(df, variable, fold):
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)

    IQR = q3 - q1

    lower_limit = q1 - (IQR * fold)
    upper_limit = q3 + (IQR * fold)

    return lower_limit, upper_limit

lower_limit, upper_limit = find_limits(X_train, "MedInc", 3)

inliers = X_train["MedInc"].ge(lower_limit)
train_t = X_train.loc[inliers]

inliers = X_test["MedInc"].ge(lower_limit)
test_t = X_test.loc[inliers]

inliers = X_train["MedInc"].le(upper_limit)
train_t = X_train.loc[inliers]

inliers = X_test["MedInc"].le(upper_limit)
test_t = X_test.loc[inliers]

trimmer = OutlierTrimmer(
    variables=["MedInc", "HouseAge", "Population"],
    capping_method="iqr",
    tail="both",
    fold=1.5,
)

trimmer.fit(X_train)

trimmer.left_tail_caps_

trimmer.right_tail_caps_

train_t = trimmer.transform(X_train)
test_t = trimmer.transform(X_test)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot_and_hist(data, variable):
    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.50, 0.85)}
    )

    sns.boxplot(x=data[variable], ax=ax_box)
    sns.histplot(data=data, x=variable, ax=ax_hist)
    plt.show()

plot_boxplot_and_hist(X_train, "MedInc")

plot_boxplot_and_hist(train_t, "MedInc")