import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from feature_engine.outliers import Winsorizer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
)

q05 = X_train.quantile(0.05).to_dict()

q95 = X_train.quantile(0.95).to_dict()

train_t = X_train.clip(lower=q05, upper=q95)
test_t = X_test.clip(lower=q05, upper=q95)

var = 'worst smoothness'
X_train[var].agg(["min", "max", "mean"])

train_t[var].agg(["min", "max", "mean"])

def plot_boxplot_and_hist(data, variable):
    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.50, 0.85)}
    )

    sns.boxplot(x=data[variable], ax=ax_box)
    sns.histplot(data=data, x=variable, ax=ax_hist)
    plt.show()

plot_boxplot_and_hist(X_train, var)

plot_boxplot_and_hist(train_t, var)

capper = Winsorizer(
    variables=["worst smoothness", "worst texture"],
    capping_method="quantiles",
    tail="both",
    fold=0.05,
)

capper.fit(X_train)

capper.left_tail_caps_

capper.right_tail_caps_

train_t = capper.transform(X_train)
test_t = capper.transform(X_test)

X_train[capper.variables_].min(), train_t[capper.variables_].max()

