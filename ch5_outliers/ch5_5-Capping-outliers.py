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

def find_limits(df, variable, fold):
    var_mean = df[variable].mean()
    var_std = df[variable].std()
    lower_limit = var_mean - fold * var_std
    upper_limit = var_mean + fold * var_std
    return lower_limit, upper_limit

var = "worst smoothness"

lower_limit, upper_limit = find_limits(X_train, var, 3)

train_t = X_train.copy()
test_t = X_test.copy()

train_t[var] = train_t[var].clip(
    lower=lower_limit, upper=upper_limit)

test_t[var] = test_t[var].clip(
    lower=lower_limit, upper=upper_limit)

X_train[var].agg(["min", "max"])

train_t["worst smoothness"].agg(["min", "max"])

capper = Winsorizer(
    variables=["worst smoothness", "worst texture"],
    capping_method="gaussian",
    tail="both",
    fold=3,
)

capper.fit(X_train)

capper.left_tail_caps_

capper.right_tail_caps_

train_t = capper.transform(X_train)
test_t = capper.transform(X_test)

print(X_train[capper.variables_].agg(["min", "max"]))

print(train_t[capper.variables_].agg(["min", "max"]))

