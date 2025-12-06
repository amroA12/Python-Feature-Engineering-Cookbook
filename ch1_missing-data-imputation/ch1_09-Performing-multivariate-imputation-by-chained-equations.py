import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import (enable_iterative_imputer)
from sklearn.impute import (IterativeImputer, SimpleImputer)

variables = ["A2", "A3", "A8", "A11", "A14", "A15", "target"]
data = pd.read_csv("credit_approval_uci_ch1.csv", usecols=variables)

x_train, x_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.3,
    random_state=0,
)

imputer = IterativeImputer(
    estimator = BayesianRidge(), 
    max_iter=10,
    random_state=0
).set_output(transform="pandas")

imputer.fit(x_train)

x_train_t = imputer.transform(x_train)
x_test_t = imputer.transform(x_test)

imputer_simple = SimpleImputer(strategy="mean").set_output(transform="pandas")
x_train_s = imputer_simple.fit_transform(x_train)
x_test_s = imputer_simple.transform(x_test)

fig, axes =plt.subplots(2, 1, figsize = (10, 10), squeeze=False)
x_test_t["A3"].hist(bins = 50, ax=axes[0, 0], color="blue")
x_test_t["A3"].hist(bins = 50, ax=axes[1, 0], color="green")

axes[0, 0].set_ylabel('Number of observations')
axes[1, 0].set_ylabel('Number of observations')

axes[0, 0].set_xlabel('A3')
axes[1, 0].set_xlabel('A3')

axes[0, 0].set_title('MICE')
axes[0, 0].set_title('Mean imputation')

plt.show()
