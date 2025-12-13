import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import FunctionTransformer
from feature_engine.transformation import PowerTransformer

x, y = fetch_california_housing(return_X_y=True, as_frame=True)

def diagonstic_plots(df, variable):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    df[variable].hist(bins=30)
    plt.title(f"Histogram of {variable}")
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.title(f"Q-Q plot of {variable}")
    plt.show()

diagonstic_plots(x, "Population")

variables = ["MedInc", "Population"]

x_tf = x.copy()
x_tf[variables] =np.power(x[variables], 0.3)

diagonstic_plots(x_tf, "Population")

transformer = FunctionTransformer(lambda x: np.power(x, 0.3))

x_tf = x.copy()
x_tf[variables] = transformer.transform(x[variables])

power_t = PowerTransformer(variables=variables, exp=0.3)
power_t.fit(x)

x_tf = power_t.transform(x)
