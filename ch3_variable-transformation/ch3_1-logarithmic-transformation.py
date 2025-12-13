import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import fetch_california_housing

x, y = fetch_california_housing(return_X_y = True, as_frame = True)

x.hist(bins = 30, figsize = (12, 12))
plt.show()

def diagnostic_plots(df, variable) :
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist(bins = 30)
    plt.title(f"Histogram of {variable}")
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.title(f"Q-Q plot of {variable}")
    plt.show()

diagnostic_plots(x, "MedInc")

x_tf = x.copy()

vars = ["MedInc", "AveRooms", "AveBedrms", "Population"]

x_tf[vars] = np.log(x[vars])

diagnostic_plots(x_tf, "MedInc")

from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log, inverse_func=np.exp)

x_tf[vars] = transformer.transform(x[vars])

x_tf[vars] = transformer.inverse_transform(x_tf[vars])

from feature_engine.transformation import LogTransformer

lt = LogTransformer(variables= vars)
lt.fit(x)

x_tf = lt.transform(x)
diagnostic_plots(x_tf, "MedInc")

x_tf = lt.inverse_transform(x_tf)
diagnostic_plots(x_tf, "MedInc")

