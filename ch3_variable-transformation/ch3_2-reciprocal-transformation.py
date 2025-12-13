import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import fetch_california_housing

x, y = fetch_california_housing(return_X_y=True, as_frame=True)

def diagnostic_plots(df, variable) :
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist(bins = 30)
    plt.title(f"Histogram of {variable}")
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.title(f"Q-Q plot of {variable}")
    plt.show()

diagnostic_plots(x, "AveOccup")

x_tf = x.copy()

x_tf["AveOccup"] = np.reciprocal(x_tf["AveOccup"])

diagnostic_plots(x_tf, "AveOccup")

from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.reciprocal)

x_tf = x.copy()
x_tf["AveOccup"] = transformer.transform(x["AveOccup"])

from feature_engine.transformation import ReciprocalTransformer

rt = ReciprocalTransformer(variables="AveOccup")
rt.fit(x)

x_tf = rt.transform(x)

diagnostic_plots(x_tf, "AveOccup")
