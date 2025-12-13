import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

df = pd.DataFrame()
df["counts1"] = stats.poisson.rvs(mu= 3, size = 10000)
df["counts2"] = stats.poisson.rvs(mu = 2 , size = 10000)

def diagnostic_plots(df, variable) :
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist(bins = 30)
    plt.title(f"Histogram of {variable}")
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.title(f"Q-Q plot of {variable}")
    plt.show()

diagnostic_plots(df, "counts1")

df_tf = df.copy()

df_tf[["counts1", "counts2"]] = np.sqrt(df[["counts1", "counts2"]])

df_tf[["counts1", "counts2"]] = np.round(df[["counts1", "counts2"]], 2)

diagnostic_plots(df_tf, "counts1")

from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.sqrt).set_output(transform="pandas")

df_tf = df.copy()
df_tf = transformer.transform(df)

from feature_engine.transformation import PowerTransformer
root_t = PowerTransformer(exp=1/2)

root_t.fit(df)

df_tf = root_t.transform(df)
