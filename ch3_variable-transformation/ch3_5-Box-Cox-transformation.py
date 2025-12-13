import numpy as np
import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PowerTransformer
from feature_engine.transformation import BoxCoxTransformer

x, y = fetch_california_housing(return_X_y=True, as_frame=True)

x.drop(labels =["Latitude", "Longitude"], axis = 1 , inplace = True)

x.hist(bins = 30, figsize = (12, 12), layout = (3, 3))
plt.show()

variables = list(x.columns)

def make_qqplot(df):
    
    plt.figure(figsize=(10, 6), constrained_layout=True)

    for i in range(6):

        # location in figure
        ax = plt.subplot(2, 3, i + 1)

        # variable to plot
        var = variables[i]

        # q-q plot
        stats.probplot((df[var]), dist="norm", plot=plt)

        # add variable name as title
        ax.set_title(var)

    plt.show()

make_qqplot(x)

transformer = PowerTransformer(method="box-cox", standardize = False).set_output(transform="pandas")
transformer.fit(x)

x_tf = transformer.transform(x)

x_tf.hist(bins=30, figsize = (12, 12), layout= (3, 3))
plt.show()

make_qqplot(x_tf)

bot = BoxCoxTransformer()
bot.fit(x)

x_tf = bot.transform(x)

bot.lambda_dict_

