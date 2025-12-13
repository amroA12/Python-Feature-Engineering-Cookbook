import numpy as np
import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PowerTransformer
from feature_engine.transformation import YeoJohnsonTransformer

x, y = fetch_california_housing(return_X_y=True, as_frame=True)

x.drop(labels =["Latitude", "Longitude"], axis = 1 , inplace = True)

transformer = PowerTransformer(method="yeo-johnson", standardize = False).set_output(transform="pandas")
transformer.fit(x)

x_tf = transformer.transform(x)

x_tf.hist(bins=30, figsize = (12, 12), layout= (3, 3))
plt.show()

yjt = YeoJohnsonTransformer()
yjt.fit(x)

x = yjt.transform(x)

yjt.lambda_dict_

