import pandas as pd
from feature_engine.creation import MathFeatures
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

df.head()

features = [
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
]
df[features].head()
df["mean_features"] = df[features].mean(axis=1)
df["mean_features"].head()
df["std_features"] = df[features].std(axis=1)
df["std_features"].head()

math_func = ["sum", "prod", "mean", "std", "max", "min"]

df_t = df[features].agg(math_func, axis="columns")
df_t.head()

new_feature_names = ["sum_f", "prod_f", "mean_f", "std_f", "max_f", "min_f"]

create = MathFeatures(
    variables=features,
    func=math_func,
    new_variables_names=new_feature_names,
)

df_t = create.fit_transform(df)
df_t[features + new_feature_names].head()

