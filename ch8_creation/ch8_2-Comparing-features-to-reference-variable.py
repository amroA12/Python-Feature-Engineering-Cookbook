import pandas as pd
from feature_engine.creation import RelativeFeatures
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

df.head()

df["difference"] = df["worst compactness"].sub(df["mean compactness"])
df["difference"].head()

df["difference"] = df["worst compactness"] - (df["mean compactness"])
df["difference"].head()

df["quotient"] = df["worst radius"].div(df["mean radius"])
df["quotient"].head()

df["quotient"] = df["worst radius"] / (df["mean radius"])
df["quotient"].head()

features = ["mean smoothness", "mean compactness", "mean concavity", "mean symmetry"]

reference = ["mean radius", "mean area"]

creator = RelativeFeatures(
    variables=features,
    reference=reference,
    func=["sub", "div"],
)

df_t = creator.fit_transform(df)

new_features = [f for f in df_t.columns if f not in creator.feature_names_in_]

df_t[new_features].head()

