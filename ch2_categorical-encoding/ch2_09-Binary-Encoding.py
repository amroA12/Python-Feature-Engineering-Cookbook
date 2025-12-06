import pandas as pd 
from sklearn.model_selection import train_test_split
from category_encoders.binary import BinaryEncoder

data = pd.read_csv("credit_approval_uci_ch2.csv")
x_train, x_test, y_train, y_test = train_test_split(
    data.drop(labels=["target"], axis=1),
    data["target"],
    test_size=0.3,
    random_state=0
)

x_train["A7"].unique()

encoder = BinaryEncoder(cols=["A7"], drop_invariant=True)

encoder.fit(x_train)

x_train_enc = encoder.transform(x_train)
x_test_enc = encoder.transform(x_test)

print("Unique values in A7:")
print(x_train["A7"].unique())

print("Shape before encoding:", x_train.shape)
print("Shape after encoding:", x_train_enc.shape)

print("First 10 rows of x_train after encoding:")
print(x_train_enc.head(10))
