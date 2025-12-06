import pandas as pd 
from sklearn.model_selection import train_test_split
from feature_engine.imputation import EndTailImputer

data = pd.read_csv("credit_approval_uci_ch1.csv")

numeric_vars = [
    var for var in data.select_dtypes(exclude="O").columns.to_list()
    if var !="target"
]

x_train, x_test, y_train, y_test = train_test_split(
    data[numeric_vars],
    data["target"],
    test_size=0.3,
    random_state=0,
)

IQR = x_train.quantile(0.75) - x_train.quantile(0.25)

imputation_dict = (
    x_train.quantile(0.75) + 1.5 * IQR).to_dict()

x_train_t = x_train.fillna(value = imputation_dict)
x_test_t = x_test.fillna(value = imputation_dict)

imputer = EndTailImputer(
    imputation_method= "iqr",
    tail= "right",
    fold=3,
    variables=None,
)

imputer.fit(x_train)

imputer.imputer_dict_

x_train = imputer.transform(x_train)
x_test = imputer.transform(x_test)

print("IQR:\n", IQR)
print("\nImputation Dictionary (Manual):\n", imputation_dict)

print("\nAfter Manual Imputation - x_train_t head:\n", x_train_t.head())
print("\nAfter Manual Imputation - x_test_t head:\n", x_test_t.head())

print("\nEndTailImputer Learned Dictionary:\n", imputer.imputer_dict_)

print("\nTransformed x_train head:\n", x_train.head())
print("\nTransformed x_test head:\n", x_test.head())

