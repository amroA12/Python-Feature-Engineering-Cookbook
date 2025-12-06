# missing_data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

#-------------------------------------------------------
# (1) Generate synthetic dataset
#-------------------------------------------------------
rng = np.random.RandomState(42)
n = 1000

age = rng.normal(40,12, size=n).clip(18,90).round().astype(float)
salary = (age * 1000 + rng.normal(0,2000, size=n)).astype(float)
score = rng.normal(50 + 0.2 * age , 10 , size=n)
cities = np.array(['Cairo' , 'Giza' , 'Alexandria' , 'Suze'])
city = rng.choice(cities , size=n , p=[0.5 , 0.2 , 0.2 , 0.1])

# Logistic model to generate target probability 
logit = -5 + 0.0002 * salary + 0.05 * (score - 50)
prob = 1 / (1 + np.exp(-logit))
target = (rng.rand(n)< prob).astype(int)

df = pd.DataFrame({
    'age': age,
    'salary': salary,
    'score': score,
    'city': city,
    'target': target
})

#-------------------------------------------------------
# (2) Introduce Missingness: MCAR - MAR - MNAR
#-------------------------------------------------------

# MCAR: random missing in score
mcar_idx = rng.choice(n , size=int(0.1 * n), replace = False)
df.loc[mcar_idx, 'score'] = np.nan

# MAR: salary missing more when city = "Suze"
prob_miss_salary = np.where(df['city'] == 'Suze' , 0.4 , 0.05)
miss_salary = rng.rand(n) < prob_miss_salary
df.loc[miss_salary, 'salary'] = np.nan

# MNAR: age missing depending on age values 
prob_miss_age = 1 / (1 + np.exp(-(df['age']-60) / 5))
miss_age = rng.rand(n) < prob_miss_age * 0.3
df.loc[miss_age, 'age'] =np.nan

#Missing in categorical varible
cat_miss_idx = rng.choice(n, size=int(0.08 * n), replace=False)
df.loc[cat_miss_idx, 'city'] = np.nan

#-------------------------------------------------------
# (3) Dispaly missingness summary
#-------------------------------------------------------
print("Missingness ratio per column:")
print(df.isnull().mean().round(3))

#-------------------------------------------------------
# (4) Train/Test split
#-------------------------------------------------------
X = df.drop('target', axis = 1)
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

numeric_features = ['age' , 'salary' , 'score']
categorical_features = ['city']

#-------------------------------------------------------
# (5) Pipeline Builder Function
#-------------------------------------------------------
def build_pipeline(imputer):
    num_pipe = Pipeline([
        ('imputer', imputer),
        ('scaler',StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    pre = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ], remainder='drop')

    pipe = Pipeline([
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    return pipe

#-------------------------------------------------------
# (6) Define all imputation methods
#-------------------------------------------------------
Pipelines = {
    'Median': build_pipeline(SimpleImputer(strategy='median', add_indicator=False)),
    'Median+Indicator': build_pipeline(SimpleImputer(strategy='median', add_indicator=True)),
    'KNN(k=5)': build_pipeline(KNNImputer(n_neighbors=5)),
    'Iteerative (MICE)': build_pipeline(IterativeImputer(random_state=42))
}

#-------------------------------------------------------
# (7) Cross-validation + evaluation
#-------------------------------------------------------
cv  = StratifiedKFold(n_splits=5 , shuffle=True , random_state=42)
results = []

for name, pipe in Pipelines.items():
    sc = cross_val_score(pipe, X_train , Y_train , cv=cv, scoring='accuracy', n_jobs=-1)
    pipe.fit(X_train, Y_train)
    preds = pipe.predict(X_test)
    test_acc = accuracy_score(Y_test, preds)

    results.append({
        'method': name,
        'cv_mean': sc.mean(),
        'cv_std': sc.std(),
        'test_accuracy': test_acc
    })

res_df = pd.DataFrame(results).round(4)
print("/nEvaluation Results:")
print(res_df)



