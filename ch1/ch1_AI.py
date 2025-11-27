# ===============================================================
#   Professional Missing Data Handling with Pipelines
#   - Synthetic dataset generation
#   - Introduce MCAR, MAR, MNAR missingness
#   - Evaluate multiple imputation strategies
# ===============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# -------------------------------------------------------
# (1) Generate synthetic dataset
# -------------------------------------------------------
def generate_dataset(n=1000, random_state=42):
    rng = np.random.RandomState(random_state)

    age = rng.normal(40, 12, size=n).clip(18, 90).round().astype(float)
    salary = (age * 1000 + rng.normal(0, 2000, size=n)).astype(float)
    score = rng.normal(50 + 0.2 * age, 10, size=n)
    cities = np.array(['Cairo', 'Giza', 'Alexandria', 'Suze'])
    city = rng.choice(cities, size=n, p=[0.5, 0.2, 0.2, 0.1])

    # Logistic model to generate target probability
    logit = -5 + 0.0002 * salary + 0.05 * (score - 50)
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        'age': age,
        'salary': salary,
        'score': score,
        'city': city,
        'target': target
    })

    # Introduce Missingness: MCAR, MAR, MNAR
    # MCAR: random missing in score
    mcar_idx = rng.choice(n, size=int(0.1 * n), replace=False)
    df.loc[mcar_idx, 'score'] = np.nan

    # MAR: salary missing more when city = "Suze"
    prob_miss_salary = np.where(df['city'] == 'Suze', 0.4, 0.05)
    miss_salary = rng.rand(n) < prob_miss_salary
    df.loc[miss_salary, 'salary'] = np.nan

    # MNAR: age missing depending on age values
    prob_miss_age = 1 / (1 + np.exp(-(df['age'] - 60) / 5))
    miss_age = rng.rand(n) < prob_miss_age * 0.3
    df.loc[miss_age, 'age'] = np.nan

    # Missing in categorical variable
    cat_miss_idx = rng.choice(n, size=int(0.08 * n), replace=False)
    df.loc[cat_miss_idx, 'city'] = np.nan

    return df


# -------------------------------------------------------
# (2) Display missingness summary
# -------------------------------------------------------
def display_missingness(df):
    print("\n=== Missingness ratio per column ===")
    print(df.isnull().mean().round(3))


# -------------------------------------------------------
# (3) Build imputation + scaling + modeling pipeline
# -------------------------------------------------------
def build_pipeline(imputer):
    numeric_features = ['age', 'salary', 'score']
    categorical_features = ['city']

    # Numeric pipeline: impute + scale
    num_pipe = Pipeline([
        ('imputer', imputer),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: impute + one-hot encoding
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine numeric and categorical
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ], remainder='drop')

    # Full pipeline: preprocessing + classifier
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced'))
    ])

    return pipe


# -------------------------------------------------------
# (4) Evaluate multiple imputation methods
# -------------------------------------------------------
def evaluate_imputers(X_train, Y_train, X_test, Y_test):
    pipelines = {
        'Median': build_pipeline(SimpleImputer(strategy='median', add_indicator=False)),
        'Median+Indicator': build_pipeline(SimpleImputer(strategy='median', add_indicator=True)),
        'KNN(k=5)': build_pipeline(KNNImputer(n_neighbors=5)),
        'Iterative (MICE)': build_pipeline(IterativeImputer(random_state=42))
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, pipe in pipelines.items():
        # Cross-validation
        cv_scores = cross_val_score(pipe, X_train, Y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        # Fit and predict
        pipe.fit(X_train, Y_train)
        preds = pipe.predict(X_test)
        prob_preds = pipe.predict_proba(X_test)[:, 1]

        # Evaluation metrics
        acc = accuracy_score(Y_test, preds)
        roc_auc = roc_auc_score(Y_test, prob_preds)

        results.append({
            'method': name,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'test_accuracy': acc,
            'test_roc_auc': roc_auc
        })

        print(f"\n=== Classification Report ({name}) ===")
        print(classification_report(Y_test, preds))

    res_df = pd.DataFrame(results).round(4)
    print("\n=== Evaluation Results Summary ===")
    print(res_df)


# -------------------------------------------------------
# (5) Main script
# -------------------------------------------------------
if __name__ == "__main__":
    df = generate_dataset(n=1000, random_state=42)
    display_missingness(df)

    X = df.drop('target', axis=1)
    Y = df['target']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42
    )

    evaluate_imputers(X_train, Y_train, X_test, Y_test)

