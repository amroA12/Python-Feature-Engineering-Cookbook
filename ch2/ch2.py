# Encoding Categorical Variables
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# sklearn encoders
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# feature_engine encoders
from feature_engine.encoding import (
    RareLabelEncoder,
    CountFrequencyEncoder
)

# category_encoders (target, loo, james-stein, woe)
import category_encoders as ce

#-------------------------------------------------------
# 1) Load example dataset
#-------------------------------------------------------
df = pd.DataFrame({
    'City': ['Cairo','Dubai','London','Dubai','Cairo','Riyadh','Jeddah'],
    'Color': ['Red','Blue','Green','Red','Red','Green','Blue'],
    'Size': ['Small','Medium','Large','Medium','Small','Large','Large'],
    'Product': ['A','B','C','A','A','D','E'],
    'Churn': [1,0,1,0,1,0,0]
})

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#-------------------------------------------------------
# 2) Define categorical feature groups
#-------------------------------------------------------

# Nominal variables → One Hot Encoding
nominal_features = ['Color']

# Ordinal variables → Ordinal Encoding
ordinal_features = ['Size']
ordinal_order = [['Small', 'Medium', 'Large']]

# High-cardinality variables → Target Encoding / Frequency Encoding
high_cardinality_features = ['Product']

# Variables requiring rare-label grouping
rare_features = ['City']

#-------------------------------------------------------
# 3) Build individual pipelines
#-------------------------------------------------------

# --- (A) Rare Label Encoding + Target Encoding ---
rare_target_pipeline = Pipeline(steps=[
    ('rare_label', RareLabelEncoder(tol=0.2, n_categories=2)),
    ('target_enc', ce.TargetEncoder())
])

# --- (B) One Hot Encoding ---
ohe_pipeline = Pipeline(steps=[
    ('ohe', OneHotEncoder(drop='first', sparse_output=False))
])

# --- (C) Ordinal Encoding ---
ordinal_pipeline = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=ordinal_order))
])

# --- (D) Frequency Encoding ---
freq_pipeline = Pipeline(steps=[
    ('freq', CountFrequencyEncoder(encoding_method='frequency'))
])

# --- (E) Leave-One-Out Encoding ---
loo_pipeline = Pipeline(steps=[
    ('loo', ce.LeaveOneOutEncoder())
])

#-------------------------------------------------------
# 4) ColumnTransformer to combine all encoders
#-------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[

        # Rare category handling + Target Encoding
        ('rare_target', rare_target_pipeline, rare_features),

        # Standard OHE
        ('onehot', ohe_pipeline, nominal_features),

        # Ordinal encoding
        ('ordinal', ordinal_pipeline, ordinal_features),

        # Frequency encoding for high cardinality variable
        ('freq', freq_pipeline, high_cardinality_features),

        # Leave-One-Out encoding applied on high-cardinality as well
        ('loo', loo_pipeline, high_cardinality_features)

    ],
    remainder='drop'   # drop unused columns
)

#-------------------------------------------------------
# 5) Fit the encoders and transform the dataset
#-------------------------------------------------------

X_train_encoded = preprocessor.fit_transform(X_train, y_train)
X_test_encoded = preprocessor.transform(X_test)

#-------------------------------------------------------
# 6) Display results
#-------------------------------------------------------
print("\n=== Encoded Training Data ===")
print(X_train_encoded)

print("\n=== Encoded Test Data ===")
print(X_test_encoded)


