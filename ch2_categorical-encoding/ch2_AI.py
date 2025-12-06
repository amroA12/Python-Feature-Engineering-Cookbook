# ===============================================================
#   Professional Categorical Encoding Script
#   - Nominal Encoding (OHE)
#   - Ordinal Encoding
#   - Rare Label Encoding
#   - Frequency Encoding
#   - Target Encoding
#   - Leave-One-Out Encoding
# ===============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Feature-engine encoders
from feature_engine.encoding import RareLabelEncoder, CountFrequencyEncoder

# category_encoders library
import category_encoders as ce


# ---------------------------------------------------------------
# (1) Sample dataset
# ---------------------------------------------------------------
df = pd.DataFrame({
    'City': ['Cairo','Dubai','London','Dubai','Cairo','Riyadh','Jeddah'],
    'Color': ['Red','Blue','Green','Red','Red','Green','Blue'],
    'Size': ['Small','Medium','Large','Medium','Small','Large','Large'],
    'Product': ['A','B','C','A','A','D','E'],
    'Churn': [1,0,1,0,1,0,0]
})

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ---------------------------------------------------------------
# (2) Feature groups
# ---------------------------------------------------------------
nominal_features = ['Color']                        # OHE
ordinal_features = ['Size']                         # Ordinal
ordinal_order = [['Small', 'Medium', 'Large']]

rare_features = ['City']                            # RareLabel + TargetEncoding
high_cardinality_features = ['Product']             # Frequency + LOO


# ---------------------------------------------------------------
# (3) Build pipelines for each encoding strategy
# ---------------------------------------------------------------

# A) Rare Label + Target Encoding
rare_target_pipeline = Pipeline([
    ('rare_label', RareLabelEncoder(tol=0.2, n_categories=2)),
    ('target_enc', ce.TargetEncoder())
])

# B) One Hot Encoding for nominal features
ohe_pipeline = Pipeline([
    ('ohe', OneHotEncoder(drop='first', sparse_output=False))
])

# C) Ordinal encoding for ordered categories
ordinal_pipeline = Pipeline([
    ('ordinal', OrdinalEncoder(categories=ordinal_order))
])

# D) Frequency Encoding for high-cardinality vars
freq_pipeline = Pipeline([
    ('freq', CountFrequencyEncoder(encoding_method='frequency'))
])

# E) Leave-One-Out Encoding for high-cardinality vars
loo_pipeline = Pipeline([
    ('loo', ce.LeaveOneOutEncoder())
])


# ---------------------------------------------------------------
# (4) Combine all encoders via ColumnTransformer
# ---------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('rare_target', rare_target_pipeline, rare_features),
        ('onehot', ohe_pipeline, nominal_features),
        ('ordinal', ordinal_pipeline, ordinal_features),
        ('freq',  freq_pipeline, high_cardinality_features),
        ('loo',   loo_pipeline, high_cardinality_features)
    ],
    remainder='drop'
)


# ---------------------------------------------------------------
# (5) Fit + Transform
# ---------------------------------------------------------------
X_train_encoded = preprocessor.fit_transform(X_train, y_train)
X_test_encoded = preprocessor.transform(X_test)


# ---------------------------------------------------------------
# (6) Display encoded results
# ---------------------------------------------------------------
print("\n=== Encoded Training Data ===")
print(pd.DataFrame(X_train_encoded))

print("\n=== Encoded Test Data ===")
print(pd.DataFrame(X_test_encoded))

