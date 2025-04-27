import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tqdm import tqdm

# 1. Load your prepared data
X_float_train = pd.read_csv('A5_train_float.csv')
X_cat_train = pd.read_csv('A5_train_category.csv')
X_train = pd.concat([X_float_train, X_cat_train], axis=1)
y_train = pd.read_csv('y_train_mapped.csv', index_col=0)['target']


# 2. Define columns
float_cols = X_float_train.columns.tolist()
cat_cols = X_cat_train.columns.tolist()

# 3. Build pipelines
float_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('float', float_pipeline, float_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 4. Define models to compare
models = {
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# 5. Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6. Compare models
results = {}

print("Running cross-validation for each model...\n")
for name, model in tqdm(models.items()):
    bcr_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Full pipeline: preprocessing + model
        pipe = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])
        
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_val)
        bcr = balanced_accuracy_score(y_val, y_pred)
        bcr_scores.append(bcr)
    
    mean_bcr = np.mean(bcr_scores)
    results[name] = mean_bcr
    print(f"{name} - Predicted BCR (BCR_hat): {mean_bcr:.4f}")

# 7. Final summary
print("\n=== Final Model BCR_hats ===")
for name, score in results.items():
    print(f"{name:20s} : BCR_hat = {score:.4f}")

