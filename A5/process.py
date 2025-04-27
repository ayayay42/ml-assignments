import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier
from tqdm import tqdm

# 1. Load the data
X_float_train = pd.read_csv('A5_train_float.csv')
X_cat_train = pd.read_csv('A5_train_category.csv')
y_train = pd.read_csv('y_train_mapped.csv', header=None).squeeze()  # Flatten to Series

X_float_test = pd.read_csv('A5_test_float.csv')
X_cat_test = pd.read_csv('A5_test_category.csv')

# 2. Concatenate floats + categories
X_train = pd.concat([X_float_train, X_cat_train], axis=1)
X_test = pd.concat([X_float_test, X_cat_test], axis=1)

# 3. Separate float and categorical feature names
float_cols = X_float_train.columns.tolist()
cat_cols = X_cat_train.columns.tolist()

# 4. Train-validation split (only on train data!)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)


# 5. Build preprocessing pipelines (updated)
float_pipeline = Pipeline([
    ('scaler', StandardScaler())  # No imputer anymore for floats
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('float', float_pipeline, float_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 6. Model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),  # to balance classes
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Train
model_pipeline.fit(X_tr, y_tr)

# 8. Validation performance
y_val_pred = model_pipeline.predict(X_val)
val_bcr = balanced_accuracy_score(y_val, y_val_pred)
print(f"Validation BCR (BCR_hat): {val_bcr:.4f}")

# 9. Final model on full training data
model_pipeline.fit(X_train, y_train)

# 10. Predict on test data
y_test_pred = model_pipeline.predict(X_test)

# 11. Save predictions
pd.DataFrame(y_test_pred, columns=['prediction']).to_csv('A5_test_predictions.csv', index=False)

# 12. Save your BCR_hat
with open('BCR_hat.txt', 'w') as f:
    f.write(str(val_bcr))