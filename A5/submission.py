import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import ParameterGrid, cross_val_score
from xgboost import XGBClassifier
from tqdm import tqdm
import joblib

#Load data
print("Loading data...")
x_train = pd.read_csv('./original_csv/A5_2025_train.csv')
x_test = pd.read_csv('./original_csv/A5_2025_test.csv')
y_train = pd.read_csv('./original_csv/A5_2025_train_labels.csv')

print("Splitting features...")
X_float_train = x_train.iloc[:, :14908].astype(np.float32)
X_float_test = x_test.iloc[:, :14908].astype(np.float32)
X_cat_train = x_train.iloc[:, 14908:]
X_cat_test = x_test.iloc[:, 14908:]

#process the data

#transform y train: inactive to 0 and active to 1
print("mapping inactive to 0 and active to 1")
y_train_mapped = y_train['target'].map({'inactive': 0, 'active': 1})

#replace missing values with the median
print("Filling missing values with median...")
num_cols = x_train.columns[:14908]
num_imputer = SimpleImputer(strategy='median')
x_train_filled = pd.DataFrame(num_imputer.fit_transform(X_float_train), columns=X_float_train.columns)
x_test_filled = pd.DataFrame(num_imputer.transform(X_float_test), columns=X_float_test.columns)

# Initialize encoder (per-column mapping)
print("Encoding categorical features...")
encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",  # Assign -1 to unseen test categories
    unknown_value=-1,                    # Custom value for unknowns
    encoded_missing_value=-2             # Handle NaNs if present
)

X_cat_train = pd.DataFrame(encoder.fit_transform(X_cat_train), columns=X_cat_train.columns)
X_cat_test = pd.DataFrame(encoder.transform(X_cat_test), columns=X_cat_test.columns)

print('PCA')

scaler = StandardScaler()
pca = PCA(n_components=1500, svd_solver='randomized')

# Scale and transform training data
X_float_train_scaled = scaler.fit_transform(X_float_train)
X_float_train_pca = pca.fit_transform(X_float_train_scaled)

# Scale and transform test data
X_float_test_scaled = scaler.transform(X_float_test)
X_float_test_pca = pca.transform(X_float_test_scaled)

# Combine with categorical features
print("Combining features...")
X_train_processed = np.hstack([X_float_train_pca, X_cat_train])
X_test_processed = np.hstack([X_float_test_pca, X_cat_test])

#train final model
print("Training final model...")

best_params = {
    'colsample_bytree': 0.8,
    'eta': 0.4,
    'gamma': 0.25,
    'max_depth': 2,
    'n_estimators': 300,
    'reg_alpha': 0.4,
    'reg_lambda': 0.1,
    'subsample': 0.95
}

final_model = XGBClassifier(
    **best_params,
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    random_state=42
)
final_model.fit(X_train_processed, y_train_mapped)

print("Evaluating final model...")
from sklearn.model_selection import cross_val_score
bcr_scores = cross_val_score(
    model, 
    X_train, 
    y_train_mapped, 
    cv=5, 
    scoring='balanced_accuracy'
)
print(f"Cross-validated BCR: {bcr_scores.mean():.4f} ± {bcr_scores.std():.4f}")

# Saving predicted y for the test set
y_pred = final_model.predict(X_test_processed)
submission = pd.DataFrame({'id': x_test['id'], 'target': y_pred})
submission.to_csv('submission.csv', index=False)

print("\n=== Final Results ===")
print(f"Best BCR: {best_score:.5f}")
print(f"Best params: {best_params}")
print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.1%}")