import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import loguniform
from tqdm import tqdm
from sklearn.model_selection import cross_val_score

# Load data
X_float_train = pd.read_csv('A5_train_float.csv')
X_cat_train = pd.read_csv('A5_train_category.csv')
X_train = pd.concat([X_float_train, X_cat_train], axis=1)

y_train = pd.read_csv('y_train_mapped.csv', index_col=0)['target']

# X_test (prepare it the same way)
X_float_test = pd.read_csv('A5_test_float.csv')
X_cat_test = pd.read_csv('A5_test_category.csv')
X_test = pd.concat([X_float_test, X_cat_test], axis=1)

print("Tuning Logistic Regression...")

logreg = LogisticRegression(max_iter=2000, random_state=42)

logreg_param_grid = {
    'C': loguniform(1e-4, 1e4)
}

logreg_search = RandomizedSearchCV(
    logreg,
    param_distributions=logreg_param_grid,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='balanced_accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

logreg_search.fit(X_train, y_train)

best_logreg = logreg_search.best_estimator_
print("Best Logistic Regression params:", logreg_search.best_params_)


print("Tuning XGBoost...")

xgb = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)

xgb_param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 300, 500],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
}

xgb_search = RandomizedSearchCV(
    xgb,
    param_distributions=xgb_param_grid,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='balanced_accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_
print("Best XGBoost params:", xgb_search.best_params_)


# Choose the best one manually depending on BCR (you can compare them)
final_model = best_xgb  # or best_logreg

# Train on full data
final_model.fit(X_train, y_train)

# Predict on test
y_test_pred = final_model.predict(X_test)

# Save predictions
pd.DataFrame({'prediction': y_test_pred}).to_csv('final_predictions.csv', index=False)
