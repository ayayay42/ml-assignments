# File: xgboost_pca_tuning.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import joblib  # To save the best model

# ------------------------
# 1. Load Data
# ------------------------
# Load numerical features (already imputed)
x_train = pd.read_csv('A5_train_pca_processed.csv').astype(np.float32) 
x_test = pd.read_csv('A5_test_pca_processed.csv').astype(np.float32) 
    
# Load mapped labels (0=inactive, 1=active)
y_train = pd.read_csv('y_train_mapped.csv', index_col=0)['target']


# ------------------------
# 3. Hyperparameter Tuning
# ------------------------
# Define parameter grid
param_grid = {
    'alpha': [0.2], #[0, +inf]    0, 0.05, 
    'colsample_bytree': [0.7, 0.8, 1.0],
    'eta': [0.05, 0.1, 0.2, 0.3], #[0,1]    
    'gamma': [0, 0.1, 0.2, 0.3], #[0, +inf]
    'lambda': [0.1, 0.5, 1], #[0,1]
    'max_depth': [3, 4, 5], #[0, +inf]
    'n_estimators': [200, 300, 400],   
    'subsample': [0.1, 0.5, 0.7, 1.0] #[0,1]
}

# Initialize XGBoost model
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',    # <--- GPU ACCELERATION
    random_state=42,
    verbosity=1
)

# Grid search with 5-fold CV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Run grid search
#grid_search.fit(x_train, y_train)

best_score = 0
best_params = None

for params in ParameterGrid(param_grid):
    model.set_params(**params)
    scores = cross_val_score(model, x_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    mean_score = scores.mean()
    
    if mean_score > best_score:
        best_score = mean_score
        best_params = params
        print(f"New Best Score: {best_score:.5f} with params: {best_params}")


# ------------------------
# 4. Results & Evaluation
# ------------------------

print("\nBest parameters:", grid_search.best_params_)
print("Best CV AUC:", grid_search.best_score)
      
# Save best model

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_xgboost_model2.pkl')

# Optional: Evaluate on test data (if labels are available)
# y_test = ... 
# y_pred = best_model.predict(x_test_pca)
# print("Test accuracy:", accuracy_score(y_test, y_pred))