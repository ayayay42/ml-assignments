from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import ParameterGrid, cross_val_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from tqdm import tqdm

x_train = pd.read_csv('A5_train_pca_processed.csv').astype(np.float32) 
x_test = pd.read_csv('A5_test_pca_processed.csv').astype(np.float32) 
y_train = pd.read_csv('y_train_mapped.csv', index_col=0)['target']

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)


# Create a BCR scorer for cross-validation
bcr_scorer = make_scorer(balanced_accuracy_score)

best_score = 0
best_params = None

# Updated parameter grid (corrected for XGBoost’s API)
param_grid = {
    'reg_alpha': [2], # 0.2, 5, 10 one alpha at a time         # Use `reg_alpha` instead of `alpha`
    'reg_lambda': [0.1, 0.2], # Use `reg_lambda` instead of `lambda`
    'colsample_bytree': [0.8, 0.85],
    'eta': [0.4, 0.45],
    'gamma': [0.20, 0.25, 0.3],
    'max_depth': [2, 3],
    'n_estimators': [300, 600],
    'subsample': [0.95]
}

total_combinations = len(ParameterGrid(param_grid))

for params in tqdm(ParameterGrid(param_grid), total=total_combinations, desc="Tuning"): 
    model = XGBClassifier(
        **params,
        objective='binary:logistic',
        eval_metric='logloss',  # Optional: Use `logloss` for training monitoring
        tree_method='hist',
        random_state=42
    )
    # Optimize for BCR
    scores = cross_val_score(
        model, 
        x_train, 
        y_train, 
        cv=3, 
        scoring=bcr_scorer,  # Use BCR scorer
        n_jobs=-1
    )
    mean_score = scores.mean()
    
    if mean_score > best_score:
        best_score = mean_score
        best_params = params
        tqdm.write(f"New Best BCR: {best_score:.5f} with params: {best_params}")

print("\nFinal Best Parameters:")
print(f"Best BCR: {best_score:.5f}")
print(f"Best params: {best_params}")