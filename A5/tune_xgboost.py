from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore")


X_float_train = pd.read_csv('A5_train_float.csv')
X_cat_train = pd.read_csv('A5_train_category.csv')
X_train = pd.concat([X_float_train, X_cat_train], axis=1)
y_train = pd.read_csv('y_train_mapped.csv', index_col=0)['target']

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

search = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='balanced_accuracy',  # BCR!
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print("Best BCR:", search.best_score_)
