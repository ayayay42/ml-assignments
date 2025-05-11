# competition_submission.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import joblib

# =============================================
# 1. DATA PREPROCESSING
# =============================================

def preprocess_data():
    """Load and preprocess the training and test data"""
    # Load data with relative paths
    x_train = pd.read_csv('./A5_2025_train.csv')
    x_test = pd.read_csv('./A5_2025_test.csv')
    y_train = pd.read_csv('./A5_2025_train_labels.csv')

    # Map target: inactive=0, active=1
    y_train_mapped = y_train['target'].map({'inactive': 0, 'active': 1})

    # Split into float and categorical features
    X_float_train = x_train.iloc[:, :14908].astype(np.float32)
    X_float_test = x_test.iloc[:, :14908].astype(np.float32)
    X_cat_train = x_train.iloc[:, 14908:]
    X_cat_test = x_test.iloc[:, 14908:]

    # Impute missing float values with median
    num_imputer = SimpleImputer(strategy='median')
    X_float_train = pd.DataFrame(num_imputer.fit_transform(X_float_train), 
                    columns=X_float_train.columns)
    X_float_test = pd.DataFrame(num_imputer.transform(X_float_test), 
                   columns=X_float_test.columns)

    # Encode categorical features
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat_train = pd.DataFrame(encoder.fit_transform(X_cat_train), 
                  columns=X_cat_train.columns)
    X_cat_test = pd.DataFrame(encoder.transform(X_cat_test), 
                 columns=X_cat_test.columns)

    # Apply PCA to float features (1500 components)
    scaler = StandardScaler()
    pca = PCA(n_components=1500, svd_solver='randomized')
    
    X_float_train_scaled = scaler.fit_transform(X_float_train)
    X_float_train_pca = pca.fit_transform(X_float_train_scaled)
    X_float_test_scaled = scaler.transform(X_float_test)
    X_float_test_pca = pca.transform(X_float_test_scaled)

    # Combine features
    X_train = np.hstack([X_float_train_pca, X_cat_train])
    X_test = np.hstack([X_float_test_pca, X_cat_test])

    return X_train, X_test, y_train_mapped, encoder, scaler, pca

# =============================================
# 2. MODEL TRAINING
# =============================================

def train_model(X_train, y_train):
    """Train XGBoost model with optimized parameters"""
    best_params = {
        'colsample_bytree': 0.8,
        'eta': 0.4,
        'gamma': 0.25,
        'max_depth': 2,
        'n_estimators': 300,
        'reg_alpha': 0.4,
        'reg_lambda': 0.1,
        'subsample': 0.95,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': 42
    }

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model

# =============================================
# 3. EVALUATION & PREDICTION
# =============================================

def predict_and_evaluate(model, X_train, y_train, X_test):
    """Generate predictions and compute expected BCR"""
    # Cross-validated BCR estimate
    from sklearn.model_selection import cross_val_score
    bcr_scores = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=5, 
        scoring='balanced_accuracy'
    )
    expected_bcr = np.mean(bcr_scores)
    
    # Generate test predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = pd.Series(
        np.where(y_pred_proba > 0.5, 'active', 'inactive'),
        name='target',
        index=range(len(X_test)))
    
    return y_pred, expected_bcr

# =============================================
# MAIN EXECUTION
# =============================================

if __name__ == '__main__':
    # 1. Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, encoder, scaler, pca = preprocess_data()
    
    # 2. Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # 3. Generate predictions and expected BCR
    print("Generating predictions and evaluating model...")
    y_pred, expected_bcr = predict_and_evaluate(model, X_train, y_train, X_test)
    
    # 4. Save predictions for Question 1
    y_pred.to_csv('predictions.csv', index=True)
    
    # 5. Output expected BCR for Question 2
    print(f"Expected BCR: {expected_bcr:.4f}")
    
    # 6. The code itself answers Question 3