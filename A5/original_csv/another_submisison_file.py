import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold
from xgboost import XGBClassifier


def preprocess_data():
    #load the datasets
    x_train = pd.read_csv('./A5_2025_train.csv')
    x_test = pd.read_csv('./A5_2025_test.csv')
    y_train = pd.read_csv('./A5_2025_train_labels.csv')

    # Mapping: inactive=0, active=1
    y_train_mapped = y_train['target'].map({'inactive': 0, 'active': 1})

    # Split the dataset between the float features and the category features
    float_cols = x_train.columns[:14908]
    cat_cols = x_train.columns[14908:]
    
    X_float_train = x_train[float_cols].astype(np.float32)
    X_float_test = x_test[float_cols].astype(np.float32)
    X_cat_train = x_train[cat_cols]
    X_cat_test = x_test[cat_cols]

    # Impute missing float values by the mean
    num_imputer = SimpleImputer(strategy='median') #idea to use the mean from: https://lakefs.io/blog/data-preprocessing-in-machine-learning/
    X_float_train = pd.DataFrame(num_imputer.fit_transform(X_float_train),columns=float_cols)
    X_float_test = pd.DataFrame(num_imputer.transform(X_float_test),columns=float_cols)

    # encodes categories into numerical values
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_train = pd.DataFrame(encoder.fit_transform(X_cat_train),columns=encoder.get_feature_names_out(cat_cols))
    X_cat_test = pd.DataFrame(encoder.transform(X_cat_test),columns=encoder.get_feature_names_out(cat_cols))

    #PCA
    scaler = StandardScaler()
    pca = PCA(n_components=1500, svd_solver='randomized')
    
    X_float_train_scaled = scaler.fit_transform(X_float_train)
    X_float_train_pca = pca.fit_transform(X_float_train_scaled)
    X_float_test_scaled = scaler.transform(X_float_test)
    X_float_test_pca = pca.transform(X_float_test_scaled)

    #Combine
    X_train = np.hstack([X_float_train_pca, X_cat_train])
    X_test = np.hstack([X_float_test_pca, X_cat_test])

    return X_train, X_test, y_train_mapped, encoder, scaler, pca


def train_best_model(X_train, y_train):

    #split data into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    best_params = { #according to what I've found soing a grid search on many parameters
        'colsample_bytree': 0.8,
        'eta': 0.4,
        'gamma': 0.25,
        'max_depth': 2,
        'n_estimators': 300,
        'reg_alpha': 0.4,
        'reg_lambda': 0.1,
        'subsample': 0.95,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',
        'random_state': 42,
        'scale_pos_weight': 9, #for unbalanced data
    }

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    return model


def get_BCR(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(model, X_train, y_train, cv=cv, method='predict',n_jobs=-1)
    
    cm = confusion_matrix(y_train, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    #formula found on this website: https://www.geeksforgeeks.org/calculate-efficiency-binary-classifier/
    negative = tn + fp  
    positive = fn + tp  
    sensitivity = tp / positive if positive > 0 else 0
    specificity = tn / negative if negative > 0 else 0 
    
    BCR = (specificity + sensitivity) / 2
    
    return BCR


def get_pred(model, X_test):
    #Get predicted probabilities and convert to class labels (0 and 1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = pd.Series(    #This part was done using the help of DeepSeek 
        np.where(y_pred_proba > 0.5, 'active', 'inactive'),
        name='target',
        index=range(len(X_test)))
    
    y_pred.to_csv('predictions.csv', index=True)
    return y_pred


if __name__ == '__main__':
    #Preprocessing
    X_train, X_test, y_train, encoder, scaler, pca = preprocess_data()

    #modeling
    model = train_best_model(X_train, y_train)

    #BCR calculation
    predicted_BCR= get_BCR(model, X_train, y_train)

    print(f"Predicted BCR: {predicted_BCR:.4f}")

    y_pred = get_pred(model, X_test)