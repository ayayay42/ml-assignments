from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

train_df = pd.read_csv("./Waveform_train.csv")
test_df = pd.read_csv("./Waveform_test.csv")

X_train = train_df.drop(columns=["labels"]) 
y_train = train_df["labels"] 

x_test = test_df.drop(columns=["labels"]) 
y_test = test_df["labels"] 

# Define the 10-fold cross-validation strategy
kf = KFold(n_splits=10, shuffle=False)  # No shuffle as per instructions

# Hyperparameter tuning for RBF and Sigmoid kernels
best_rbf_params = {}
best_rbf_score = 0.800
best_sigm_params = {}
best_sigm_score = 0.8660

C_values = np.arange(96, 110, 2)  
gamma_values = np.arange(0.001, 0.05, 0.001)
coef_values = np.arange(10, 10, 1)  # Coef parameter for Sigmoid kernel

param_grid_rbf = [(C, gamma) for C in C_values for gamma in gamma_values]
param_grid_sigm = [(C, gamma) for C in C_values for gamma in gamma_values]
# Sigmoid Kernel Tuning
for C, gamma in tqdm(param_grid_sigm, desc="Tuning Sigmoid kernel"):
    sigm_svm = SVC(kernel='sigmoid', C=C, gamma=gamma)
    sigm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', sigm_svm)
    ])
    sigm_score = cross_val_score(sigm_pipeline, X_train, y_train, cv=kf, scoring='accuracy').mean()
    
    if sigm_score > best_sigm_score:
        best_sigm_score = sigm_score
        best_sigm_params = {'C': C, 'gamma': gamma}
        print(f"New best Sigmoid parameters: {best_sigm_params} with accuracy {best_sigm_score:.4f}")

# RBF Kernel Tuning
for C, gamma in tqdm(param_grid_rbf, desc="Tuning RBF kernel"):
    rbf_svm = SVC(kernel='rbf', C=C, gamma=gamma)
    rbf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', rbf_svm)
    ])
    rbf_score = cross_val_score(rbf_pipeline, X_train, y_train, cv=kf, scoring='accuracy').mean()
    
    if rbf_score > best_rbf_score:
        best_rbf_score = rbf_score
        best_rbf_params = {'C': C, 'gamma': gamma}
        print(f"New best RBF parameters: {best_rbf_params} with accuracy {best_rbf_score:.4f}")


# Instantiate the best RBF and Sigmoid SVMs
my_rbf_svm = SVC(kernel='rbf', C=best_rbf_params['C'], gamma=best_rbf_params['gamma'])
my_sigm_svm = SVC(kernel='sigmoid', C=best_sigm_params['C'], gamma=best_sigm_params['gamma'])

print(f"Final best RBF parameters: {best_rbf_params} with accuracy {best_rbf_score:.4f}")
print(f"Final best Sigmoid parameters: {best_sigm_params} with accuracy {best_sigm_score:.4f}")


'''
New best Sigmoid parameters: {'C': np.int64(95), 'gamma': np.float64(0.0004)} with accuracy 0.8680

C_values = np.arange(0.1, 3, 0.1)
gamma_values = np.arange(0.005, 0.3, 0.005) 
Final best RBF parameters: {'C': np.float64(1.0), 'gamma': np.float64(0.01)} with accuracy 0.8660
Final best Sigmoid parameters: {'C': np.float64(2.5000000000000004), 'gamma': np.float64(0.01)} with accuracy 0.8650

Final best Sigmoid parameters: {'C': np.int64(6), 'gamma': np.float64(0.01)} with accuracy 0.8660

pour sigmoid il faut coef 


Final best RBF parameters: {'C': np.int64(9), 'gamma': np.float64(0.0004)} with accuracy 0.8642
Final best Sigmoid parameters: {'C': np.int64(5), 'gamma': np.float64(0.0006000000000000001), 'coef0': np.int64(-1)} with accuracy 0.8610
avec coef de -10 à 10

New best RBF parameters: {'C': np.int64(210), 'gamma': np.float64(0.0001)} with accuracy 0.8690
'''
