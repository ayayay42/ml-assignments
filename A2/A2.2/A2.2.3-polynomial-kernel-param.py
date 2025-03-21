from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import numpy as np

# Load the dataset
train_df = pd.read_csv("./Waveform_train.csv")
test_df = pd.read_csv("./Waveform_test.csv")

X_train = train_df.drop(columns=["labels"]) 
y_train = train_df["labels"] 

x_test = test_df.drop(columns=["labels"]) 
y_test = test_df["labels"] 

folds = KFold(n_splits=10, shuffle=False)  # No shuffle as per instructions

# Find the optimal polynomial degree and hyperparameters
best_params = {}
best_score = 0

degree_range = np.arange(1, 10, 1) #non negative, only for poly
coef0_range = np.arange(11, 21, 2)  #for poly and sigmoid 
C_range = np.arange(5,7,1)  # strictly positive 
gamma_range = np.arange(0.0001, 0.0007, 0.0001)  # non negative 

# Create parameter grid including gamma
param_grid = [(degree, coef0, C, gamma) 
              for degree in degree_range 
              for coef0 in coef0_range 
              for C in C_range 
              for gamma in gamma_range]

for degree, coef0, C, gamma in tqdm(param_grid, desc="Tuning hyperparameters"):
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='poly', degree=degree, coef0=coef0, C=C, gamma=gamma))
    ])
    
    cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=folds, scoring='accuracy')
    mean_score = cv_scores.mean()
    
    if mean_score > best_score:
        best_score = mean_score
        best_params = {'degree': degree, 'coef0': coef0, 'C': C, 'gamma': gamma}
        print(f"New best parameters found: {best_params} with accuracy {best_score:.4f}")

# Instantiate the best polynomial SVM with gamma
my_poly_svm = SVC(kernel='poly', degree=best_params['degree'], coef0=best_params['coef0'], 
                  C=best_params['C'], gamma=best_params['gamma'])

# Implement SVM with the best polynomial degree
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', my_poly_svm)
])

# Perform cross-validation
cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=folds, scoring='accuracy')

# Compute mean cross-validation accuracy
cv_acc = cv_scores.mean()

print(f"Optimal polynomial parameters: {best_params}")
print(f"Cross-validation accuracy: {cv_acc:.4f}")



'''
param_grid = [(degree, coef0, C) for degree in range(2, 5) for coef0 in [0.5,0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85] for C in [0.1, 0.2, 0.3, 0.4, 0.5]]
Optimal polynomial parameters: {'degree': 2, 'coef0': 0.6, 'C': 0.1}
Cross-validation accuracy: 0.8645

Optimal polynomial parameters: {'degree': np.int64(2), 'coef0': np.float64(0.7), 'C': np.float64(0.09)}
Cross-validation accuracy: 0.8645

#0.0001 for gamme
C 6
Coef grand dizaine 
c range plus petit


New best parameters found: {'degree': np.int64(2), 'coef0': np.int64(14), 'C': np.int64(5), 'gamma': np.float64(0.00030000000000000003)} with accuracy 0.8693
'''