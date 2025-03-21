from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
train_df = pd.read_csv("./Waveform_train.csv")
test_df = pd.read_csv("./Waveform_test.csv")

X_train = train_df.drop(columns=["labels"]) 
y_train = train_df["labels"] 

x_test = test_df.drop(columns=["labels"]) 
y_test = test_df["labels"] 

kf = KFold(n_splits=10, shuffle=False)  # No shuffle as per instructions

# Find the optimal polynomial degree in the range [2,10]
best_degree = None
best_score = 0
for degree in range(2, 11):
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='poly', degree=degree))
    ])
    
    cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=kf, scoring='accuracy')
    mean_score = cv_scores.mean()
    
    if mean_score > best_score:
        best_score = mean_score
        best_degree = degree

# Implement SVM with the best polynomial degree
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='poly', degree=best_degree))
])

# Perform cross-validation
cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=kf, scoring='accuracy')

# Compute mean cross-validation accuracy
cv_acc = cv_scores.mean()

print(f"Optimal polynomial degree: {best_degree}")
print(f"Cross-validation accuracy: {cv_acc:.4f}")