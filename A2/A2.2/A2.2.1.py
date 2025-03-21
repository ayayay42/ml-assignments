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

# Create a pipeline with StandardScaler and SVM (linear kernel)
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear'))  # Ensure linear SVM
])

# Perform cross-validation
cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=kf, scoring='accuracy')

# Compute mean cross-validation accuracy
cv_acc = cv_scores.mean()

print(f"Cross-validation accuracy: {cv_acc:.4f}")
