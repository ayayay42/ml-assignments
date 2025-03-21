import numpy as np
from sklearn.svm import SVC
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

# Train a linear SVM on the full dataset
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('svm', SVC(kernel='linear'))  # Linear SVM
])

svm_pipeline.fit(X_train, y_train)

# Extract feature coefficients (absolute values)
feature_importance = np.abs(svm_pipeline.named_steps['svm'].coef_).mean(axis=0)

# Get the top 4 feature indices
top_features = np.argsort(feature_importance)[-4:]

# Get feature names
feature_names = X_train.columns[top_features]

# Print the top 4 features
print(",".join(feature_names))

#x6,x11,x15,x16