import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

train_df = pd.read_csv("./files/Parkinson_train.csv")
test_df = pd.read_csv("./files/Parkinson_test.csv")

x_train = train_df.drop(columns=["label"]) 
y_train = train_df["label"] 

x_test = test_df.drop(columns=["label"]) 
y_test = test_df["label"] 

# Store test accuracies
test_accuracies = []

# Train 30 different models
for i in range(30):
    clf = DecisionTreeClassifier(random_state=i) 
    clf.fit(x_train, y_train)
    test_accuracies.append(clf.score(x_test, y_test))

# Compute the median test accuracy
median_test_accuracy = np.median(test_accuracies)

print(f"Median Test Accuracy over 30 runs: {median_test_accuracy:.4f}")
