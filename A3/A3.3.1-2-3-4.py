import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("HeartFailure_train.csv")
test_df = pd.read_csv("HeartFailure_test.csv")

# Sample 5% of the training set
small_train = train_df.sample(frac=0.05, random_state=0)
X_train = small_train.drop(columns=['HeartFailure'])
y_train = small_train['HeartFailure']

# Sample 100 examples from the test set
test_sample = test_df.sample(n=100, random_state=0)
X_test = test_sample.drop(columns=['HeartFailure'])
y_test = test_sample['HeartFailure']

# Fit the decision tree model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

# Print to verify the value
print("Test Accuracy:", test_acc)

from sklearn.metrics import accuracy_score

test_accs = []

# Loop over 200 test folds, using random_state = i for the i-th fold.
for i in range(200):
    test_sample = test_df.sample(n=100, random_state=i)
    X_test = test_sample.drop(columns=['HeartFailure'])
    y_test = test_sample['HeartFailure']
    
    # Use the already trained decision tree clf from question 1.
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    test_accs.append(acc)

# Compute the mean test accuracy over the 200 folds.
mean_test_acc = sum(test_accs) / len(test_accs)

print("List of Test Accuracies (first 10):", test_accs[:10])
print("Mean Test Accuracy:", mean_test_acc)

import numpy as np

# Compute the 2.5 and 97.5 percentiles from the 200 test accuracies
observed_lower_bound = np.percentile(test_accs, 2.5)
observed_upper_bound = np.percentile(test_accs, 97.5)

print("Observed Lower Bound:", round(observed_lower_bound, 3))
print("Observed Upper Bound:", round(observed_upper_bound, 3))


