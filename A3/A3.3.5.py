import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm

train_df = pd.read_csv("HeartFailure_train.csv")
test_df = pd.read_csv("HeartFailure_test.csv")

# Create an empty DataFrame
frame = pd.DataFrame(columns=["indiv_test_acc", "CI_lower_bound", "CI_upper_bound", 
                              "mean_test_acc", "observed_lower_bound", "observed_upper_bound"])

# Total training samples
train_size = int(0.05 * len(train_df))  # 5% of training data
test_size = 100  # Each test set has 100 samples

for i in range(50):
    # Step 1: Sample 5% of training data and train a decision tree
    train_sample = train_df.sample(n=train_size, random_state=i)
    X_train = train_sample.drop(columns=["HeartFailure"])
    y_train = train_sample["HeartFailure"]
    
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    # Step 2: Evaluate on a single random test set (100 examples)
    test_sample = test_df.sample(n=test_size, random_state=i)
    X_test = test_sample.drop(columns=["HeartFailure"])
    y_test = test_sample["HeartFailure"]
    
    y_pred = clf.predict(X_test)
    indiv_test_acc = accuracy_score(y_test, y_pred)
    
    # Step 3: Compute 95% Confidence Interval for single test set
    z = norm.ppf(0.975)  # 1.96 for 95% confidence
    se = np.sqrt((indiv_test_acc * (1 - indiv_test_acc)) / test_size)
    CI_lower_bound = indiv_test_acc - z * se
    CI_upper_bound = indiv_test_acc + z * se

    # Step 4: Compute mean accuracy across 200 test folds
    test_accs = []
    for j in range(200):
        test_fold = test_df.sample(n=test_size, random_state=(i+1)*j)
        X_test_fold = test_fold.drop(columns=["HeartFailure"])
        y_test_fold = test_fold["HeartFailure"]
        
        y_pred_fold = clf.predict(X_test_fold)
        test_accs.append(accuracy_score(y_test_fold, y_pred_fold))
    
    mean_test_acc = np.mean(test_accs)
    observed_lower_bound = np.percentile(test_accs, 2.5)
    observed_upper_bound = np.percentile(test_accs, 97.5)

    # Store results in DataFrame
    frame.loc[i] = [indiv_test_acc, CI_lower_bound, CI_upper_bound, 
                    mean_test_acc, observed_lower_bound, observed_upper_bound]

# Print first 5 rows to verify output
print(frame.head())
print(frame.shape)
print(frame.describe())
