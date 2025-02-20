import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

train_df = pd.read_csv("./files/Parkinson_train.csv")
test_df = pd.read_csv("./files/Parkinson_test.csv")


x_train = train_df.drop(columns=["label"]) 
y_train = train_df["label"] 

x_test = test_df.drop(columns=["label"]) 
y_test = test_df["label"] 


def bagging_fit(x_train, y_train, nb_trees, i_run):
    tree_list = []
    for i in range(nb_trees):
        x_bootstrap = x_train.sample(frac=1, replace=True, random_state=i)
        y_bootstrap = y_train.loc[x_bootstrap.index]
        clf = DecisionTreeClassifier(random_state=i_run)
        clf.fit(x_bootstrap, y_bootstrap)
        tree_list.append(clf)
    return tree_list


def bagging_predict(tree_list, x_test):
    #the following array was done using chatGPT answers, as for the advice on using Counter
    pred = np.array([clf.predict(x_test) for clf in tree_list])
    output = []
    for sample in pred.T:
        votes = Counter(sample)
        prediction = max(sorted(votes.items()), key=lambda x: x[1])[0]
        output.append(prediction)
    return output


"""
for samples in pred.T:
        votes = {}
        for samp in samples:
            votes[samp] = votes.get(samp, 0) + 1
        output.append(max(votes, key=votes.get))


for samples in pred.T:
        most_voted = sorted(Counter(samples).items(), key=lambda x: x[1], reverse=True)
        #other: most_voted = sorted(Counter(votes).items())[0][0]
        output.append(most_voted[0][0])


label count = counter(col)
sorted
pred = max(sorted, key= lambda x: x[1])[0]
final.append 
"""


# Set number of trees and seed for reproducibility
nb_trees = 10
i_run = 42  

# Train the bagging model
tree_list = bagging_fit(x_train, y_train, nb_trees, i_run)

# Check if the correct number of trees were created
print(f"Number of trees trained: {len(tree_list)}")
print(f"Example tree structure:\n {tree_list[0]}")

############################################

# Predict on test data
y_pred = bagging_predict(tree_list, x_test)

# Check predictions
print("Predicted labels (first 10 samples):", y_pred[:10])
print("Actual labels (first 10 samples):", y_test.values[:10])

############################################

from sklearn.metrics import accuracy_score

# Compute accuracy
bagging_acc = accuracy_score(y_test, y_pred)
print(f"Bagging Model Accuracy: {bagging_acc:.4f}")

# Compare with a single Decision Tree
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(x_train, y_train)
single_tree_pred = single_tree.predict(x_test)

single_tree_acc = accuracy_score(y_test, single_tree_pred)
print(f"Single Decision Tree Accuracy: {single_tree_acc:.4f}")
