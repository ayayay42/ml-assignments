import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

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

#plots

nb_trees_values = [1, 5, 10, 20, 50, 100]  # Number of trees to test
median_accuracies = []

# Run experiment
for nb_trees in nb_trees_values:
    accuracies = []
    for i_run in range(30):
        trees = bagging_fit(x_train, y_train, nb_trees, i_run)
        y_pred = bagging_predict(trees, x_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    median_accuracies.append(np.median(accuracies))
    print(f"nb_trees={nb_trees}, Median Test Accuracy: {median_accuracies[-1]:.4f}")

# Plot accuracy vs. nb_trees
plt.figure(figsize=(8, 5))
plt.plot(nb_trees_values, median_accuracies, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Trees in Bagging")
plt.ylabel("Median Test Accuracy")
plt.title("Bagging: Accuracy vs. Number of Trees")
plt.grid()
plt.savefig("./files/Bagging_Accuracy_vs_Number_of_Trees.png")

