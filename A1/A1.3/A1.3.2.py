import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Data
train_df = pd.read_csv("./files/Parkinson_train.csv")
test_df = pd.read_csv("./files/Parkinson_test.csv")

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# Training fractions to test
training_sizes = [0.05, 0.10, 0.20, 0.50, 0.99]

# Pruning values to test
pruning_values = [0.0, 0.02, 0.05, 0.1]

# DataFrame to store results
frame = pd.DataFrame(columns=["Frac", "Run", "Pruning", "NodeCount", "TrainAcc", "TestAcc"])

# Run experiment
for frac in training_sizes:
    for pruning in pruning_values:
        for run in range(100):
            # Sample fraction of training data
            sampled_df = train_df.sample(frac=frac, random_state=run)
            X_train = sampled_df.drop(columns=["label"])
            y_train = sampled_df["label"]

            # Train model with pruning
            clf = DecisionTreeClassifier(random_state=0, min_impurity_decrease=pruning)
            clf.fit(X_train, y_train)

            # Predictions
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)

            # Compute metrics
            num_nodes = clf.tree_.node_count
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            # Store results
            frame.loc[len(frame)] = [frac, run, pruning, num_nodes, train_acc, test_acc]

# Convert data types
frame["Frac"] = frame["Frac"].astype(float)
frame["Pruning"] = frame["Pruning"].astype(float)

# Plot tree size vs. training size
plt.figure(figsize=(12, 5))
sns.boxplot(x="Frac", y="NodeCount", hue="Pruning", data=frame)
plt.title("Tree Size vs. Training Set Size (With and Without Pruning)")
plt.xlabel("Training Set Fraction")
plt.ylabel("Number of Nodes")
plt.legend(title="Pruning (min_impurity_decrease)")
plt.savefig("./files/TreeSize_vs_TrainingSize.png")

# Plot test accuracy vs. training size (learning curve)
plt.figure(figsize=(12, 5))
sns.boxplot(x="Frac", y="TestAcc", hue="Pruning", data=frame)
plt.title("Learning Curve: Test Accuracy vs. Training Set Size (With and Without Pruning)")
plt.xlabel("Training Set Fraction")
plt.ylabel("Test Accuracy")
plt.legend(title="Pruning (min_impurity_decrease)")
plt.savefig("./files/TestAcc_vs_TrainingSize.png")


#plot the train accuracy vs test accuracy
plt.figure(figsize=(12, 5))
sns.scatterplot(x="TrainAcc", y="TestAcc", hue="Pruning", data=frame)
plt.title("Train Accuracy vs Test Accuracy")
plt.xlabel("Train Accuracy")
plt.ylabel("Test Accuracy")
plt.legend(title="Pruning (min_impurity_decrease)")
plt.savefig("./files/TrainAcc_vs_TestAcc.png")

