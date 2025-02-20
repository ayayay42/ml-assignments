####### Question 4 #######
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv("./files/Parkinson_train.csv")
test_df = pd.read_csv("./files/Parkinson_test.csv")

frame = pd.DataFrame(columns = ["Run", "NodeCount", "TrainAcc", "TestAcc"])

runs = 100

for i in range(runs):
    train_sample = train_df.sample(frac=0.25, random_state=i)

    x_train_sample = train_sample.drop(columns=["label"])
    y_train_sample = train_sample["label"]

    x_test = test_df.drop(columns=["label"]) 
    y_test = test_df["label"]

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train_sample, y_train_sample)

    node_count = clf.tree_.node_count  # Number of nodes in the tree
    train_accur = clf.score(x_train_sample, y_train_sample)  # Accuracy on sampled training data
    test_accur = clf.score(x_test, y_test)  # Accuracy on full test set

    # Store results in DataFrame
    frame.loc[i] = [i, node_count, train_accur, test_accur]

mean_nb_nodes = frame["NodeCount"].mean()
mean_train_accur = frame["TrainAcc"].mean()
mean_test_accur = frame["TestAcc"].mean()

print("mean number of nodes: ", mean_nb_nodes)
print("mean train accuracy: ", mean_train_accur)
print("mean test accuracy: ", mean_test_accur)