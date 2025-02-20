######## Question 1 ########
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train_df = pd.read_csv("./files/Parkinson_train.csv")
test_df = pd.read_csv("./files/Parkinson_test.csv")

frame = pd.DataFrame(columns = ["min_impurity_decrease", "NodeCount", "TrainAcc", "TestAcc"])

x_train = train_df.drop(columns=["label"]) 
y_train = train_df["label"] 

x_test = test_df.drop(columns=["label"]) 
y_test = test_df["label"] 

impurity_score = np.arange(0, 0.11, 0.001)

for i in impurity_score:
    clf = DecisionTreeClassifier(random_state=0, min_impurity_decrease = i)
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train) 
    y_test_pred = clf.predict(x_test)
    
    nb_nodes = clf.tree_.node_count
    train_accur = clf.score(x_train, y_train)
    test_accur = clf.score(x_test, y_test)
    
    frame = pd.concat([frame, pd.DataFrame({
        "min_impurity_decrease": [i],
        "NodeCount": [nb_nodes],
        "TrainAcc": [train_accur],
        "TestAcc": [test_accur]
    })], ignore_index=True)

frame.to_csv("./files/results_min_impurity2.csv", index=False)
best_model = frame.loc[frame["TestAcc"].idxmax()]
worst_model = frame.loc[frame["TestAcc"].idxmin()]
mean_test_acc = frame["TestAcc"].mean()
average_model = frame.iloc[(frame["TestAcc"] - mean_test_acc).abs().idxmin()]

print("Best Model:\n", best_model)
print("Worst Model:\n", worst_model)
print("Average Model:\n", average_model)
