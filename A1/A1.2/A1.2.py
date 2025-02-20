import pandas as pd
from sklearn.tree import DecisionTreeClassifier

####### Question 1 #######

train_df = pd.read_csv("./files/Parkinson_train.csv")
test_df = pd.read_csv("./files/Parkinson_test.csv")

clf = DecisionTreeClassifier(random_state=0) #initiation (source: website of sklearn)

x_train = train_df.drop(columns=["label"]) 
y_train = train_df["label"] 

x_test = test_df.drop(columns=["label"]) 
y_test = test_df["label"] 

clf.fit(x_train, y_train) 

y_train_pred = clf.predict(x_train) 
y_test_pred = clf.predict(x_test) 

####### Question 2 #######

nb_nodes = clf.tree_.node_count
train_accur = clf.score(x_train, y_train)
test_accur = clf.score(x_test, y_test)

print("number of nodes: ", nb_nodes)
print("train accuracy: ", train_accur)
print("test accuracy: ", test_accur)

####### Question 3 #######

#only correct if data is consistent

