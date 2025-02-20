###### Question 5 #######
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("./files/Parkinson_train.csv")
test_df = pd.read_csv("./files/Parkinson_test.csv")

frame = pd.DataFrame(columns = ["Frac", "Run", "NodeCount", "TrainAcc", "TestAcc"])

fractions_size = [0.01, 0.05, 0.10, 0.20, 0.50, 0.99]

runs = 100

for frac in fractions_size:
    for i in range(runs):
        train_sample = train_df.sample(frac=frac, random_state=i)
        
        x_train_sample = train_sample.drop(columns=["label"])
        y_train_sample = train_sample["label"]
        
        x_test = test_df.drop(columns=["label"])
        y_test = test_df["label"]
        
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(x_train_sample, y_train_sample)
        
        node_count = clf.tree_.node_count
        train_accur = clf.score(x_train_sample, y_train_sample)
        test_accur = clf.score(x_test, y_test)
        
        frame.loc[len(frame)] = [frac, i, node_count, train_accur, test_accur]

#print(frame.head())
#print(frame.tail())
summary = frame.groupby("Frac")[["NodeCount", "TrainAcc", "TestAcc"]].mean()
#print(summary)

####### Question 6 #######

# Set the style
sns.set(style="whitegrid")

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1️⃣ Boxplot: Number of Nodes vs. Training Set Size
sns.boxplot(data=frame, x="Frac", y="NodeCount", ax=axes[0])
axes[0].set_title("Tree Size vs. Training Set Size")
axes[0].set_xlabel("Training Set Fraction")
axes[0].set_ylabel("Number of Nodes")

# 2️⃣ Boxplot: Test Accuracy vs. Training Set Size (Learning Curve)
sns.boxplot(data=frame, x="Frac", y="TestAcc", ax=axes[1])
axes[1].set_title("Learning Curve: Test Accuracy vs. Training Set Size")
axes[1].set_xlabel("Training Set Fraction")
axes[1].set_ylabel("Test Accuracy")

# Show the plots
plt.tight_layout()
plt.savefig("./files/plot.png")

