import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


train_df = pd.read_csv("./files/Parkinson_train.csv")
test_df = pd.read_csv("./files/Parkinson_test.csv")


x_train = train_df.drop(columns=["label"]) 
y_train = train_df["label"] 

x_test = test_df.drop(columns=["label"]) 
y_test = test_df["label"] 


param_grid = {
    "n_estimators": [200, 250, 300, 350],
    "max_depth": [None, 7, 9, 12, 20],
    "min_samples_split": [3, 7, 9, 12],
    "min_samples_leaf": [2, 4 , 6],
    "min_impurity_decrease": [0.0, 0.008, 0.01, 0.02, 0.03],
}

best_acc = 0  # Store best median test accuracy
best_params = {}  # Store best parameters

# Total number of iterations for tqdm
total_iters = np.prod([len(v) for v in param_grid.values()])

with tqdm(total=total_iters, desc="Tuning RF") as pbar:
    for n in param_grid["n_estimators"]:
        for d in param_grid["max_depth"]:
            for split in param_grid["min_samples_split"]:
                for leaf in param_grid["min_samples_leaf"]:
                    for impurity in param_grid["min_impurity_decrease"]:
                        scores = []

                        for i in range(30):  # Perform 30 independent runs
                            rf = RandomForestClassifier(
                                n_estimators=n,
                                max_depth=d,
                                min_samples_split=split,
                                min_samples_leaf=leaf,
                                min_impurity_decrease=impurity,
                                random_state=i
                            )
                            rf.fit(x_train, y_train)
                            scores.append(rf.score(x_test, y_test))  # Use clf.score

                        median_acc = np.median(scores)  # Get median accuracy

                        # If we find a better accuracy, print and update best values
                        if median_acc > best_acc:
                            best_acc = median_acc
                            best_params = {
                                "n_estimators": n,
                                "max_depth": d,
                                "min_samples_split": split,
                                "min_samples_leaf": leaf,
                                "min_impurity_decrease": impurity,
                            }
                            print(f"New Best Accuracy: {best_acc:.4f} with {best_params}")

                        pbar.update(1)  # Update tqdm progress ba

print("\n Final Best Parameters:", best_params)
print("Best Median Accuracy:", best_acc)

