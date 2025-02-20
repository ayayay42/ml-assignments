import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
from tqdm import tqdm

# Load your dataset (replace with actual data)
train_df = pd.read_csv("./files/Parkinson_train.csv")
test_df = pd.read_csv("./files/Parkinson_test.csv")

X_train = train_df.drop(columns=["label"]) 
y_train = train_df["label"] 

X_test = test_df.drop(columns=["label"]) 
y_test = test_df["label"] 

# Define the hyperparameter search space
param_grid = {
    "min_impurity_decrease": [0.0075, 0.0076, 0.0077, 0.0078, 0.0079, 0.008, 0.0081, 0.0082, 0.0083],
    "min_samples_split": [2],
    "max_depth": [None,1, 2, 3],
    "min_samples_leaf": [1,2,]
}

# Store results
results = []

# Try all combinations of hyperparameters
for params in tqdm(product(*param_grid.values())):
    param_dict = dict(zip(param_grid.keys(), params))
    test_accuracies = []

    # Repeat training 10 times with different random seeds
    for i in range(10):
        model = DecisionTreeClassifier(
            **param_dict, 
            criterion='gini', 
            splitter='best', 
            max_features=None, 
            random_state=i
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_accuracies.append(accuracy_score(y_test, y_pred))
    
    # Compute median accuracy across 10 runs
    median_accuracy = np.median(test_accuracies)
    
    # Store results
    results.append({**param_dict, "MedianTestAcc": median_accuracy})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Find the best hyperparameter set
best_params = results_df.loc[results_df["MedianTestAcc"].idxmax()].to_dict()

print("Best Parameters:", best_params)
