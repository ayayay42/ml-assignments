from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC


# Load the test dataset
train_df = pd.read_csv("./Waveform_train.csv")
test_df = pd.read_csv("./Waveform_test.csv")

X_train = train_df.drop(columns=["labels"]) 
y_train = train_df["labels"] 

x_test = test_df.drop(columns=["labels"]) 
y_test = test_df["labels"] 

# Standardize using training set statistics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(x_test)

# Define the best models
best_linear_svm = SVC(kernel="linear")  # Assuming C=1 was used
best_poly_svm = SVC(kernel="poly", degree=2, coef0=14, C=5, gamma=0.0003)
best_rbf_svm = SVC(kernel="rbf", C=210, gamma=0.0001)
best_sigm_svm = SVC(kernel="sigmoid", C=95, gamma=0.01)

# Train models on the full training set
best_linear_svm.fit(X_train_scaled, y_train)
best_poly_svm.fit(X_train_scaled, y_train)
best_rbf_svm.fit(X_train_scaled, y_train)
best_sigm_svm.fit(X_train_scaled, y_train)

# Evaluate models on the test set
acc_linear = accuracy_score(y_test, best_linear_svm.predict(X_test_scaled))
acc_poly = accuracy_score(y_test, best_poly_svm.predict(X_test_scaled))
acc_rbf = accuracy_score(y_test, best_rbf_svm.predict(X_test_scaled))
acc_sigm = accuracy_score(y_test, best_sigm_svm.predict(X_test_scaled))

# Determine the best-performing model
accuracies = {"linear": acc_linear, "poly": acc_poly, "rbf": acc_rbf, "sigmoid": acc_sigm}
kernel_choice = max(accuracies, key=accuracies.get)

print("Test set accuracies:", accuracies)
print("Best performing model:", kernel_choice)


'''

best_linear_svm = SVC(kernel="linear")  # Assuming C=1 was used
best_poly_svm = SVC(kernel="poly", degree=2, coef0=0.6, C=0.1,)
best_rbf_svm = SVC(kernel="rbf", C=1, gamma=0.01)
best_sigm_svm = SVC(kernel="sigmoid", C=2.5, gamma=0.01)

Test set accuracies: {'linear': 0.871, 'poly': 0.87, 'rbf': 0.871, 'sigmoid': 0.868}
Best performing model: linear

note: rbf does best with C=2, i don't get it
'''
