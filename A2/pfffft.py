import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Define the training data: coordinates and class labels
X = np.array([[0, 3], [3, 0], [2, 1], [0, 2], [3, 3]])  # Coordinates (x1, x2)
y = np.array([1, -1, -1, 1, 1])  # Class labels (+1 for positive class, -1 for negative class)

# Train a linear SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Get the equation of the hyperplane: w.x + b = 0
w = clf.coef_[0]  # coefficients of the hyperplane
b = clf.intercept_[0]  # intercept term

# Print the equation of the hyperplane with x1 and x2 replaced by their coefficients
print(f"Equation of the maximal margin hyperplane:")
print(f"+ {w[0]:.1f} x1 + {w[1]:.1f} x2 + {b:.1f} = 0")

# Plot the points and the hyperplane
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o', label='Training points')

# Plot the hyperplane (w.x + b = 0)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.title('SVM Maximal Margin Hyperplane')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.savefig('ca.png')
