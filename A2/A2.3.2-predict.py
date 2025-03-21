import numpy as np


def predict(w, x):
    x = np.hstack((np.ones((x.shape[0], 1)), x))  
    predictions = np.dot(x, w)
    
    labels = np.zeros_like(predictions, dtype=int)
    for i in range(len(predictions)):
        if predictions[i] > 0:
            labels[i] = 1
    
    return labels