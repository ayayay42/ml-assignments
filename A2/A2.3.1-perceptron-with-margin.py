import numpy as np

def perceptron(x, y, b, w_init, eta, epoch, decay=False):
    """
    Learn a binary classifier: single-example perceptron with margin
    Inputs:
        x : a feature matrix containing an example on each row [pandas DataFrame of shape n x d]
        y : a vector with the class (either 1 or 0) of each example  [numpy array of size n]
        b : a margin [float]
        w_init : a vector with the initial weight values (intercept in w_init[0]) [numpy array of size d+1]
        eta : an (initial) learning rate [float]
        epoch : the maximal number of iterations (1 epoch = 1 iteration of the "repeat" loop in the lecture slides) [int]
        decay : a boolean [default=False], when True the learning rate at iteration k should be equal to eta/k.
                When False, the learning rate should remain constant.
    Output:
        A weight vector [numpy array of size d+1] (intercept in w[0])
    """
    y = 2 * y - 1  #so we have -1 and 1
    N = x.shape[0]
    
    X = np.hstack((np.ones((N, 1)), x)) 
    w = w_init.copy() #initialize bold w
    
    n = X.shape[0]

    k = 0
    i = 0
    
    for _ in range(epoch):
        i = (k%n) #index of training example
        k = k + 1 #iteration counter 

        #decay
        if decay:
            eta_bis = eta/k
        else:
            eta_bis = eta

        #is xi misclassified or classified with a too small margin?
        if np.dot(np.transpose(w), X[i])*y[i] <= b:
            w = w + eta_bis * y[i] * X[i] #weight update

        for j in range(n):
            if np.dot(np.transpose(w), X[j])*y[j] > b:
                break
    return w