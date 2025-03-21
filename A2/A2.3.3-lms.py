import numpy as np

def lms(x, y, b, w_init, eta, epoch, decay=True):
    """
    Learn a binary classifier: single-example perceptron with margin
    Inputs:    x : a feature matrix containing an example on each row [pandas DataFrame of shape n x d]
               y : a vector with the class (either 1 or 0) of each example  [numpy array of size n]
               b : equality constraint value (same value for each training point) [float]
               w_init : a vector with the initial weight values (intercept in w_init[0]) [numpy array of size d+1]
               eta : an (initial) learning rate [float]
               epoch : the maximal number of iterations (1 epoch = 1 iteration
                       of the "repeat" loop in the lecture slides) [int]
               decay : a boolean [default=True], when True the learning rate at iteration k should be equal to eta/k. When False,
                       the learning rate should be equal to eta and should remain constant.
    Output:    A weight vector [list or numpy array of size d+1] (intercept in w[0])
    """

    y = 2 * y - 1

    N = x.shape[0]

    X = np.hstack((np.ones((N, 1)), x))
    w = w_init.copy()

    n = X.shape[0]

    k = 0

    for _ in range(epoch):
        i = (k % n)  # index of training example
        k = k + 1 # iteration counter

        # decay
        if decay:
            eta_bis = eta / k
        else:
            eta_bis = eta

        w_old = w.copy()

        # weight update 
        w = w + eta_bis * y[i] * X[i] * (1- (np.dot(np.transpose(w), X[i])*y[i]))

        # rescale to unit norm vector
        w = w / np.linalg.norm(w)

        termination_criterion = False

        if np.linalg.norm(w - w_old) < epsilon:
            break

    return w 

        
