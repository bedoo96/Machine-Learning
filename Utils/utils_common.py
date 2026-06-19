import matplotlib.pyplot as plt
import numpy as np

###########################################
#    Regression Routines
###########################################

# Cost function for linear regression calculation
def compute_cost(X, y, w, b, verbose=False):
    """
    Computes the cost function for linear regression.

    Args:
        X (ndarray): Input features, shape (m, n).
        y (ndarray): Target values, shape (m,).
        w (ndarray): Weights, shape (n,).
        b (float): Bias term.
        verbose (bool): If True, prints the cost value f_wb.
    Returns:
        float: The cost function value.
    """

    m = X.shape[0] # Number of training examples
    # calculate the cost function for all examples
    f_wb = X @ w + b  # Predicted values
    error = f_wb - y # Error between predicted and actual values
    total_cost = (1/(2*m)) * np.sum (error**2) # Mean Squared Error Cost Function

    if verbose:
        print(f"Cost at w={w}, b={b}: {total_cost}")

    return total_cost


def compute_gradient_matrix(X, y, w, b):
    """
    Computes the gradient for linear regression.

    Args:
        X (ndarray (m.n)): Input features, shape (m, n), m examples with n features.
        y (ndarray (m,)): Target values, shape (m,).
        w (ndarray (n, )): Weights, shape (n,).
        b (scalar): Bias term.

    Returns:
        dj_dw (ndarray (n, 1)): Gradient with respect to weights, shape (n,).
        dj_db (scalar): Gradient with respect to bias.
    """

    m,n = X.shape # Number of training examples and number of features
    f_wb = X @ w + b #
    error = f_wb - y
    dj_dw = (1/m) * (X.T @ error)# Gradient with respect to weights
    dj_db = (1/m) * np.sum(error) # Gradient with respect to bias

    return dj_dw, dj_db

# loop version of multiple-variable compute cost

def compute_cost(X, y, w, b) : 
    """
    Computes the cost function for linear regression using a loop.

    Args:
        X (ndarray (m,n)): Input features, shape (m, n).
        y (ndarray (m,)): Target values, shape (m,).
        w (ndarray (n, )): Weights, shape (n,).
        b (scalar): Bias term.

    Returns:
        scalar: The cost function value.
    """
    
    m = X.shape[0] # Number of training examples
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b # Predicted value for the i-th example  
        cost = cost + (f_wb_i - y[i]) ** 2 # Squared error for the i-th example
    cost = cost / (2 * m) # Average cost for all examples
    return cost


# Compute gradient (X, y, w, b):

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression using a loop.

    Args:
        X (ndarray (m,n)): Input features, shape (m, n).
        y (ndarray (m,)): Target values, shape (m,).
        w (ndarray (n, )): Weights, shape (n,).
        b (scalar): Bias term.

    Returns:
        dj_dw (ndarray (n, )): Gradient with respect to weights, shape (n,).
        dj_db (scalar): Gradient with respect to bias.
    """
    m,n = X.shape # Number of training examples and number of features
    dj_dw = np.zeros(n) # Initialize gradient for weights
    dj_db = 0.0 # Initialize gradient for bias

    for i in range(m):
        err = np.dot(X[i], w) + b - y[i] # Error for the i-th example
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X [i,j] # Gradient with respect to weights
        dj_db = dj_db + err # Gradient with respect to bias
    dJ_dw = dj_dw / m # Average gradient for weights
    dJ_db = dj_db / m # Average gradient for bias

    return dJ_dw, dJ_db








