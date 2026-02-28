# I edited this file in github , haha
import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    y = y.reshape(-1,1)
    
    # Initial params
    w = np.zeros((X.shape[1], 1))
    b = 0.0

    m = X.shape[0]
    
    for i in range(steps):
        z = X @ w + b
        y_hat = _sigmoid(z)
        
        dz = y_hat - y
        db = dz.mean()
        dw = X.T @ dz / m
        
        b -= lr*db
        w -= lr*dw
    w = w.flatten()
    print(w.shape)
    return w, b
        

    
