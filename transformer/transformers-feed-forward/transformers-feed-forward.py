import numpy as np


def ReLU(x):
    return np.maximum(0.0, x)
    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here

    z = ReLU(np.matmul(x,W1) + b1)
    return np.matmul(z,W2) + b2

    