import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A = np.asarray(A)

    range = np.arange(len(A))
    
    return np.sum(A[range, range])

    
