import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    res = np.ndarray((seq_length, d_model), dtype=np.float64)
    for i in range(len(res)):
        for j in range(0,len(res[i]), 2):
            res[i][j] = np.sin(i / np.power(10000, j/d_model))
            if j < len(res[i]) - 1:
                res[i][j + 1] = np.cos(i / np.power(10000, j/d_model))
    return res
            
    