import numpy as np

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Convert to array
    x_t = np.array(x_t, dtype=np.float32)
    h_prev = np.array(h_prev, dtype=np.float32)
    Wx = np.array(Wx, dtype=np.float32)
    Wh = np.array(Wh, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    return tanh(Wx.T @ x_t + Wh.T @ h_prev + b)
    # return x_t

# rnn_step_forward(x_t, h_prev, Wx, Wh, b)
