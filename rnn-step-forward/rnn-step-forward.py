import numpy as np

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Convert to array
   

    return tanh(Wx.T @ x_t + Wh.T @ h_prev + b)
    # return x_t

# rnn_step_forward(x_t, h_prev, Wx, Wh, b)
