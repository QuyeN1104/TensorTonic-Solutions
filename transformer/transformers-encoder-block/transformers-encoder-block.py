import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    return gamma * (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.var(x, axis=-1, keepdims=True) + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    Q = np.matmul(Q, W_q)
    K = np.matmul(K, W_k)
    V = np.matmul(V, W_v)

    batch_sz, sequences, dim_models = Q.shape
    Q = Q.reshape(batch_sz, sequences, num_heads, dim_models // num_heads)
    K = K.reshape(batch_sz, sequences, num_heads, dim_models // num_heads)
    V = V.reshape(batch_sz, sequences, num_heads, dim_models // num_heads)

    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    result_concat = (softmax(np.matmul(Q, K.transpose(0, 1, 3, 2))/ np.sqrt(dim_models // num_heads)) @ V).transpose(0,2,1,3).reshape(batch_sz, sequences, -1)

    return result_concat @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    return np.matmul(np.maximum(0, np.matmul(x, W1) + b1), W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    x = layer_norm(x + multi_head_attention(x, x, x,
                                           W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)

    return layer_norm(x + feed_forward(x, W1, b1, W2, b2), gamma2, beta2)