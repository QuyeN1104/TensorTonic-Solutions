import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"W_q shape: {W_q.shape}")
    print(f"W_k shape: {W_k.shape}" )
    print(f"W_v shape: {W_v.shape}")

    print(f"W_o shape: {W_o.shape}")

    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    batch_size, seqs, d_models = Q.shape
    d_k = d_models // num_heads

    Q_proj = Q_proj.reshape(batch_size, seqs, num_heads, d_k)
    K_proj = K_proj.reshape(batch_size, seqs, num_heads, d_k)
    V_proj = V_proj.reshape(batch_size, seqs, num_heads, d_k)

    Q_proj = Q_proj.transpose(0, 2, 1, 3)
    K_proj = K_proj.transpose(0, 2, 1, 3)
    V_proj = V_proj.transpose(0, 2, 1, 3)

    res = softmax((np.matmul(Q_proj, K_proj.transpose(0, 1, 3, 2)))/ np.sqrt(d_k)) @ V_proj
    res = res.transpose(0, 2, 1, 3).reshape(batch_size, seqs, -1)
    final_res = np.matmul(res, W_o)
    return final_res
