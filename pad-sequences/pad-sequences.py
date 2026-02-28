import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    # Bước 1: Xác định max_len (vẫn nên làm riêng để code sạch)
    if max_len is None:
        max_len = max(len(s) for s in seqs) if seqs else 0

    # Bước 2: List Comprehension
    # Logic: Nếu dài hơn -> cắt (truncate), nếu ngắn hơn -> bù (pad)
    result = [
        np.array(s[:max_len]) if len(s) >= max_len 
        else np.append(s, np.full(max_len - len(s), pad_value)) 
        for s in seqs
    ]

    return np.array(result)