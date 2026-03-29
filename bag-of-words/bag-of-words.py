import numpy as np
from collections import Counter

def bag_of_words_vector(tokens, vocab):
    # Trả về mảng 0 nếu vocab rỗng, đảm bảo dtype là int
    if not vocab: 
        return np.array([], dtype=int)
    
    # Đếm số lần xuất hiện của mỗi token một lần duy nhất: O(N)
    counts = Counter(tokens)
    
    # Tạo vector dựa trên vocab: O(M)
    return np.array([counts[word] for word in vocab], dtype=int)