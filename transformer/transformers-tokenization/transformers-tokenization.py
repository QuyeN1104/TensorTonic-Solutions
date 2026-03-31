import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # 1. Initialize special tokens
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        
        cnt = 4 
        
        # 2. Iterate through sentences, then split into words
        for text in texts:
            for word in text.split():
                # 3. Only add the word if it doesn't already exist
                if word not in self.word_to_id:
                    self.word_to_id[word] = cnt
                    self.id_to_word[cnt] = word
                    cnt += 1
                    
        # 4. Update vocabulary size
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        encoded_id = []
        for word in text.split():
            encoded_id.append(self.word_to_id.get(word, 1)) # 1 is UNK ID
            
        return encoded_id
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # Using join prevents trailing spaces and .get() protects against KeyError
        words = [self.id_to_word.get(token_id, self.unk_token) for token_id in ids]
        return " ".join(words)