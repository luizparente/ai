import numpy as np
import math

class AttentionModel:
    def __init__(self):
        pass

    def compute_attention(self, sentence, embedding_dim=4):
        # Tokenizing the sentence into words.
        tokens = sentence.split()
        n = len(tokens)
        
        # Generating random embeddings (seeded for reproducibility).
        np.random.seed(0)
        embeddings = np.random.rand(n, embedding_dim)
        
        # Defining weight matrices as identity.
        W_Q = np.eye(embedding_dim)
        W_K = np.eye(embedding_dim)
        W_V = np.eye(embedding_dim)
        
        # Computing Q, K, V.
        Q = embeddings.dot(W_Q)
        K = embeddings.dot(W_K)
        V = embeddings.dot(W_V)
        
        # Computing scaled dot-product scores.
        d_k = embedding_dim
        scores = Q.dot(K.T) / math.sqrt(d_k)
        
        # Softmax function for normalization.
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)
        
        # Generating attention matrix.
        attention_matrix = softmax(scores)
        
        # Computing output as weighted sum.
        output = attention_matrix.dot(V)
        
        return tokens, attention_matrix, output