import torch
import numpy as np
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask_matrix):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        mask_matrix: [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = K.size(-1)
        # [batch_size, n_heads, len_q, len_k]
        self_attention = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 将self_attention中与掩码mask_matrix对应的True位置置为一个较大的负数
        if mask_matrix is not None:
            self_attention.masked_fill_(mask_matrix, -1e9)

        self_attention = nn.Softmax(dim=-1)(self_attention)
        output = torch.matmul(self_attention, V)  # [batch_size, n_heads, len_q, d_v]
        return (
            output,         # [batch_size, n_heads, len_q, d_v]
            self_attention  # [batch_size, n_heads, len_q, len_k]
        )
