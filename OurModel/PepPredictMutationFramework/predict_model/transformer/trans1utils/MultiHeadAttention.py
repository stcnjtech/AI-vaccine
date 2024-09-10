import torch
import torch.nn as nn
from OurModel.PepPredictMutationFramework.predict_model.transformer.trans1utils.ScaledDotProductAttention import  ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads=1,d_model=64,d_k=64,d_v=64,device=torch.device("cuda:0")):
        super(MultiHeadAttention,self).__init__()
        self.device = device

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # [d_model, d_k * n_heads]
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)  # [d_model, d_k * n_heads]
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)  # [d_model, d_v * n_heads]

        self.FC = nn.Linear(n_heads * d_v, d_model, bias=False)   # [n_heads * d_v, d_k * n_heads]

    def forward(self,q,k,v,mask_matrix):
        """
        q: [batch_size, len_q, d_model]
        k: [batch_size, len_k, d_model]
        v: [batch_size, len_v(=len_k), d_model]
        mask_matrix: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = q, q.size(0)

        # Q=q * W_Q => 拆分 => 交换1号位和2号位:
        # [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # K=k * W_K => 拆分 => 交换1号位和2号位:
        # [batch_size, n_heads, len_k, d_k]
        K = self.W_K(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # V=v * W_V => 拆分 => 交换1号位和2号位:
        # [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # mask_matrix : [batch_size, n_heads, seq_len, seq_len]
        if mask_matrix is not None:
            mask_matrix = mask_matrix.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # output: [batch_size, n_heads, len_q, d_v]
        # self_attention: [batch_size, n_heads, len_q, len_k]
        output, self_attention = ScaledDotProductAttention()(Q, K, V, mask_matrix)
        # self_attention: [batch_size, len_q, n_heads * d_v]
        output = output.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)

        # [batch_size, len_q, d_model]
        output = self.FC(output)
        output = nn.LayerNorm(self.d_model).to(self.device)(output + residual)
        return (
            output,  # [batch_size, len_q, d_model]
            self_attention  # [batch_size, n_heads, len_q, len_k]
        )