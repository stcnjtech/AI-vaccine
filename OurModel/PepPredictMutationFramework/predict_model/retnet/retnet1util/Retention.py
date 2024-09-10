import torch
import torch.nn as nn
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet1util.RotaryPositionEmbedding import RotaryPositionEmbedding
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet1util.Get_D import get_D

class Retention(nn.Module):
    def __init__(self, gamma,d_model=64,d_k=64,d_v=64,dropout_rate=0.1,max_len=512,device=torch.device("cuda:0")):
        super(Retention, self).__init__()
        self.device = device
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(d_model, d_k) / d_model)
        self.W_K = nn.Parameter(torch.randn(d_model, d_k) / d_model)
        self.W_V = nn.Parameter(torch.randn(d_model, d_v) / d_model)

        self.RoPE = RotaryPositionEmbedding(d_model,dropout_rate,max_len)

    def forward(self, x):
        """
        x: [batch_size, len, d_model]
        """
        seq_len = x.shape[1]

        D = get_D(self.gamma,seq_len,self.device)                       # [len,len]

        Q = (x @ self.W_Q)                                  # [batch_size,len,d_k]
        K = (x @ self.W_K)                                  # [batch_size,len,d_k]

        Q = self.RoPE(Q)                                    # [batch_size,len,d_k]
        K = self.RoPE(K, downscale=True)                    # [batch_size,len,d_k]

        V = x @ self.W_V                                    # [batch_size,len,d_v]
        self_retention = Q @ K.transpose(-1, -2)            # [batch_size,len,len]
        self_retention = self_retention * D.unsqueeze(0)    # [batch_size,len,len]
        output = self_retention @ V                         # [batch_size,len,d_v]

        return (
            output,             # [batch_size,len,d_v]
            self_retention      # [batch_size,len,len]
        )

    def forward_recurrent(self, x_n, s_n_1, n):
        """
        x_n: [batch_size, 1, d_model]
        s_n_1: [batch_size, d_k, d_v]
        """

        Q = (x_n @ self.W_Q)                    # [batch_size, 1, d_k]
        K = (x_n @ self.W_K)                    # [batch_size, 1, d_k]

        Q = self.xpos(Q, n + 1)                 # [batch_size, 1, d_k]
        K = self.xpos(K, n + 1, downscale=True) # [batch_size, 1, d_k]

        V = x_n @ self.W_V                      # [batch_size, 1, d_v]

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)    # [batch_size, d_k, d_v]
        output = Q @ s_n # [batch_size, 1, d_v]

        return (
            output, # [batch_size, 1, d_v]
            s_n     # [batch_size, d_k, d_v]
        )