import math
import torch
import torch.nn as nn
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet1util.Retention import Retention

class MultiScaleRetention(nn.Module):
    def __init__(self, n_heads=1,d_model=64,d_k=64,d_v=64,dropout_rate=0.1,max_len=512,device=torch.device("cuda:0")):
        super(MultiScaleRetention, self).__init__()
        self.device = device

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        # [n_heads]
        self.gammas = (1-torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), n_heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(d_model, n_heads * d_v) / d_model)
        self.W_O = nn.Parameter(torch.randn(n_heads * d_v, d_model) / d_model)

        self.retentions = nn.ModuleList([Retention(gamma,d_model,d_k,d_v,dropout_rate,max_len,device) for gamma in self.gammas])

    def forward(self, x):
        """
        x: [batch_size,len,d_model]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        output_list = []
        self_retention_list = []

        for i in range(self.n_heads):
            # output_list: [batch_size,len,d_v] * n_heads
            # self_retention_list: [batch_size,len,len] * n_heads
            ret_output,self_retention = self.retentions[i](x)
            output_list.append(ret_output)
            self_retention_list.append(self_retention)

        # [batch_size,len,n_heads * len]
        self_retentions = torch.cat(self_retention_list,dim=2)
        # [batch_size,n_heads,len,len]
        self_retentions = self_retentions.view(batch_size, -1, self.n_heads,seq_len).transpose(1, 2)

        # [batch_size,len,n_heads * d_v]
        output = torch.cat(output_list, dim=2)
        output_shape = output.shape

        # [batch_size,len,n_heads * d_v]
        Y = nn.GroupNorm(self.n_heads, self.n_heads).to(self.device)(output.reshape(-1, self.n_heads)).reshape(output_shape)

        # [batch_size,len,d_model]
        output = (self.swish(x @ self.W_G) * Y) @ self.W_O

        return (
            output,             # [batch_size,len,d_model]
            self_retentions     # [batch_size,n_heads,len,len]
        )

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        x_n: [batch_size, 1, d_model]
        s_n_1s: [n_heads, batch_size, d_k, d_v]
        """
        Y = []
        s_ns = []
        for i in range(self.n_heads):
            # y: [batch_size,1,d_v]; s_n: [batch_size, d_k, d_v]
            y, s_n = self.retentions[i].forward_recurrent(x_n, s_n_1s[i], n)
            Y.append(y)         # [batch_size,1,d_v] * n_heads
            s_ns.append(s_n)    # [n_heads, batch_size, d_k, d_v]

        Y = torch.cat(Y, dim=2) # [batch_size,1,n_heads * d_v]
        Y_shape = Y.shape
        Y = nn.GroupNorm(self.n_heads, self.n_heads).to(self.device)(Y.reshape(-1, self.n_heads)).reshape(Y_shape)
        output = (self.swish(x_n @ self.W_G) * Y) @ self.W_O    # [batch_size, 1, d_model]
        return (
            output,     # [batch_size, 1, d_model]
            s_ns        # [n_heads, batch_size, d_k, d_v]
        )