import torch
import torch.nn as nn

class PosWiseFeedForwardNet(nn.Module):

    def __init__(self,d_model=64,d_ff=512,device=torch.device("cuda:0")):
        super(PosWiseFeedForwardNet, self).__init__()
        self.device = device
        self.d_model = d_model

        self.FC = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), # [d_model,d_ff]
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)  # [d_ff,d_model]
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, input_len, d_model]
        """
        residual = inputs

        # [batch_size, input_len, d_model]
        output = self.FC(inputs)

        # 对残差进行LayerNorm
        # [batch_size, input_len, d_model]
        output = nn.LayerNorm(self.d_model).to(self.device)(output + residual)
        return  output