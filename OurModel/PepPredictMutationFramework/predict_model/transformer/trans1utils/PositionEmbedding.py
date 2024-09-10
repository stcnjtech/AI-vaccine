import math
import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    def __init__(self,d_model=64,dropout_rate=0.1,max_len=5000):
        super(PositionEmbedding,self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        # 0矩阵:[max_len,d_model]
        pe = torch.zeros(max_len, d_model)
        # 计算position:[max_len,1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算div_term:[1,d_model/2=32]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 构建位置矩阵
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len,d_model/2=32]偶数
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len,d_model/2=32]奇数
        # [max_len,d_model]->[1,max_len,d_model]->[max_len,1,d_model]:正面翻转为侧面
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 注册为模型的缓冲区:不需要训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # 位置编码矩阵pe被添加到输入张量x的前seq_len行上,以使模型能够考虑序列中每个位置的位置信息。
        # [seq_len, batch_size, d_model] + [seq_len,1,d_model](=>[seq_len,batch_size,d_model])=[seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :, :]
        x = self.dropout(x) # 位置编码通过dropout,以减少过拟合
        return x # [seq_len, batch_size, d_model]