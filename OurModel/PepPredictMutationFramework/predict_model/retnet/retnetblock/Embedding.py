import torch.nn as nn
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet1util.RotaryPositionEmbedding import RotaryPositionEmbedding

class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model,dropout_rate,max_len):
        super(Embedding,self).__init__()
        # 嵌入d_model层
        self.src_emb = nn.Embedding(vocab_size, d_model)
        # 旋转位置编码(RoPE)
        self.RoPE_emb = RotaryPositionEmbedding(d_model,dropout_rate,max_len)

    def forward(self,emb_inputs):
        """
        emb_inputs: [batch_size, len]
        """
        # [batch_size, len, d_model]
        emb_outputs = self.src_emb(emb_inputs)
        # [batch_size, len, d_model]
        emb_outputs = self.RoPE_emb(emb_outputs)
        return emb_outputs # [batch_size, len, d_model]

if __name__ == '__main__':
    import torch
    import numpy as np
    x = np.ones((1024, 15))
    x = torch.LongTensor(x)
    print(x.shape)
    emb = Embedding(20,4,0.1,512)
    res = emb(x)
    print(res.shape)  # 结果shape与输入的shape保持不变