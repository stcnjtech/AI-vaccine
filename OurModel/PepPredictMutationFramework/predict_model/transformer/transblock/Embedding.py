import torch.nn as nn
from OurModel.PepPredictMutationFramework.predict_model.transformer.trans1utils.PositionEmbedding import PositionEmbedding

class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model,dropout_rate,max_len):
        super(Embedding,self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model,dropout_rate,max_len)

    def forward(self,emb_inputs):
        """
        emb_inputs: [batch_size, input_len]
        """
        # [batch_size, input_len, d_model]
        emb_outputs = self.src_emb(emb_inputs)
        # [batch_size, input_len, d_model]
        emb_outputs = self.pos_emb(emb_outputs.transpose(0, 1)).transpose(0, 1)
        return emb_outputs # [batch_size, input_len, d_model]

