import torch.nn as nn
from OurModel.PepPredictMutationFramework.predict_model.transformer.trans1utils.MultiHeadAttention import MultiHeadAttention
from OurModel.PepPredictMutationFramework.predict_model.transformer.trans1utils.PosWiseFeedForwardNet import PosWiseFeedForwardNet
from OurModel.PepPredictMutationFramework.predict_model.transformer.trans1utils.PositionEmbedding import PositionEmbedding

class DecoderLayer(nn.Module):
    def __init__(self,n_heads,d_model,d_k,d_v,d_ff,device):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads,d_model,d_k,d_v,device)
        self.pos_wise_feed_forward_net = PosWiseFeedForwardNet(d_model,d_ff,device)

    def forward(self, dec_inputs, mask_matrix):
        """
        dec_inputs: [batch_size, concat_len, d_model]
        mask_matrix: [batch_size, concat_len, concat_len]
        """
        # dec_outputs: [batch_size, concat_len, d_model]
        # dec_self_attention: [batch_size, n_heads, concat_len, concat_len]
        dec_outputs, dec_self_attention \
            = self.multi_head_attention(dec_inputs, dec_inputs, dec_inputs, mask_matrix)
        # [batch_size, concat_len, d_model]
        dec_outputs = self.pos_wise_feed_forward_net(dec_outputs)

        return (
            dec_outputs,        # [batch_size, concat_len, d_model]
            dec_self_attention  # [batch_size, n_heads, concat_len, concat_len]
        )


class Decoder(nn.Module):
    def __init__(self,n_layers,n_heads,d_model,d_k,d_v,d_ff,dropout_rate,max_len,device):
        super(Decoder, self).__init__()

        self.device = device
        self.pos_emb = PositionEmbedding(d_model,dropout_rate,max_len)
        # 向Decoder中添加n_layers层DecoderLayer
        self.layers = nn.ModuleList([DecoderLayer(n_heads,d_model,d_k,d_v,d_ff,device) for _ in range(n_layers)])


    def forward(self, dec_inputs):
        """
        dec_inputs: [batch_size, concat_len,d_model]
        """
        # 位置编码
        # [batch_size, concat_len, d_model]
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(self.device)

        dec_self_attentions = []

        for layer in self.layers:
            # dec_outputs:[batch_size, concat_len, d_model]
            # dec_self_attn:[batch_size, n_heads, concat_len, concat_len]
            dec_outputs, dec_self_attention = layer(dec_outputs, None)
            dec_self_attentions.append(dec_self_attention)

        return (
            dec_outputs,            # [batch_size, concat_len, d_model],
            dec_self_attentions     # [n_layers,batch_size, n_heads, concat_len, concat_len]
        )