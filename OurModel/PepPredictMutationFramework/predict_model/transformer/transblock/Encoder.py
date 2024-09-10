import torch.nn as nn
from OurModel.PepPredictMutationFramework.predict_model.transformer.trans1utils.Get_attn_pad_mask import get_attn_pad_mask
from OurModel.PepPredictMutationFramework.predict_model.transformer.trans1utils.PosWiseFeedForwardNet import PosWiseFeedForwardNet
from OurModel.PepPredictMutationFramework.predict_model.transformer.trans1utils.MultiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self,n_heads,d_model,d_k,d_v,d_ff,device):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads,d_model,d_k,d_v,device)
        self.pos_wise_feed_forward_net = PosWiseFeedForwardNet(d_model,d_ff,device)

    def forward(self, enc_inputs, mask_matrix):
        """
        enc_inputs: [batch_size, input_len, d_model]  , enc_inputs to same Q,K,V
        mask_matrix: [batch_size, input_len, input_len]
        """
        # enc_outputs: [batch_size, input_len, d_model]
        # self_attention: [batch_size, n_heads, input_len, input_len]
        enc_outputs, self_attention = self.multi_head_attention(enc_inputs, enc_inputs, enc_inputs,mask_matrix)
        # enc_outputs: [batch_size, input_len, d_model]
        enc_outputs = self.pos_wise_feed_forward_net(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return (
            enc_outputs,    # [batch_size, input_len, d_model]
            self_attention  # [batch_size, n_heads, input_len, input_len]
        )


class Encoder(nn.Module):
    def __init__(self,n_layers,n_heads,d_model,d_k,d_v,d_ff,device):
        super(Encoder, self).__init__()
        # 向Encoder中添加n_layers层EncoderLayer
        self.layers = nn.ModuleList([EncoderLayer(n_heads,d_model,d_k,d_v,d_ff,device) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, input_len,d_model]
        """

        # [batch_size, input_len, input_len]
        mask_matrix = get_attn_pad_mask(enc_inputs[:,:,0], enc_inputs[:,:,0])

        enc_outputs = enc_inputs
        enc_self_attentions = []

        for layer in self.layers:
            # enc_outputs: [batch_size, input_len, d_model]
            # enc_self_attention: [batch_size, n_heads, input_len, input_len]
            enc_outputs, enc_self_attention = layer(enc_outputs, mask_matrix)
            enc_self_attentions.append(enc_self_attention)
        return (
            enc_outputs,        # [batch_size, input_len, d_model]
            enc_self_attentions # [n_layers, batch_size, n_heads, input_len, input_len]
        )