import torch.nn as nn
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet1util.RotaryPositionEmbedding import  RotaryPositionEmbedding
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet1util.MultiScaleRetention import MultiScaleRetention
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet1util.FeedForwardNet import FeedForwardNet

class DecoderLayer(nn.Module):
    def __init__(self,n_heads,d_model,d_k,d_v,d_ff,dropout_rate,max_len,device):
        super(DecoderLayer, self).__init__()
        self.device = device
        self.d_model = d_model

        self.multi_scale_retention = MultiScaleRetention(n_heads,d_model,d_k,d_v,dropout_rate,max_len,device)
        self.feed_forward_net = FeedForwardNet(d_model,d_ff,device)

    def forward(self, dec_inputs):
        """
        dec_inputs: [batch_size, concat_len, d_model]
        """
        # dec_outputs: [batch_size, concat_len, d_model]
        # dec_self_attention: [batch_size, n_heads, concat_len, concat_len]
        dec_outputs, dec_self_attention = self.multi_scale_retention(nn.LayerNorm(self.d_model).to(self.device)(dec_inputs))
        dec_outputs = dec_outputs + dec_inputs
        # [batch_size, concat_len, d_model]
        dec_outputs = self.feed_forward_net(nn.LayerNorm(self.d_model).to(self.device)(dec_outputs))

        return (
            dec_outputs,        # [batch_size, concat_len, d_model]
            dec_self_attention  # [batch_size, n_heads, concat_len, concat_len]
        )

class Decoder(nn.Module):
    def __init__(self,n_layers,n_heads,d_model,d_k,d_v,d_ff,dropout_rate,max_len,device):
        super(Decoder, self).__init__()

        self.device = device
        self.RoPE_emb = RotaryPositionEmbedding(d_model,dropout_rate,max_len)
        # 向Decoder中添加n_layers层DecoderLayer
        self.layers = nn.ModuleList([DecoderLayer(n_heads,d_model,d_k,d_v,d_ff,dropout_rate,max_len,device) for _ in range(n_layers)])


    def forward(self, dec_inputs):
        """
        dec_inputs: [batch_size, concat_len,d_model]
        """
        # 位置编码
        # [batch_size, concat_len, d_model]
        dec_outputs = self.RoPE_emb(dec_inputs).to(self.device)

        dec_self_retention_list = []

        for layer in self.layers:
            # dec_outputs: [batch_size, concat_len, d_model]
            # dec_self_retention: [batch_size, n_heads, concat_len, concat_len]
            dec_outputs, dec_self_retention = layer(dec_outputs)
            dec_self_retention_list.append(dec_self_retention)

        return (
            dec_outputs,                # [batch_size, concat_len, d_model],
            dec_self_retention_list     # [n_layers,batch_size, n_heads, concat_len, concat_len]
        )