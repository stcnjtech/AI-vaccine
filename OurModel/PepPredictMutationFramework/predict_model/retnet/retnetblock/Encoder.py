import torch.nn as nn
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet1util.MultiScaleRetention import MultiScaleRetention
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet1util.FeedForwardNet import FeedForwardNet

class EncoderLayer(nn.Module):
    def __init__(self,n_heads,d_model,d_k,d_v,d_ff,dropout_rate,max_len,device):
        super(EncoderLayer, self).__init__()
        self.device = device
        self.d_model = d_model

        self.multi_scale_retention = MultiScaleRetention(n_heads,d_model,d_k,d_v,dropout_rate,max_len,device)
        self.feed_forward_net = FeedForwardNet(d_model,d_ff,device)

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, len, d_model], enc_inputs to same Q,K,V
        """
        # enc_outputs: [batch_size, len, d_model]
        # self_retention: [batch_size, n_heads, len, len]
        enc_outputs, self_retention = self.multi_scale_retention(nn.LayerNorm(self.d_model).to(self.device)(enc_inputs))
        # enc_outputs: [batch_size, len, d_model]
        enc_outputs = enc_outputs + enc_inputs
        # enc_outputs: [batch_size, len, d_model]
        enc_outputs = self.feed_forward_net(nn.LayerNorm(self.d_model).to(self.device)(enc_outputs)) + enc_outputs
        return (
            enc_outputs,    # [batch_size, len, d_model]
            self_retention  # [batch_size, n_heads, len, len]
        )

class Encoder(nn.Module):
    def __init__(self,n_layers,n_heads,d_model,d_k,d_v,d_ff,device,dropout_rate,max_len):
        super(Encoder, self).__init__()
        # 向Encoder中添加n_layers层EncoderLayer
        self.layers = nn.ModuleList([EncoderLayer(n_heads,d_model,d_k,d_v,d_ff,dropout_rate,max_len,device) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, len, d_model]
        """

        enc_outputs = enc_inputs
        enc_self_retentions = []

        for layer in self.layers:
            # enc_outputs: [batch_size, len, d_model]
            # enc_self_attention: [batch_size, n_heads, len, len]
            enc_outputs, enc_self_retention = layer(enc_outputs) # D掩码在层内的MultiScaleRetention的Retention机制中实现
            enc_self_retentions.append(enc_self_retention)
        return (
            enc_outputs,        # [batch_size, len, d_model]
            enc_self_retentions # [n_layers, batch_size, n_heads, len, len]
        )
