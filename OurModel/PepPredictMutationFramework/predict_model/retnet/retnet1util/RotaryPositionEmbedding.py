import torch
import torch.nn as nn


def fixed_pos_embedding(scale):
    """
    scale:[len,d_model/2]
    """
    seq_len, d_model_half = scale.shape
    # 计算每个维度的频率因子inv_freq
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model_half) / d_model_half))
    # 生成正弦和余弦的位置编码
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(scale)
    return (
        torch.sin(sinusoid_inp),        # [len, d_model/2]
        torch.cos(sinusoid_inp)         # [len, d_model/2]
    )
def rotate_every_two(x):
    # [batch_size, len, d_model]
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2) # [batch_size, len * d_model]
def duplicate_interleave(m):
    dim0 = m.shape[0]
    m = m.view(-1, 1)
    m = m.repeat(1, 2)
    m = m.view(dim0, -1)
    return m
def apply_rotary_pos_emb(x, sin, cos, scale):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    return (x * cos) + (rotate_every_two(x) * sin)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model=64, dropout_rate=0.1, max_len=512):
        super(RotaryPositionEmbedding,self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        # self.scale: [d_model/2]
        self.register_buffer("scale", (torch.arange(0, d_model, 2) + 0.4 * d_model) / (1.4 * d_model))

    def forward(self, x, offset=0, downscale=False):
        """
        x: [batch_size, len, d_model]
        """
        length = x.shape[1]
        # 扩充length
        max_pos = length + offset

        # scale: [len, d_model/2]
        scale = self.scale ** torch.arange(0, max_pos, 1).to(self.scale).div(self.max_len)[:, None]

        # sin: [len, d_model/2]
        # cos: [len, d_model/2]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)    # [batch_size,len,d_model]
        x = self.dropout(x)                             # [batch_size,len,d_model]
        return x    # [batch_size, len, d_model]

if __name__ == '__main__':
    RoPE = RotaryPositionEmbedding(d_model=4)
    x = torch.ones((1,20,4))
    res = RoPE(x,0,False)
    print(res.shape)
