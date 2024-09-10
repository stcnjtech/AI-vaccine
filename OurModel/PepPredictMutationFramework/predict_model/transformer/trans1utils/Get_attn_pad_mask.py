def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be input_len or it could be concat_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # [batch_size, 1, len_k]
    # True is masked
    pad_attn_mask = seq_k.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True(0) is masked
    # mask矩阵拓展到[batch_size, len_q, len_k]
    mask_matrix = pad_attn_mask.expand(batch_size, len_q, len_k) # [batch_size, len_q, len_k]
    return mask_matrix # [batch_size, len_q, len_k]