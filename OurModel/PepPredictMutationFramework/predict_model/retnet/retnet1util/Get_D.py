import torch

def get_D(gamma,seq_len,device):
    n = torch.arange(seq_len).unsqueeze(1)
    m = torch.arange(seq_len).unsqueeze(0)
    D = (gamma ** torch.abs(n - m))
    # D = (gamma ** (n - m)) * (n >= m).float()
    D[D != D] = 0
    D = D.to(device)
    return D    # [len,len]
