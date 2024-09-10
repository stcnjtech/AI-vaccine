import torch
import random
import warnings
import numpy as np

def vocab_config():
    vocab = {'C': 1,'W': 2,'V': 3,'A': 4,'H': 5,'T': 6,'E': 7,'K': 8,'N': 9,'P': 10,'I': 11,'L': 12,'S': 13,'D': 14,'G': 15,'Q': 16,'R': 17,'Y': 18,'F': 19,'M': 20,'O': 21,'X': 22,'B':23,'J':24,'U':25,'Z':26,'-': 0}
    return vocab,len(vocab)

def data_config():
    hla_max_len, pep_max_len, tcr_max_len = 34, 15, 34
    hla_pep_concat_len = hla_max_len + pep_max_len
    pep_tcr_concat_len = pep_max_len + tcr_max_len
    return hla_max_len,pep_max_len,tcr_max_len,hla_pep_concat_len,pep_tcr_concat_len

def model_config():
    d_model, d_ff, d_k, d_v, n_layers, n_heads = 64, 512, 64, 64, 1, 1
    epochs, batch_size, threshold, dropout_rate, max_len = 50, 1024, 0.5, 0.1, 5000
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    return d_model, d_ff, d_k, d_v, n_layers,n_heads,epochs, batch_size, threshold, dropout_rate, max_len,device

def run_config():
    warnings.filterwarnings("ignore")
    seed = 19961231
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed
