import pandas as pd
import torch.utils.data as Data
from predict_model.retnet.retnet1util.Config import *


seed = run_config()
vocab,vocab_size = vocab_config()
hla_max_len,pep_max_len,tcr_max_len,hla_pep_concat_len,pep_tcr_concat_len = data_config()
d_model, d_ff, d_k, d_v, _,_,epochs, batch_size, threshold, dropout_rate, max_len,device = model_config()


def read_predict_data(predict_data, batch_size):
    hla_inputs,pep_inputs = hla_pep_make_data(vocab,predict_data,hla_max_len,pep_max_len)
    data_loader = Data.DataLoader(HLA_PEP_Final_DataSet(hla_inputs,pep_inputs), batch_size, shuffle=False, num_workers=0)
    return predict_data, hla_inputs, pep_inputs, data_loader

class HLA_PEP_Final_DataSet(Data.Dataset):
    def __init__(self, hla_inputs, pep_inputs):
        super(HLA_PEP_Final_DataSet, self).__init__()
        self.hla_inputs = hla_inputs
        self.pep_inputs = pep_inputs
    # 样本数
    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx]

class PEP_TCR_Final_DataSet(Data.Dataset):
    def __init__(self, pep_inputs, tcr_inputs):
        super(PEP_TCR_Final_DataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.tcr_inputs = tcr_inputs
    # 样本数
    def __len__(self):
        return self.pep_inputs.shape[0]
    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.tcr_inputs[idx]

def hla_pep_make_data(vocab, data, hla_max_len=34, pep_max_len=15):
    hla_inputs, pep_inputs = [], []
    for hla, pep in zip(data.hla, data.peptide):
        hla, pep = hla.ljust(hla_max_len, '-'), pep.ljust(pep_max_len, '-')
        hla_input = [[vocab[n] for n in hla]]
        pep_input = [[vocab[n] for n in pep]]
        hla_inputs.extend(hla_input)
        pep_inputs.extend(pep_input)
    return (
        torch.LongTensor(hla_inputs),   # [data_size, hla_max_len]
        torch.LongTensor(pep_inputs),   # [data_size, pep_max_len]
    )

def pep_tcr_make_data(vocab, data, pep_max_len=15, tcr_max_len=24):
    pep_inputs, tcr_inputs = [], []
    for pep, tcr in zip(data.peptide, data.tcr):
        pep, tcr = pep.ljust(pep_max_len, '-'), tcr.ljust(tcr_max_len, '-')
        pep_input = [[vocab[n] for n in pep]]
        tcr_input = [[vocab[n] for n in tcr]]
        pep_inputs.extend(pep_input)
        tcr_inputs.extend(tcr_input)
    return (
        torch.LongTensor(pep_inputs),   # [data_size, pep_max_len]
        torch.LongTensor(tcr_inputs),   # [data_size, tcr_max_len]
    )

def hla_pep_data_loader(file_name,vocab, hla_max_len=34, pep_max_len=15, batch_size=1024, train_rate=0.8):
    data = pd.read_csv("./data/{}.csv".format(file_name))
    data_size = data.shape[0]

    train_hla_inputs, train_pep_inputs = hla_pep_make_data(vocab, data[:int(data_size * train_rate)],hla_max_len, pep_max_len)
    test_hla_inputs, test_pep_inputs = hla_pep_make_data(vocab, data[int(data_size * train_rate):],hla_max_len, pep_max_len)

    train_loader = Data.DataLoader(HLA_PEP_Final_DataSet(train_hla_inputs, train_pep_inputs), batch_size,shuffle=False, num_workers=0)
    test_loader = Data.DataLoader(HLA_PEP_Final_DataSet(test_hla_inputs, test_pep_inputs), batch_size, shuffle=False,num_workers=0)

    return (
        train_loader,   # pep和hla:batch_num * [batch_size,xx_max_lex];  label:batch_num * [batch_size]
        test_loader     # pep和hla:batch_num * [batch_size,xx_max_lex];  label:batch_num * [batch_size]
    )

def pep_tcr_data_loader(file_name,vocab, pep_max_len=15, tcr_max_len=34, batch_size=1024, train_rate=0.8):
    data = pd.read_csv("./data/{}.csv".format(file_name))
    data_size = data.shape[0]

    train_pep_inputs, train_tcr_inputs = pep_tcr_make_data(vocab, data[:int(data_size * train_rate)], pep_max_len, tcr_max_len)
    test_pep_inputs, test_tcr_inputs = pep_tcr_make_data(vocab, data[int(data_size * train_rate):], pep_max_len, tcr_max_len)

    train_loader = Data.DataLoader(PEP_TCR_Final_DataSet(train_pep_inputs, train_tcr_inputs), batch_size,shuffle=False, num_workers=0)

    test_loader = Data.DataLoader(PEP_TCR_Final_DataSet(test_pep_inputs, test_tcr_inputs), batch_size, shuffle=False,num_workers=0)

    return (
        train_loader,   # [data_size]
        test_loader,     # pep和hla:batch_num * [batch_size,xx_max_lex];  label:batch_num * [batch_size]
    )
