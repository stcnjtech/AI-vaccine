import torch
import pandas as pd
import torch.utils.data as Data

class HLA_PEP_DataSet(Data.Dataset):
    def __init__(self, hla_inputs, pep_inputs, hla_pep_labels):
        super(HLA_PEP_DataSet, self).__init__()
        self.hla_inputs = hla_inputs
        self.pep_inputs = pep_inputs
        self.hla_pep_labels = hla_pep_labels
    # 样本数
    def __len__(self):
        return self.pep_inputs.shape[0]
    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx], self.hla_pep_labels[idx]

class PEP_TCR_DataSet(Data.Dataset):
    def __init__(self, pep_inputs, tcr_inputs, pep_tcr_labels):
        super(PEP_TCR_DataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.tcr_inputs = tcr_inputs
        self.pep_tcr_labels = pep_tcr_labels
    # 样本数
    def __len__(self):
        return self.pep_inputs.shape[0]
    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.tcr_inputs[idx], self.pep_tcr_labels[idx]


def hla_pep_make_data(vocab, data, hla_max_len=34, pep_max_len=15):
    hla_inputs, pep_inputs, labels = [], [], []
    for hla, pep, label in zip(data.hla, data.peptide, data.label):
        hla, pep = hla.ljust(hla_max_len, '-'), pep.ljust(pep_max_len, '-')
        hla_input = [[vocab[n] for n in hla]]
        pep_input = [[vocab[n] for n in pep]]
        hla_inputs.extend(hla_input)
        pep_inputs.extend(pep_input)
        labels.append(label)
    return (
        torch.LongTensor(hla_inputs),   # [data_size, hla_max_len]
        torch.LongTensor(pep_inputs),   # [data_size, pep_max_len]
        torch.LongTensor(labels)        # [data_size]
    )
def pep_tcr_make_data(vocab, data, pep_max_len=15, tcr_max_len=24):
    pep_inputs, tcr_inputs, labels = [], [], []
    for pep, tcr, label in zip(data.peptide, data.tcr, data.label):
        pep, tcr = pep.ljust(pep_max_len, '-'), tcr.ljust(tcr_max_len, '-')
        pep_input = [[vocab[n] for n in pep]]
        tcr_input = [[vocab[n] for n in tcr]]
        pep_inputs.extend(pep_input)
        tcr_inputs.extend(tcr_input)
        labels.append(label)
    return (
        torch.LongTensor(pep_inputs),   # [data_size, pep_max_len]
        torch.LongTensor(tcr_inputs),   # [data_size, hla_max_len]
        torch.LongTensor(labels)        # [data_size]
    )


def data_with_loader(vocab,type_ = 'train',fold = None,hla_max_len=34,pep_max_len=15,batch_size = 1024):
    data = None
    if type_ == 'train':
        data = pd.read_csv('./data/train_data_fold{}.csv'.format(fold), index_col = 0)
    elif type_ == 'val':
        data = pd.read_csv('./data/val_data_fold{}.csv'.format(fold), index_col = 0)

    hla_inputs, pep_inputs, labels = hla_pep_make_data(vocab,data,hla_max_len,pep_max_len)
    loader = Data.DataLoader(HLA_PEP_DataSet(hla_inputs, pep_inputs, labels), batch_size, shuffle = False, num_workers = 0)

    return loader
def hla_pep_data_loader(file_name,vocab, hla_max_len=34, pep_max_len=15, batch_size=1024, train_rate=0.8):
    data = pd.read_csv("./data/{}.csv".format(file_name))
    data_size = data.shape[0]

    train_hla_inputs, train_pep_inputs, train_labels = hla_pep_make_data(vocab, data[:int(data_size * train_rate)],hla_max_len, pep_max_len)
    test_hla_inputs, test_pep_inputs, test_labels = hla_pep_make_data(vocab, data[int(data_size * train_rate):],hla_max_len, pep_max_len)

    train_loader = Data.DataLoader(HLA_PEP_DataSet(train_hla_inputs, train_pep_inputs, train_labels), batch_size,shuffle=False, num_workers=0)
    test_loader = Data.DataLoader(HLA_PEP_DataSet(test_hla_inputs, test_pep_inputs, test_labels), batch_size, shuffle=False,num_workers=0)

    return (
        train_loader,   # pep和hla:batch_num * [batch_size,xx_max_lex];  label:batch_num * [batch_size]
        test_loader     # pep和hla:batch_num * [batch_size,xx_max_lex];  label:batch_num * [batch_size]
    )
def pep_tcr_data_loader(file_name,vocab, pep_max_len=15, tcr_max_len=34, batch_size=1024, train_rate=0.8):
    data = pd.read_csv("D:/ProjectsSTC/PepPredictMutationFramework/predict_model/data/{}.csv".format(file_name))
    data_size = data.shape[0]

    train_pep_inputs, train_tcr_inputs, train_labels = pep_tcr_make_data(vocab, data[:int(data_size * train_rate)], pep_max_len, tcr_max_len)
    test_pep_inputs, test_tcr_inputs, test_labels = pep_tcr_make_data(vocab, data[int(data_size * train_rate):], pep_max_len, tcr_max_len)

    train_loader = Data.DataLoader(PEP_TCR_DataSet(train_pep_inputs, train_tcr_inputs, train_labels), batch_size,shuffle=False, num_workers=0)

    test_loader = Data.DataLoader(PEP_TCR_DataSet(test_pep_inputs, test_tcr_inputs, test_labels), batch_size, shuffle=False,num_workers=0)

    return (
        train_loader,   # [data_size]
        test_loader,     # pep和hla:batch_num * [batch_size,xx_max_lex];  label:batch_num * [batch_size]
    )