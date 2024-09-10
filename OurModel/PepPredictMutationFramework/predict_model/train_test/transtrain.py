import os
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from transformer.trans1utils.Config import *
from transformer.transformer.Transformer import Transformer
from loader.DataGenerater import *

vocab,vocab_size = vocab_config()
hla_max_len,pep_max_len,tcr_max_len,hla_pep_concat_len,pep_tcr_concat_len = data_config()
d_model, d_ff, d_k, d_v, _, n_heads, epochs, batch_size, threshold, dropout_rate, max_len, device = model_config()
seed = run_config()

train_pep_tcr_loader,test_pep_tcr_loader = pep_tcr_data_loader('pep_tcr_dataset',vocab,pep_max_len,tcr_max_len,batch_size,0.9)


kf = KFold(n_splits=5)
n_layers = 1
for n_heads in range(1, 10):
    for fold, (train_index, val_index) in enumerate(kf.split(train_pep_tcr_loader.dataset)):

        train_pep_tcr_set = train_pep_tcr_loader.dataset[train_index]
        val_pep_tcr_set = train_pep_tcr_loader.dataset[val_index]

        train_pep_tcr = Data.DataLoader(PEP_TCR_DataSet(train_pep_tcr_set[0], train_pep_tcr_set[1], train_pep_tcr_set[2]),
                              batch_size, shuffle=False, num_workers=0)
        val_pep_tcr = Data.DataLoader(PEP_TCR_DataSet(val_pep_tcr_set[0], val_pep_tcr_set[1], val_pep_tcr_set[2]),
                              batch_size, shuffle=False, num_workers=0)

        model = Transformer(
            vocab_size=vocab_size,
            d_model=64,
            n_enc_layers=n_layers,
            n_enc_heads=n_heads,
            n_dec_layers=n_layers,
            n_dec_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            hla_pep_concat_len=hla_pep_concat_len,
            pep_tcr_concat_len=pep_tcr_concat_len,
            dropout_rate=dropout_rate,
            max_len=max_len,
            device=device).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        dir_saver = './transformer/model_1layers/'
        path_saver = './transformer/model_1layers/model_pt_layer{}_head{}_fold{}.pth'.format(n_layers, n_heads, fold)

        metric_best, ep_best, time_train = 0, -1, 0
        for epoch in range(1, epochs + 1):
            _, _, _ = model.train_per_epoch(train_pep_tcr, epoch, epochs, criterion, optimizer, num_group=2,
                                        threshold=threshold, metrics_print_=True)

            _, _, metrics_val = model.eval_per_epoch(val_pep_tcr, epoch, epochs, criterion, num_group=2, threshold=threshold,
                                       metrics_print_=True)

            metrics_ep_avg = sum(metrics_val[:4]) / 4

            if metrics_ep_avg > metric_best:
                metric_best, ep_best = metrics_ep_avg, epoch
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                print('Best epoch = {} | Best metrics_ep_avg = {:.4f}--------'.format(ep_best, metric_best))
                print('Saving model Path saver: {} --------'.format(path_saver))
                torch.save(model.state_dict(), path_saver)


for n_heads in range(1, 10):
    for fold in range(5):

        train_loader = data_with_loader(vocab, 'train', fold, hla_max_len, pep_max_len, batch_size)
        val_loader = data_with_loader(vocab, 'val', fold, hla_max_len, pep_max_len, batch_size)

        model = Transformer(
            vocab_size=vocab_size,
            d_model=64,
            n_enc_layers=n_layers,
            n_enc_heads=n_heads,
            n_dec_layers=n_layers,
            n_dec_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            hla_pep_concat_len=hla_pep_concat_len,
            pep_tcr_concat_len=pep_tcr_concat_len,
            dropout_rate=dropout_rate,
            max_len=max_len,
            device=device).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        dir_saver = './transformer/model_1layers/'
        path_saver = './transformer/model_1layers/model_hp_layer{}_head{}_fold{}.pth'.format(n_layers, n_heads, fold)

        metric_best, ep_best, time_train = 0, -1, 0
        for epoch in range(1, epochs + 1):
            _, _, _ = model.train_per_epoch(train_loader, epoch, epochs, criterion, optimizer, num_group=1,
                                        threshold=threshold, metrics_print_=True)

            _, _, metrics_val = model.eval_per_epoch(val_loader, epoch, epochs, criterion, num_group=1, threshold=threshold,
                                       metrics_print_=True)

            metrics_ep_avg = sum(metrics_val[:4]) / 4

            if metrics_ep_avg > metric_best:
                metric_best, ep_best = metrics_ep_avg, epoch
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                print('Best epoch = {} | Best metrics_ep_avg = {:.4f}--------'.format(ep_best, metric_best))
                print('Saving model Path saver: {} --------'.format(path_saver))
                torch.save(model.state_dict(), path_saver)