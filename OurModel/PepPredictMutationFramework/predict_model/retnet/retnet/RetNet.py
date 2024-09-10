import torch
import torch.nn as nn
from tqdm import tqdm
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnetblock.Embedding import Embedding
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnetblock.Encoder import Encoder
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnetblock.Decoder import Decoder
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet1util.Criterion import *
import torch.nn.functional as F

class RetNet(nn.Module):
    def __init__(self,vocab_size,d_model=64,n_enc_layers=1,n_enc_heads=1,n_dec_layers=1,n_dec_heads=1,d_k=64,d_v=64,d_ff=512,hla_pep_concat_len=49,pep_tcr_concat_len=49,dropout_rate=0.1,max_len=512,device=torch.device("cuda:0")):
        super(RetNet, self).__init__()
        self.device = device

        # Embedding block:
        self.hla_embedding = Embedding(vocab_size,d_model,dropout_rate,max_len)
        self.pep_embedding = Embedding(vocab_size,d_model,dropout_rate,max_len)
        self.tcr_embedding = Embedding(vocab_size,d_model,dropout_rate,max_len)

        # Encoder block:
        self.hla_encoder = Encoder(n_enc_layers,n_enc_heads,d_model,d_k,d_v,d_ff,device,dropout_rate,max_len)
        self.pep_encoder = Encoder(n_enc_layers,n_enc_heads,d_model,d_k,d_v,d_ff,device,dropout_rate,max_len)
        self.tcr_encoder = Encoder(n_enc_layers,n_enc_heads,d_model,d_k,d_v,d_ff,device, dropout_rate, max_len)

        # Decoder block:
        self.hla_pep_decoder = Decoder(n_dec_layers,n_dec_heads,d_model,d_k,d_v,d_ff,dropout_rate,max_len,device)
        self.pep_tcr_decoder = Decoder(n_dec_layers,n_dec_heads,d_model,d_k,d_v,d_ff,dropout_rate,max_len,device)

        # Projection block:
        self.hla_pep_projection = nn.Sequential(
            nn.Linear(hla_pep_concat_len * d_model, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        ).to(self.device)

        self.pep_tcr_projection = nn.Sequential(
            nn.Linear(pep_tcr_concat_len * d_model, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        ).to(self.device)

    def forward(self, first_inputs,second_inputs,num_group):
        """
        first_inputs:   [batch_size,input_max_len]
        second_inputs:  [batch_size,input_max_len]
        """
        if num_group == 1:
            '''
            1.Embedding block:
            1) pep_inputs:[batch_size,pep_max_len,d_model]
            2) hla_inputs:[batch_size,hla_max_len,d_model]
            '''
            hla_inputs = self.hla_embedding(first_inputs)
            pep_inputs = self.pep_embedding(second_inputs)

            '''
            2.Encoder block:
            1) hla_enc_outputs:         [batch_size,hla_max_len,d_model]
            2) hla_self_retentions:     [n_layers,batch_size,n_heads,hla_max_len,hla_max_len]
            1) pep_enc_outputs:         [batch_size,pep_max_len,d_model]
            2) pep_self_retentions:     [n_layers,batch_size,n_heads,pep_max_len,pep_max_len]
            '''
            hla_enc_outputs, hla_self_retentions = self.hla_encoder(hla_inputs)
            pep_enc_outputs, pep_self_retentions = self.pep_encoder(pep_inputs)

            '''
            3.Concat block:
            dec_hla_pep_concat_inputs:  [batch_size,hla_pep_concat_len,d_model]
            '''
            dec_hla_pep_concat_inputs = torch.cat((hla_enc_outputs,pep_enc_outputs), 1)

            '''
            4.Decoder block:
            1) dec_hla_pep_concat_outputs:      [batch_size,hla_pep_concat_len,d_model]
            2) hla_pep_concat_self_retentions:  [n_layers,batch_size,n_heads,hla_pep_concat_len,hla_pep_concat_len]
            '''
            dec_hla_pep_concat_outputs, hla_pep_concat_self_retentions = self.hla_pep_decoder(dec_hla_pep_concat_inputs)

            # 5.Projection block:
            # Flatten to [batch_size, hla_pep_concat_len * d_model]
            hla_pep_proj_inputs = dec_hla_pep_concat_outputs.view(dec_hla_pep_concat_outputs.shape[0], -1)
            # hla_pep_proj_outputs:[batch_size,2]
            hla_pep_proj_outputs = self.hla_pep_projection(hla_pep_proj_inputs)

            return (
                hla_pep_proj_outputs,           # [batch_size,2]
                hla_self_retentions,            # [n_layers,batch_size,n_heads,hla_max_len,hla_max_len]
                pep_self_retentions,            # [n_layers,batch_size,n_heads,peptide_max_len,peptide_max_len]
                hla_pep_concat_self_retentions, # [n_layers,batch_size,n_heads,hla_pep_concat_len,hla_pep_concat_len]
            )
        else:
            '''
            1.Embedding block:
            1) pep_inputs:[batch_size,pep_max_len,d_model]
            2) tcr_inputs:[batch_size,tcr_max_len,d_model]
            '''
            pep_inputs = self.pep_embedding(first_inputs)
            tcr_inputs = self.tcr_embedding(second_inputs)

            '''
            2.Encoder block:
            1) pep_enc_outputs:         [batch_size,pep_max_len,d_model]
            2) pep_self_retentions:     [n_layers,batch_size,n_heads,pep_max_len,pep_max_len]
            1) tcr_enc_outputs:         [batch_size,tcr_max_len,d_model]
            2) tcr_self_retentions:     [n_layers,batch_size,n_heads,tcr_max_len,tcr_max_len]
            '''
            pep_enc_outputs, pep_self_retentions = self.pep_encoder(pep_inputs)
            tcr_enc_outputs, tcr_self_retentions = self.tcr_encoder(tcr_inputs)

            '''
            3.Concat block:
            dec_pep_tcr_concat_inputs:  [batch_size,pep_tcr_concat_len,d_model]
            '''
            dec_pep_tcr_concat_inputs = torch.cat((pep_enc_outputs, tcr_enc_outputs), 1)

            '''
            4.Decoder block:
            1) dec_pep_tcr_concat_outputs:      [batch_size,pep_tcr_concat_len,d_model]
            2) pep_tcr_concat_self_retentions:  [n_layers,batch_size,n_heads,pep_tcr_concat_len,pep_tcr_concat_len]
            '''
            dec_pep_tcr_concat_outputs, pep_tcr_concat_self_retentions = self.pep_tcr_decoder(dec_pep_tcr_concat_inputs)

            # 5.Projection block:
            # Flatten to [batch_size, pep_tcr_concat_len * d_model]
            pep_tcr_proj_inputs = dec_pep_tcr_concat_outputs.view(dec_pep_tcr_concat_outputs.shape[0], -1)
            # proj_outputs:[batch_size,2]
            pep_tcr_proj_outputs = self.pep_tcr_projection(pep_tcr_proj_inputs)

            return (
                pep_tcr_proj_outputs,           # [batch_size,2]
                pep_self_retentions,            # [n_layers,batch_size,n_heads,pep_max_len,pep_max_len]
                tcr_self_retentions,            # [n_layers,batch_size,n_heads,tcr_max_len,tcr_max_len]
                pep_tcr_concat_self_retentions, # [n_layers,batch_size,n_heads,pep_tcr_concat_len,pep_tcr_concat_len]
            )

    def train_per_epoch(self,train_loader,epoch,epochs,criterion,optimizer,num_group=1,threshold=0.5,metrics_print_=True):
        print("******开始第{}个epoch/共{}个epochs训练******".format(epoch,epochs))
        # 训练模式
        self.train()
        y_true_train_list, y_prob_train_list, loss_train_list = [], [], []
        """
        train_loader:
        pep和hla:batch_num * [batch_size,xx_max_lex]; label: batch_num * [batch_size]
        """
        for train_first_inputs, train_second_inputs, train_labels in tqdm(train_loader):
            """
            train_first_inputs: [batch_size,pep_max_len]
            train_second_inputs:[batch_size,hla_max_len]
            train_labels:       [batch_size]
            """

            train_first_inputs, train_second_inputs, train_labels = train_first_inputs.to(self.device), train_second_inputs.to(self.device), train_labels.to(self.device)

            # train_proj_outputs:           [batch_size,2]
            # train_concat_self_retentions: [n_layers,batch_size,n_heads,concat_len,concat_len]
            train_proj_outputs, _, _, _ = self.forward(train_first_inputs, train_second_inputs,num_group)

            # 计算loss值
            train_loss = criterion(train_proj_outputs, train_labels)

            # 优化器优化
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # 真实标签
            y_true_train = train_labels.cpu().numpy()
            # 计算训练预测概率
            y_prob_train = nn.Softmax(dim=1)(train_proj_outputs)[:, 1].cpu().detach().numpy()

            y_true_train_list.extend(y_true_train)  # 真实标签列表
            y_prob_train_list.extend(y_prob_train)  # 训练预测概率列表
            loss_train_list.append(train_loss)      # loss值列表

        # 计算训练预测标签列表
        y_pred_train_list = transfer(y_prob_train_list, threshold)
        # y_true_train_list:真实标签列表; y_pred_train_list:训练预测标签列表; y_prob_train_list:训练预测概率列表
        ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

        if metrics_print_:
            print("以下是评估得分:")
        metrics_train = performance(y_true_train_list, y_pred_train_list, y_prob_train_list, print_=metrics_print_)

        print('******结束第{}个epoch/共{}个epochs训练: Loss = {:.4f}******'.format(epoch, epochs,f_mean(loss_train_list)))

        return (
            ys_train,
            loss_train_list,
            metrics_train,
        )

    # 迭代一个epoch
    def eval_per_epoch(self, val_loader, epoch=None, epochs=None, criterion=None, num_group=1,threshold=0.5,metrics_print_=True):
        if epoch is not None:
            print("******开始第{}个epoch/共{}个epochs验证******".format(epoch,epochs))
        else:
            print("******开始验证******")
        # 评价模式
        self.eval()
        with torch.no_grad():
            y_true_val_list, y_prob_val_list, loss_val_list = [], [], []
            for val_first_inputs, val_second_inputs, val_labels in tqdm(val_loader):

                val_first_inputs, val_second_inputs, val_labels = val_first_inputs.to(self.device), val_second_inputs.to(self.device), val_labels.to(self.device)

                # val_outputs:  [batch_size,2]
                val_outputs, _, _, _ = self.forward(val_first_inputs, val_second_inputs,num_group)

                # 计算loss值
                val_loss = criterion(val_outputs, val_labels)
                # 真实标签
                y_true_val = val_labels.cpu().numpy()
                # 计算验证预测概率
                y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()

                y_true_val_list.extend(y_true_val)  # 真实标签列表
                y_prob_val_list.extend(y_prob_val)  # 训练预测概率列表
                loss_val_list.append(val_loss)      # loss值列表

            # 计算验证预测标签列表
            y_pred_val_list = transfer(y_prob_val_list, threshold)
            # y_true_val_list:真实标签列表;   y_pred_val_list:验证预测标签列表;   y_prob_val_list:验证预测概率列表
            ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

            if metrics_print_:
                print("以下是评估得分:")
            metrics_val = performance(y_true_val_list, y_pred_val_list, y_prob_val_list, print_=metrics_print_)
            if epoch is not None:
                print('******结束第{}个epoch/共{}个epochs验证: Loss = {:.6f}******'.format(epoch, epochs, f_mean(loss_val_list)))
            else:
                print('******结束验证: Loss = {:.6f}******'.format(f_mean(loss_val_list)))
        return (
            ys_val,
            loss_val_list,
            metrics_val
        )

    def predict(self, loader,num_group=1,threshold=0.5,n_layer_idx=0):
        # 评价模式
        self.eval()
        with torch.no_grad():
            val_y_prob_list, val_dec_concat_self_retentions_list  = [], []
            for val_first_inputs, val_second_inputs in tqdm(loader):
                val_first_inputs, val_second_inputs = val_first_inputs.to(self.device), val_second_inputs.to(self.device)

                # val_outputs:  [batch_size,2]
                # dec_concat_self_retentions:[n_layers,batch_size,n_heads,concat_len,concat_len]
                val_outputs, _, _, val_dec_concat_self_retentions = self.forward(val_first_inputs, val_second_inputs,num_group)

                # 计算预测概率
                # [batch_size,1]
                y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
                # [data_size,1]
                val_y_prob_list.extend(y_prob_val) # 预测概率列表

                if num_group == 1:
                    # (34,15)=(hla,peptide)
                    # val_dec_concat_self_retentions_list:[data_size,n_heads,hla_max_len,peptide_max_len]
                    val_dec_concat_self_retentions_list.extend(val_dec_concat_self_retentions[n_layer_idx][:, :, :34, 34:])
                else:
                    # (15,34)=(peptide,tcr)
                    # val_dec_concat_self_retentions_list:[data_size,n_heads,peptide_max_len,tcr_max_len]
                    val_dec_concat_self_retentions_list.extend(val_dec_concat_self_retentions[n_layer_idx][:, :, 15:, :15])
            # 预测标签列表
            val_y_pred_list = transfer(val_y_prob_list, threshold)
        return (
            val_y_pred_list,                        # 预测标签列表:[data_size,1]
            val_y_prob_list,                        # 预测概率列表:[data_size,1]
            val_dec_concat_self_retentions_list     # 自注意力张量:[data_size,n_heads,first_len,second_len]
        )
    def eval_one(self,input1,input2,num_group=1,layer_dix=-1):
        """
            input1 : (batch_size,34)
            input2 : (batch_size,15)
        """
        self.eval()
        with torch.no_grad():
            val_first_inputs, val_second_inputs = input1.to(self.device), input2.to(self.device)
            if num_group==1:
                val_outputs, _, pep_self_attn, val_dec_concat_self_attentions = self.forward(val_first_inputs, val_second_inputs, num_group)
                val_dec_concat_self_attentions=val_dec_concat_self_attentions[layer_dix][:,:,:34,34:]
            else:
                val_outputs, pep_self_attn, _, val_dec_concat_self_attentions = self.forward(val_second_inputs, val_first_inputs,
                                                                                  num_group)
                val_dec_concat_self_attentions = val_dec_concat_self_attentions[layer_dix][:, :, 15:, :15]

            val_dec_concat_self_attentions = nn.Softmax(dim=-1)(val_dec_concat_self_attentions)

        return F.softmax(val_outputs,dim=-1).detach(),val_dec_concat_self_attentions.detach(),pep_self_attn[layer_dix].detach()


