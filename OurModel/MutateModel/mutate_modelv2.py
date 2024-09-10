"""
    突变模型v2版
"""

from OurModel.MutateModel.model import *
from OurModel.MutateModel.mutate_policy import *

class Mutationv2(nn.Module):
    """
        突变模型第二版
    """
    def __init__(self,critic,Train=True,num_group=1,device=None):
        """
            critic : 评论家模型
            Train : 是否训练
            num_group
        """
        super(Mutationv2, self).__init__()
        self.device=device
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Train = Train
        #评论家
        self.critic=critic.to(self.device)
        #行为者
        self.actor=MutateBlockv2(device=device,is_train=self.Train).to(self.device)
        self.num_group=num_group
        #损失函数
        self.loss_fn1=nn.BCELoss(reduction="mean")
        self.loss_fn2=nn.CrossEntropyLoss(reduction="mean")

    def input_processing(self, position_pd, position_num_pd, attn_of_phla_pd,peptide_self_attn):
        # 对输入数据进行处理
        # 通过position_pd position_num_pd 得到 contrib矩阵
        #print(position_pd)
        batch_size = len(position_pd)
        contrib = torch.zeros(size=(batch_size, len(cfg.acid_list), cfg.max_peptide_length))
        for i, posi_pd in enumerate(position_pd):
            posi_num_pd = position_num_pd[i]
            for a, acid in enumerate(posi_pd.index[:-1]):
                for p, p_ in enumerate(posi_pd.columns[:-1]):
                    posi_pd.loc["sum"] = posi_pd.sum(axis=0)
                    posi_pd["sum"] = posi_pd.sum(axis=1)
                    posi_num_pd.loc["sum"] = posi_num_pd.sum(axis=0)
                    posi_num_pd["sum"] = posi_num_pd.sum(axis=1)
                    t1 = posi_pd.loc[acid, p_] / (posi_num_pd.loc[acid, p_] + 1e-5)
                    t2 = posi_pd.loc["sum", "sum"] / (posi_num_pd.loc["sum", "sum"] + 1e-5)
                    contrib[i, a, p] = t1 * t2

        if isinstance(attn_of_phla_pd, list):
            attn_of_pHla = torch.from_numpy(np.array(attn_of_phla_pd)).to(torch.float)
        elif isinstance(attn_of_phla_pd, np.ndarray):
            attn_of_pHla = torch.from_numpy(attn_of_phla_pd).to(torch.float)
        else:
            attn_of_pHla = attn_of_phla_pd

        B, l1, l2 = attn_of_pHla.shape

        # attn 进行pad操作
        if self.num_group == 1:
            attn_of_pHla = F.pad(attn_of_pHla, pad=(0, cfg.max_peptide_length - l2, 0, cfg.max_hla_length - l1),
                                 value=0)
        else:
            attn_of_pHla = F.pad(attn_of_pHla, pad=(0, cfg.max_peptide_length - l2, 0, cfg.max_tcr_length - l1),
                                 value=0)

        if isinstance(peptide_self_attn, list):
            peptide_self_attn = torch.tensor([item.cpu().detach().numpy() for item in peptide_self_attn]).squeeze(dim=1)
        else:
            peptide_self_attn=peptide_self_attn
        peptide_self_attn=torch.sum(peptide_self_attn,dim=1)
        #print(torch.sum(peptide_self_attn,dim=1).shape)
        return contrib, attn_of_pHla,peptide_self_attn
    def mask2number(self,seq,max_len):
        return F.pad(torch.LongTensor([cfg.vocab[t] for t in seq]),pad=(0,max_len-len(seq)),value=0).unsqueeze(0)

    def mutate_policy(self,mutate_matrix,peptide_self_attn,hla,peptides,is_train=True):
        """
            mutate_matrix : (batch_size,4,20,15,2)
            peptide_self_attn : (batch_size,4,15,15)
            hla         :   (batch_size,)
            peptides    :   (batch_size,)
            mutated_peptides : list
        """
        mutated_peptides=[[peptide] for peptide in peptides]
        mutated_result1,mutated_peptides=mutate_one_position(mutate_matrix[:,0,:,:,:],peptide_self_attn[:,0,:,:],hla,peptides,mutated_peptides)
        mutated_result2, mutated_peptides = mutate_two_position(mutate_matrix[:, 1, :, :, :],
                                                                peptide_self_attn[:, 1, :, :], hla, peptides,
                                                                mutated_peptides)
        mutated_result3, mutated_peptides = mutate_three_position(mutate_matrix[:, 2, :, :, :],
                                                                peptide_self_attn[:, 2, :, :], hla, peptides,
                                                                mutated_peptides)
        mutated_result4, mutated_peptides = mutate_four_position(mutate_matrix[:, 3, :, :, :],
                                                                peptide_self_attn[:, 3, :, :], hla, peptides,
                                                                mutated_peptides)
        return mutated_result1+mutated_result2+mutated_result3+mutated_result4

    def remark(self,mutate_matrix,pep_self_matrix,mutated_peptides):
        """
            突变评估
            mutate_matrix : (batch_size,4,20,15,2)
            peptide_self_attn : (batch_size,4,15,15)
        """
        mutated_mask=torch.zeros_like(mutate_matrix)       #(batch_size,4,20,15,2)
        mutated_score=torch.zeros_like(mutate_matrix)      #(batch_size,4,20,15,2)
        self_attn_mask=torch.zeros_like(pep_self_matrix) #(batch_size,4,15,15)
        mutated_attn_score=torch.zeros_like(pep_self_matrix)
        mutated_peptides=np.array(mutated_peptides,dtype=object)

        #获得hla ,mutated_peptide
        hla=mutated_peptides[:,-2]
        peptide=mutated_peptides[:,-1]
        if self.num_group==1:
            DataSet = torch.cat([self.mask2number(hla_, cfg.max_hla_length) for hla_ in hla], dim=0)
        else:
            DataSet = torch.cat([self.mask2number(hla_, cfg.max_tcr_length) for hla_ in hla], dim=0)
        peptideDataSet = torch.cat([self.mask2number(peptide_, cfg.max_peptide_length) for peptide_ in peptide], dim=0)
        #实际得分
        score,_,_=self.critic.eval_one(DataSet,peptideDataSet,num_group=self.num_group)  #(N,2)

        for i in range(mutated_peptides.shape[0]):
            batch=mutated_peptides[i,0]
            strategy=mutated_peptides[i,1]
            mutate_acids=mutated_peptides[i,2]       #突变氨基酸
            mutate_pep_posi=mutated_peptides[i,3]    #突变位置
            for j,mutate_acid in enumerate(mutate_acids):
                mutate_posi=mutate_pep_posi[j]
                mutated_mask[batch,strategy,mutate_acid,mutate_posi,:]=1
                mutated_score[batch,strategy,mutate_acid,mutate_posi,0]=mutated_score[batch,strategy,mutate_acid,mutate_posi,0]+score[i,0]
                mutated_score[batch,strategy,mutate_acid,mutate_posi,1]=mutated_score[batch,strategy,mutate_acid,mutate_posi,1]+score[i,1]

            if strategy==1:
                posi1,posi2=mutate_pep_posi

                self_attn_mask[batch,strategy,posi1,posi2]=1
                mutated_attn_score[batch,strategy,posi1,posi2]=mutated_attn_score[batch,strategy,posi1,posi2]+(score[i,1]-score[i,0])

            elif strategy==2:
                posi1,posi2,posi3=mutate_pep_posi
                self_attn_mask[batch, strategy, posi1, posi2] = 1
                self_attn_mask[batch, strategy, posi1, posi3] = 1

                mutated_attn_score[batch, strategy, posi1, posi2] = mutated_attn_score[batch, strategy, posi1, posi2]+(score[i, 1] - score[i, 0])
                mutated_attn_score[batch, strategy, posi1, posi3] = mutated_attn_score[batch, strategy, posi1, posi3]+(score[i, 1] - score[i, 0])

            elif strategy==3:

                posi1, posi2, posi3 ,posi4= mutate_pep_posi
                self_attn_mask[batch, strategy, posi1, posi2] = 1
                self_attn_mask[batch, strategy, posi1, posi3] = 1
                self_attn_mask[batch, strategy, posi1, posi4] = 1

                mutated_attn_score[batch, strategy, posi1, posi2] = mutated_attn_score[batch, strategy, posi1, posi2]+(score[i, 1] - score[i, 0])
                mutated_attn_score[batch, strategy, posi1, posi3] = mutated_attn_score[batch, strategy, posi1, posi3]+(score[i, 1] - score[i, 0])
                mutated_attn_score[batch, strategy, posi1, posi4] = mutated_attn_score[batch, strategy, posi1, posi4]+(score[i, 1] - score[i, 0])
        return F.softmax(mutated_score,dim=-1).clone(),mutated_mask,F.softmax(mutated_attn_score,dim=-1).clone(),self_attn_mask

    def CalculateLoss(self,mutate_matrix,pep_self_matrix,mutated_score,mutated_mask,mutated_attn_score,self_attn_mask):
        """
            计算损失
            mutate_matrix : (batch_size,4,20,15,2)
            peptide_self_attn : (batch_size,4,15,15)
            mutated_score : (batch_size,4,20,15,2)
            mutated_mask  : (batch_size,4,20,15,2)
            mutated_attn_score: (batch_size,4,15,15)
            self_attn_mask : (batch_size,4,15,15)
        """
        mutated_mask=mutated_mask>=1
        self_attn_mask=self_attn_mask>=1

        selected_mutate_matrix=mutate_matrix.masked_select(mutated_mask).reshape(-1,2)

        #print(self_attn_mask.shape)
        selected_peptide_self_attn=pep_self_matrix.masked_select(self_attn_mask)

        selected_mutated_score=mutated_score.masked_select(mutated_mask).reshape(-1,2)
        selected_mutated_attn_score=mutated_attn_score.masked_select(self_attn_mask)

        l1=self.loss_fn1(selected_mutate_matrix,selected_mutated_score)
        l2=self.loss_fn2(selected_peptide_self_attn,selected_mutated_attn_score)

        return 0.5*l1+0.5*l2

    def forward(self,position_pd,position_num_pd,attn_of_phla_pd,peptide_self_attn,hla,peptide):
        batch_size = len(hla)
        if attn_of_phla_pd is None:
            # 通过模型生成注意力
            hlaDataSet = torch.cat([self.mask2number(hla_, cfg.max_hla_length) for hla_ in hla], dim=0)
            peptideDataSet = torch.cat([self.mask2number(peptide_, cfg.max_peptide_length) for peptide_ in peptide],
                                       dim=0)
            scores, attns,peptide_self_attn = self.critic.eval_one(hlaDataSet, peptideDataSet, num_group=self.num_group)
            attn_of_phla_pd = torch.sum(attns, dim=1)  # 多个头累加
        contrib, attn_of_pHla,peptide_self_attn = self.input_processing(position_pd, position_num_pd, attn_of_phla_pd,peptide_self_attn)
        #进行突变
        _,mutate_matrix,pep_self_matrix=self.actor(contrib,attn_of_pHla,peptide_self_attn,[len(hla_) for hla_ in hla],[len(pep) for pep in peptide])
        mutated_peptides=self.mutate_policy(mutate_matrix,pep_self_matrix,hla,peptide)
        if self.Train == False:
            return list(np.array(mutated_peptides,dtype=object)[:,-1])
        mutated_score,mutated_mask,mutated_attn_score,self_attn_mask=self.remark(mutate_matrix,pep_self_matrix,mutated_peptides)
        loss=self.CalculateLoss(mutate_matrix,pep_self_matrix,mutated_score,mutated_mask,mutated_attn_score,self_attn_mask)
        return loss










