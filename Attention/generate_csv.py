import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from OurModel.PepPredictMutationFramework.predict_model.transformer.transformer.Transformer import *
import torch.cuda
vocab={'C': 1,'W': 2,'V': 3,'A': 4,'H': 5,'T': 6,'E': 7,'K': 8,'N': 9,'P': 10,'I': 11,'L': 12,'S': 13,'D': 14,'G': 15,'Q': 16,'R': 17,'Y': 18,'F': 19,'M': 20,'O': 21,'X': 22,'B':23,'J':24,'U':25,'Z':26,'-': 0}
acid_list=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

max_tcr_length=34
max_pep_length=15
vocab_size=len(vocab)

file_path= "../OurModel/MutateModel/dataset/pep_tcr_dataset.csv"
data=pd.read_csv(file_path)
peptide_list=data.peptide
tcr_list=data.tcr
label_list=data.label

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
predictblock=Transformer(vocab_size, n_enc_heads=9,n_dec_heads=9,device=device)

predictblock.load_state_dict(torch.load("./OurModel/PepPredictMutationFramework/predict_model/transformer/model/model_pt_layer1_head9_fold4.pth",map_location="cpu"))
print("模型加载完毕")

position_pd_dict={}
position_num_pd_dict={}

for i,pep in enumerate(peptide_list):
    pep_length=len(pep)
    pep_key=str(pep_length)
    if pep_key not in position_pd_dict.keys():
        position_pd_dict[pep_key]=pd.DataFrame(np.zeros(shape=(len(acid_list),pep_length)),index=acid_list,columns=np.arange(1,pep_length+1))
        position_num_pd_dict[pep_key] = pd.DataFrame(np.zeros(shape=(len(acid_list), pep_length)), index=acid_list,
                                                    columns=np.arange(1, pep_length + 1))
    #print(position_pd_dict[pep_key].columns)
    #print(position_pd_dict[pep_key].loc["A",1])
    tcr=tcr_list[i]
    label=label_list[i]
    if label != 1:
        continue
    tcrDataSet=F.pad(torch.LongTensor([vocab[t] for t in tcr]),pad=(0,max_tcr_length-len(tcr)),value=0).unsqueeze(0)
    peptideDataset=F.pad(torch.LongTensor([vocab[t] for t in pep]),pad=(0,max_pep_length-len(pep)),value=0).unsqueeze(0)
    score,attn,_=predictblock.eval_one(tcrDataSet,peptideDataset)
    attn=torch.sum(attn,dim=1)
    score=score.numpy()
    #columns

    for j,w in enumerate(pep):
        position_pd_dict[pep_key].loc[w,j+1]+=score[0][-1]
        position_num_pd_dict[pep_key].loc[w, j + 1] += 1

position_pd_dir= "tcr_pep_position_pd"
position_num_pd_dir= "tcr_pep_position_num_pd"

for key in position_pd_dict.keys():
    position_pd_dict[key].to_csv(os.path.join(position_pd_dir,"tcr-peptide-length="+key+"position_pd.csv"))
    position_num_pd_dict[key].to_csv(os.path.join(position_num_pd_dir, "tcr-peptide-length=" + key + "position_num_pd.csv"))
