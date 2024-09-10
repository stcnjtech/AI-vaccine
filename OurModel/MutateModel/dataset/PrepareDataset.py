"""
    制作强化学习的数据集
"""
import os
import pandas as pd
import numpy as np
data=pd.read_csv("../../../common_hla_sequence.csv")
hla_name_list=[]
hla_seq_list=[]
for hla_name in list(data.HLA):
    hla_name_list.append(hla_name.replace("*","_").replace(":","_"))
# hla_name_list=list(set(hla_name_list))
hla_seq_list=list(data.HLA_sequence)
peptide_list=list(pd.read_csv("./val_data_fold0.csv")["peptide"].unique())
note={}

dataset_hla_list=[]
dataset_peptide_list=[]
dataset_hla_name_list=[]

# 将hla_seq_list 与 peptide_list 进行配对
for i in range(len(hla_name_list)):
    hla_name=hla_name_list[i]
    hla_seq=hla_seq_list[i]
    if hla_name not in note.keys():
        note[hla_name]=[]
    for j in range(len(peptide_list)):
        peptide=peptide_list[j]
        if len(peptide) in note[hla_name]:
            dataset_hla_list.append(hla_seq)
            dataset_peptide_list.append(peptide)
            dataset_hla_name_list.append(hla_name)
        else:
            dir_path1 = "../../../Attention/peptideAAtype_peptidePosition"
            dir_path2 = "../../../Attention/peptideAAtype_peptidePosition_NUM"
            file_path1 = os.path.join(dir_path1, f"{hla_name}_Length{len(peptide)}.npy")
            file_path2 = os.path.join(dir_path2, f"{hla_name}_Length{len(peptide)}_num.npy")
            if os.path.exists(file_path1) and os.path.exists(file_path2):
                note[hla_name].append(len(peptide))
                dataset_hla_list.append(hla_seq)
                dataset_peptide_list.append(peptide)
                dataset_hla_name_list.append(hla_name)

dataset=pd.DataFrame({"hla":dataset_hla_list,"hla_name":dataset_hla_name_list,"peptide": dataset_peptide_list,},
                     columns=["hla","hla_name","peptide"])
dataset.to_csv("./dataset.csv")





