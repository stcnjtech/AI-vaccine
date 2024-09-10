"""
    模型接口
"""
import os.path
from Aomp import *
import numpy as np
ts_hp_weight_path='./OurModel/PepPredictMutationFramework/predict_model/transformer/model/model_hp_layer1_head9_fold4.pth'
ts_tp_weight_path='./OurModel/PepPredictMutationFramework/predict_model/transformer/model/model_pt_layer1_head9_fold4.pth'
rt_hp_weight_path="./OurModel/PepPredictMutationFramework/predict_model/retnet/model/model_hp_layer1_head9_fold4.pth"
rt_tp_weight_path="./OurModel/PepPredictMutationFramework/predict_model/retnet/model/model_pt_layer1_head9_fold4.pth"
db_path="./db/aomp_hp.db"
rl_weight_path= "OurModel/MutateModel/weight_path/mutation-Model.pth"
print(os.path.exists((rl_weight_path)))
def read_file_hp(file_path:str):
    """
        读取hla-pep数据文件
    """
    if os.path.exists(file_path)==False:
        #文件不存在
        return None
    link_list = None
    start_list = None
    file_type=file_path.split(".")[-1]
    if file_type=="csv":
        #读取csv文件
        data=pd.read_csv(file_path)
        if 'HLA_sequence' in data.columns:
            hla_list=list(data.HLA_sequence)
        else:
            hla_list=list(data.HLA)
        peptide_list=list(data.peptide)
        if "listid" in data.columns:
            link_list=list(data.listid)
        if "startid" in data.columns:
            start_list=list(data.startid)
    elif file_type=="txt":
        datas = np.loadtxt(file_path)
        hla_list=[]
        peptide_list=[]
        for data in datas:
            hla_list.append(data[0])
            peptide_list.append(data[1])
    else:
        print("当前文件格式不支持")
        return None

    return hla_list, peptide_list,link_list,start_list

def read_file_tp(file_path:str):
    """
        读取tcr-pep数据文件
    """
    if os.path.exists(file_path)==False:
        #文件不存在
        return None
    link_list = None
    start_list=None
    file_type=file_path.split(".")[-1]
    if file_type=="csv":
        #读取csv文件
        data=pd.read_csv(file_path)
        if 'TCR_sequence' in data.columns:
            tcr_list=list(data.TCR_sequence)
        else:
            tcr_list=list(data.TCR)
        peptide_list=list(data.peptide)
        if "listid" in data.columns:
            link_list=list(data.listid)
        if "startid" in data.columns:
            start_list=list(data.startid)

    elif file_type=="txt":
        datas = np.loadtxt(file_path)
        tcr_list=[]
        peptide_list=[]
        for data in datas:
            tcr_list.append(data[0])
            peptide_list.append(data[1])
    else:
        print("当前文件格式不支持")
        return None

    return tcr_list, peptide_list,link_list,start_list

def predict(hla_or_tcr_list,peptide_list,num_group=1,use_retnet=False):
    """
        hla_list,peptide_list : 均为列表形式
        num_group == 1 => hla-peptide
        num_group == 0 => tcr-peptide  hla_list内存放tcr序列
    """
    if use_retnet==True:
        model=Model(hp_weight_path=rt_hp_weight_path,tp_weight_path=rt_tp_weight_path,
                    num_group=num_group,db_path=db_path,use_retnet=use_retnet)
    else:
        model=Model(hp_weight_path=ts_hp_weight_path,
            tp_weight_path=ts_tp_weight_path,
            num_group=num_group, use_retnet=use_retnet)
    data,attn_pd_list=model.predict(hla_or_tcr_list,peptide_list,is_mutated=False)
    return data,attn_pd_list

def mutation(hla_or_tcr_list=None,peptide_list=None,target_hla=None,target_peptide=None,num_group=1,use_retnet=False,use_RL=True):
    """
        hla_list ,peptide_list : 均为列表形式 这两个参数用于生成position_pd 和 position_num_pd ，可以为空
        target_hla,target_peptide : 均为字符串,用于进行突变,不能为空
        num_group ==1 => hla-peptide
        num_group ==0 => tcr-peptide  target_hla存放tcr序列
    """
    if use_retnet==True:
        aomp=Aomp(rt_hp_weight_path,rt_tp_weight_path,num_group,db_path,use_RL=use_RL,rl_weight_path=rl_weight_path,use_retnet=use_retnet)
    else:
        aomp = Aomp(ts_hp_weight_path, ts_tp_weight_path, num_group, db_path, use_RL=use_RL,
                    rl_weight_path=rl_weight_path,use_retnet=use_retnet)
    mutate_peptides,mutate_data,mutate_attn,positive_mutate_rate=aomp.predict_and_optim(hla_or_tcr_list,peptide_list,target_hla,target_peptide,read_npy=True)
    return mutate_peptides,mutate_data,mutate_attn,positive_mutate_rate

def output_processing(datas,attns,link_list=None,start_list=None):
    result_list=[]
    row_count=datas.shape[0]
    label_list=list(datas.columns)
    datas = np.array(datas).tolist()
    for i in range(row_count):
        t_dict={}
        attn=attns[i]
        for j,label in enumerate(label_list):
            if label=="y_prob":
                t_dict[label]=np.around(datas[i][j],3)
            else:
                t_dict[label]=datas[i][j]
        if link_list is not None:
            t_dict["listid"]=link_list[i]
        else:
            t_dict["listid"] = ""
        if start_list is not None:
            t_dict["startid"]=start_list[i]
        else:
            t_dict["startid"] = ""
        x=list(attn.index)
        y=list(attn.columns)
        t_dict['max'] = np.max(np.array(attn, dtype=float), axis=(0, 1))
        t_dict['min'] = np.min(np.array(attn, dtype=float), axis=(0, 1))
        attn_li=np.around(np.array(attn).astype(float),3).tolist()
        t_dict["xAxis"]=x
        y.reverse()
        t_dict['yAxis']=y
        a=[]
        for x_ in range(len(attn_li)):
            for y_ in range(len(attn_li[x_])):
                a.append([x_,len(y)-1-y_,attn_li[x_][y_]])
        t_dict["data"]= a
        result_list.append(t_dict)
    return result_list

if __name__ == "__main__":
    tcr="CBSLMSGLTGELF"
    hla="YFAMYGEKVAHTHVDTLYVRYHYYTWAVQAYTWY"
    peptide="AEAFIQPI"

    print("\n==============hla-pep=================")

    data,attn_pd_list=predict([hla,hla,hla,hla,hla],[peptide,"AFAEIQPG","AEAFSQPY","MMPFIQPI","AMAIIMPI"],num_group=1,use_retnet=False)
    print(data)
    mutate_peptides,mutate_data,mutate_attn,positive_mutate_rate=mutation(target_hla=hla,target_peptide=peptide,num_group=1,use_retnet=False,use_RL=True)
    print(mutate_data[['mutated_peptide',"y_prob","similarity"]])
    print(positive_mutate_rate)

    print("\n==============tcr-pep=================")

    data, attn_pd_list = predict([tcr, tcr, tcr, tcr, tcr], [peptide, "AFAEIQPG", "AEAFSQPY", "MMPFIQPI", "AMAIIMPI"],
                                 num_group=0, use_retnet=False)
    print(data)
    mutate_peptides, mutate_data, mutate_attn, positive_mutate_rate = mutation(target_hla=tcr, target_peptide=peptide,
                                                                               num_group=0, use_retnet=False,
                                                                               use_RL=True)
    print(mutate_data[['mutated_peptide', "y_prob", "similarity"]])
