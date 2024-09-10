import os.path
import pandas as pd
import torch.cuda
#from mutation import *
from OurModel.PepPredictMutationFramework.predict_model.transformer.transformer.Transformer import *
from OurModel.PepPredictMutationFramework.predict_model.retnet.retnet.RetNet import *
from OurModel.MutateModel.mutate_modelv2 import *
import sqlite3
class Model:
    def init_model(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.use_retnet==True:
            # 初始化retnet网络
            self.model = RetNet(self.vocab_size,n_enc_heads=9,n_dec_heads=9,device=self.device)
        else:
            #初始化transformer网络
            self.model = Transformer(self.vocab_size, n_enc_heads=9,n_dec_heads=9,device=self.device)
        if self.num_group == 1 and self.hp_weight_path != None and os.path.exists(self.hp_weight_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.hp_weight_path, map_location='cpu')) if not torch.cuda.is_available() else \
                    self.model.load_state_dict(torch.load(self.hp_weight_path))
            except:
                    self.model.load_state_dict(torch.load(self.hp_weight_path,map_location="cpu"))
        elif self.num_group != 1 and self.tp_weight_path != None and os.path.exists(self.tp_weight_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.tp_weight_path, map_location="cpu")) if not torch.cuda.is_available() else \
                    self.model.load_state_dict(torch.load(self.tp_weight_path))
            except:
                self.model.load_state_dict(torch.load(self.tp_weight_path, map_location="cpu"))
        self.model=self.model.to(self.device)
    def __init__(self,hp_weight_path,tp_weight_path,num_group=1,db_path="./db/aomp_hp.db",use_retnet=True):
        """
            hp_weight_path  hla-peptide
            tp_weight_path  tcr-peptide
            num_group :  peptide or tcr
            db_path :  数据库地址
            use_retnet : 是否使用retnet網絡
        """
        #氨基酸列表
        self.acid_list=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
        self.conn=sqlite3.connect(db_path)
        self.hla_max_len=34
        self.pep_max_len=15
        self.tcr_max_len=34
        self.hp_weight_path=hp_weight_path
        self.tp_weight_path=tp_weight_path
        self.num_group=num_group
        self.use_retnet=use_retnet
        self.vocab={'C': 1,'W': 2,'V': 3,'A': 4,'H': 5,'T': 6,'E': 7,'K': 8,'N': 9,'P': 10,'I': 11,'L': 12,'S': 13,'D': 14,'G': 15,'Q': 16,'R': 17,'Y': 18,'F': 19,'M': 20,'O': 21,'X': 22,'B':23,'J':24,'U':25,'Z':26,'-': 0}
        self.vocab_size=len(self.vocab)
        #初始化模型
        self.init_model()

    def mask2number(self,seq,max_len):
        return F.pad(torch.LongTensor([self.vocab[t] for t in seq]),pad=(0,max_len-len(seq)),value=0).unsqueeze(0)

    def predict(self,hla_or_tcr_list,peptide_list,is_mutated=False):
        if self.num_group==1:
           dataSet = torch.cat([self.mask2number(hla, self.hla_max_len) for hla in hla_or_tcr_list], dim=0)
        else:
            dataSet = torch.cat([self.mask2number(tcr, self.tcr_max_len) for tcr in hla_or_tcr_list], dim=0)
        peptide_dataSet = torch.cat([self.mask2number(peptide, self.pep_max_len) for peptide in peptide_list], dim=0)
        result, attns,_ = self.model.eval_one(dataSet, peptide_dataSet, self.num_group)
        #print(attns)
        label = torch.argmax(result, dim=-1)
        prob = result[:, -1]
        if is_mutated==False:
            if self.num_group==1:
                data = pd.DataFrame(data={"HLA": hla_or_tcr_list, "peptide": peptide_list, "y_pred": label, "y_prob": prob},
                                    index=np.arange(label.shape[0]), columns=("HLA", "peptide", "y_pred", "y_prob"))
            else:
                data = pd.DataFrame(data={"TCR": hla_or_tcr_list, "peptide": peptide_list, "y_pred": label, "y_prob": prob},
                                    index=np.arange(label.shape[0]), columns=("TCR", "peptide", "y_pred", "y_prob"))
        else:
            if self.num_group==1:
                data = pd.DataFrame(data={"HLA": hla_or_tcr_list, "mutated_peptide": peptide_list, "y_pred": label, "y_prob": prob},
                                    index=np.arange(label.shape[0]), columns=("HLA", "mutated_peptide", "y_pred", "y_prob"))
            else:
                data = pd.DataFrame(
                    data={"TCR": hla_or_tcr_list, "mutated_peptide": peptide_list, "y_pred": label, "y_prob": prob},
                    index=np.arange(label.shape[0]), columns=("TCR", "mutated_peptide", "y_pred", "y_prob"))
        #对attn裁剪,并且转为pd.DataFrame
        attns=torch.sum(attns,dim=1)    #对头累加
        attn_pd_list=[]
        for i in range(len(hla_or_tcr_list)):
            hla_or_tcr=hla_or_tcr_list[i]
            peptide=peptide_list[i]
            attn=attns[i,:len(hla_or_tcr),:len(peptide)]
            attn_pd=pd.DataFrame(attn,index=list(hla_or_tcr),columns=list(peptide))
            attn_pd_list.append(attn_pd)

        return data,attn_pd_list

    def read_npy_file(self,target_hla_or_tcr,target_peptide,label=1):
        """
            读取npy文件
        """
        if label==None:
            label="all"
        elif label==1:
            label="positive"
        else:
            label="negative"
        peptide_length=len(target_peptide)

        if self.num_group != 1:
            dir_path1 = "./Attention/tcr_pep_position_pd"
            dir_path2 = "./Attention/tcr_pep_position_num_pd"
            pep_length=len(target_peptide)
            file_path1=f"tcr-peptide-length={pep_length}position_pd.csv"
            file_path2=f"tcr-peptide-length={pep_length}position_num_pd.csv"
            position_pd=pd.read_csv(os.path.join(dir_path1,file_path1))
            position_num_pd=pd.read_csv(os.path.join(dir_path2,file_path2))
            position_pd.set_index("Unnamed: 0", inplace=True)
            position_num_pd.set_index("Unnamed: 0", inplace=True)
            position_pd.loc['sum'] = position_pd.sum(axis=0)
            position_pd["sum"] = position_pd.sum(axis=1)
            position_num_pd.loc['sum'] = position_num_pd.sum(axis=0)
            position_num_pd["sum"] = position_num_pd.sum(axis=1)
            return position_pd,position_num_pd

        dir_path1="./Attention/peptideAAtype_peptidePosition"
        dir_path2="./Attention/peptideAAtype_peptidePosition_NUM"
        #搜索数据库
        cursor=self.conn.cursor()
        length=len(target_peptide)
        hla_name_list=cursor.execute(f"select hla_name from hla_table where hla_seq == '{target_hla_or_tcr}'").fetchall()
        if len(hla_name_list)==0:
            position_pd=np.load("./Attention/peptideAAtype_peptidePosition/Allsamples_Alllengths.npy",
                                allow_pickle=True).item()[length][label]
            position_num_pd=np.load("./Attention/peptideAAtype_peptidePosition_NUM/Allsamples_Alllengths_num.npy",
                                    allow_pickle=True).item()[length][label]
            position_pd.loc['sum'] = position_pd.sum(axis=0)
            position_pd["sum"] = position_pd.sum(axis=1)
            position_num_pd.loc['sum'] = position_num_pd.sum(axis=0)
            position_num_pd["sum"] = position_num_pd.sum(axis=1)
        else:
            hla_name=hla_name_list[0][0]
            try:
                position_pd_path=os.path.join(dir_path1,f"{hla_name}_Length{peptide_length}.npy")
                position_num_pd_path=os.path.join(dir_path2,f"{hla_name}_Length{peptide_length}_num.npy")
                position_pd=np.load(position_pd_path,allow_pickle = True).item()[label]
                position_num_pd=np.load(position_num_pd_path,allow_pickle = True).item()[label]
                position_pd.loc['sum'] = position_pd.sum(axis=0)
                position_pd["sum"] = position_pd.sum(axis=1)
                position_num_pd.loc['sum'] = position_num_pd.sum(axis=0)
                position_num_pd["sum"] = position_num_pd.sum(axis=1)
            except:
                position_pd = np.load("./Attention/peptideAAtype_peptidePosition/Allsamples_Alllengths.npy",
                                      allow_pickle=True).item()[length][label]
                position_num_pd = np.load("./Attention/peptideAAtype_peptidePosition_NUM/Allsamples_Alllengths_num.npy",
                                          allow_pickle=True).item()[length][label]
                position_pd.loc['sum'] = position_pd.sum(axis=0)
                position_pd["sum"] = position_pd.sum(axis=1)
                position_num_pd.loc['sum'] = position_num_pd.sum(axis=0)
                position_num_pd["sum"] = position_num_pd.sum(axis=1)
        cursor.close()
        return position_pd,position_num_pd

    def run_model(self,hla_or_tcr_list,peptide_list,target_hla_or_tcr,target_peptide,read_npy=False):
        """
            hla_list : hla数据集  (batch_size,hla_length)
            peptide_list : peptide数据集 (batch_size,peptide_length)
            target_hla : 目标hla
            target_peptide : 目标peptide
        """
        if self.num_group==1:
            dataSet=torch.cat([self.mask2number(hla_or_tcr,self.hla_max_len) for hla_or_tcr in [target_hla_or_tcr]],dim=0)
        else:
            dataSet=torch.cat([self.mask2number(hla_or_tcr,self.tcr_max_len) for hla_or_tcr in [target_hla_or_tcr]],dim=0)
        peptide_dataSet=torch.cat([self.mask2number(peptide,self.pep_max_len) for peptide in [target_peptide]],dim=0)

        peptide_length=len(target_peptide)
        hla_or_tcr_length=len(target_hla_or_tcr)

        result,attn,pep_self_attn=self.model.eval_one(dataSet,peptide_dataSet,self.num_group)
        attn = attn[:, :, :hla_or_tcr_length, :peptide_length]
        #数据处理
        #print(attn.shape)
        data,attn_pd=self.data_processing(hla_or_tcr_list,peptide_list,attn,target_hla_or_tcr,target_peptide,result)
        if read_npy==False and hla_or_tcr_list!=None and peptide_list!=None:
            position_pd,position_num_pd=self.generate_position_pd_and_position_num_pd(data,attn,peptide_list,target_hla_or_tcr,target_peptide)
        else:
            #读取已有的npy文件
            position_pd,position_num_pd=self.read_npy_file(target_hla_or_tcr,target_peptide)
        return data,attn_pd,pep_self_attn,position_pd,position_num_pd

    def data_processing(self,hla_or_tcr_list,peptide_list,attn,target_hla_or_tcr,target_peptide,result):
        """
            hla_list
            peptide_list
            attn_list (batch_size,num_head,max_hla_length,max_peptide_length)
        """
        peptide_length=len(target_peptide)
        hla_or_tcr_length=len(target_hla_or_tcr)
        label = torch.argmax(result, dim=-1)
        prob = result[:, -1]

        data = pd.DataFrame(data={"HLA": [target_peptide], "peptide": [target_peptide], "y_pred": label, "y_prob": prob},
                            index=np.arange(label.shape[0]), columns=("HLA", "peptide", "y_pred", "y_prob"))
        attn=attn[0]
        attn_head_sum=torch.sum(attn,dim=0)
        attn_pd=pd.DataFrame(data=np.array(attn_head_sum),index=list(target_hla_or_tcr),columns=list(target_peptide))
        attn_pd.loc['sum'] = attn_pd.sum(axis=0)
        attn_pd.loc['posi'] = range(1, peptide_length + 1)
        attn_pd.loc['contrib'] = attn_pd.loc['sum'] / attn_pd.loc['sum'].sum()
        return data,attn_pd

    def generate_position_pd_and_position_num_pd(self,data,attn_list,peptide_list,target_hla_or_tcr,target_peptide):
        peptide_length=len(target_peptide)
        hla_length=len(target_hla_or_tcr)
        idx_list=data[data.HLA==target_hla_or_tcr].index[:]
        position_pd=pd.DataFrame(np.zeros(shape=(len(self.acid_list),peptide_length)),index=self.acid_list,columns=np.arange(1,peptide_length+1))
        position_num_pd=pd.DataFrame(np.zeros(shape=(len(self.acid_list),peptide_length)),index=self.acid_list,columns=np.arange(1,peptide_length+1))
        #从attn_list 中获得注意力矩阵,并且进行累加
        for idx in idx_list:
            attn=np.array(torch.sum(attn_list[idx],dim=(0,1)))
            peptide=peptide_list[idx]
            for posi,score in enumerate(attn):
                peptide_acid=peptide[posi]
                position_pd.loc[peptide_acid,posi+1]+=score
                position_num_pd.loc[peptide_acid,posi+1]+=1
        position_pd.loc['sum']=position_pd.sum(axis=0)
        position_pd["sum"]=position_pd.sum(axis=1)
        position_num_pd.loc['sum']=position_num_pd.sum(axis=0)
        position_num_pd["sum"]=position_num_pd.sum(axis=1)
        return position_pd,position_num_pd

class Aomp:
    def __init__(self,hp_weight_path,tp_weight_path,num_group=1,db_path="./db/aomp_hp.db",use_RL=False,rl_weight_path=None,use_retnet=False):
        self.hp_weight_path=hp_weight_path
        self.tp_weight_path=tp_weight_path
        self.num_group=num_group
        self.db_path=db_path
        self.use_RL=use_RL   #是否使用强化学习
        self.use_retnet=use_retnet
        self.model=Model(self.hp_weight_path,self.tp_weight_path,self.num_group,self.db_path,use_retnet=self.use_retnet)
        if self.use_RL==True:
            self.mutate_model = Mutationv2(self.model.model,Train=False,num_group=self.num_group)
            if rl_weight_path is not None and os.path.exists(rl_weight_path):
                try:
                    self.mutate_model.actor.load_state_dict(
                        torch.load(rl_weight_path, map_location={'0': "cup"})) if not \
                        torch.cuda.is_available() else self.mutate_model.mutateblock.load_state_dict(
                        torch.load(rl_weight_path))
                except:
                    self.mutate_model.actor.load_state_dict(torch.load(rl_weight_path,map_location="cpu"))
        else:
            self.mutate_model=None

    def mutate_position_acid(self,mutated_peptides,original_peptides):
        result=[]
        for i,mutated_peptide in enumerate(mutated_peptides):
            original_peptide = original_peptides[i]
            t=""
            for j,m_w in enumerate(mutated_peptide):
                o_w=original_peptide[j]
                if o_w != m_w:
                    t+=f"{j+1}|"+o_w+"/"+m_w+","
            result.append(t[:-1])
        return result

    def predict_and_optim(self,hla_list=None,peptide_list=None,target_hla=None,target_peptide=None,read_npy=True):
        """
            hla  : str hla 序列
            peptide : str  序列

            read_csv : 如果为True 那么从文件中读取 position_pd 和 position_num_pd
                       如果未False 那么根据传入的hla_list peptide_list 生成 position_pd position_num_pd
        """
        if target_hla==None or target_peptide==None:
            print("请输入目标hla 和 目标peptide")
            return None
        length=len(target_peptide)
        #将单个hla,peptide 送入模型,得到预测值data,attn
        data,attn,peptide_self_attn, aatype_position_pd,aatype_position_num_pd=self.model.run_model(hla_list,peptide_list,target_hla,target_peptide,read_npy=read_npy)
        #使用强化学习
        # mutate_model 为了方便训练 hla和peptide输入的是列表,而aomp接受的输入都是单个序列,因此要将target_hla,target_peptide 放入到列表中
        mutation_peptides=self.mutate_model([aatype_position_pd],[aatype_position_num_pd],[attn],peptide_self_attn,[target_hla],[target_peptide])
        mutation_peptides=[target_peptide]+mutation_peptides
        #将新生成的肽送入模型中，得到得分和注意力矩阵
        #scores,attns=self.model.predict(hla,mutation_peptides)
        peptide_list=list(mutation_peptides)
        hla_list=[target_hla] * len(peptide_list)
        mutate_data,mutate_attn=self.model.predict(hla_list,peptide_list,is_mutated=True)
        mutate_data["original_peptide"]=target_peptide

        before_filter = mutate_data.shape[0] - 1
        #进行筛选
        mutate_data_ = mutate_data[(mutate_data.y_pred == 1)]
        if mutate_data_.shape[0] == 0 :
            #当前未获得有益突变
            print("!!!!!!!!!!!!当前未找到有益突变!!!!!!!!!!!!")
            #对亲和度进行排序,找到前几个较大的值
            mutate_data_=mutate_data.sort_values(by="y_prob",ascending=False).iloc[:10]
        mutate_data=mutate_data_
        after_filter = mutate_data.shape[0] - 1
        # 计算有益突变占比
        positive_mutate_rate = after_filter / before_filter
        mutation_peptides_ = [mutation_peptides[idx] for idx in mutate_data.index]
        mutate_attn_ = [mutate_attn[idx] for idx in mutate_data.index]
        #计算相似度
        # mutate_data["similarity"]=0.
        similarity_list=[]
        for row in range(mutate_data.shape[0]):
            same_count=0
            mutated_peptide=mutate_data.iloc[row]["mutated_peptide"]
            for i,w in enumerate(mutated_peptide):
                if w == target_peptide[i]:
                    same_count+=1
            similarity_list.append(same_count/len(target_peptide))
            #mutate_data.iloc[row]["similarity"]=same_count/len(target_peptide)
        mutate_data['similarity']=similarity_list
        #添加突变位点以及突变氨基酸
        posi_acid=self.mutate_position_acid(list(mutate_data.mutated_peptide),list(mutate_data.original_peptide))
        mutate_data["Mutation amino-acid site"]=posi_acid
        return mutation_peptides_,mutate_data,mutate_attn_,positive_mutate_rate

if __name__ == '__main__':
    hla=""
    t = pd.read_csv("./results/attention/HLA-A_11_01_AEAFIQPI_attention.csv")
    t.set_index("Unnamed: 0", inplace=True)
    for w in t.index:
        if w == "sum":
            break
        hla+=w
    #print(hla)
    peptide="BSAFIQPI"
    model=Aomp(hp_weight_path="./OurModel/PepPredictMutationFramework/predict_model/transformer/model/model_hp_layer1_head9_fold4.pth",
               tp_weight_path="./OurModel/PepPredictMutationFramework/predict_model/transformer/model/model_pt_layer1_head9_fold4.pth",use_RL=True,num_group=1,use_retnet=False)
    mutate_peptides,mutate_data,mutate_attn=model.predict_and_optim([hla,hla],[peptide,"AEAFIQSA"],hla,peptide,read_npy=False)
    print(mutate_attn)
    print(mutate_data)




