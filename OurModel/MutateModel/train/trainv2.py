import matplotlib.pyplot as plt
from torch.optim.adam import Adam
import os
from OurModel.MutateModel.mutation_model import *
from OurModel.MutateModel.mutate_modelv2 import *
from torch.utils.data import Dataset,DataLoader
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from OurModel.PepPredictMutationFramework.predict_model.transformer.transformer.Transformer import Transformer

#训练次数
epochs=2000
#批大小
batch_size=16
#hla-peptide 预测模型参数文件
hp_weight_path= '../../PepPredictMutationFramework/predict_model/transformer/model/model_hp_layer1_head9_fold4.pth'
tp_weight_path= "../../PepPredictMutationFramework/predict_model/transformer/model/model_pt_layer1_head9_fold4.pth"
num_group=1
#学习率
lr=0.001
#数据集地址
data_path= "../dataset/dataset1.csv"
#预测模型
device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
predictblock=Transformer(cfg.vocab_size, n_enc_heads=9,n_dec_heads=9,device=device)

if num_group == 1 and hp_weight_path != None and os.path.exists(hp_weight_path):
    predictblock.load_state_dict(
        torch.load(hp_weight_path, map_location='cpu')) if not torch.cuda.is_available() else \
        predictblock.load_state_dict(torch.load(hp_weight_path))
    print("已加载模型")
elif num_group != 1 and tp_weight_path != None and os.path.exists(tp_weight_path):
    predictblock.load_state_dict(
        torch.load(tp_weight_path, map_location="cpu")) if not torch.cuda.is_available() else \
        predictblock.load_state_dict(torch.load(tp_weight_path))
    print("已加载模型")
#训练模型
#mutation=Mutation(predictblock,True,num_group)

mutation=Mutationv2(predictblock,True,num_group,device).to(device)
#优化器
optim=Adam(mutation.actor.parameters(),lr=lr)

class MyDataSet(Dataset):
    def __init__(self,data,label="positive"):
        super(MyDataSet, self).__init__()
        self.hla_list=list(data.hla)
        self.hla_name_list=list(data.hla_name)
        self.peptide_list=list(data.peptide)
        self.label=label

    def __getitem__(self, item):
        hla=self.hla_list[item]
        hla_name=self.hla_name_list[item]
        peptide=self.peptide_list[item]
        dir_path1 = "../../../Attention/peptideAAtype_peptidePosition"
        dir_path2 = "../../../Attention/peptideAAtype_peptidePosition_NUM"
        file_path1 = os.path.join(dir_path1, f"{hla_name}_Length{len(peptide)}.npy")
        file_path2 = os.path.join(dir_path2, f"{hla_name}_Length{len(peptide)}_num.npy")
        position_pd=np.load(file_path1,allow_pickle=True).item()[self.label]
        position_num_pd=np.load(file_path2,allow_pickle=True).item()[self.label]
        return position_pd,position_num_pd,hla,peptide

    def __len__(self):
        return len(self.hla_list)

data=pd.read_csv(data_path)
idx=np.random.choice(data.index,size=2000)
data=data.iloc[idx]

dataset=MyDataSet(data)
print("已加载数据")
def collate_fn(batch_list):
    hla=[]
    peptide=[]
    position_pd_list=[]
    position_num_pd_list=[]
    for batch in batch_list:
        position_pd_list.append(batch[0])
        position_num_pd_list.append(batch[1])
        hla.append(batch[2])
        peptide.append(batch[3])
    return position_pd_list,position_num_pd_list,hla,peptide
dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)

loss_list=[]
print_=True
with torch.autograd.set_detect_anomaly(True):
    for epoch in range(epochs):
        for i,data in enumerate(dataloader):
            position_pd,position_num_pd,hla,peptide=data
            loss=mutation(position_pd,position_num_pd,None,None,hla,peptide)
            #print("-------")
            if i%1000==0:
                loss_list.append(loss.data)
                if print_:
                    print(f"{epoch} - {i} loss : {loss.data}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        if epoch%200==0:
            torch.save(mutation.actor.state_dict(), f"./mutation-epoch={epoch}.pth")

#保存模型
torch.save(mutation.actor.state_dict(),f"./mutation-num_group={num_group}.pth")
plt.plot(np.arange(len(loss_list)),loss_list)
plt.savefig(f"./rl_loss-num_group={num_group}.png")
