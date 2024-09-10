from OurModel.MutateModel.common import *
from OurModel.MutateModel.rl_config import cfg

class MutateBlock(nn.Module):
    def __init__(self,device,num_head=8,is_train=True):
        super(MutateBlock, self).__init__()
        self.device=device
        self.num_head=num_head
        self.is_train=is_train
        self.attnblock=Attn(cfg.max_peptide_length,cfg.max_peptide_length,num_head)
        self.convblock=nn.Sequential(
            Conv(self.num_head,32),
            Residual(32),
            Conv(32,64),
            Residual(64),
            Conv(64,128),
            Residual(128),
            Conv(128,64),
            Residual(64),
            Conv(64,32),
            Conv(32,2*cfg.max_mutate_position),
        )
    def forward(self,contrib,attn):
        """
            contrib : (batch_size,20,max_peptide_length)
            attn :   (batch_size,max_hla_length,max_peptide_length)
        """
        print(attn.shape)
        if np.random.uniform(0,1)<cfg.random_mutate and self.is_train==True:
            contrib=torch.ones_like(contrib)
        #对输入数据做归一化处理
        contrib=F.normalize(contrib,dim=(1,2))
        attn=F.normalize(attn,dim=(1,2))
        batch_size,n,l=contrib.shape
        contrib = contrib.to(self.device)
        attn = attn.to(self.device)
        matrix=self.convblock(self.attnblock(contrib,attn))   #(batch_size,2*max_mutate_position,20,max_peptide_length)
        matrix=matrix.reshape(batch_size,cfg.max_mutate_position,n,l,2)
        #print(F.softmax(matrix.reshape(batch_size,cfg.max_mutate_position,n,l,2),dim=-1))
        if self.is_train:
            #return F.sigmoid(matrix)
            return F.softmax(matrix,dim=-1).clone()
        else:
            #return F.sigmoid(matrix).detach()
            return F.softmax(matrix,dim=-1).clone().detach()
class MutateBlockv2(nn.Module):
    """
        突变模型v2版本
    """
    def __init__(self,device,is_train=True):
        super(MutateBlockv2, self).__init__()
        #self.graphConv=GraphConv(in_feature=len(cfg.acid_list),out_feature=len(cfg.acid_list))
        #图卷积层
        self.device=device
        self.graphConv_list=nn.ModuleList(
            [GraphConv(len(cfg.acid_list),len(cfg.acid_list)) for _ in range(cfg.gcn_num)]
        )
        #注意力层
        self.attn_model1=Attn(cfg.max_peptide_length,cfg.max_peptide_length,num_head=8)
        self.attn_model2=Attn(cfg.max_peptide_length,cfg.max_peptide_length,num_head=8)
        #向量卷积层
        self.vector_convs=nn.Sequential(
            VectorConv(8,16,cfg.max_peptide_length,len(cfg.acid_list)),
            VectorConv(16,64,cfg.max_peptide_length,len(cfg.acid_list)),
            VectorConv(64,128,cfg.max_peptide_length,len(cfg.acid_list)),
            #通道压缩
            Residual(128,kernel_size=1,padding=0,stride=1,activation=nn.LeakyReLU(0.1)),
            Conv(128,64,kernel_size=1,padding=0,stride=1,activation=nn.LeakyReLU(0.1)),
            Residual(64,kernel_size=1,padding=0,stride=1,activation=nn.LeakyReLU(0.1)),
            Conv(64,cfg.max_mutate_position*2,kernel_size=1,padding=0,stride=1,activation=nn.LeakyReLU(0.1))
        )
        #GoogleNet层
        self.g_net=GoogleNet(8,4,horizontal=cfg.max_peptide_length,vertical=cfg.max_peptide_length)
        self.is_train=is_train

    def get_self_attn_mask(self,self_attn,peptide_len):
        """
            self_attn : (batch_size,15,15)
            peptide_len : (batch_size,)
        """
        mask=torch.zeros_like(self_attn)
        for i,len in enumerate(peptide_len):
            mask[i,:len,:len]=1
        return mask

    def get_cross_attn_mask(self,cross_attn,lens1,lens2):
        """
            cross_attn : (batch_size,34,15)
            hla_or_tcr_len : (batch_size,)
            peptide_len : (batch_size,)
        """
        mask=torch.zeros_like(cross_attn)
        for i,h_or_t_len in enumerate(lens1):
            p_len=lens2[i]
            mask[i,:h_or_t_len,:p_len]=1
        return mask

    def forward(self,contrib,attn,peptide_self_attn,hla_or_tcr_len,peptide_len):
        """
            contrib:   (batch_size,20,15)
            attn   :   (batch_size,34,15)
            peptide_self_attn : (batch_size,15,15)
            hla_len  : (batch_size,)
            peptide_len : (batch_size,)
        """
        contrib=contrib.to(self.device)
        peptide_self_attn=peptide_self_attn.to(self.device)
        batch_size,_,_=contrib.shape
        cross_attn_mask = self.get_cross_attn_mask(attn, hla_or_tcr_len, peptide_len)
        self_attn_mask = self.get_self_attn_mask(peptide_self_attn, peptide_len)
        attn = attn * cross_attn_mask
        peptide_self_attn=peptide_self_attn*self_attn_mask
        contrib=contrib.transpose(-1,-2)    # (batch_size,15,20)
        #随机抖动
        if self.is_train==True and np.random.uniform(0,1)<cfg.random_mutate:
            contrib=torch.ones_like(contrib)
        #多层图卷积
        for gcn in self.graphConv_list:
            contrib=gcn(peptide_self_attn,contrib)   #(batch_size,15,20)
        #使用注意力机制将contrib图和attn图关联起来
        contrib=contrib.transpose(-1,-2)
        cross_matrix=self.attn_model1(contrib,attn)   #(batch_size,num_head,20,15)
        #使用注意力机制将peptide_self_attn与attn图关联起来
        mask_=self.get_cross_attn_mask(peptide_self_attn @ attn.transpose(-1,-2) ,peptide_len,hla_or_tcr_len)
        """
            peptide_self_attn : (batch_size,15,15)
            attn              : (batch_size,34,15)
        """
        peptide_self_matrix=self.attn_model2(peptide_self_attn,attn,mask_)
        #向量卷积
        cross_matrix=self.vector_convs(cross_matrix) #(batch_size,8,20,15)
        #使用googlenet
        #peptide_self_attn=peptide_self_attn.unsqueeze(dim=1)
        peptide_self_matrix=F.softmax(self.g_net(peptide_self_matrix).masked_fill(self_attn_mask.unsqueeze(dim=1)<=0,-1e2),dim=-1).clone()
        #peptide_self_matrix=peptide_self_matrix.masked_fill(self_attn_mask.unsqueeze(dim=1)<=0,1e-9)
        _,_,h,w=cross_matrix.shape
        cross_matrix=cross_matrix.reshape(batch_size,-1,h,w,2)
        cross_matrix_prob=F.softmax(cross_matrix,dim=-1).clone()

        return cross_matrix,cross_matrix_prob,peptide_self_matrix

if __name__ == '__main__':
   batch_size=4
   contrib=torch.randn(size=(batch_size,20,15))
   attn=torch.randn(size=(batch_size,34,15))
   peptide_self_attn=torch.randn(size=(batch_size,15,15))
   hla_len=torch.randint(30,35,size=(batch_size,))
   peptide_len=torch.randint(8,16,size=(batch_size,))
   mutate2=MutateBlockv2(torch.device("cpu"))
   mutate2(contrib,attn,peptide_self_attn,hla_len,peptide_len)