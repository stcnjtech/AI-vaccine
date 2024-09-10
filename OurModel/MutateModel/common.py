import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv(nn.Module):
    def __init__(self,C_in,C_out,kernel_size=3,padding=1,stride=1,activation=nn.ReLU()):
        super(Conv, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(C_in,C_out,kernel_size=kernel_size,padding=padding,stride=stride),
            nn.BatchNorm2d(C_out),
            activation,
        )
    def forward(self,inputs):
        return self.conv(inputs)

class AttnCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(AttnCell, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.Linear_k=nn.Linear(input_size,hidden_size)
        self.Linear_q=nn.Linear(input_size,hidden_size)
        self.Linear_v=nn.Linear(input_size,hidden_size)

    def forward(self,inputs1,inputs2,mask=None):
        """
            inputs1 : (batch_size,N1,input_size)
            inputs2 :  (batch_size,N2,input_size)
            output: (batch_size,N1,hidden_size)
            mask : None or (batch_size,N1,N2)
        """
        k=self.Linear_k(inputs2)  #(batch_size,N2,hidden_size)
        q=self.Linear_q(inputs1)  #(batch_size,N1,hidden_size)
        v=self.Linear_v(inputs2)  #(batch_size,N2,hidden_size)
        if mask!=None:
            mask = mask <= 0
            mat=torch.bmm(q,k.transpose(-1,-2))/np.sqrt(self.hidden_size)
            mat=mat.masked_fill(mask,1e-9)
            return torch.bmm(F.softmax(mat,dim=-1),v)
        return torch.bmm(F.softmax(torch.bmm(q,k.transpose(-1,-2))/np.sqrt(self.hidden_size),dim=-1),v)

class Residual(nn.Module):

    def __init__(self,feature_size,kernel_size=3,padding=1,stride=1,activation=nn.ReLU()):
        super(Residual, self).__init__()
        self.feature_size=feature_size
        self.model=nn.Sequential(
            Conv(feature_size,feature_size,activation=activation,kernel_size=kernel_size,padding=padding,stride=stride),
            Conv(feature_size,feature_size,activation=activation,kernel_size=kernel_size,padding=padding,stride=stride),
        )
    def forward(self,inputs):
        return inputs + self.model(inputs)

class Attn(nn.Module):

    def __init__(self,input_size,hidden_size,num_head):
        super(Attn, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_head=num_head
        self.attns=nn.ModuleList(
            [AttnCell(self.input_size,self.hidden_size) for _ in range(self.num_head)]
        )
    def forward(self,inputs1,inputs2,mask=None):
        """
            inputs1 : (batch_size,N1,L)
            inputs2 : (batch_size,N2,L)
            outputs:  (batch_Size,num_head,N1,L)
        """
        attn_list=[]
        for attn in self.attns:
            attn_list.append(torch.unsqueeze(attn(inputs1,inputs2,mask),dim=1))
        return torch.cat(attn_list,dim=1)

class GraphConv(nn.Module):
    """
        图卷积神经网络
    """
    def __init__(self,in_feature,out_feature,activation=nn.ReLU()):
        """
            in_C 输入通道
            out_C 输出通道
        """
        super(GraphConv,self).__init__()
        self.in_feature=in_feature
        self.out_C=out_feature
        self.w=nn.Parameter(torch.randn(in_feature,out_feature),requires_grad=True)
        self.activation=activation

    def forward(self,relation_matrix,inputs):
        """
            relation_matrix : 关系矩阵  （batch_size,N,N)
            inputs :         (batch_size,N,in_feature)
        """
        B,N,_=relation_matrix.shape
        #对角矩阵
        diagonal_matrix=torch.zeros_like(relation_matrix)
        for i in range(N):
            diagonal_matrix[:,i,i]=torch.sum(relation_matrix[:,i,:],dim=-1)
        #对角矩阵加上单位矩阵
        diagonal_matrix=diagonal_matrix+torch.eye(N)
        relation_matrix=relation_matrix+torch.eye(N)
        diagonal_matrix=torch.inverse(diagonal_matrix)
        z=diagonal_matrix @(relation_matrix @ inputs) @ self.w
        return self.activation(z)

class VectorConv(nn.Module):
    def __init__(self,in_C,out_C,horizontal,vertical):
        super(VectorConv, self).__init__()
        self.horizontal=horizontal
        self.vertical=vertical
        self.horizontal_conv=Conv(in_C,out_C,kernel_size=(1,horizontal),padding=0,stride=1,activation=nn.LeakyReLU(0.1))
        self.vertical_conv=Conv(in_C,out_C,kernel_size=(vertical,1),padding=0,stride=1,activation=nn.LeakyReLU(0.1))
        self.bn=nn.BatchNorm2d(out_C)

    def forward(self,inputs):
        """
            inputs  : (batch_size,C_in,20,15)
        """
        h=self.horizontal_conv(inputs)          #(batch_size,C_out,20,1)
        v=self.vertical_conv(inputs)            #(batch_size,C_out,1,15)
        return self.bn(h*v)                     #(batch_Size,C_out,20,15)

class GoogleNet(nn.Module):
    def __init__(self,C_in,C_out,horizontal,vertical):
        super(GoogleNet, self).__init__()
        #k1
        self.conv1=Conv(C_in,C_out,kernel_size=1,padding=0,stride=1)
        #k2
        self.conv2=nn.Sequential(
            Conv(C_in,C_out//2,kernel_size=1,padding=0,stride=1),
            VectorConv(C_out//2,C_out,horizontal=horizontal,vertical=vertical),
        )
        self.conv=Conv(2*C_out,C_out,kernel_size=1,padding=0,stride=1)
    def forward(self,inputs):
        k1=self.conv1(inputs)
        k2=self.conv2(inputs)
        return self.conv(torch.cat([k1,k2],dim=1))
