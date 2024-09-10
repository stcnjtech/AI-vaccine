from easydict import EasyDict

cfg=EasyDict()

#hla 最大长度
cfg.max_hla_length=34

#peptide 最大长度
cfg.max_peptide_length=15

#tcr 最大长度
cfg.max_tcr_length=34

#氨基酸列表
cfg.acid_list=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

#最多突变位点
cfg.max_mutate_position=4

#随机突变概率
cfg.random_mutate=0.15

#衰减
cfg.gamma=0.02

#放大因子
cfg.factor=2.5

cfg.vocab={'C': 1,'W': 2,'V': 3,'A': 4,'H': 5,'T': 6,'E': 7,'K': 8,'N': 9,'P': 10,'I': 11,'L': 12,'S': 13,'D': 14,'G': 15,'Q': 16,'R': 17,'Y': 18,'F': 19,'M': 20,'O': 21,'X': 22,'B':23,'J':24,'U':25,'Z':26,'-': 0}
cfg.vocab_size=len(cfg.vocab)

#随机数种子
#45
cfg.seed=45

#图卷积层数
cfg.gcn_num=4

