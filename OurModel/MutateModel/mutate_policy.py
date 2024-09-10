"""
    突变策略
"""
import torch
from OurModel.MutateModel.rl_config import cfg


def mutate_one_position(mutate_matrix,peptide_self_attn,hla,peptides,mutated_peptides,is_train=True):
    """
        mutate_matrix : (batch_size,20,15,2)
        peptide_self_attn : (batch_size,15,15)
        hla         :   (batch_size,)
        peptides    :   (batch_size,)
        mutated_peptides : (batch_size,list)
    """
    mutate_result=[]
    label_matrix=torch.argmax(mutate_matrix,dim=-1)      #(batch_size,20,15)
    batch_size,acid_num,_=label_matrix.shape
    for i in range(batch_size):
        cur_hla=hla[i]
        cur_peptide=peptides[i]
        for acid in range(acid_num):
            for pep in range(len(cur_peptide)):
                if label_matrix[i,acid,pep]>=1:
                   target_acid=cfg.acid_list[acid]
                   if target_acid != cur_peptide[pep]:
                       new_peptide=cur_peptide[:pep]+target_acid+cur_peptide[pep+1:]
                       if new_peptide not in mutated_peptides[i]:
                           mutate_result.append([i,0,(int(acid),),(int(pep),),cur_hla,new_peptide])
                           mutated_peptides[i].append(new_peptide)
    return mutate_result,mutated_peptides


def mutate_two_position(mutate_matrix,peptide_self_attn,hla,peptides,mutated_peptides,is_train=True):
    """
        mutate_matrix : (batch_size,20,15,2)
        peptide_self_attn : (batch_size,15,15)
        hla         :   (batch_size,)
        peptides    :   (batch_size,)
        mutated_peptides : list
    """
    mutate_result = []
    label_matrix = torch.argmax(mutate_matrix, dim=-1)  # (batch_size,20,15)
    score_matrix=torch.argsort(mutate_matrix[:,:,:,1],dim=1,descending=True)[:,:3,:]                 # (batch_size,4,15)
    batch_size, acid_num, _ = label_matrix.shape
    mask_mat=torch.eye(15)   #（15,15）
    peptides_cross_matrix=torch.argsort(peptide_self_attn.masked_fill(mask_mat==1,0),dim=-1,descending=True)[:,:,:2]   #(batch_size,15,2)
    for i in range(batch_size):
        cur_hla=hla[i]
        cur_peptide=peptides[i]
        for pep in range(len(cur_peptide)):
            for _pep in peptides_cross_matrix[i,pep,:]:
                """
                    pep : 第一个突变位置
                    _pep : 第二个突变位置
                """
                #找到第一个突变位置前几个可突变的氨基酸
                one_acid_list=score_matrix[i,:,pep]    #(batch_size,2)
                two_acid_list=score_matrix[i,:,_pep]
                for one_acid in one_acid_list:
                    for two_acid in two_acid_list:
                        mutate_one_acid=cfg.acid_list[one_acid]
                        mutate_two_acid=cfg.acid_list[two_acid]
                        if mutate_one_acid != cur_peptide[pep] and mutate_two_acid != cur_peptide[_pep]:
                            new_peptide=cur_peptide[:pep]+mutate_one_acid+cur_peptide[pep+1:]
                            new_peptide=new_peptide[:_pep]+mutate_two_acid+new_peptide[_pep+1:]
                            if new_peptide not in mutated_peptides[i]:
                                mutate_result.append([i, 1, (int(one_acid),int(two_acid)), (int(pep),int(_pep)), cur_hla, new_peptide])
                                mutated_peptides[i].append(new_peptide)
        return mutate_result,mutated_peptides

def mutate_three_position(mutate_matrix,peptide_self_attn,hla,peptides,mutated_peptides,is_train=True):
    """
        mutate_matrix : (batch_size,20,15,2)
        peptide_self_attn : (batch_size,15,15)
        hla         :   (batch_size,)
        peptides    :   (batch_size,)
        mutated_peptides : list
    """
    mutate_result = []
    label_matrix = torch.argmax(mutate_matrix, dim=-1)  # (batch_size,20,15)
    score_matrix = torch.argsort(mutate_matrix[:, :, :, 1], dim=1,descending=True)[:, :3, :]  # (batch_size,2,15)
    batch_size, acid_num, _ = label_matrix.shape
    mask_mat = torch.eye(15)  # （15,15）
    peptides_cross_matrix = torch.argsort(peptide_self_attn.masked_fill(mask_mat == 1, 0), dim=-1, descending=True)[:,
                            :, :2]  # (batch_size,15,2)
    for i in range(batch_size):
        cur_hla = hla[i]
        cur_peptide = peptides[i]
        for pep in range(len(cur_peptide)):
            # print(peptides_cross_matrix[batch_size,pep,:].shape)
            pep2,pep3=peptides_cross_matrix[i,pep,:]  #pep2第二个突变位置,pep3第三个突变位置
            one_acid_list = score_matrix[i, :, pep]            #(2)
            two_acid_list = score_matrix[i, :, pep2]
            three_acid_list=score_matrix[i,:,pep3]
            for one_acid in one_acid_list:
                for two_acid in two_acid_list:
                    for three_acid in three_acid_list:
                        mutate_one_acid=cfg.acid_list[one_acid]
                        mutate_two_acid=cfg.acid_list[two_acid]
                        mutate_three_acid=cfg.acid_list[three_acid]
                        if mutate_one_acid != cur_peptide[pep] and mutate_two_acid != cur_peptide[
                            pep2] and mutate_three_acid != cur_peptide[pep3]:
                            new_peptide = cur_peptide[:pep] + mutate_one_acid + cur_peptide[pep + 1:]
                            new_peptide = new_peptide[:pep2] + mutate_two_acid + new_peptide[pep2 + 1:]
                            new_peptide = new_peptide[:pep3] + mutate_three_acid + new_peptide[pep3+1:]
                            if new_peptide not in mutated_peptides[i]:
                                mutate_result.append(
                                    [i, 2, (int(one_acid), int(two_acid),int(three_acid)), (int(pep), int(pep2) , int(pep3)), cur_hla, new_peptide])
                                mutated_peptides[i].append(new_peptide)
    return mutate_result,mutated_peptides

def mutate_four_position(mutate_matrix,peptide_self_attn,hla,peptides,mutated_peptides,is_train=True):
    """
        mutate_matrix : (batch_size,20,15,2)
        peptide_self_attn : (batch_size,15,15)
        hla         :   (batch_size,)
        peptides    :   (batch_size,)
        mutated_peptides : list
    """
    mutate_result = []
    label_matrix = torch.argmax(mutate_matrix, dim=-1)  # (batch_size,20,15)
    score_matrix = torch.argsort(mutate_matrix[:, :, :, 1], dim=1,descending=True)[:, :3, :]  # (batch_size,2,15)
    batch_size, acid_num, _ = label_matrix.shape
    mask_mat = torch.eye(15)  # （15,15）
    peptides_cross_matrix = torch.argsort(peptide_self_attn.masked_fill(mask_mat == 1, 0), dim=-1, descending=True)[:,
                            :, :3]  # (batch_size,15,2)
    for i in range(batch_size):
        cur_hla = hla[i]
        cur_peptide = peptides[i]
        for pep in range(len(cur_peptide)):
            pep2, pep3,pep4 = peptides_cross_matrix[i, pep, :]  # pep2第二个突变位置,pep3第三个突变位置
            one_acid_list = score_matrix[i, :, pep]  # (2)
            two_acid_list = score_matrix[i, :, pep2]
            three_acid_list = score_matrix[i, :, pep3]
            four_acid_list=score_matrix[i,:,pep4]
            for one_acid in one_acid_list:
                for two_acid in two_acid_list:
                    for three_acid in three_acid_list:
                        for four_acid in four_acid_list:
                            mutate_one_acid = cfg.acid_list[one_acid]
                            mutate_two_acid = cfg.acid_list[two_acid]
                            mutate_three_acid = cfg.acid_list[three_acid]
                            mutate_four_acid=cfg.acid_list[four_acid]
                            if mutate_one_acid != cur_peptide[pep] and mutate_two_acid != cur_peptide[pep2] and mutate_three_acid != cur_peptide[pep3] and mutate_four_acid != cur_peptide[pep4]:
                                new_peptide = cur_peptide[:pep] + mutate_one_acid + cur_peptide[pep + 1:]
                                new_peptide = new_peptide[:pep2] + mutate_two_acid + new_peptide[pep2 + 1:]
                                new_peptide = new_peptide[:pep3] + mutate_three_acid + new_peptide[pep3 + 1:]
                                new_peptide = new_peptide[:]
                                if new_peptide not in mutated_peptides[i]:
                                    mutate_result.append(
                                        [i, 3, (int(one_acid), int(two_acid), int(three_acid),int(four_acid)),
                                         (int(pep), int(pep2), int(pep3),int(pep4)), cur_hla, new_peptide])
                                    mutated_peptides[i].append(new_peptide)
    return mutate_result, mutated_peptides














































