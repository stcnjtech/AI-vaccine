import pymol
from pathlib import Path
from Bio.SeqUtils import seq3
import re
import os
import multiprocessing
from multiprocessing.pool import Pool
from functools import partial


def replace_invalid_chars(string):
    pattern = r'[^a-zA-Z0-9_]'
    replaced_string = re.sub(pattern, '_', string)
    return replaced_string


def convertTo3Letter(amino_acid_code):
    # 将氨基酸单个字母转换为三字母缩写
    three_letter_code = seq3(amino_acid_code)
    if three_letter_code:
        return three_letter_code.upper()
    else:
        print("无效的氨基酸单字母缩写。")


def getData(origPeptide, mutPeptide, chain, index):
    MutDict = {chain: {}}
    for i in range(len(origPeptide)):
        if (origPeptide[i] != mutPeptide[i]):
            MutDict[chain][index + i] = convertTo3Letter(mutPeptide[i])
    return MutDict


def changeData(data):
    for origPeptide, value1 in data.items():
        for mutPepttide, value2 in value1.items():
            index = value2[1]
            chain = value2[0]
            value1[mutPepttide] = getData(origPeptide, mutPepttide, chain, index)
    return data


def drawPic(pdbFile, mutations, path):
    # 创建文件夹
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    # 加载初始文件
    pymol.pymol_argv = ['pymol', '-qc']
    pymol.finish_launching()
    pymol.cmd.load(pdbFile, "protein")
    pymol.cmd.color("cyan", "protein")

    # 更好文件格式
    name = pdbFile.split('/')
    name = name[-1].split('.')
    mutFile = ''
    if name[1] == 'cif':
        s = name[0] + '.pdb'
        pdbFile = path + s
        mutFile = path + "mut" + s
        pymol.cmd.save(pdbFile, format='pdb', selection="protein")
    else:
        name = pdbFile.split('/')
        mutFile = path + "mut" + name[-1]

    # 创建新的蛋白质结构并进行突变
    mutated_structure = pymol.cmd.get_session()
    for chain_id, mutation in mutations.items():
        for position, aminoacid in mutation.items():
            selection = f"chain {chain_id} and resi {position}"
            pymol.cmd.alter(selection, f"resn='{aminoacid}'")
            pymol.cmd.select("modified_residues", selection)
            pymol.cmd.color("red", "modified_residues")

    #pymol.cmd.set('opengl', '1')

    # 保存修改后的蛋白质结构为.pdb文件
    mut_pdb_file = mutFile
    pymol.cmd.save(mut_pdb_file, format='pdb', selection="protein")
    # 生成修改后的蛋白质结构图
    pymol.cmd.load(mut_pdb_file, 'protein')
    mut_image_file = path + "Mut.png"
    pymol.cmd.png(mut_image_file, width=800, height=600, dpi=300)
    # pymol.cmd.delete("protein")
    pymol.cmd.quit()
    return mut_pdb_file


def draw(pdbFile, data):
    picPath = []
    mutFilePath = []
    data = changeData(data)

    filePeptide = list(data.keys())[0]
    fileMutPeptide = list(list(data.values())[0].keys())[0]
    mutations = list(list(data.values())[0].values())[0]

    # for filePeptide, value2 in data.items():
    #     for fileMutPeptide, mutations in value2.items():

    path = 'D:/mutResult/'
    filePeptide = replace_invalid_chars(filePeptide)
    fileMutPeptide = replace_invalid_chars(fileMutPeptide)
    path += filePeptide + '/' + fileMutPeptide + '/'
    mut_file_path = drawPic(pdbFile, mutations, path)
    picPath.append(os.path.abspath(f'{path}Mut.png'))
    mutFilePath.append(os.path.abspath(f'{mut_file_path}'))

    return picPath, mutFilePath


def process_environment(pdbFile, data):
    # 在每个环境中执行绘制任务
    result = draw(pdbFile, data)
    return result


def drawPdbPic(pdbFile, data):
    num_processes = 1  # 设置要创建的进程数量
    pool = Pool(processes=num_processes)
    tasks = [(pdbFile, data)] * num_processes
    results = pool.starmap(process_environment, tasks)
    pool.close()
    pool.join()
    return results[0][0], results[0][1]


if __name__ == '__main__':
    pdbFile = '../draw/ori_pep.cif'  # .cif 或者 .pdb
    data = {  # 数据必须按照下述结构传入
        "AEAFIQSAQAQTIQPI": {  # 初始Peptide
            "PAQTIQPIIAQTIQPI": ['C', 37]  # 突变Peptide
        }
    }

    imagePath, filePath = drawPdbPic(pdbFile, data)

    print(imagePath)
    print(filePath)
