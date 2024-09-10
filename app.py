"""
    web接口
"""
import json
from flask import Flask, request, jsonify
from model_interface import *
from GeneratePic.genderPic import *

app = Flask(__name__)

# 预测模型
@app.route('/upload/<modelId>', methods=['POST'])
def upload_file(modelId):
    file = request.files['file']
    if file:
        filename = file.filename
        path = os.path.join('./file/', filename)
        file.save(path)
        hla_list, peptide_list,link_list,start_list = read_file_hp('./file/{}'.format(filename))
        data,attn_pd_list = predict(hla_list,peptide_list,num_group=1,use_retnet=(modelId == '1')) # 预测结果
        result_list = output_processing(data, attn_pd_list,link_list,start_list) # 后处理
        return json.dumps(result_list, ensure_ascii=False, indent=4, sort_keys=True)
    else:
        return jsonify({'message': 'No file selected'})

# 突变模型
@app.route('/mutation', methods=['POST'])
def mutate():
    request_data = request.form

    hla = request_data.get('hla')
    pep = request_data.get('pep')

    mutate_peptides, mutate_data, mutate_attn, positive_mutate_rate = mutation(target_hla=hla, target_peptide=pep,num_group=1,use_RL=True)
    result_list = output_processing(mutate_data,mutate_attn)

    return json.dumps(result_list)

# 蛋白质结构图
@app.route("/draw_upload/",methods=['POST'])
def draw_upload():
    file = request.files['file']
    if file:
        (name, suffix) = os.path.splitext(file.filename)
        filename = 'ori_pep' + suffix
        path = os.path.join('./draw/', filename)
        file.save(path)
        return jsonify({'message': 'ok'})
    else:
        return jsonify({'message': 'No file selected'})

@app.route("/draw/",methods=['POST'])
def draw():
    path = os.path.join('./draw/', 'ori_pep.cif')
    flag = os.path.exists(path)
    pdbFile = None
    if flag:
        pdbFile = os.path.join('./draw/', 'ori_pep.cif')
    else:
        pdbFile = os.path.join('./draw/', 'ori_pep.pdb')

    data = request.form
    ori_pep = data.get('ori_pep')
    mutate_pep = data.get('mut_pep')
    list_id = data.get('list_id')
    start_id = data.get('start_id')
    start_id = int(start_id)
    data_draw = {
        ori_pep:{
            mutate_pep:[list_id, start_id]
        }
    }

    png, file_path = drawPdbPic(pdbFile, data_draw)
    return {'png_path': png[0], "pdb_path": file_path[0]}

@app.route('/upload2/<modelId>', methods=['POST'])
def upload_file2(modelId):
    file = request.files['file']
    if file:
        filename = file.filename
        path = os.path.join('./file2/', filename)
        file.save(path)
        tcr_list, peptide_list,link_list,start_list = read_file_tp('./file2/{}'.format(filename))
        data,attn_pd_list = predict(tcr_list,peptide_list,num_group=2,use_retnet=(modelId == '1')) # 预测结果
        result_list = output_processing(data, attn_pd_list,link_list,start_list) # 后处理
        return json.dumps(result_list, ensure_ascii=False, indent=4, sort_keys=True)
    else:
        return jsonify({'message': 'No file selected'})

# 突变模型
@app.route('/mutation2/', methods=['POST'])
def mutate2():
    request_data = request.form

    tcr = request_data.get('tcr')
    pep = request_data.get('pep')

    mutate_peptides, mutate_data, mutate_attn, positive_mutate_rate = mutation(target_hla=tcr, target_peptide=pep,num_group=2,use_RL=True)
    result_list = output_processing(mutate_data,mutate_attn)

    return json.dumps(result_list)


@app.route("/draw2_upload/",methods=['POST'])
def draw2_upload():
    file = request.files['file']
    if file:
        (name, suffix) = os.path.splitext(file.filename)
        filename = 'ori_pep' + suffix
        path = os.path.join('./draw2/', filename)
        file.save(path)
        return jsonify({'message': 'ok'})
    else:
        return jsonify({'message': 'No file selected'})


@app.route("/draw2/",methods=['POST'])
def draw2():
    path = os.path.join('./draw2/', 'ori_pep.cif')
    flag = os.path.exists(path)
    pdbFile = None
    if flag:
        pdbFile = os.path.join('./draw/', 'ori_pep.cif')
    else:
        pdbFile = os.path.join('./draw/', 'ori_pep.pdb')

    data = request.form
    ori_pep = data.get('ori_pep')
    mutate_pep = data.get('mut_pep')
    list_id = data.get('list_id')
    start_id = data.get('start_id')
    start_id = int(start_id)

    data_draw = {
        ori_pep:{
            mutate_pep:[list_id,start_id]
        }
    }

    png, file_path = drawPdbPic(pdbFile, data_draw)
    return {'png_path': png[0], "pdb_path": file_path[0]}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8590)