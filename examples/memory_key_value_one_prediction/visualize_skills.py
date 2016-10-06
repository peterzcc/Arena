import os
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from arena.utils import *

def softrelu(a):
    return np.log(1 + np.exp(a))

def l2_normalize(a, axis=1):
    return a / np.linalg.norm(a, axis=axis, keepdims=True)

def fc_forward(a, weight, bias):
    return np.dot(a, weight.T) + bias

def read_assistmapping(path='D:\\HKUST\\3-1\\KVMN\\data\\assist2009_old\\assist2009_old_skill_mapping'):
    ret = dict()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            ret[int(line[0])] = " ".join(line[1:])
    return ret

#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\KDDal0506\\key_value_one_qembed50qaembed200qdim50qadim200msize30mkdim200mvdim200h1std0.1lr0.1mmt0.9gn50.0\\200_0.815431426956'
#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\KDDal0506\\key_value_one_qembed50qaembed200qdim50qadim200msize30mkdim25mvdim200h1std0.1lr0.2mmt0.9gn50.0\\200_0.811471436206'
#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\KDDal0506\\key_value_one_qembed50qaembed200qdim50qadim200msize10mkdim200mvdim200h1std0.1lr0.1mmt0.9gn50.0\\199_0.811695282869'
#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\KDDal0506\\key_value_one_qembed50qaembed200qdim50qadim200msize50mkdim200mvdim200h1std0.1lr0.1mmt0.9gn50.0\\199_0.81316966251'
#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\KDDal0506\\nonorm_key_value_one_qembed50qaembed200qdim50qadim200msize30mkdim25mvdim200h1std0.1lr0.2mmt0.9gn50.0\\199_0.811707523499'


#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\KDDal0506\\nonorm_nobeta_novaluekeytanh_key_value_one_qembed25qaembed200qdim50qadim200msize10mkdim25mvdim200h1std0.1lr0.2mmt0.9gn50.0\\14_0.790419002745'
#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\STATICS\\_key_value_qembed50qaembed100qdim25qadim50msize50mkdim25mvdim50h1std0.1lr0.01mmt0.9gn50.0\\29_0.830373212804'
#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\KDDal0506\\key_value_qembed50qaembed200qdim50qadim200msize50mkdim50mvdim200h1std0.1lr0.1mmt0.9gn50.0\\99_0.802574654345'

#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\KDDal0506\\nonorm_nobeta_novaluekeytanh_key_value_one_qembed25qaembed200qdim50qadim200msize10mkdim25mvdim200h1std0.1lr0.2mmt0.9gn50.0\\41_0.806558686103'
#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\KDDal0506\\nonorm_nobeta_novaluekeytanh_key_value_one_qembed25qaembed200qdim50qadim200msize10mkdim25mvdim200h1std0.1lr0.2mmt0.9gn50.0\\93_0.81129977328'

#dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\assist2009_old\\nonorm_nobeta_novaluekeytanh_key_value_one_qembed20qaembed100qdim20qadim50msize10mkdim20mvdim50h1std0.05lr0.1mmt0.9gn50.0\\12_0.851483820253'
#base_dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\assist2009_old\\nonorm_nobeta_novaluekeytanh_key_value_one_qembed20qaembed100qdim20qadim50msize10mkdim20mvdim50h1std0.05lr0.1mmt0.9gn50.0'
#base_dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\assist2009_old\\nonorm_nobeta_novaluekeytanh_shortcut_key_value_one_qembed20qaembed100qdim20qadim100msize5mkdim20mvdim100h1std0.1lr0.05mmt0.9gn50.0'

#base_dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\assist2009_old\\nonorm_nobeta_novaluekeytanh_shortcut_key_value_one_qembed20qaembed100qdim20qadim100msize10mkdim20mvdim100h1std0.1lr0.05mmt0.9gn50.0'
#base_dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\assist2009_updated\\rnn_nonorm_nobeta_novaluekeytanh_shortcut_key_value_one_qembed20qaembed100qdim20qadim100msize5mkdim20mvdim100h1std0.1lr0.05mmt0.9gn50.0'
base_dir_path = 'D:\\HKUST\\3-1\\KVMN\\memory_key_value_one_prediction\\model\\assist2009_updated\\reg_nonorm_nobeta_novaluekeytanh_shortcut_key_value_one_qembed20qaembed100qdim20qadim100msize10mkdim20mvdim100h1std0.1lr0.05mmt0.9gn50.0'
dir_names = os.listdir(base_dir_path)
'''
params = nd.load(os.path.join(dir_path, 'KVMN_KT.params'))
key_weight = params['arg:KVMN->key_head:key_weight'].asnumpy()
key_bias = params['arg:KVMN->key_head:key_bias'].asnumpy()
key_beta_weight = params['arg:KVMN->key_head:beta_weight'].asnumpy()
key_beta_bias = params['arg:KVMN->key_head:beta_bias'].asnumpy()
key_init_memory = params['arg:init_memory_key'].asnumpy()
q_embed = params['arg:q_embed_weight'].asnumpy()

q_embed = np.tanh(q_embed)
key_init_memory = np.tanh(key_init_memory)
q_key = np.tanh(fc_forward(q_embed, key_weight, key_bias))
q_beta = softrelu(fc_forward(q_embed, key_beta_weight, key_beta_bias))
similarity_score = np.dot(l2_normalize(q_key), l2_normalize(key_init_memory).T) * q_beta
#similarity_score = np.dot(q_key, key_init_memory.T) * q_beta
q_concept_weight = npy_softmax(similarity_score, axis=1)
'''
assist_mapping = read_assistmapping()

epoch_num = 200
for name in dir_names:
    if name.startswith("%d_"%epoch_num) and 'similarity' not in name:
        dir_path = os.path.join(base_dir_path, name)
        params = nd.load(os.path.join(dir_path, 'KVMN_KT.params'))
        key_init_memory = params['arg:init_memory_key'].asnumpy()
        q_embed = params['arg:q_embed_weight'].asnumpy()
        similarity_score = np.dot(q_embed, key_init_memory.T)
        q_concept_weight = npy_softmax(similarity_score, axis=1)
        q_class = q_concept_weight.argmax(axis=1)
        skill_clusters = []
        for concept_id in range(q_concept_weight.shape[1]):
            max_score_questions = q_concept_weight[:, concept_id].argsort()[::-1]
            skill_names = []
            for q in max_score_questions:
                if q not in assist_mapping:
                    print concept_id, q
                    continue
                if q_class[q] == concept_id:
                    skill_names.append(assist_mapping[q])
            skill_clusters.append(skill_names)

