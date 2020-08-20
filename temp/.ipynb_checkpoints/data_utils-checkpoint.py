# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import random
from collections import defaultdict


def Count(train_path):
    entity_dict = defaultdict(int)
    relation_dict = defaultdict(int)
    with open(train_path, 'r') as f:
        datas = f.readlines()
        for data in datas:
            [e1, e2, r] = data.strip().split('\t')
            entity_dict[e1] += 1
            entity_dict[e2] += 1
            relation_dict[r] += 1
    return entity_dict, relation_dict


def path_select(path_num_dict, threshold):
    path_num_dict = path_num_dict
    threshold = threshold
    path_threshold_list = []
    for key in path_num_dict.keys():
        if path_num_dict[key] > threshold:
            path_threshold_list.append([key])
    return path_threshold_list


def load_all(data_feature_dict, metapath_len):
    feature = []
    label = []
    for key in data_feature_dict.keys():
        data = data_feature_dict[key]
        for n, d in enumerate(data):
            data[n] = float(d)
            if len(data) == metapath_len + 1:
                feature.append(np.array(data[1:], dtype=np.float32))
                label.append(np.array(data[0], dtype=np.int64))
    return feature, label


class PRAData(Dataset):
    def __init__(self, data_feature_dict, metapath_len):
        super(PRAData, self).__init__()
        self.feature, self.label = load_all(data_feature_dict, metapath_len)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]


class GenerateData:

    def __init__(self, train_path):
        self.train_path = train_path  # 训练数据集三元组存储路径
        self.neg_pairs = []  # 负样本对
        self.train_pairs = []
        #self.pos_pairs_dict = defaultdict(list)

    def gen_pos_data(self):   # 产生对应数据集每一个关系的正样本对并保存在相应关系的文件夹下 ./relation_pairs/relation1/pos.txt
        pos_pairs_dict = defaultdict(list)
        path = './relation_pairs'
        _, relation_dict = Count(self.train_path)
        relation_list = relation_dict.keys()
        for relation in relation_list:
            '''
            if os.path.exists(path + '/' + relation):
                pass
            else:
                os.mkdir(path + '/' + relation)  # 产生相应的文件夹
            pos_set = []
            '''
            with open(self.train_path, 'r') as f:  # 读取数据集的三元组文件
                datas = f.readlines()
                for data in datas:
                    [e1, e2, r] = data.strip().split('\t')
                    pos_pairs_dict[r].append((e1, e2))
            '''
                    r = r.replace('concept:', '')
                    if relation == r:
                        pos_set.append((e1, e2))
                    else:
                        continue
                with open(path + '/' + relation + '/' + 'pos', 'w') as f2:  # 在相应关系文件夹下生成对应该关系的正样本
                    for (e1, e2) in pos_set:
                     f2.write('thing$' + e1 + ',' + e2 + ': +' + '\n')  # thing$concept:journalist:cynthia_mcfadden,concept:city:abc: +
            '''
        return pos_pairs_dict  # 这个是以关系作为键的实体对字典存储的

    def sample_neg_data(self):
        train_entity = []
        train_pairs = []  # ['concept:chemical:companies', 'concept:dateliteral:n2005', 'concept:atdate']
        raw_neg_pairs = []
        with open(self.train_path, 'r') as f:
            datas = f.readlines()
            for data in datas:
                [e1, e2, r] = data.strip().split('\t')
                train_pairs.append([e1, e2, r])
                if e1 not in train_entity:
                    train_entity.append(e1)
                if e2 not in train_entity:
                    train_entity.append(e2)
        for item in train_pairs:  # 随机替换已有训练集三元组的头实体或者尾实体
            pr = random.random()  # 计算替换头实体或者是替换尾实体的概率
            tem_entity = random.sample(train_entity, 1)[0]  # 在所有实体中随机选取待替换实体
            if pr > 0.5:
                item[0] = tem_entity
            else:
                item[1] = tem_entity
            raw_neg_pairs.append(item)  # 将随机替换后的三元组作为待用的负样本
        self.neg_pairs = raw_neg_pairs
        self.train_pairs = train_pairs
        '''
        with open('./raw_neg_pairs.txt', 'w') as f:
            for item in raw_neg_pairs:
                f.write(item[0] + '\t' + item[1] + '\t' + item[2] + '\n')
        '''
        return raw_neg_pairs
    
    def gen_tuple_pairs(self):
        
        return self.train_pairs

    def gen_true_neg_data(self):  # 由于随机替换了头实体以及尾实体，有可能产生的负样本实际为真，统计出负样本中替换后变为正样本的三元组的假负样本
        neg_pairs = self.neg_pairs  # 本函数返回并存储筛选后的负样本三元组以及假的负样本三元组。
        #false_neg_pairs = []
        true_neg_pairs = []
        for item in neg_pairs:
            if item not in self.train_pairs:
                true_neg_pairs.append(item)
        '''
        with open('./true_neg.txt', 'w') as f:
            for item in true_neg_pairs:
                [e1, e2, r] = item
                f.write(e1 + '\t' + e2 + '\t' + r + '\n')

        with open('./false_neg.txt', 'w') as f:
            for item in false_neg_pairs:
                [e1, e2, r] = item
                f.write(e1 + '\t' + e2 + '\t' + r + '\n')
        '''
        return true_neg_pairs
