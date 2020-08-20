# -*- coding: utf-8 -*-
from data_utils import Count, path_select, GenerateData, PRAData
from GraphDFS import *
from GetFeature import *
import torch.optim as optim
import torch.nn as nn
import random
import argparse
from collections import Counter
from torch.utils.data import DataLoader
from TrainData import LogisticRegression
import pdb


parser = argparse.ArgumentParser(description='please select the train mode: raw or filter')
parser.add_argument('train_mode', type=str, help='raw or filter')
args = parser.parse_args()

train_path = './NELL/train.txt'  # 数据集的训练三元组路径
alpha = 0.2  # 筛选路径的惩罚项系数
entity_dict, relation_dict = Count(train_path)
gen_data = GenerateData(train_path)
pos_pairs_dict = gen_data.gen_pos_data()  # 以关系为字典的正样本对{relation1:[(entity1, entity2)...()...],r2:[].....}
raw_neg_pairs = gen_data.sample_neg_data()  # 整个数据集的负样本三元组[(entity1, entity2, r1),......
pdb.set_trace()
true_neg_pairs = gen_data.gen_true_neg_data()  # 整个数据集真正的负样本对[(entity1, entity2, r1), ....]
pdb.set_trace()
tuple_data = gen_data.gen_tuple_pairs() # 存储着所有的待训练三元组

kg = Graph()
for data in tuple_data:
    [node, next_node, relation] = data   # ['concept:chemical:companies', 'concept:dateliteral:n2005', 'concept:atdate']
    kg.add_node(node, relation, next_node)
pdb.set_trace()
for relation in pos_pairs_dict.keys():  # 一个大的循环遍历每一个关系，生成每个关系下的正负样本
    print('strat' + relation)
    relation_pos_pairs = pos_pairs_dict[relation]
    neg_pairs = []  # 每个关系下的采样后的负样本对 [[e1, e2, 0],....]
    train_pairs = []  # 每个关系下的正样本对和负样本对的总和 [[e1, e2, 1],....]
    pos_pairs_num = len(relation_pos_pairs)
    if args == 'raw':
        neg_pairs = random.sample(raw_neg_pairs, int(pos_pairs_num*(4 + random.random())))
    if args == 'filter':
        neg_pairs = random.sample(true_neg_pairs, int(pos_pairs_num*(4 + random.random())))
    for item in relation_pos_pairs:
        e1, e2, _ = item
        train_pairs.append([e1, e2, 1])
    for item in neg_pairs:
        e1, e2, _ = item
        train_pairs.append([e1, e2, 0])
    threshold = len(train_pairs) * alpha  # 筛选路径的依据
    paths = []  # 用来记录所有正样本节点之间的路径
    max_length = 4
    for n, data in enumerate(train_pairs):
        node1, node2, flag = data
        if flag == 1:
            begin_node = node1
            end_node = node2
            kg.set_init_state(begin_node, end_node, max_length)  # 每次循环初始化参数
            print('第%d节点对是正样本，下面开始进行搜索' % n)
            kg.dfs(begin_node)
            kg.extract_relation_path()  # 一次dfs找出的某对节点下所有路径
            paths.extend(kg.relation_paths)  # 进行extend，[node1,node2之间的所有关系路径,.....]
        else:
            continue

    path_counter = Counter(paths)
    path_threshold_list = path_select(path_counter, threshold)  # 选取筛选过后的路径列表

    feature = GetFeature(tuple_data, train_pairs, path_threshold_list)
    data_feature_dict = feature.get_probs()
    metapath_len = len(path_threshold_list)
    learning_rate = 0.001
    batch_size = 8
    #test_path = ''
    #validation_path = ''
    input_size = metapath_len
    #num_classes = 2
    epoch_num = 200
    pra_data = PRAData(data_feature_dict, metapath_len)
    train_loader = DataLoader(pra_data, batch_size=batch_size)
    model = LogisticRegression(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        print('epoch is %d:' % epoch)
        for i, (path_feature, label) in enumerate(train_loader):
            # print(path_feature.dtype, label.dtype)
            optimizer.zero_grad()
            outputs = model(path_feature)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('under the' + relation + 'Epoch: [%d/%d], Step:[%d/%d], Loss: %.4f'
                      % (epoch+1, epoch_num, i+1, len(pra_data)//batch_size, loss.data))
