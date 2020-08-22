# -*- coding: utf-8 -*-
from data_utils import Count, path_select, GenerateData, PRAData, load_tuple
from GraphDFS import *
from GetFeature import *
import torch.optim as optim
import torch.nn as nn
import random
import argparse
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from TrainData import LogisticRegression
import pdb


parser = argparse.ArgumentParser(description='please select the train mode: raw or filter')
parser.add_argument('train_mode', type=str, help='raw or filter')
args = parser.parse_args()

train_path = './NELL/train.txt'  # 数据集的训练三元组路径
raw_neg_pairs_path = ''  # 原始负样本三元组的路径
true_neg_pairs_path = '' #真实的负样本的三元组路径
train_rules_path = ''  #一次划分下的训练集规则路径
train_valid_rules_path = ''  #一次划分下train和valid一起的规则路径
alpha = 0.2  # 筛选路径的惩罚项系数
#raw_neg_pairs = gen_data.sample_neg_data()  # 整个数据集的负样本三元组[(entity1, entity2, r1),......
#true_neg_pairs = gen_data.gen_true_neg_data()  # 整个数据集真正的负样本对[(entity1, entity2, r1), ....]
raw_neg_pairs = []
true_neg_pairs = []
train_rules_len = 0  #训练集的规则长度
train_valid_rules_len = 0  #训练集加验证集的规则长度

'''
事先处理好负样本的文件，包含raw和ture两个
'''
raw_neg_pairs = load_tuple(raw_neg_pairs_path)
true_neg_pairs = load_tuple(true_neg_pairs_path)

relation_paths = defaultdict[list]  # 存储着每一个关系下面的metapath{relation1:[[r1, r2, r3],[],......], ....}


'''
实现处理好每个数据集中的metapth按关系存储
'''
with open(train_rules_path, 'r') as f:
    datas = f.readlines()
    train_rules_len = datas[0].split('\t') #训练集的规则长度具体的获取方法见数据最终的存储格式  这是加了限制之后的长度
    for data in datas[1:]:
        query_relation = data  #具体解析方式看数据存储格式
        paths = []  # 具体解析方式看数据存储格式
        relation_paths[query_relation].append(paths)  # 最后的格式应该是类似于{reltaion1 : [['root\tconcept:atdate\tconcept:subpartof\tconcept:atdate\t'],       ['root\tconcept:atdate\tconcept:proxyfor\tconcept:atdate\t'], ['root\tconcept:atdate\tconcept:proxyfor\tconcept:istallerthan\tconcept:atdate\t'], ['root\tconcept:atdate\tconcept:proxyfor\tconcept:statelocatedingeopoliticallocation\tconcept:atdate\t']], relation2:[[]].....}
        
with open(train_valid_rules_path, 'r') as f:
    datas = f.readlines()
    train_valid_rules_len = datas[0].split('\t') #训练集加验证集的规则长度具体的获取方法见数据最终的存储格式 这是加了限制之后的长度
    
    
#entity_dict, relation_dict = Count(train_path)
gen_data = GenerateData(train_path)
pos_pairs_dict = gen_data.gen_pos_data()  # 以关系为字典的正样本对{relation1:[(entity1, entity2)...()...],r2:[].....}
tuple_data = gen_data.gen_tuple_pairs() # 存储着所有的待训练三元组

'''
kg = Graph()
for data in tuple_data:
    [node, next_node, relation] = data   # ['concept:chemical:companies', 'concept:dateliteral:n2005', 'concept:atdate']
    kg.add_node(node, relation, next_node)

for relation in pos_pairs_dict.keys():  # 一个大的循环遍历每一个关系，生成每个关系下的正负样本
    print('strat\t' + relation + '\t' + 'iter\n')
    relation_pos_pairs = pos_pairs_dict[relation]
    neg_pairs_01 = []  # 每个关系下的采样后的负样本对 [[e1, e2, 0],....]
    train_pairs_01 = []  # 每个关系下的正样本对和负样本对的总和 [[e1, e2, 1],....]
    pos_pairs_num = len(relation_pos_pairs)
    if args.train_mode == 'raw':
        neg_pairs_01 = random.sample(raw_neg_pairs, int(pos_pairs_num*(4 + random.random())))
    if args.train_mode == 'filter':
        neg_pairs_01 = random.sample(true_neg_pairs, int(pos_pairs_num*(4 + random.random())))
    for item in relation_pos_pairs:
        e1, e2 = item
        train_pairs_01.append([e1, e2, 1])
    for item in neg_pairs_01:
        e1, e2, _ = item
        train_pairs_01.append([e1, e2, 0])
    threshold = len(train_pairs_01) * alpha  # 筛选路径的依据
    paths = []  # 用来记录所有正样本节点之间的路径
    max_length = 4
    pdb.set_trace()
    for n, data in enumerate(train_pairs_01):
        node1, node2, flag = data
        if flag == 1:
            begin_node = node1
            end_node = node2
            kg.set_init_state(begin_node, end_node, max_length, relation)  # 每次循环初始化参数
            print('在关系%s下，一共有正负样本对%d，第%d节点对是正样本，下面开始进行搜索' %(relation, len(train_pairs_01), n))
            kg.dfs(begin_node)
            kg.extract_relation_path()  # 一次dfs找出的某对节点下所有路径
            paths.extend(kg.relation_paths)  # 进行extend，[node1,node2之间的所有关系路径,.....]
        else:
            continue 
'''

for relation in pos_pairs_dict.keys():
    #paths = relation_paths[relation]  # 是某个关系下的所有paths集合[[r1, r2, ...],...[]]
    #path_counter = Counter(paths)
    relation_pos_pairs = pos_pairs_dict[relation]  #某个关系的正样本对[(entity1, entity2)...()...]
    train_pairs_01 = []
    neg_pairs_01 = []
    pos_pairs_num = len(relation_pos_pairs)
    if args.train_mode == 'raw':
        neg_pairs_01 = random.sample(raw_neg_pairs, int(pos_pairs_num*(4 + random.random())))
    if args.train_mode == 'filter':
        neg_pairs_01 = random.sample(true_neg_pairs, int(pos_pairs_num*(4 + random.random())))
    for item in relation_pos_pairs:
        e1, e2 = item
        train_pairs_01.append([e1, e2, 1])
    for item in neg_pairs_01:
        e1, e2, _ = item
        train_pairs_01.append([e1, e2, 0])
    '''    
    threshold = len(train_pairs_01) * alpha  # 筛选路径的依据
    path_threshold_list = path_select(path_counter, threshold)  # 选取筛选过后的路径列表 [['root\tconcept:atdate\tconcept:subpartof\tconcept:atdate\t'],       ['root\tconcept:atdate\tconcept:proxyfor\tconcept:atdate\t'], ['root\tconcept:atdate\tconcept:proxyfor\tconcept:istallerthan\tconcept:atdate\t'], ['root\tconcept:atdate\tconcept:proxyfor\tconcept:statelocatedingeopoliticallocation\tconcept:atdate\t'],
    '''
    feature = GetFeature(tuple_data, train_pairs_01, relation_paths[relation])  # 这个地方在getfeature函数的时候可能还要根据路径实际格式改一点函数解析方法
    data_feature_dict = feature.get_probs()
    metapath_len = len(relation_paths[relation])
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
    torch.save(model.state_dict(), './model/' + relation + '_model.pkl')