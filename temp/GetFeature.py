# -*- coding: utf-8 -*-
from collections import defaultdict


class Node:
    def __init__(self, NodeName):
        self.name = NodeName
        self.info = defaultdict(list)    # 记录从实体NodeName出发，经关系relation,能到达的实体

    def add(self, relation, subnode_name):
        self.info[relation].append(subnode_name)


class GetFeature:
    def __init__(self, tuple_data, entity_pairs, metapath):  # tuple_data 是训练的所有三元组，entity_pairs是某个关系下的正负样本对，matepath是之前筛选过后的路径
        self.tuple_data = tuple_data
        self.metapath = metapath
        self.entity_pairs = entity_pairs
        self.nodes = {}  # 记录节点的关系信息
        self.data_dict_feature = defaultdict(list)  # 记录了每一个实体对的特征以及标签
        self.set_range()

    def set_range(self):
        for data in self.tuple_data:
            node1_name, node2_name, relation = data
            if node1_name not in self.nodes.keys():
                temp = Node(node1_name)
                self.nodes[node1_name] = temp
            self.nodes[node1_name].add(relation, node2_name)
            if node2_name not in self.nodes.keys():
                temp = Node(node2_name)
                self.nodes[node2_name] = temp

    def _prob(self, begin, end, relation_path):
        prob = 0
        length = len(relation_path)
        if length == 1:
            if end in self.nodes[begin].info[relation_path[0]]:
                prob = 1/len(self.nodes[begin].info[relation_path[0]])
            else:
                prob = 0
            return prob
        elif length == 0:
            return 0
        else:
            if self.nodes[begin].info[relation_path[0]] == []:
                return 0
            else:
                for item in self.nodes[begin].info[relation_path[0]]:
                    prob += (1/len(self.nodes[begin].info[relation_path[0]]))*self._prob(item, end, relation_path[1:])
                return prob

    def get_probs(self):
        for i, data in enumerate(self.entity_pairs):
            [node1, node2, flag] = data
            node1 = node1.replace('thing$', '')
            if node1 not in self.nodes.keys():
                print('发现非法实体%s' % node1)
                continue
            else:
                node2 = node2.replace('thing$', '')
                if flag == 1:
                    self.data_dict_feature[node1, node2].append(1)
                else:
                    self.data_dict_feature[node1, node2].append(0)
                for path in self.metapath:
                    tem_prob = self._prob(node1, node2, path)
                    self.data_dict_feature[node1, node2].append(tem_prob)
            # print('第%d个数据结束\n' % i)
        return self.data_dict_feature
