# -*- coding: utf-8 -*-
from collections import defaultdict
from collections import Counter

class Node:
    def __init__(self, NodeName):
        self.name = NodeName
        self.adjust_info = []


class Graph:
    def __init__(self):
        self.nodes = {}
        self.path = []
        self.all_path = []
        self.step = 0
        self.max_length = 5
        self.begin_node = ''
        self.end_node = ''
        self.relation_paths = []

    def set_init_state(self, begin_node, end_node, max_length, relation):
        self.begin_node = begin_node
        self.end_node = end_node
        self.max_length = max_length
        self.relation = relation #设置一个flag，去除头尾直接相连的path
        self.path = [('root', self.begin_node)]  # 存储两节点之间的某条路径[('root', self.begin_node), （relation， next_node)....]
        self.all_path = []  # 两个正样本节点之间，可能存在多条路径，dfs的时候我们将所有路径进行存储[('root', self.begin_node), (relation， next_node)....]
        self.relation_paths = []  # 将all_path中的所有relation提取出来

    def add_node(self, node, relation, next_node):
        if node in self.nodes:
            if next_node not in self.nodes:
                self.nodes[next_node] = Node(next_node)
            self.nodes[node].adjust_info.append((relation, next_node))
        else:
            self.nodes[node] = Node(node)
            self.add_node(node, relation, next_node)

    def dfs(self, begin_node):
        if begin_node == self.end_node:
            tem = []
            if len(self.path) == 2 and self.path[1][0] == self.relation: # 判断
                return
            else:
                for item in self.path:
                    tem.append(item)
                self.all_path.append(tem)
                return
        try:
            if self.nodes[begin_node].adjust_info is None:
                return
            if len(self.path) == self.max_length + 1:
                return
            for (_relation, _next_node) in self.nodes[begin_node].adjust_info:
                if (_relation, _next_node) not in self.path:
                    self.path.append((_relation, _next_node))
                    self.dfs(_next_node)
                    self.path.remove((_relation, _next_node))
        except:
            print('存在非法实体%s\n' % begin_node)
        return

    def extract_relation_path(self):
        for path in self.all_path:
            tem = ''
            for i in path:
                tem = tem + i[0] + '\t'
            self.relation_paths.append(tem)
        return


if __name__ == '__main__':
    save_path = 'lx_path_dfs_all.txt'
    paths = []  # 用来记录所有正样本节点之间的路径
    kg = Graph()
    data_path = './Nell995_data/graph.txt'
    train_path = './Nell995_data/train.pairs'
    max_length = 4
    relation = 'worksfor' #每种关系都需要不同的设置

    with open(data_path, 'r') as f:
        datas = f.readlines()
        for data in datas:
            [node, relation, next_node] = data.strip().split('\t')
            kg.add_node(node, relation, next_node)
    with open(train_path, 'r') as f:
        datas = f.readlines()
        for n, data in enumerate(datas):  # 迭代所有的正样本对节点node1，node2，找出两节点之间有可能存在的路径
            [node1, node2] = data.strip()[0:-3].split(',')
            node1 = node1.replace('thing$', '')
            node2 = node2.replace('thing$', '')
            flag = data.strip()[-1]
            if flag == '+':
                begin_node = node1
                end_node = node2
                kg.set_init_state(begin_node, end_node, max_length, relation)  # 每次循环初始化参数
                print('第%d节点对是正样本，下面开始进行搜索' % n)
                kg.dfs(begin_node)
                kg.extract_relation_path()  # 一次dfs找出的某对节点下所有路径
                
                paths.extend(kg.relation_paths)  # 进行extend，[node1,node2之间的所有关系路径,.....]
            else:
                continue

    path_counter = Counter(paths)
    with open(save_path, 'w') as f:
        for path in path_counter.keys():
            f.write(path + '%d' % path_counter[path] + '\n')
