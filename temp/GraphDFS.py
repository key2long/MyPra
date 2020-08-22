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
        self.relation = relation
        self.max_length = max_length
        self.path = [('root', self.begin_node)]  # 存储两节点之间的某条路径[('root', self.begin_node), （relation， next_node)....]
        self.all_path = []  # 两个正样本节点之间，可能存在多条路径，dfs的时候我们将所有路径进行存储[('root', self.begin_node), （relation， next_node)....]
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
            if len(self.path) == 2 and self.path[1][0] == self.relation:
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
