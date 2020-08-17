from collections import defaultdict


class Node:
    def __init__(self, NodeName):
        self.name = NodeName
        self.info = defaultdict(list)    # 记录从实体NodeName出发，经关系relation,能到达的实体

    def add(self, relation, subnode_name):
        self.info[relation].append(subnode_name)


class GetFeature:
    def __init__(self):
        self.data_file = "./Nell995_data/Graph.txt"
        self.path_file = "paths_threshold.txt"
        self.train_file = "./Nell995_data/train.pairs"
        self.nodes = {}  # 记录节点的关系信息
        self.train_data = defaultdict(list)
        self.set_range()

    def set_range(self):
        with open(self.data_file, "r") as f:
            datas = f.readlines()
            for data in datas:
                node1_name, relation, node2_name = data.strip().split("\t")
                if node1_name not in self.nodes.keys():
                    temp = Node(node1_name)
                    self.nodes[node1_name] = temp
                self.nodes[node1_name].add(relation, node2_name)

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
        relation_paths = []
        with open(self.path_file, 'r') as f:
            paths = f.readlines()
            for path in paths:
                relation_paths.append(path.strip().split('\t')[1:])
        with open(self.train_file, 'r') as f:
            datas = f.readlines()
            for i, data in enumerate(datas):
                [node1, node2] = data.strip()[0:-3].split(',')
                node1 = node1.replace('thing$', '')
                if node1 not in self.nodes.keys():
                    print('发现非法实体%s'%node1)
                    continue
                else:
                    node2 = node2.replace('thing$', '')
                    flag = data.strip()[-1]
                    if flag == '+':
                        self.train_data[node1, node2].append(0)
                    else:
                        self.train_data[node1, node2].append(1)
                    for path in relation_paths:
                        tem_prob = self._prob(node1, node2, path)
                        self.train_data[node1, node2].append(tem_prob)
                print('第%d个数据结束\n'%i)
        with open('train_data.txt', 'w') as f:
            for key in self.train_data:
                f.write(str(key) + '\t' + str(self.train_data[key]) + '\n')
        return


if __name__ == '__main__':
    feature = GetFeature()
    feature.get_probs()