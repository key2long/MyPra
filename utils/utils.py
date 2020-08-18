import os
import random


class GenerateData:

    def __init__(self, train_path, relation_path, train_entity_path):
        self.train_path = train_path  # 训练数据集三元组存储路径
        self.relation_path = relation_path  # 训练数据集relation的存储路径
        self.neg_pairs = []  # 负样本的存储路径

    def pos_data(self):   # 产生对应数据集每一个关系的正样本对并保存在相应关系的文件夹下 ./relation_pairs/relation1/pos.txt
        path = './relation_pairs'
        with open(self.relation_path, 'r') as f1:
            datas = f1.readlines()
            for data in datas:
                relation = data.strip().split('\t')[0]
                os.mkdir(path + '/' + relation)  # 产生相应的文件夹
                pos = []
                with open(self.train_path, 'r') as f2:  # 读取数据集的三元组文件
                    datas2 = f2.readlines()
                    for data2 in datas2:
                        [e1, e2, r] = data2.strip().split('\t')
                        r = r.replace('concept:', '')
                        if relation == r:
                            pos.append((e1, e2))
                        else:
                            continue
                    with open(path + '/' + relation + '/' + 'pos', 'w') as f3:  # 在相应关系文件夹下生成对应该关系的正样本
                        for (e1, e2) in pos:
                            f3.write('thing$' + e1 + ',' + e2 + ': +' + '\n')  # thing$concept:journalist:cynthia_mcfadden,concept:city:abc: +

    def sample_neg_data(self):
        train_entity = []
        train_pairs = []
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
        with open('./raw_neg_pairs.txt', 'w') as f:
            for item in raw_neg_pairs:
                f.write(item[0] + '\t' + item[1] + '\t' + item[2] + '\n')

    def true_neg_data(self):  # 由于随机替换了头实体以及尾实体，有可能产生的负样本实际为真，统计出负样本中替换后变为正样本的三元组的假负样本
        neg_pairs = self.neg_pairs  # 本函数返回并存储筛选后的负样本三元组以及假的负样本三元组。
        false_neg = []
        true_neg = []
        train_pairs = []
        with open(self.train_path, 'r') as f:
            datas = f.readlines()
            for data in datas:
                [e1, e2, r] = data.strip().split('\t')
                train_pairs.append([e1, e2, r])
        for item in neg_pairs:
            for item2 in train_pairs:
                if item == item2:
                    false_neg.append(item)
                    break
                else:
                    true_neg.append(item)
        with open('./true_neg.txt', 'w') as f:
            for item in true_neg:
                [e1, e2, r] = item
                f.write(e1 + '\t' + e2 + '\t' + r + '\n')

        with open('./true_neg.txt', 'w') as f:
            for item in false_neg:
                [e1, e2, r] = item
                f.write(e1 + '\t' + e2 + '\t' + r + '\n')