import operator
import torch
import torch.optim as optim
import torch.nn as nn
import random, os
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from mhyao_src.GraphManager import ProcessedGraphManager
from mhyao_src.PRAModel import LogisticRegression
from temp.GetFeature import GetFeature
from temp.data_utils import PRAData


class GraphExperiments:
    def __init__(self,
                 query_graph_pt: ProcessedGraphManager,
                 predict_graph_pt: ProcessedGraphManager = None,
                 model_pt: LogisticRegression = None,
                 hit_range: int = 10):
        self.query_graph_pt = query_graph_pt
        self.predict_graph_pt = predict_graph_pt
        self.model_pt = model_pt
        self.hit_range = hit_range
        self.hit_percent = None
        self.MR = None
        self.MRR = None

    def tail_predict(self):
        hits = 0
        mr = 0
        mrr = 0
        for triple in self.predict_graph_pt.fact_list:
            entity_rank_dict = {}
            for entity in self.query_graph_pt.entity_set:
                score = self.model_pt.rank_score(head_mid=triple[0],
                                                 relation=triple[1],
                                                 tail_mid=entity)
                entity_rank_dict[(triple[0], triple[1], entity)] = score
            rank_tail_sorted = sorted(entity_rank_dict.items(), key=operator[1], reverse=False)
            for rank, (tail, score) in enumerate(rank_tail_sorted):
                if triple[2] == tail:
                    if rank < self.hit_range:
                        hits += 1
                    mr += rank
                    mrr += 1 / rank
                    break
        self.hit_percent = hits / len(self.predict_graph_pt.fact_list)
        self.MR = mr / len(self.predict_graph_pt.fact_list)
        self.MRR = mrr / len(self.predict_graph_pt.fact_list)
        return self.hit_percent, self.MR, self.MRR


class Validation(GraphExperiments):
    def __init__(self,
                 model_pt: LogisticRegression,
                 query_graph_pt: ProcessedGraphManager,
                 predict_graph_pt: ProcessedGraphManager,
                 hit_range):
        super().__init__(model_pt=model_pt,
                         query_graph_pt=query_graph_pt,
                         predict_graph_pt=predict_graph_pt)
        self.hit_range = hit_range


class Test(GraphExperiments):
    def __init__(self,
                 model_pt: LogisticRegression,
                 query_graph_pt: ProcessedGraphManager,
                 predict_graph_pt: ProcessedGraphManager,
                 hit_range):
        super().__init__(model_pt=model_pt,
                         query_graph_pt=query_graph_pt,
                         predict_graph_pt=predict_graph_pt)
        self.hit_range = hit_range


class PRATrain(GraphExperiments):
    def __init__(self,
                 query_graph: ProcessedGraphManager,
                 neg_pairs_path: str,
                 meta_path_file: str,
                 hold_out_path: Path,
                 hyper_param: float):
        super().__init__(query_graph_pt=query_graph)
        self.meta_path_file = meta_path_file
        self.hold_out_path = hold_out_path
        self.relation_meta_paths = defaultdict(list)
        self.alpha = hyper_param
        self.neg_pairs_path = neg_pairs_path

    def get_relation_paths(self):
        self.relation_meta_paths = defaultdict(list)
        with open(self.meta_path_file, "r") as f:
            datas = f.readlines()
            for data in datas:
                data = data.strip("\n").split("\t")
                query_relation = data[1]
                meta_path = data[3:]
                self.relation_meta_paths[query_relation].append(meta_path)

    def get_neg_pairs(self):
        return ProcessedGraphManager(file_path=self.neg_pairs_path).fact_list

    def train_this_hold_out(self):
        print(f"超参alpha为{self.alpha},开始训练{self.query_graph_pt.file_path}中的三元组:")
        self.get_relation_paths()  # get all the pre-computed meta paths
        neg_pairs = self.get_neg_pairs()  # get all the negative triples
        for relation in self.query_graph_pt.relation_set:
            print(f"预测关系:{relation};")
            relation_pos_pairs = self.query_graph_pt.relation_pos_sample_dict[relation]
            train_pairs_01 = []
            pos_pairs_num = len(relation_pos_pairs)
            neg_pairs_01 = random.sample(neg_pairs, int(pos_pairs_num * (4 + random.random())))
            for item in relation_pos_pairs:
                e1, e2 = item
                train_pairs_01.append([e1, e2, 1])
            for item in neg_pairs_01:
                e1, e2, _ = item
                train_pairs_01.append([e1, e2, 0])
            feature = GetFeature(tuple_data=self.query_graph_pt.fact_list,
                                 entity_pairs=train_pairs_01,
                                 metapath=self.relation_meta_paths[relation])
            data_feature_dict = feature.get_probs()
            metapath_len = len(self.relation_meta_paths[relation])
            input_size = metapath_len
            learning_rate = 0.001
            batch_size = 8
            epoch_num = 2
            pra_data = PRAData(data_feature_dict=data_feature_dict,
                               metapath_len=metapath_len)
            train_loader = DataLoader(pra_data, batch_size=batch_size)
            model = LogisticRegression(input_size=input_size, num_classes=1)
            criterion = nn.BCELoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            for epoch in range(epoch_num):
                for i, (path_feature, label) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = model(path_feature)
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()
                    if (i + 1) % 500 == 0:
                        print('\t\tEpoch: [%d/%d], Step:[%d/%d], Loss: %.4f'
                              % (epoch + 1, epoch_num, i + 1, len(pra_data) // batch_size, loss.data))
            print(f"\t为关系{relation}保存模型.")
            model_save_path = self.hold_out_path / 'model/'
            if os.path.exists(model_save_path) is False:
                os.makedirs(model_save_path)
            model_save_path = model_save_path / (relation.replace("/", '') + f"_{self.alpha}_model.pkl")
            torch.save(model.state_dict(), model_save_path)
