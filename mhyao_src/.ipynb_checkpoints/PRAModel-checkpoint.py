import sys
sys.path.append('../')
import torch.nn as nn
import torch
from GraphManager import ProcessedGraphManager
from temp.GetFeature import GetFeature
from temp.data_utils import PRAData
from torch.utils.data import DataLoader


class LogisticRegression(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 query_graph_pt: ProcessedGraphManager = None):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        output = self.linear(x)
        return torch.sigmoid(output)


class ModelWrapper:
    def __init__(self):
        self.dump = 1


class PRAModelWrapper(ModelWrapper):
    def __init__(self, query_graph_pt: ProcessedGraphManager = None):
        super().__init__()
        self.relation_torch_model_dict = {}
        self.query_graph_pt = query_graph_pt

    def rank_score(self,
                   head_mid: str,
                   relation: str,
                   tail_mid: str):
        entity_pairs_1 = [head_mid, tail_mid, 1]
        feature = GetFeature(tuple_data=self.query_graph_pt.fact_list,
                             entity_pairs=entity_pairs_1,
                             metapath=self.query_graph_pt.relation_meta_paths[relation])
        data_feature = feature.get_probs()
        py_feature = data_feature[(head_mid, tail_mid)]
        torch_feature = torch.tensor(data=py_feature,
                                     dtype=torch.float32)
        result = self.relation_torch_model_dict[relation].forward(torch_feature)
        return result


class MyModel(nn.Module):
    def rank_score(self,
                   head_mid: str,
                   relation: str,
                   tail_mid: str):
        pass
