import torch.nn as nn
import torch
from mhyao_src.GraphManager import ProcessedGraphManager
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
        self.query_graph_pt = query_graph_pt

    def forward(self, x):
        output = self.linear(x)
        return torch.sigmoid(output)

    def rank_score(self,
                   head_mid: str,
                   relation: str,
                   tail_mid: str):
        entity_pairs_1 = [head_mid, tail_mid, 1]
        feature = GetFeature(tuple_data=self.query_graph_pt.fact_list,
                             entity_pairs=entity_pairs_1,
                             metapath=self.query_graph_pt.relation_meta_paths[relation])
        data_feature = feature.get_probs()
        metapath_len = len(self.query_graph_pt.relation_meta_paths[relation])
        batch_size = 1
        pra_data = PRAData(data_feature_dict=data_feature,
                           metapath_len=metapath_len)
        train_loader = DataLoader(pra_data, batch_size=batch_size)
        results = []
        for i, (path_feature, label) in enumerate(train_loader):
            results.append(self.forward(path_feature))
        return results[0]
