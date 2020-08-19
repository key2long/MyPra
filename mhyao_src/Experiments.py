from .GraphManager import GraphDatasetManager
import operator


class GraphExperiments:
    def __init__(self,
                 model_pt=None,
                 graph_pt=None,
                 hit_range: int = 10):
        self.model_pt = model_pt
        self.graph_pt = graph_pt
        self.hit_range = hit_range
        self.hit_percent = None
        self.MR = None
        self.MRR = None

    def tail_predict(self):
        hits = 0
        mr = 0
        mrr = 0
        for triple in self.graph_pt.fact_list:
            entity_rank_dict = {}
            for entity in self.graph_pt.entity_dict.keys():
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
        self.hit_percent = hits / len(self.graph_pt.fact_list)
        self.MR = mr / len(self.graph_pt.fact_list)
        self.MRR = mrr / len(self.graph_pt.fact_list)
        return self.hit_percent, self.MR, self.MRR


class Validation(GraphExperiments):
    def __init__(self,
                 model_pt,
                 graph_pt,
                 hit_range):
        super().__init__(model_pt=model_pt, graph_pt=graph_pt)
        self.hit_range = hit_range


class Test(GraphExperiments):
    def __init__(self,
                 model_pt,
                 graph_pt,
                 hit_range):
        super().__init__(model_pt=model_pt,
                         graph_pt=graph_pt)
        self.hit_range = hit_range
