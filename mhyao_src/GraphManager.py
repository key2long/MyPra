import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from .raw_graph_utils import parse_fb15k237_data, parse_wn18rr_data, parse_yago310_data
from .raw_graph_utils import py2nxGraphStyle, py2nxGraph
from .Param import Param


class GraphDatasetManager:
    class GDMHyperParam(Param):
        def __init__(self, config_path):
            """
            :param config_path: 非强制参数,但是不为空时,会检查是否存在.
            """
            if config_path is not None and os.path.exists(config_path) is False:
                raise Exception(f"Can't find {config_path}! Please create such yml file.")
            super().__init__(config_path=config_path)

    def __init__(self,
                 config_path: str = None,
                 data_dir: str = None,
                 train_val_test: str = None,
                 model_asses_k: int = None,
                 split_mode=StratifiedKFold,
                 holdout_test_size: float = None):
        graph_config_pt = self.GDMHyperParam(config_path=config_path)
        self.root_dir = Path(graph_config_pt.ifInConfig(param_dict=dict(data_dir=data_dir),
                                                        param_type="str"))
        self.train_val_test = graph_config_pt.ifInConfig(param_dict=dict(train_val_test=train_val_test),
                                                         param_type="str")
        self.model_asses_k = graph_config_pt.ifInConfig(param_dict=dict(model_asses_k=model_asses_k),
                                                        param_type="int")
        self.holdout_test_size = graph_config_pt.ifInConfig(param_dict=dict(holdout_test_size=holdout_test_size),
                                                            param_type="float")
        self.raw_dir = self.root_dir / "raw"
        if not self.raw_dir.exists():
            print("找不到raw_dir")
        self.processed_dir = self.root_dir / "processed"
        if not self.processed_dir.exists():
            print("找不到processed_dir")
        # 检测图数据是否已经以pt的形式保存了下来,没有时,则需要进一步处理
        if not (self.processed_dir / f"{self.name}.graph").exists():
            self.nx_graph = self._process()
        else:
            self.nx_graph = None
            self.entity_dict = None
            self.fact_list = None

        # 经过上面_process的处理,已经有了相应的graph文件.将该文件load到dataset里
        # self.geo_sty_graphs = GraphDataset(torch.load(self.processed_dir / f"{self.name}.pt"))

    def _process(self):
        raise NotImplementedError

    def getTrainQueries(self):
        raise NotImplementedError

    def saveStandGraph(self):
        stand_graph_path = self.processed_dir / f"{self.name + '_' + self.train_val_test}.graph"
        with open(stand_graph_path, "w") as f:
            for (head_mid, tail_mid, relation) in self.nx_graph.edges.data():
                name_relation = relation["relation"]
                content_to_write = head_mid + "\t"
                content_to_write += name_relation + "\t"
                content_to_write += tail_mid + "\n"
                f.write(content_to_write)
        f.close()


class WN18RRManager(GraphDatasetManager):
    name = "WN18RR"

    def _process(self):
        fact_list =parse_wn18rr_data(self.name,
                                     self.train_val_test,
                                     self.raw_dir)
        nx_graph = py2nxGraph(fact_list=fact_list)
        return nx_graph


class YAGO310Manager(GraphDatasetManager):
    name = "YAGO3_10"

    def _process(self):
        fact_list = parse_yago310_data(self.name,
                                       self.train_val_test,
                                       self.raw_dir)
        nx_graph = py2nxGraph(fact_list=fact_list)
        return nx_graph


class FB15k237Manager(GraphDatasetManager):
    name = "fb15k237"

    def _process(self):
        mid_nid_dict, nid_mid_dict, \
        relation_rid_dict, rid_relation_dict, \
        fact_list = parse_fb15k237_data(self.name,
                                        self.train_val_test,
                                        self.raw_dir)
        self.entity_dict = mid_nid_dict
        self.fact_list = fact_list
        # 保存nid_mid_dict和rid_relation_dict
        nid_mid_path = self.processed_dir / f"{self.name+'_'+self.train_val_test}.mid"
        rid_relation_path = self.processed_dir / f"{self.name+'_'+self.train_val_test}.relation"
        try:
            with open(nid_mid_path, "w") as nidf, open(rid_relation_path, "w") as ridf:
                for nid in range(len(nid_mid_dict)):
                    nidf.write(nid_mid_dict[nid] + "\n")
                for rid in range(len(rid_relation_dict)):
                    tmp_dict = rid_relation_dict[rid]  # {"relation": str(relation), "count": int(count)}
                    ridf.write(tmp_dict["relation"] + "\t" + str(tmp_dict["count"]) + "\n")
            nidf.close()
            ridf.close()
        except:
            print(f"上述try模块的某行出现了错误")

        # 将py_style的知识图谱转化为networkx_style
        nx_graph = py2nxGraphStyle(mid_nid_dict=mid_nid_dict,
                                   fact_list=fact_list)
        return nx_graph

        # geo_graph = self._to_Geo_sty_graph(nx_graph)

    def _to_Geo_sty_graph(self, nx_graph):
        a = 1
        return a

    def getTrainQueries(self):
        """
        从self.nx_graph生成queries_pt.
        数据格式为:
        queries_pt = {"relation": dict<head_mid: Query>}
        其中Query = {"good": [tail_mid], "bad": []}
        :return: queries_pt
        """
        queries_pt = {}
        for (head_mid, tail_mid, relation) in self.nx_graph.edges.data():
            name_relation = relation["relation"]
            if name_relation not in queries_pt:
                queries_pt[name_relation] = {head_mid: [tail_mid]}
            else:
                if head_mid not in queries_pt[name_relation]:
                    queries_pt[name_relation][head_mid] = [tail_mid]
                else:
                    queries_pt[name_relation][head_mid].append(tail_mid)
        return queries_pt
