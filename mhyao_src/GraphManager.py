from pathlib import Path
from .raw_graph_utils import parse_fb15k237_data, parse_wn18rr_data, parse_yago310_data


class RawGraphManager:

    def __init__(self,
                 root_dir: str = None,
                 train_val_test: str = None):
        """
        :param root_dir: 该路径下应该有raw以及processed文件夹.
                        其中raw存放未经处理,拆分的原始三元组文件；
                        processed存放处理过的,且拆分过的三元组文件.
        :param train_val_test: raw文件夹中三元组文件的后缀,表示是否为train/valid/test文件.
        """
        self.root_dir = Path(root_dir)
        self.train_val_test = train_val_test
        self.raw_dir = self.root_dir / "raw"
        self.processed_dir = self.root_dir / "processed"
        self.fact_list = None

        if not self.raw_dir.exists():
            print(f"找不到raw_dir:{self.raw_dir}")

        if not self.processed_dir.exists():
            print(f"找不到processed_dir:{self.processed_dir}")

        self._process()

    def _process(self):
        raise NotImplementedError

    def save_stand_graph(self):
        """
        将原始三元组保存为统一格式的文件,文件名为:self.name + '_' + self.train_val_test + .graph
        :return: None
        """
        stand_graph_path = self.processed_dir / f"{self.name + '_' + self.train_val_test}.graph"
        with open(stand_graph_path, "w") as f:
            for fact in self.fact_list:
                name_relation = fact[1]
                head_mid = fact[0]
                tail_mid = fact[2]
                content_to_write = head_mid + "\t"
                content_to_write += name_relation + "\t"
                content_to_write += tail_mid + "\n"
                f.write(content_to_write)
        f.close()


class WN18RRRawGraph(RawGraphManager):
    name = "WN18RR"

    def _process(self):
        self.fact_list = parse_wn18rr_data(self.name,
                                           self.train_val_test,
                                           self.raw_dir)


class YAGO310RawGraph(RawGraphManager):
    name = "YAGO3_10"

    def _process(self):
        self.fact_list = parse_yago310_data(self.name,
                                            self.train_val_test,
                                            self.raw_dir)


class FB15k237RawGraph(RawGraphManager):
    name = "fb15k237"

    def _process(self):
        self.fact_list = parse_fb15k237_data(self.name,
                                             self.train_val_test,
                                             self.raw_dir)
