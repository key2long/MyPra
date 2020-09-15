import os
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Process, Pool
from raw_graph_utils import parse_fb15k237_data, parse_wn18rr_data, parse_yago310_data


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


class GraphNode:
    def __init__(self,
                 node_name: str):
        self.name = node_name
        self.neighbours_list = []


class ProcessedGraphManager:
    """同RawGraphManager相比,本类要处理的对象格式已经统一,不同数据集之间差别很小,比如它们可以使用完全相同的parse方法.
    另外将Graph分为RawGraph和ProcessedGraph的主要原因为:两种图的构造函数完全不同.
    python对构造函数没有重载机制,因此干脆拆成两个类.此外两个类的方法也有很多差异:
    RawGraph需要save_stand_graph方法；ProcessedGraph不应该有这个方法；
    ProcessedGraph可以考虑merge方法来合并不同的标准图；RawGraph因为格式还没统一,因而不可能做到合并；
    最重要的是ProcessedGraph的图采用的数据结构不一样,同时自带一个很重要的DFS方法.
    总结而言,合成一个类收益很小,麻烦很多,可读性很差.
    """
    def __init__(self,
                 file_path: str,
                 if_add_reverse_relation: bool = False):
        self.file_path = file_path
        self.reverse_relation = if_add_reverse_relation
        if os.path.isfile(self.file_path) is False:
            print(f"找不到文件{self.file_path}.")
        else:
            print(f"读取数据{self.file_path}.")
        self.graph_nodes = {}
        self.fact_list = []
        self.entity_set = set()
        self.relation_set = set()
        self.relation_pos_sample_dict = defaultdict(list)
        self.relation_meta_paths = None

        self.begin_node = ""
        self.end_node = ""
        self.max_depth = 2
        self.min_hit = 5
        self.relation_of_blocked_edge = ""
        self.path = []
        self.all_paths = []
        self.meta_paths = []

        self._init_graph()

    def add_graph_node(self,
                       head_mid: str,
                       relation: str,
                       tail_mid: str):
        if head_mid in self.graph_nodes:
            if tail_mid not in self.graph_nodes:
                self.graph_nodes[tail_mid] = GraphNode(tail_mid)
            if (relation, tail_mid) not in self.graph_nodes[head_mid].neighbours_list:
                self.graph_nodes[head_mid].neighbours_list.append((relation, tail_mid))
        else:
            self.graph_nodes[head_mid] = GraphNode(head_mid)
            self.add_graph_node(head_mid, relation, tail_mid)

    def _report_statistic_info(self):
        output_info = f"图数据{self.file_path}:\n" \
                      f"\t有{len(self.fact_list)}条边；\n" \
                      f"\t有{len(self.entity_set)}个实体；\n" \
                      f"\t有{len(self.relation_set)}种关系."
        print(output_info)

    def _init_graph(self):
        with open(self.file_path, "r") as f:
            all_facts = f.readlines()
            for fact in all_facts:
                fact = fact.strip().split("\t")
                head_mid = fact[0]
                tail_mid = fact[2]
                relation = fact[1]
                self.add_graph_node(head_mid, relation, tail_mid)
                self.fact_list.append([head_mid, tail_mid, relation])
                self.entity_set.add(head_mid)
                self.entity_set.add(tail_mid)
                self.relation_set.add(relation)
                self.relation_pos_sample_dict[relation].append([head_mid, tail_mid])
                if self.reverse_relation is True:
                    relation = "_" + relation
                    self.relation_set.add(relation)
                    self.relation_pos_sample_dict[relation].append([head_mid, tail_mid])
                    self.add_graph_node(tail_mid, relation, head_mid)
                    self.fact_list.append([tail_mid, head_mid, relation])

        f.close()
        self._report_statistic_info()

    def merge(self,
              graph_pt):
        output_info = f"开始合并:"
        print(output_info)
        for fact in tqdm(graph_pt.fact_list):
            temp = (fact[2], fact[1])
            if temp not in self.graph_nodes[fact[0]].neighbours_list:
                content = fact.copy()
                self.fact_list.append(content)
                self.entity_set.add(content[0])
                self.entity_set.add(content[1])
                self.relation_set.add(content[2])
                self.add_graph_node(head_mid=content[0],
                                    relation=content[2],
                                    tail_mid=content[1])
        self._report_statistic_info()

    def entity_set_difference(self, graph_pt):
        return self.entity_set - graph_pt.entity_set

    def relation_set_difference(self, graph_pt):
        return self.relation_set - graph_pt.relation_set

    def set_dfs_search_state(self,
                             begin_node: str,
                             end_node: str,
                             max_depth: int,
                             relation_of_blocked_edge: str):
        self.begin_node = begin_node
        self.end_node = end_node
        self.max_depth = max_depth
        self.relation_of_blocked_edge = relation_of_blocked_edge
        self.path = [("root", self.begin_node)]
        self.all_paths = []
        self.meta_paths = []

    def dfs(self,
            begin_node: str):
        if begin_node == self.end_node:
            tem = []
            if len(self.path) == 2 and self.path[1][0] == self.relation_of_blocked_edge:
                return
            else:
                for item in self.path:
                    tem.append(item)
                self.all_paths.append(tem)
                return
        try:
            if self.graph_nodes[begin_node].neighbours_list is None:
                return
            if len(self.path) == self.max_depth + 1:
                return
            for (_relation, _next_node) in self.graph_nodes[begin_node].neighbours_list:
                if (_relation, _next_node) not in self.path:
                    self.path.append((_relation, _next_node))
                    self.dfs(_next_node)
                    self.path.remove((_relation, _next_node))
        except:
            print(f"存在非法实体{begin_node}")
        return

    def extract_relation_path(self):
        for path in self.all_paths:
            tem = self.relation_of_blocked_edge + "\t"
            for i in path:
                tem = tem + i[0] + "\t"
            self.meta_paths.append(tem)

    def def_for_parallel(self,
                         begin_node: str,
                         end_node: str,
                         path: list,
                         relation_of_blocked_edge: list,
                         all_paths: list,
                         max_depth: int):
        if begin_node == end_node:
            tem = []
            if len(path) == 2:
                for blocked in relation_of_blocked_edge:
                    if path[1][0] == blocked:
                        return
            else:
                for item in path:
                    tem.append(item)
                all_paths.append(tem)
                return
        try:
            if self.graph_nodes[begin_node].neighbours_list is None:
                return
            if len(path) == max_depth + 1:
                return
            for (_relation, _next_node) in self.graph_nodes[begin_node].neighbours_list:
                if (_relation, _next_node) not in path:
                    path.append((_relation, _next_node))
                    self.def_for_parallel(begin_node=_next_node,
                                          end_node=end_node,
                                          path=path,
                                          relation_of_blocked_edge=relation_of_blocked_edge,
                                          all_paths=all_paths,
                                          max_depth=max_depth)
                    path.remove((_relation, _next_node))
        except:
            print(f"存在非法实体{begin_node}")
        return

    def dfs_in_one_process(self,
                           part_of_fact_list: list):
        part_of_all_meta_paths = []
        for fact in tqdm(part_of_fact_list):
            query_relation = fact[2]
            # self.set_dfs_search_state(begin_node=fact[0],
            #                           end_node=fact[2],
            #                           max_depth=self.max_depth,
            #                           relation_of_blocked_edge=query_relation)
            begin_node = fact[0]
            end_node = fact[1]
            max_depth = self.max_depth

            if self.reverse_relation is True:
                if query_relation[0] != "_":
                    relation_of_blocked_edge = [query_relation, "_" + query_relation]
                else:
                    relation_of_blocked_edge = [query_relation, query_relation[1:]]
            else:
                relation_of_blocked_edge = [query_relation]

            path = [("root", begin_node)]
            all_paths = []
            meta_paths = []
            self.def_for_parallel(begin_node=begin_node,
                                  end_node=end_node,
                                  path=path,
                                  relation_of_blocked_edge=relation_of_blocked_edge,
                                  all_paths=all_paths,
                                  max_depth=max_depth)
            # self.extract_relation_path()
            for tmp_path in all_paths:
                tem = relation_of_blocked_edge[0] + "\t"
                for i in tmp_path:
                    tem = tem + i[0] + "\t"
                meta_paths.append(tem)
            part_of_all_meta_paths.extend(meta_paths)
        # print(f"父:{os.getppid()}；子:{os.getpid()} done.")
        return part_of_all_meta_paths

    def parallel_dfs_search(self,
                            fact_list: list,
                            num_of_process: int):
        process_pool = Pool(processes=num_of_process)
        results = []
        len_of_fact_list = len(fact_list)
        batch_size = round(len_of_fact_list / num_of_process)
        end_idx = 0
        for i in range(num_of_process):
            begin_idx = i*batch_size
            end_idx = min((i+1)*batch_size, len_of_fact_list)
            results.append(process_pool.apply_async(self.dfs_in_one_process,
                                                    args=(fact_list[begin_idx:end_idx],)))
        if end_idx < len_of_fact_list:
            results.append(process_pool.apply_async(self.dfs_in_one_process,
                                                    args=(fact_list[end_idx:len_of_fact_list],)))
        process_pool.close()
        process_pool.join()
        all_meta_paths = []
        for part_of_all_meta_paths in results:
            tmp = part_of_all_meta_paths.get()
            all_meta_paths.extend(tmp)
        return all_meta_paths

    def write_down_meta_paths(self,
                              write_file_path: str,
                              max_depth: int = 4):
        self.max_depth = max_depth
        output_info = f"开始搜索{self.file_path}中的meta path:"
        print(output_info)
        num_of_process = 12
        all_meta_paths = self.parallel_dfs_search(fact_list=self.fact_list,
                                                  num_of_process=num_of_process)
        meta_path_counter = Counter(all_meta_paths)
        with open(write_file_path, "w") as f:
            for meta_path in meta_path_counter.keys():
                if meta_path_counter[meta_path] >= self.min_hit:
                    f.write(f"{meta_path_counter[meta_path]}\t" + meta_path + "\n")
        f.close()