# -*- coding: utf-8 -*-
import os
from collections import defaultdict


class ProcessedGraph:
    """Load graph data from file.

    Args:
        file_path: str. Path to ".graph" file, where one triple per line.

    Attributes:
        file_path: str. Path to ".graph" file, where one triple per line.
        triple_list: list. A list of all the triples in file_path,
                            and each triple looks like [head_mid, relation, tail_mid].
                            Example: [[h_1, r_1, t_1], ..., [h_i, r_i, t_i], ..., ].
        relation_pos_samp_dict: dict. A dict storing all the triples in triple_list,
                            where each key is a relation, and its value is a list of triples.
                            Example: {key=r_i: value=[h_i, t_i]}.
        adjacency_dict: dict. A dict storing all the edges connecting to each entity in this graph.
                            where each key is an entity, and its value is a list of edges.
                            Example: {key=h_j: value=[[r_j, t_j], ..., [r_k, t_k], ..., ]}.
        relation_adj_dict: dict. A dict storing all the head entity through one relation connect to all
                            the other tail entity, where each key is an entity name(string), and the value
                            is a relation dict, which key is relation_i connect to another entity_name.
                            Example: {key=h_i: {relation_j: [e_i,..., e_k], ...}, key=h_i+1: {relation_j: [e_i,..., e_k], ...},...}

    Methods:
        _check_file: Check if the file_path exists and print help information.
        _parse_triple: Parse the original triple, and fill in "triple_list", "relation_pos_samp_dict", and "adjacency_dict".
        _report_statistic_info: Print the statistical information of this graph data.
    """
    def __init__(self,
                 file_path: str):
        self.file_path = file_path
        self._check_file()
        self.triple_list = []
        self.relation_pos_samp_dict = defaultdict(list)
        self.adjacency_dict = defaultdict(list)
        self.relation_adj_dict = defaultdict()
        self._parse_triple()

    def _check_file(self):
        """Check if the file_path exists and print help information.

        :return: None
        """
        if os.path.isfile(self.file_path) is False:
            print(f"找不到{self.file_path}.")
        else:
            print(f"读取{self.file_path}")

    def _report_statistic_info(self):
        """Print the statistical information of this graph data.

        :return: None
        """
        output_info = f"图数据{self.file_path}:\n" \
                      f"\t有{len(self.triple_list)}条边；\n" \
                      f"\t有{len(self.adjacency_dict.keys())}个实体；\n" \
                      f"\t有{len(self.relation_pos_samp_dict.keys())}种关系."
        print(output_info)

    def _parse_triple(self):
        """Parse the original triple, and fill in this instance's attributes.

        The attributes includes "triple_list", "relation_pos_samp_dict", "adjacency_dict", and "relation_adj_dict".

        :return: None
        """
        with open(self.file_path, "r") as f:
            for triple in f.readlines():
                triple = triple.strip().split("\t")
                head_mid, relation, tail_mid = triple[0], triple[1], triple[2]
                # fill in attributes.
                self.triple_list.append([head_mid, relation, tail_mid])
                self.relation_pos_samp_dict[relation].append([head_mid, tail_mid])
                self.adjacency_dict[head_mid].append([relation, tail_mid])
                # Using "extend" instead of "append",
                # because "append" will add an empty list to "tail_mid"'s edge list.
                self.adjacency_dict[tail_mid].extend([])
                self._fill_r_adj_dict(head_mid, relation, tail_mid)

        f.close()
        self._report_statistic_info()

    def _fill_r_adj_dict(self,
                         head_mid: str,
                         relation: str,
                         tail_mid: str):
        """self.relation_adj_dict[head_mid][relation].append(tail_mid)

        :param head_mid: str.
        :param relation: str.
        :param tail_mid: str.
        :return: None
        """
        if head_mid not in self.relation_adj_dict:
            self.relation_adj_dict[head_mid] = defaultdict(list)
        if tail_mid not in self.relation_adj_dict:
            self.relation_adj_dict[tail_mid] = defaultdict(list)
        if relation not in self.relation_adj_dict[head_mid]:
            self.relation_adj_dict[head_mid][relation] = []
        self.relation_adj_dict[head_mid][relation].append(tail_mid)
