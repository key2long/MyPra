# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
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
                 file_path: str,
                 entity_embedding_path: str = "entity2vec.bin",
                 entity_embedding_id_path: str = "entity2id.txt",
                 relation_embedding_path: str = "relation2vec.bin",
                 relation_embedding_id_path: str = "relation2id.txt"):
        self.file_path = file_path
        self.entity2vec_path = entity_embedding_path
        self.entity2id_path = entity_embedding_id_path
        self.parsed_entity2vec = False
        self.relation2vec_path = relation_embedding_path
        self.relation2id_path = relation_embedding_id_path
        self.parsed_relation2vec = False
        self._check_file()
        self.triple_list = []
        self.relation_pos_samp_dict = defaultdict(list)
        self.adjacency_dict = defaultdict(list)
        self.relation_adj_dict = defaultdict()
        self._parse_triple()

    def load_entity2vec(self):
        if self.parsed_entity2vec is not True:
            self.parsed_entity2vec = True
            self.entity2vec = torch.from_numpy(np.memmap(self.entity2vec_path,
                                                         dtype='float32',
                                                         mode='r')).reshape([-1, 50])
            self.entity2id_dict = {}
            with open(self.entity2id_path, "r") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip().split('\t')
                    entity_mid = line[0]
                    entity_id = int(line[1])
                    self.entity2id_dict[entity_mid] = entity_id

    def load_relation2vec(self):
        if self.parsed_relation2vec is not True:
            self.parsed_relation2vec = True
            self.relation2vec = torch.from_numpy(np.memmap(self.relation2vec_path,
                                                           dtype='float32',
                                                           mode='r')).reshape([-1, 50])
            self.relation2id_dict = {}
            with open(self.relation2id_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split('\t')
                    relation_mid = line[0]
                    relation_id = int(line[1])
                    self.relation2id_dict[relation_mid] = relation_id

    def get_entity2vec(self, mid: str):
        self.load_entity2vec()
        if mid in self.entity2id_dict:
            return self.entity2vec[self.entity2id_dict[mid], :]
        else:
            return None

    def get_relation2vce(self, relation: str):
        self.load_relation2vec()
        if relation in self.relation2id_dict:
            return self.relation2vec[self.relation2id_dict[relation], :]
        else:
            return None

    def get_target_tail_vec(self, head_mid: str,
                            relation: str):
        head_mid = ".".join(head_mid.split('/')[1:])
        head_vec = self.get_entity2vec(mid=head_mid)
        relation = ".".join(relation.split('/')[1:])
        relation_vec = self.get_entity2vec(mid=relation)
        if head_vec is not None and relation_vec is not None:
            target_tail_vec = head_vec + relation_vec
            return target_tail_vec
        else:
            return None

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
