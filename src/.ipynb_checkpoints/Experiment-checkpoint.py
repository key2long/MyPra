# -*- coding: utf-8 -*-
import argparse
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from ProcessedGraph import ProcessedGraph
from model import MhYaoPRA
from utils import handle_error, get_train_probs, MhYaoPRAData, get_predict_probs
import sys
sys.path.append("../FeatureBuffer")
from FeatureBuffer import ValidFeatureBuffer
import time 


class GraphExperiments:
    """The parent class of PRATrain, Valid, and Test class.

    Both subclasses will inherit _load_meta_path methods, which will open the meta path file,
     and store all meta path in a dict. The key of this dict is relation, and the value of
     this dict is the list of all possible meta path of this relation.

     Attributes:
         query_graph_pt: ProcessedGraph. The pointer to a ProcessedGraph instance, such that this class can have access
                                        to this instance's attributes.
                                        This graph is based on train graph or train+valid graph.
         hold_out_fold_path: Path. The path to hold out folder. A Path instance.
         meta_path_file: str. The path to meta_path_file, which will be loaded by _load_meta_path method in __init__().
         predict_graph_pt: ProcessedGraph. It's similar to query_graph_pt except that only Valid subclass will have this
                                        attributes, and it is based on the valid graph.
         model_pt: MhYaoPRA. The pointer to this model. Only specified in its subclass.
         args_pt: argparse. The pointer to the args instance from main() function.
         relation_meta_path_dict: defaultdict(list). It stores all relation's meta path.
                                                    Example: {key=relation, value=[p_1, p_2, ...]}
                                                                                   p_1:[r_1, r_2, r_3,...].
         relation_id_list: list. A list of relation. The order is based on each relation's meta path number.
                                This list is used to index each relation.

     Methods:
        _load_meta_path: Load the meta_path file in meta_path_file, and return a dict to self.relation_meta_path_dict.
        _sort_and_index_relation: Based on each relation's meta path number, sort and index each relation.

    """
    def __init__(self,
                 query_graph_pt: ProcessedGraph,
                 hold_out_fold_path: Path = None,
                 meta_path_file: str = None,
                 args_pt: argparse = None,
                 predict_graph_pt: ProcessedGraph = None,
                 valid_feature_buffer_pt: ValidFeatureBuffer=None,
                 model_pt: MhYaoPRA = None):
        self.query_graph_pt = query_graph_pt
        self.hold_out_fold_path = hold_out_fold_path
        self.meta_path_file = meta_path_file
        self.predict_graph_pt = predict_graph_pt
        self.valid_feature_buffer_pt = valid_feature_buffer_pt
        self.model_pt = model_pt
        self.args_pt = args_pt

        self.relation_meta_path_dict = self._load_meta_path()
        self.relation_id_list = self._sort_and_index_relation()
        self.entity_id_list = self._sort_and_index_entity()

    def _load_meta_path(self):
        """Load meta path in self.meta_path_file.

        The default format of meta path in self.meta_path_file is:
        "65 234 relation    #SEP#   r_1 r_2 r_3".
        There are possibly several meta_path_files which contains meta path with different length.

        :return: relation_meta_path_dict.
        """
        relation_meta_path_dict = defaultdict(list)
        max_meta_path_len = 4
        for length in range(2, max_meta_path_len+1):
            meta_path_file_name = self.hold_out_fold_path / (self.meta_path_file + "." + str(length))
            with open(meta_path_file_name, "r") as f:
                datas = f.readlines()
                for data in datas:
                    data = data.strip("\n").split("\t")
                    relation = data[2]
                    meta_path = data[4:]
                    relation_meta_path_dict[relation].append(meta_path)
            f.close()
        return relation_meta_path_dict

    def _sort_and_index_relation(self):
        """Based on each relation's meta path number, sort and index each relation.

        :return: relation_id_list. A list of relation.
        """
        relation_mp_num_list = [[item[0], len(item[1])] for item in self.relation_meta_path_dict.items()]
        sorted_rc_list = sorted(relation_mp_num_list, key= lambda item:item[1], reverse=True)
        relation_id_list = [item[0] for item in sorted_rc_list]
        return relation_id_list

    def _sort_and_index_entity(self):
        entity_num_list = [[item[0], len(item[1])] for item in self.query_graph_pt.adjacency_dict.items()]
        sorted_en_list = sorted(entity_num_list, key=lambda item: item[1], reverse=True)
        entity_id_list = [item[0] for item in sorted_en_list]
        return entity_id_list

    def _tail_predict(self,
                      process_rank: int,
                      hold_out_rank: int):
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=self.args_pt.gpu_num_per_hold,
                                rank=process_rank)
        gpu_id = hold_out_rank * self.args_pt.hold_out_num + process_rank
        print(f"hold_out_id/gpu_id/pid:[{hold_out_rank}/{gpu_id}/{os.getpid()}].")
        self.model_pt.cuda(gpu_id)
        model = torch.nn.parallel.DistributedDataParallel(self.model_pt,
                                                          device_ids=[gpu_id],
                                                          output_device=gpu_id)
        # 还在CPU上的三元组， 每个Torch进程都有一份指向它们的指针
        # 以Batch(大小为gpu_num_per_hold)形式遍历所有测试三元组,一张卡计算一条三元组
        # 每条三元组先由CPU生成对应的矩阵，然后传入GPU，并计算
        valid_triple_list = self.predict_graph_pt.triple_list
        # 根据gpu数量，将valid_triple_list划分为多个batch。每个batch只有gpu_num个triple。
        # 这样每个gpu就只运算一个triple生成的feature_matrix。保护了内存。
        batch_triple_list = self._my_valid_data_partition(valid_triple_list=valid_triple_list,
                                                          batch_size=self.args_pt.gpu_num_per_hold)
        hit_rank_list = []
        for batch_triple in tqdm(batch_triple_list):
            data, label = self._get_valid_feature_matrix(triple_list=batch_triple)
            valid_dataset = MhYaoPRAData(data=data, label=label)
            valid_sampler = torch.utils.data.DistributedSampler(dataset=valid_dataset,
                                                                num_replicas=self.args_pt.gpu_num_per_hold,
                                                                rank=process_rank)
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                       batch_size=int(self.args_pt.batch_size / self.args_pt.gpu_num_per_hold),
                                                       shuffle=False,
                                                       sampler=valid_sampler,
                                                       pin_memory=True,
                                                       num_workers=0)
            for feature_matrix_with_rid, label in valid_loader:
                feature_matrix_with_rid = feature_matrix_with_rid.cuda(gpu_id)
                results = model(feature_matrix_with_rid)
                hit_rank, tmp_results = self._calculate_hit_mr_mrr(results, label)
                hit_rank_list.append(hit_rank)
                # print(f"process_rank/hit_rank/tmp_results:{process_rank}/{hit_rank}/{tmp_results}.")
        hits_1 = 0
        hits_3 = 0
        hits_10 = 0
        mr = 0
        mrr = 0
        mrr_list = []
        mr_list = []
        for hit_rank in hit_rank_list:
            if hit_rank > 50 or hit_rank==0:
                continue
            if hit_rank <= self.args_pt.hit_range[0]:
                hits_1 += 1
            if hit_rank <= self.args_pt.hit_range[1]:
                hits_3 += 1
            if hit_rank <= self.args_pt.hit_range[2]:
                hits_10 += 1            
            mr += hit_rank
            mr_list.append(hit_rank)
            mrr += 1 / hit_rank
            mrr_list.append(1/hit_rank)
            # print(mrr_list, mr_list)
        rank_len = len(hit_rank_list)
        valid_rank_log_path = self.args_pt.train_log_path / "valid_rank.log"
        # if os.path.exists(valid_rank_log_path):
        #     with open(valid_rank_log_path, "a") as f:
        #         t=time.gmtime()
        #         log_time = time.strftime("%Y-%m-%d  %H:%M:%S",t)
        #         count = 0
        #         log_info = ''
        #         for item_rank in mrr_list:
        #             log_info = log_info + str(item_rank) + '\t'
        #             count += 1
        #             if count % 8 == 0:
        #                 log_info = log_info + '\n'
        #         log_rank_info = (f"This experiment logs on:{log_time}\n{log_info}")
        #         f.write(log_rank_info)
        # else:
        #     with open(valid_rank_log_path, 'w') as f:
        #         t=time.gmtime()
        #         log_time = time.strftime("%Y-%m-%d  %H:%M:%S",t)
        #         count = 0
        #         log_info = ''
        #         for item_rank in mrr_list:
        #             log_info = log_info + str(item_rank) + '\t'
        #             count += 1
        #             if count % 8 == 0:
        #                 log_info = log_info + '\n'
        #         log_rank_info = (f"This experiment logs on:{log_time}\n{log_info}")
        #         f.write(log_rank_info)
        #
        # valid_result_log_path = self.args_pt.train_log_path / "valid_result.log"
        # if os.path.exists(valid_result_log_path):
        #     with open(valid_result_log_path, "a") as f:
        #         t=time.gmtime()
        #         log_time = time.strftime("%Y-%m-%d  %H:%M:%S",t)
        #         log_info = (f"This experiment logs on:{log_time}\t"
        #                     f"Datasets is:{self.args_pt.graph_names[0]}\n"
        #                     f"hit@10 is:{hits_10/rank_len}, hit@3 is:{hits_3/rank_len}, hit@1 is:{hits_1/rank_len},"
        #                     f"MR is:{mr/rank_len}, MRR is:{mrr/rank_len}"
        #                     f"本次实验共{rank_len}个样本。\n\n")
        #         f.write(log_info)
        # else:
        #     with open(valid_result_log_path, 'w') as f:
        #         t=time.gmtime()
        #         log_time = time.strftime("%Y-%m-%d  %H:%M:%S",t)
        #         log_info = (f"This experiment logs on:{log_time}\t"
        #                     # f"Datasets is:{self.args_pt.graph_names[0]}\n"
        #                     # f"hit@10 is:{hits_10/rank_len}, hit@3 is:{hits_3/rank_len}, hit@1 is:{hits_1/rank_len},"
        #                     # f"MR is:{mr/rank_len}, MRR is:{mrr/rank_len}"
        #                     f"本次实验共{rank_len}个样本。\n\n")
        #         f.write(log_info)
        if rank_len == 0:
            print('rank_len is zero')
        else:
            print(f"Datasets is:{self.args_pt.graph_names[0]}\n"
            f"hit@10 is:{hits_10/rank_len}, hit@3 is:{hits_3/rank_len}, hit@1 is:{hits_1/rank_len},"
            f"MR is:'{mr/rank_len}, MRR is:{mrr/rank_len},"
            f"本次实验共{rank_len}个样本。\n")

    def _calculate_hit_mr_mrr(self,
                              results: torch.tensor,
                              label: torch.tensor):
        tmp_results = results.detach().cpu().numpy().tolist()
        tmp_label = int(label.detach().cpu().numpy().tolist()[0][0])
        candidate_entity_id_list = [int(id) for id in label.detach().cpu().numpy().tolist()[0][1:]]
        if tmp_label not in candidate_entity_id_list:
            return len(candidate_entity_id_list), tmp_results[0:10]
        score = tmp_results[candidate_entity_id_list.index(tmp_label)]
        results_with_id = [(tmp_results[i], candidate_entity_id_list[i]) for i in range(len(tmp_results))]
        results_with_id = sorted(results_with_id, key=lambda item: item[0], reverse=True)
        hit_rank = results_with_id.index((score, tmp_label))
        # tmp_results = sorted(tmp_results, reverse=True)
        # hit_rank = tmp_results.index(score)
        return hit_rank + 1, tmp_results[0:10]

    def _get_valid_feature_matrix(self,
                                  triple_list: list):
        valid_data_list = []
        valid_label_list = []
        for triple in triple_list:
            head_mid, relation, tail_mid = triple[0], triple[1], triple[2]
            key = head_mid + "#" + relation + "#" + tail_mid
            # 查询当前key是否在feature buffer中有记录，没有则跳过
            dense_feature_matrix = self.valid_feature_buffer_pt.get_feature_from_csv(triple_as_key=key)
            if dense_feature_matrix is False:
                continue
            if relation not in self.relation_id_list:
                continue
            relation_id = self.relation_id_list.index(relation)
            row_num = dense_feature_matrix.shape[0]
            col_num = dense_feature_matrix.shape[1] + 1 - 1
            feature_matrix_with_rid = torch.zeros((row_num, col_num))
            feature_matrix_with_rid[:, 0] = relation_id
            feature_matrix_with_rid[:, 1:] = dense_feature_matrix[:, 1:]
            tail_label_idx = self.entity_id_list.index(tail_mid)
            tmp_label = torch.from_numpy(np.array([tail_label_idx] + dense_feature_matrix[:, 0].numpy().tolist()))
            valid_data_list.append(feature_matrix_with_rid)
            valid_label_list.append(tmp_label)
        return valid_data_list, valid_label_list

    def _substitute_tail_entity(self,
                                triple: list):
        one_triple_with_all_tail_entity = []
        for entity_id, tail_mid in enumerate(self.entity_id_list):
            one_triple_with_all_tail_entity.append([triple[0], triple[1], tail_mid])
        return one_triple_with_all_tail_entity

    def _my_valid_data_partition(self,
                                 valid_triple_list: list,
                                 batch_size: int):
        data_len = len(valid_triple_list)
        all_batch_num = int(data_len / batch_size)
        batch_list = []
        for i in range(all_batch_num):
            batch_list.append(valid_triple_list[i * batch_size:(i + 1) * batch_size])
        if data_len % batch_size != 0:
            batch_list.append(valid_triple_list[all_batch_num * batch_size:])
        return batch_list


class PRAValid(GraphExperiments):
    """The subclass of GraphExperiments. It will organize the validation process for the previously trained model.

    The difference between PRATrain and PRAValid are that:
    1) It will also have predict_graph_pt, which contains all the validation triples. In the meanwhile, it also
        have query_graph_pt, which is used to generate the
    2) It will

    """
    def __init__(self,
                 query_graph_pt: ProcessedGraph,
                 hold_out_path: Path,
                 meta_path_file: str,
                 args_pt: argparse,
                 predict_graph_pt: ProcessedGraph,
                 valid_feature_buffer_pt: ValidFeatureBuffer,
                 model_pt: MhYaoPRA):
        super(PRAValid, self).__init__(query_graph_pt=query_graph_pt,
                                       hold_out_fold_path=hold_out_path,
                                       meta_path_file=meta_path_file,
                                       args_pt=args_pt,
                                       predict_graph_pt=predict_graph_pt,
                                       valid_feature_buffer_pt=valid_feature_buffer_pt)
        self.model_pt = model_pt

    def tail_predict(self,
                     hold_out_rank: int):
        os.environ['MASTER_ADDR'] = '172.17.0.2'
        os.environ['MASTER_PORT'] = str(hold_out_rank + 8886)
        print(f"开始验证模型。")
        mp.spawn(fn=self._tail_predict,
                 args=(hold_out_rank,),
                 nprocs=self.args_pt.gpu_num_per_hold)


class PRATrain(GraphExperiments):
    """The subclass of GraphExperiments. It will organize the training process for specific hold out and hyper parameters.

    Args:
        query_graph_pt: ProcessedGraph. Passed to its parent class.
        neg_samp_graph: ProcessedGraph. Pointer to negative samples.
        meta_path_file: str. Passed to its parent class. In it's parent class, this file would be load.
        args_pt: argparse. Passed to its parent class.
        hold_out_path: Path. Path to the hold out fold.
        hyper_param: dict. A dict containing possible hyper param such as lr.

    Attributes:
        All attributes inherited from it's parent class.
        neg_samp_graph: ProcessedGraph. Load the neg samp triples in neg_samp_path.
        hyper_param: Check it in Args.

    Methods:
        train_this_hold_out: Organize training process based on a specific hyper param for a specific hold out.
        train_on_each_gpu: Organize training process for each gpu.
        _get_feature_matrix_and_labels: Turn the pos (access via the query_graph_pt)
                                        and neg samples of triples into feature matrix and labels.
        _each_relation_fmal:

    """
    def __init__(self,
                 query_graph_pt: ProcessedGraph,
                 neg_graph_pt: ProcessedGraph,
                 hold_out_path: Path,
                 meta_path_file: str,
                 args_pt: argparse,
                 hyper_param: dict):
        super(PRATrain, self).__init__(query_graph_pt=query_graph_pt,
                                       hold_out_fold_path=hold_out_path,
                                       meta_path_file=meta_path_file,
                                       args_pt=args_pt)
        self.neg_samp_graph = neg_graph_pt
        self.hyper_param = hyper_param

    def train_this_hold_out(self,
                            hold_out_rank: int,
                            if_save_model: bool=False):
        """Organize training process based on a specific hyper param for a specific hold out.

        For each specific huper param, the training process consists of the following several stages:
        Stage 1: Load meta path of each relation into a dict. This is done in the __init__ method of its parent class.
            Stage 1.1: Determine the relation's id according to it's meta path number.
        Stage 2: Load negative samples of triples stored in neg_samp_path file, and return a ProcessedGraph instance.
                This is done in the __init__ method of its own.
        Stage 3: Turn the pos (access via the query_graph_pt) and neg samples of triples into feature matrix and labels.
            Stage 3.1: In this stage, the feature_size is determined and kept in args_pt.
            Stage 3.2: Using multiprocessing to search feature for each relation.
        Stage 4. Transfer the feature matrix and labels in CPU to GPU, and train the model_pt.

        :param hold_out_rank: int.
        :param if_save_model: int.
        :return: None.
        """
        # Print necessary info.
        if hold_out_rank is not None and hold_out_rank == 0:
            print(f"超参为{self.hyper_param},开始训练{self.query_graph_pt.file_path}中的三元组")
        # 将原始三元组转化为对应的torch特征矩阵 这里删除训练步骤时需要注释掉
        # train_feature_matrix_and_labels = self._get_feature_matrix_and_labels(hold_out_rank=hold_out_rank)
        # relation_num = len(train_feature_matrix_and_labels[0])
        # 确定feature_size的长度
        max_meta_path_num = max([len(self.relation_meta_path_dict[relation])
                                 for relation in self.relation_meta_path_dict.keys()])
        self.args_pt.feature_size = min(max_meta_path_num, self.args_pt.feature_size)
        # 开多进程
        relation_num = len(self.relation_id_list)
        # print(f"fsz:{self.args_pt.feature_size};feature_matrix:{train_feature_matrix_and_labels[0][1][0,:]}")
        # 定义模型
        self.model_pt = MhYaoPRA(self.args_pt.feature_size, relation_num)
        # os.environ['MASTER_ADDR'] = '172.17.0.2'
        # os.environ['MASTER_PORT'] = str(hold_out_rank + 8887)
        # print(f"开始训练数据。")
        # mp.spawn(fn=self.train_on_each_gpu,
        #          args=(hold_out_rank,
        #                train_feature_matrix_and_labels,),
        #          nprocs=self.args_pt.gpu_num_per_hold)

    def train_on_each_gpu(self,
                          process_rank: int,
                          hold_out_rank: int,
                          train_data: list):
        """Train one relation on a GPU each time.

        :param process_rank:
        :param hold_out_rank:
        :param train_data:
        :return:
        """
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=self.args_pt.gpu_num_per_hold,
                                rank=process_rank)
        gpu_id = hold_out_rank * self.args_pt.hold_out_num + process_rank
        print(f"hold_out_id/gpu_id/pid:[{hold_out_rank}/{gpu_id}/{os.getpid()}].")
        torch.manual_seed(gpu_id)
        self.model_pt.cuda(gpu_id)
        model = torch.nn.parallel.DistributedDataParallel(self.model_pt,
                                                          device_ids=[gpu_id],
                                                          output_device=gpu_id)
        feature_matrix_with_rid, labels = train_data
        train_data = MhYaoPRAData(data=feature_matrix_with_rid, label=labels)
        train_sampler = torch.utils.data.DistributedSampler(dataset=train_data,
                                                            num_replicas=self.args_pt.gpu_num_per_hold,
                                                            rank=process_rank)
        valid_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=int(self.args_pt.batch_size / self.args_pt.gpu_num_per_hold),
                                                   shuffle=False,
                                                   sampler=train_sampler,
                                                   pin_memory=True,
                                                   num_workers=0)
        criterion = nn.BCELoss().cuda(gpu_id)
        optimizer = torch.optim.Adam(model.parameters())
        loss = 0
        loss_list = []
        loss_per_epo_list = []
        for epo in range(self.args_pt.epoch):
            for feature_matrix_with_rid, label in valid_loader:
                feature_matrix_with_rid = feature_matrix_with_rid.cuda(gpu_id).float()
                label = label.cuda(gpu_id).squeeze().float()
                results = model(feature_matrix_with_rid)
                loss = criterion(results, label)
                loss_per_epo_list.append([int(feature_matrix_with_rid[0, 0, 0]), loss.item()])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_list.append(loss_per_epo_list)
            loss_per_epo_list = []
            if hold_out_rank == 0 and process_rank == 0:
                print(f"Step:[{epo + 1}/{self.args_pt.epoch}];loss:{loss}.")
        with open(self.args_pt.train_log_path / ("GPU." + str(gpu_id) + "loss.log"), "w") as f:
            for epo, loss_per_epo in enumerate(loss_list):
                log_info = f"Step:[{epo+1}/{self.args_pt.epoch}];"
                for relation in loss_per_epo:
                    log_info += f"\t{relation[0]}:{relation[1]:.4}"
                log_info += "\n"
                f.write(log_info)
        f.close()
        # "\t" + str(relation[0]) + ":" + str(relation[1])

    def _get_feature_matrix_and_labels(self, hold_out_rank: int):
        """Turn the pos (access via the query_graph_pt) and neg samples of triples into feature matrix and labels.

        :param hold_out_rank: int.
        :return: (train_data_list, train_label_list). Where each entry in train_data_list is the feature_matrix of all
                                            pos and neg samples of triples of one relation. The type is torch.tensor.
                                            train_label_list is similar.

        """
        # 确定feature_size的长度
        max_meta_path_num = max([ len(self.relation_meta_path_dict[relation])
                                  for relation in self.relation_meta_path_dict.keys() ])
        self.args_pt.feature_size = min(max_meta_path_num, self.args_pt.feature_size)
        # 开多进程
        relation_num = len(self.relation_id_list)
        process_pool = Pool(processes=relation_num)
        collect_feature_matrix_and_labels = []
        train_data_list = []
        train_label_list = []
        for relation_id in range(relation_num):
            collect_feature_matrix_and_labels.append(
                process_pool.apply_async(func=self._each_relation_fmal,
                                         args=(hold_out_rank,
                                               relation_id,),
                                         error_callback=handle_error)
            )
        process_pool.close()
        process_pool.join()
        for result in collect_feature_matrix_and_labels:
            train_data, train_label = result.get()
            train_data_list.append(train_data)
            train_label_list.append(train_label)
        return train_data_list, train_label_list

    def _each_relation_fmal(self,
                            hold_out_rank: int,
                            relation_id: int):
        """Get fmal for each relation by calling the get_probs function in utils.

        :param hold_out_rank: int. Used to print info.
        :param relation_id: int. The index of relation that this process handles with.
        :param neg_samp_triple_list: list. A list of triples that is neg sample.
                                            Each process will only sample part of it.
        :return: (feature_matrix_with_rid, labels_tensor). Both are torch tensor.
        """
        if relation_id == 0 and hold_out_rank == 0:
            print(f"开始提取特征")
        # Get relation name by indexing self.relation_id_list.
        relation = self.relation_id_list[relation_id]
        print(f"关系{relation};", end="")
        # 随机抽取出2*args.pos_sample_size个正/负样本
        sample_size = min(self.args_pt.pos_samp_size,
                          len(self.query_graph_pt.relation_pos_samp_dict[relation]),
                          len(self.neg_samp_graph.relation_pos_samp_dict[relation]))
        relation_pos_samp = random.sample(self.query_graph_pt.relation_pos_samp_dict[relation],
                                          sample_size)
        relation_neg_samp = random.sample(self.neg_samp_graph.relation_pos_samp_dict[relation],
                                          sample_size)
        train_pairs_01 = []
        for item in relation_pos_samp:
            e1, e2 = item
            train_pairs_01.append([e1, e2, 1])
        for item in relation_neg_samp:
            e1, e2 = item
            train_pairs_01.append([e1, e2, 0])
        # 打乱这批训练数据
        random.shuffle(train_pairs_01)
        # 生成样本特征
        feature_matrix_with_rid, labels_tensor = get_train_probs(query_graph_pt=self.query_graph_pt,
                                                                 rid=relation_id,
                                                                 entity_pairs=train_pairs_01,
                                                                 args_pt=self.args_pt,
                                                                 relation_meta_path=self.relation_meta_path_dict[relation])
        return feature_matrix_with_rid, labels_tensor
