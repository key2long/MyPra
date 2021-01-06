# -*- coding: utf-8 -*-
import sys
import argparse
import torch
import numpy as np
import multiprocessing.pool
from pathlib import Path
from torch.utils.data import Dataset
from ProcessedGraph import ProcessedGraph
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm


class MhYaoPRAData(Dataset):
    def __init__(self, data, label):
        super(MhYaoPRAData, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


class NoDaemonProcess(multiprocessing.Process):
    """自定义非守护进程

    Python自带的multiprocessing默认生成守护进程，从而导致无法在子进程中再开启子进程。
    改为非守护进程可以解决这个问题。

    """
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    """对Pool对象的重新包装。

    """
    Process = NoDaemonProcess


def handle_error(error: str):
    """Immediately print the error message.

    :param error: The segment fault message.
    :return: None.
    """
    print(error)
    sys.stdout.flush()


def find_and_save_meta_path(query_graph_pt: ProcessedGraph,
                            process_num: int,
                            max_depth: int,
                            hit_appear_ratio: float,
                            save_path: str):
    """Find all the meta paths for each relation in the query_graph, which is a  ProcessedGraph instance, and save them.

    Use multiprocessing to find each relation's meta paths, and save them.
    To balance the load of each process, we split queru_graph_pt's triple_list evenly according to the process number.
    Each process iterates its own share, and returns a relation_meta_path_dict.
    At last, this function will merge all the dicts for each relation, and save the meta paths in file.

    :param query_graph_pt: A pointer to an instance of ProcessedGraph class.
    :param process_num: The number of accessible processes.
    :param max_depth: The maximum length of each meta path.
    :param hit_appear_ratio: The minimum ratio of the number of path that fit in a meta path
                            over this meta path's appear count.
    :param save_path: Where to save all the meta paths.
    :return: One meta path per line in save_path file.
              Each meta path is a string and it looks like: "65 234 relation    #SEP#   r_1 r_2 r_3"
              The number "65" indicates how many path fit in this meta path.
              The number "234" indicates how many times that this meta path appears.
              The "relation" indicates which relation this meta path belongs to.
              The "#SEP#" is a special string, indicating that the remaining string are meta path itself.
              The "r_1 r_2 r_3" is the meta path itself, and its length is less than max_depth.
              All the string above are separated by "\t".
    """
    print(f"开始搜索{query_graph_pt.file_path}中的meta path")
    collect_relation_meta_path_dict = []
    process_pool = Pool(processes=process_num)
    for process_rank in range(process_num):
        batch_triple_list = get_batch_triple_list(query_graph_pt.triple_list,
                                                  process_rank,
                                                  process_num)
        collect_relation_meta_path_dict.append(
            process_pool.apply_async(func=batch_meta_path_search,
                                     args=(process_rank,
                                           batch_triple_list,
                                           max_depth,
                                           query_graph_pt.adjacency_dict,),
                                     error_callback=handle_error)
        )
    process_pool.close()
    process_pool.join()
    # Merge all batch's rmpd into one.
    print(f"开始合并meta path统计数据")
    relation_meta_path_dict = defaultdict(defaultdict)
    for result in tqdm(collect_relation_meta_path_dict):
        tmp_rmpd = result.get()
        merge_rmpd(relation_meta_path_dict, tmp_rmpd)
    # Save the Global rmpd into save_path file.
    print(f"开始写入meta path")
    with open(save_path, "w") as f:
        for relation in tqdm(relation_meta_path_dict.keys()):
            for meta_path_as_key in relation_meta_path_dict[relation].keys():
                # Check if the hit/appear ratio is higher enough
                hit = relation_meta_path_dict[relation][meta_path_as_key][0]
                appear = relation_meta_path_dict[relation][meta_path_as_key][1]
                ratio = hit / appear
                if ratio >= hit_appear_ratio and appear > 5:
                    std_format_meta_path = parse_mp(meta_path_as_key, relation, hit, appear)
                    f.write(std_format_meta_path)
    f.close()


def parse_mp(meta_path_as_key: str,
             relation: str,
             hit: int,
             appear: int):
    """Turn meta_path_as_key into standard format of meta path.

    "&#SEP#"+"&r_1"+...+"&r_i" --> "hit appear relation #SEP# r_1 r_2 r_3".

    :param meta_path_as_key: "&#SEP#"+"&r_1"+...+"&r_i"
    :return: std_format_meta_path. It looks like this: "hit appear relation #SEP# r_1 r_2 r_3".
    """
    std_format_meta_path = str(hit) + "\t" + str(appear) + "\t" + relation
    mp_list = meta_path_as_key.split("&")
    for r in mp_list:
        if r != "":
            std_format_meta_path += "\t" + r
    std_format_meta_path += "\n"
    return std_format_meta_path


def merge_rmpd(relation_meta_path_dict: defaultdict,
               tmp_rmpd: defaultdict):
    """Merge the tmp_rmpd into relation_meta_path_dict by calling the increase_rmph_dict function.

    :param relation_meta_path_dict:  A dict of a dict.
    :param tmp_rmpd: A dict of a dict.
    :return: None
    """
    for relation in tmp_rmpd.keys():
        for meta_path_as_key in tmp_rmpd[relation].keys():
            hit = tmp_rmpd[relation][meta_path_as_key][0]
            app = tmp_rmpd[relation][meta_path_as_key][1]
            increase_rmph_dict(relation_meta_path_dict,
                               relation, meta_path_as_key,
                               increase_hit=hit,
                               increase_app=app)


def get_batch_triple_list(triple_list: list,
                          process_rank: int,
                          process_num: int):
    """Return the process_rank th batch of triple_list, whose length is  1/process_num.

    :param triple_list: The triple_list attribute of ProcessedGraph instance.
    :param process_rank: Return this batch.
    :param process_num: total batch num.
    :return: a batch of triple_list.
    """
    batch_size = int(len(triple_list) / process_num)
    batch_begin_idx = process_rank * batch_size
    batch_end_idx = min((process_rank + 1)*batch_size, len(triple_list))
    return triple_list[batch_begin_idx:batch_end_idx]


def batch_meta_path_search(process_rank: int,
                           batch_triple_list: list,
                           max_depth: int,
                           adjacency_dict: defaultdict):
    """Iterate over each triple in batch_triple_list, and get meta path for each triple's relation.

    :param process_rank: int. Make sure that only the process_rank=0 will print message.
    :param batch_triple_list: list. The batch of triples to iterate over.
    :param max_depth: int. The maximum length of a meta path.
    :param adjacency_dict: defaultdict. An attribute of the ProcessedGraph class.
    :return: relation_meta_path_hit_dict: A dict of a dict.
                                        Example: dict_out = {key="relation": value=dict_in}
                                                 dict_in = {key="r_1+r_2+r_3": value=[hit_count, appear_count]}
    """
    if process_rank == 0:
        print(f"\t开始遍历每个batch中的三元组")
    # 每个进程独自维护一个relation_meta_path_hit_dict字典, 不存在原子操作之类的问题。
    relation_meta_path_hit_dict = defaultdict()
    for triple in tqdm(batch_triple_list):
        head_mid, relation, tail_mid = triple[0], triple[1], triple[2]
        current_path = [["#SEP#", head_mid]]
        dfs(max_depth=max_depth,
            blocked_edge=triple,
            current_path=current_path,
            adjacency_dict=adjacency_dict,
            relation_meta_path_hit_dict=relation_meta_path_hit_dict)
    return relation_meta_path_hit_dict


def dfs(max_depth: int,
        blocked_edge: list,
        current_path: list,
        adjacency_dict: defaultdict,
        relation_meta_path_hit_dict: defaultdict):
    """Search meta path based on one triple, which is blocked.

    分情况讨论如下：
    # 检查current_path的最后一个节点current_entity是否为tail_mid？
    # case 1: 是. 则current_path中的meta path击中次数增加一次；同时它的出现次数增加一次。
    #             搜索结束。需要先pop掉current_path中的最后一条边，然后return 到上一层中继续搜索下一条边。
    # case 2: 否. 检查current_path的长度是否达到了最大？
    #             case 2.1: 是. 则current_path中的meta path出现次数增加一次。
    #                           搜索结束。需要先pop掉current_path中的最后一条边，然后return 到上一层中继续搜索下一条边。
    #             case 2.2: 否. 则current_path中的meta path出现次数增加一次。
    #                           搜索继续。需要遍历current_entity的连边，并检查当前边是否为blocked_edge或者是否回头了？
    #                           case 2.2.1: 是. 则continue。
    #                           case 2.2.2: 否. 则将当前边压入current_path，递归调用dfs函数。

    :param max_depth: maximum search depth.
    :param blocked_edge: [head_mid, relation, tail_mid].
    :param current_path: [[#SEP#, head_mid], [r_1, t_1], ..., [r_i, current_entity]].
    :param adjacency_dict: The attributes of a ProcessedGraph instance.
                            Example: {key=h_j: value=[[r_j, t_j], ..., [r_k, t_k], ..., ]}
    :param relation_meta_path_hit_dict: A dict of a dict.
                                        Example: dict_out = {key="relation": value=dict_in}
                                                 dict_in = {key="r_1+r_2+r_3": value=[hit_count, appear_count]}
    :return: 维护relation_meta_path_hit_dict.一个进程一个字典，进程间不相互共享。
    """
    # 代码思路见上方“分情况讨论”注释。
    target_entity = blocked_edge[2]
    relation = blocked_edge[1]
    current_entity = current_path[-1][1]
    meta_path_as_key = get_meta_path(current_path)
    if current_entity == target_entity:  # case 1
        increase_rmph_dict(relation_meta_path_hit_dict,
                           relation, meta_path_as_key,
                           increase_hit=1,
                           increase_app=None)
        increase_rmph_dict(relation_meta_path_hit_dict,
                           relation, meta_path_as_key,
                           increase_app=1)
        current_path.pop()
        return
    elif len(current_path) > max_depth:  # case 2.1
        increase_rmph_dict(relation_meta_path_hit_dict,
                           relation, meta_path_as_key,
                           increase_app=1)
        current_path.pop()
        return
    else:  # case 2.2
        increase_rmph_dict(relation_meta_path_hit_dict,
                           relation, meta_path_as_key,
                           increase_app=1)
        for (r, entity) in adjacency_dict[current_entity]:
            if [current_entity, r, entity] == blocked_edge:
                continue  # case 2.2.1
            else:  # case 2.2.2
                current_path.append([r, entity])
                dfs(max_depth, blocked_edge, current_path,
                    adjacency_dict, relation_meta_path_hit_dict)


def get_meta_path(current_path: list):
    """Concatenate all the relation in current_path,and return this str as a key.

    "&" is the separation char between relation.

    :param current_path: [[#SEP#, head_mid], [r_1, t_1], ..., [r_i, current_entity]].
    :return: key="&#SEP#"+"&r_1"+...+"&r_i".
    """
    key = ""
    for (r, _) in current_path:
        key += "&" + r
    return key


def increase_rmph_dict(relation_meta_path_hit_dict: defaultdict,
                       relation: str,
                       meta_path_as_key: str,
                       increase_hit: int = None,
                       increase_app: int = None):
    """Increase relation_meta_path_hit_dict's record about meta path's hit and appear count.

    :param relation_meta_path_hit_dict: A dict of a dict.
    :param relation: str.
    :param meta_path_as_key: str.
    :param increase_hit: If is not None, then increase increase_hit amount.
    :param increase_app: If is not None, then increase increase_app amount.
    :return: None.
    """
    if relation not in relation_meta_path_hit_dict:
        relation_meta_path_hit_dict[relation] = defaultdict(list)
    if meta_path_as_key not in relation_meta_path_hit_dict[relation]:
        relation_meta_path_hit_dict[relation][meta_path_as_key] = [0, 0]
    if increase_hit is not None:
        relation_meta_path_hit_dict[relation][meta_path_as_key][0] += increase_hit
    if increase_app is not None:
        relation_meta_path_hit_dict[relation][meta_path_as_key][1] += increase_app


def _probs(begin_entity: str,
           end_entity: str,
           one_metapath: list,
           query_graph_pt: ProcessedGraph):
    """本函数的功能是接收一对实体对和一条metapath，生成相应的由该metapath指导，从头实体转移到尾实体的概率。

    转移概率情况讨论：
    1：关系长度=1：
        1.1：头节点能够通过该关系转移到尾实体：
            概率为该关系下连接实体总数的倒数
        1.2：头节点不能通过该关系转移到尾实体：
            概率为0
        这也是跳出跳出递归的边界条件
    2：关系长度=0：返回概率0
    3：关系长度不唯一：
        3.1：头节点找不到关系0：
            返回概率0
        3.2：能够从头结点通过关系0转移到下一个节点：
            遍历该关系0的可达节点集合，将可达节点作为头节点从下一个关系递归调用本函数

    :param begin_entity: 头实体
    :param end_entity: 尾实体
    :param one_metapath: [r_1, r_2, r_3....]一条关系路径
    :param query_graph_pt: 查询图对象，里面存储着整个图的各种信息
    :return prob: float. 头实体按照metapath转移到尾实体的概率。
    """
    prob = 0
    length = len(one_metapath)
    if length == 1:
        if end_entity in query_graph_pt.relation_adj_dict[begin_entity][one_metapath[0]]:
            prob = 1/len(query_graph_pt.relation_adj_dict[begin_entity][one_metapath[0]])
        else:
            prob = 0
        return prob
    elif length == 0:
        return 0
    else:
        if query_graph_pt.relation_adj_dict[begin_entity][one_metapath[0]] == []:
            return 0
        else:
            for item in query_graph_pt.relation_adj_dict[begin_entity][one_metapath[0]]:
                prob += (1/len(query_graph_pt.relation_adj_dict[begin_entity][one_metapath[0]])) \
                        *_probs(item, end_entity, one_metapath[1:], query_graph_pt)
            return prob


# 由于在训练和做预测时的数据格式不同，考虑构造两个生成实体对特征的函数
# get_train_probs 是生成训练所用的数据特征格式
def get_train_probs(query_graph_pt: ProcessedGraph,
                    rid: int,
                    entity_pairs: list,
                    args_pt: argparse,
                    relation_meta_path: list):
    """生成某个关系下的所有正负样本对的特征矩阵.

    :param query_graph_pt 指向query_graph的指针，存储着图的各种信息
    :param rid 某个关系的id
    :param entity_pairs 某个关系下的正负样本对
        example:    [[h_1, e_1, 1], [h_2, e_2, 0], ...]
    :param args_pt argparse对象，传递最大特征长度数值
    :param relation_meta_path 某个关系下的metapath
        example:    [p_1, p_2, ...]
                    p_1:[r_1, r_2, r_3,...]
    :return (feature_matrix_with_rid, labels_tensor): Both are torch tensor.
    """
    feature_size = args_pt.feature_size
    feature_matrix_with_rid = torch.from_numpy(np.zeros(shape=[len(entity_pairs), 1 + feature_size]))
    feature_matrix_with_rid[:, 0] = rid
    labels = []
    row = 0
    for data in tqdm(entity_pairs):
        [h_entity, t_entity, tmp_label] = data
        labels.append(float(tmp_label))
        if h_entity not in query_graph_pt.adjacency_dict.keys():
            row += 1
            continue
        tmp_feature_vec = []
        meta_path_count = 0
        for id, path in enumerate(relation_meta_path):
            if id >= feature_size: # 截断多余的metapath
                meta_path_count = id
                break
            else:
                tmp_prob = _probs(h_entity, t_entity, path, query_graph_pt)
                tmp_feature_vec.append(tmp_prob)
                meta_path_count = id
        # 对metapath数量比较少的模型作padding
        if meta_path_count < feature_size:
            for i in range(0, feature_size - meta_path_count - 1):
                tmp_feature_vec.append(0.0)
        feature_matrix_with_rid[row, 1:] = torch.from_numpy(np.array(tmp_feature_vec))
        row += 1
    labels_tensor = torch.from_numpy(np.array(labels))
    non_zero_count = torch.nonzero(feature_matrix_with_rid[:,1:]).shape[0]
    print(f"relation:{rid} is done;"
          f"nonzero/total:"
          f"{non_zero_count}/{feature_matrix_with_rid[:,1:].shape[0]*feature_matrix_with_rid[:,1:].shape[1]}")
    return feature_matrix_with_rid, labels_tensor


# get_train_probs 是生成训练所用的数据特征格式
def get_predict_probs(query_graph_pt: ProcessedGraph,
                      rid: int,
                      one_triple_with_all_tail_entity: list,
                      args_pt: argparse,
                      relation_meta_path: list):
    '''
    :param rid: 关系的id标识
    :param one_entity_extend_list: 某个待预测的实体对，将尾实体扩充为整个实体集合构成一个大的二维列表
        example: [[h_1, r_1, e_1], [h_1, r_1, e_2],....[h_1, r_1, e_i]] i为实体集合数目
    :param args_pt argparse对象，传递最大特征长度数值
    :param relation_meta_path 某个关系下的metapath
        example:    [p_1, p_2, ...]
                    p_1:[r_1, r_2, r_3,...]
    '''
    feature_size = args_pt.feature_size
    feature_matrix_with_rid = torch.from_numpy(np.zeros(shape=[len(one_triple_with_all_tail_entity), 2 + feature_size]))
    feature_matrix_with_rid[:, 0] = rid
    row = 0
    for item in one_triple_with_all_tail_entity:
        [h_entity, relation, t_entity, t_entity_id, t_entity_dis] = item
        tmp_feature_vec = []
        meta_path_count = 0
        for id, path in enumerate(relation_meta_path):
            if id >= feature_size:
                meta_path_count = id
                break
            else:
                tmp_prob = _probs(h_entity, t_entity, path, query_graph_pt)
                tmp_feature_vec.append(tmp_prob)
                meta_path_count = id
    # 对metapath数量比较少的模型作padding
        if meta_path_count < feature_size:
            for i in range(0, feature_size - meta_path_count - 1):
                tmp_feature_vec.append(0.0)
        feature_matrix_with_rid[row, 2:] = torch.from_numpy(np.array(tmp_feature_vec))
        feature_matrix_with_rid[row, 1] = t_entity_id
        row += 1
    # print(f"head_mid/relation:{one_triple_with_all_tail_entity[0][0]}\{one_triple_with_all_tail_entity[0][1]} is done.")
    return feature_matrix_with_rid

def load_meta_path(hold_out_path: Path,
                   args: argparse):
    relation_meta_path_dict = defaultdict(list)
    for length in range(args.min_mp_length, args.max_mp_length + 1):
        meta_path_file_name = hold_out_path / ("train.MetaPath." + str(length))
        with open(meta_path_file_name, "r") as f:
            datas = f.readlines()
            for data in datas:
                data = data.strip("\n").split("\t")
                relation = data[2]
                meta_path = data[4:]
                relation_meta_path_dict[relation].append(meta_path)
        f.close()
    return relation_meta_path_dict


def batch_partition(triple_list: list,
                    all_batch_num: int):
    data_len = len(triple_list)
    batch_size = int(data_len / all_batch_num)
    batch_list = []
    for i in range(all_batch_num):
        batch_list.append(triple_list[i * batch_size:(i + 1) * batch_size])
    if data_len % batch_size != 0:
        batch_list.append(triple_list[all_batch_num * batch_size:])
    return batch_list


def substitute_tail_entity(triple: list,
                           entity_id_list: list):
    one_triple_with_all_tail_entity = []
    for entity_id, tail_mid in enumerate(entity_id_list):
        one_triple_with_all_tail_entity.append([triple[0], triple[1], tail_mid])
    return one_triple_with_all_tail_entity


def screen_tail_entity(triple: list,
                       entity_id_list: list,
                       query_graph: ProcessedGraph,
                       k_nearest: int):
    target_tail_vec = query_graph.get_target_tail_vec(triple[0], triple[2])
    if target_tail_vec is not None:
        local_id_mid_dis_list = []
        for entity_id, tail_mid in enumerate(entity_id_list):
            parsed_tail_mid = ".".join(tail_mid.split('/')[1:])
            tail_embedding = query_graph.get_entity2vec(parsed_tail_mid)
            if tail_embedding is not None:
                local_id_mid_dis_list.append([triple[0], triple[1], tail_mid,
                                              entity_id, torch.norm(target_tail_vec - tail_embedding)])
            else:
                # print(f"Entity {tail_mid} is missing in KGE.")
                local_id_mid_dis_list.append([triple[0], triple[1], tail_mid,
                                              entity_id, 0.0])
        one_triple_with_screened_tail_entity = sorted(local_id_mid_dis_list, key=lambda item: item[-1])[:k_nearest]
        return one_triple_with_screened_tail_entity
    else:
        print(f"Not screened at all for head_relation:{triple[0], triple[2]}.")
        one_triple_with_all_tail_entity = []
        for entity_id, tail_mid in enumerate(entity_id_list):
            one_triple_with_all_tail_entity.append([triple[0], triple[1], tail_mid])
        return one_triple_with_all_tail_entity


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return (hours, minutes, seconds)
