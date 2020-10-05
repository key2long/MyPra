import argparse
from pathlib import Path
from collections import defaultdict
from ..src.ProcessedGraph import ProcessedGraph
from ..src.utils import handle_error, get_predict_probs
from multiprocessing import Pool


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


# 多进程生成feature_matrix
def generate_feature_matrix_each_batch(process_rank: int,
                                       batch_triple: list,
                                       query_graph: ProcessedGraph,
                                       relation_meta_path_dict: defaultdict,
                                       args: argparse):
    # Get entity id list
    entity_num_list = [[item[0], len(item[1])] for item in query_graph.adjacency_dict.items()]
    sorted_en_list = sorted(entity_num_list, key=lambda item: item[1], reverse=True)
    entity_id_list = [item[0] for item in sorted_en_list]
    # Get relation id list
    relation_mp_num_list = [[item[0], len(item[1])] for item in relation_meta_path_dict.items()]
    sorted_rc_list = sorted(relation_mp_num_list, key=lambda item: item[1], reverse=True)
    relation_id_list = [item[0] for item in sorted_rc_list]
    # Get sparse feature matrix and store them in a dict
    triple_feature_coo_matrix_dict = defaultdict(list)
    for triple in batch_triple:
        head_mid, relation, tail_mid = triple[0], triple[1], triple[2]
        if relation not in relation_id_list:
            continue
        relation_id = relation_id_list.index(relation)
        one_triple_with_all_tail_entity = substitute_tail_entity(triple, entity_id_list)
        feature_matrix = get_predict_probs(query_graph_pt=query_graph,
                                           rid=relation_id,
                                           one_triple_with_all_tail_entity=one_triple_with_all_tail_entity,
                                           args_pt=args,
                                           relation_meta_path=relation_meta_path_dict[relation])[:, 1:]
        sparse_feature = feature_matrix.to_sparse()
        feature_position = sparse_feature.indices().numpy()
        feature_value = sparse_feature.values().numpy()
        non_zero_count = sparse_feature.size()[1]
        key = triple[0] + "#" + triple[1] + "#" + triple[2]
        value = [feature_position, feature_value, non_zero_count]

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_size', default=500, type=int)
    parser.add_argument('--process_num', default=20, type=int)
    args = parser.parse_args()
    args.graph_names = ['fb15k237']
    args.processed_path = Path("../DATA/processed")
    args.hold_out_fold = "hold_out_"
    args.files_suffix = ["train", "valid", "test"]
    hold_out_rank = 0
    args.min_mp_length = 2
    args.max_mp_length = 4
    # 打开图和metapath。
    hold_out_path = args.processed_path / args.graph_names[0] / (args.hold_out_fold + str(hold_out_rank))

    train_graph_name =  hold_out_path / (args.graph_names[0] + str(hold_out_rank) + args.files_suffix[0])
    train_graph = ProcessedGraph(file_path=train_graph_name)

    valid_graph_name = hold_out_path / (args.graph_names[0] + str(hold_out_rank) + args.files_suffix[1])
    valid_graph = ProcessedGraph(file_path=valid_graph_name)

    test_graph_name = hold_out_path / (args.graph_names[0] + str(hold_out_rank) + args.files_suffix[2])
    test_graph = ProcessedGraph(file_path=test_graph_name)

    relation_meta_path_dict = load_meta_path(hold_out_path=hold_out_path, args=args)

    # 遍历valid_graph中的所有triple，从而为每条triple生成对应的feature_matrix。
    process_pool = Pool(processes=args.process_num)
    collect_feature_matrix_dict = []
    batch_triple_list = batch_partition(triple_list=valid_graph.triple_list,
                                        all_batch_num=args.process_num)
    for process_rank in range(args.process_num):
        batch_triple = batch_triple_list[process_rank]
        collect_feature_matrix_dict.append(
            process_pool.apply_async(func=generate_feature_matrix_each_batch,
                                     args=(process_rank,
                                           batch_triple,
                                           train_graph,
                                           relation_meta_path_dict,
                                           args,),
                                     error_callback=handle_error)
        )
    process_pool.join()
    process_pool.close()


if __name__ == "__main__":
    main()
