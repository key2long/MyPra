from utils import r_path_random_walk_with_select, parse_rp
from ProcessedGraph import ProcessedGraph
import argparse
import traceback
from pathlib import Path
import random
import os
from tqdm import tqdm


def GenRelationPath(hold_out_rank: int,
                    args_pt: argparse,
                    data_name: str):
    '''生成关系语料数据，将不同长度的关系语料数据写入文件
        example：path_length /t r_1 /t r_2 ... /t r_direct /n

    '''
    hold_out_fold_path = args_pt.processed_path / data_name / (args_pt.hold_out_fold + str(hold_out_rank))
    train_graph_name = data_name + "." + str(hold_out_rank) + "." + args_pt.files_suffix[0] + "." + "graph"
    train_graph = ProcessedGraph(file_path=hold_out_fold_path / train_graph_name)
    path_num = args_pt.path_num
    relation_path_dict = {}
    for path_len in tqdm(range(args_pt.min_path_length, args_pt.max_path_length+1)):
        all_legal_path_with_label = []
        while len(all_legal_path_with_label) < path_num:
            legal_path_with_label_list = r_path_random_walk_with_select(train_graph, path_len, 500)
            if len(legal_path_with_label_list) == 0:
                 pass
            else:
                for one_legal_path_with_label in legal_path_with_label_list:
                    all_legal_path_with_label.append(one_legal_path_with_label)
        relation_path_dict[path_len] = all_legal_path_with_label
        path_num = path_num // 2

    # print(relation_path_dict)
    save_path = hold_out_fold_path / 'train.RelationPath'
    if os.path.exists(save_path):
        with open(save_path, 'a') as f:
            for key, relation_path_with_label_list in relation_path_dict.items():
                all_std_format_rp = parse_rp(relation_path_with_label_list=relation_path_with_label_list,
                                             path_len=key)
                f.write(all_std_format_rp)
    else:
        with open(save_path, 'w') as f:
            for key, relation_path_with_label_list in relation_path_dict.items():
                all_std_format_rp = parse_rp(relation_path_with_label_list=relation_path_with_label_list,
                                             path_len=key)
                f.write(all_std_format_rp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_path_length', default=5, type=int)
    parser.add_argument('--min_path_length', default=2, type=int)
    parser.add_argument('--path_num', default=8000, type=int)
    parser.add_argument('--hold_out_num', default=1, type=int)
    #parser.add_argument('--save_path', default='../DATA/')
    args = parser.parse_args()
    args.graph_names = ['fb15k237']
    args.processed_path = Path('../DATA/processed')
    args.hold_out_fold = "hold_out_"
    args.files_suffix = ["train", "valid", "test"]
    #hold_out_fold_path = "../DATA/processed/fb15k237/test_case/fb15k237/hold_out_0/fb15k237.0.train.graph"
    # hold_out_fold_path = "../DATA/processed/fb15k237/hold_out_0/fb15k237.0.train.graph"
    for data_name in args.graph_names:
        for hold_out_id in range(args.hold_out_num):
            GenRelationPath(hold_out_rank=hold_out_id,
                            args_pt=args,
                            data_name=data_name)
            