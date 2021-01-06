# -*- coding: utf-8 -*-
import argparse
import traceback
from pathlib import Path
from utils import MyPool, handle_error
from ProcessedGraph import ProcessedGraph
from Experiment import PRATrain, PRAValid
import sys
sys.path.append("../FeatureBuffer")
from FeatureBuffer import ValidFeatureBuffer


def one_hold_out_experiment(hold_out_rank: int,
                            args_pt: argparse,
                            data_name: str):
    """Using Valid data to search the best hyper param, and based on this param, a model will trained using train + valid data.

    Every hold out will have one process. Different hold out test in parallel.

    :param hold_out_rank: int. The rank of this hold out.
    :param args_pt: argparse. Contains arguments like feature_size, et,el.
    :param data_name: str. Data set name.
    :return: None
    """
    try:
        # Load training data. The train_graph will also generate feature_matrix and the labels in torch.tensor format.
        hold_out_fold_path = args_pt.processed_path / data_name / (args_pt.hold_out_fold + str(hold_out_rank))
        args_pt.train_log_path = hold_out_fold_path
        train_graph_name = data_name + "." + str(hold_out_rank) + "." + args_pt.files_suffix[0] + "." + "graph"
        train_graph = ProcessedGraph(file_path=hold_out_fold_path / train_graph_name)

        # 更换数据集时，需要稍微修改一下此处的代码
        neg_graph = ProcessedGraph(file_path=args_pt.processed_path / data_name / "fb15k237.neg.graph")
        valid_graph_name = data_name + "." + str(hold_out_rank) + "." + args_pt.files_suffix[1] + "." + "graph"
        valid_graph = ProcessedGraph(file_path=hold_out_fold_path / valid_graph_name)
        valid_feature_buffer = ValidFeatureBuffer(csv_path=args_pt.valid_feature_path)

        # Search through the hyper parameter's grid to find the best one using validation graph.
        hyper_param_grid = [{"lr":0.01}] #0.1,0.005,0.001
        for id, hyper_param in enumerate(hyper_param_grid):
            one_hold_out_experiment = PRATrain(query_graph_pt=train_graph,
                                               hold_out_path=hold_out_fold_path,
                                               meta_path_file="train.MetaPath",
                                               args_pt=args_pt,
                                               neg_graph_pt=neg_graph,
                                               hyper_param=hyper_param)
            one_hold_out_experiment.train_this_hold_out(hold_out_rank)
            valid_experiment = PRAValid(query_graph_pt=train_graph,
                                        hold_out_path=hold_out_fold_path,
                                        meta_path_file="train.MetaPath",
                                        args_pt=args_pt,
                                        predict_graph_pt=valid_graph,
                                        valid_feature_buffer_pt=valid_feature_buffer,
                                        model_pt=one_hold_out_experiment.model_pt)
            valid_experiment.tail_predict(hold_out_rank)
    except:
        stack_error = traceback.format_exc()
        raise Exception(stack_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="PRA", type=str)
    parser.add_argument('--hold_out_num', default=1, type=int)
    parser.add_argument('--gpu_num_per_hold', default=2, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--feature_size', default=500, type=int)
    parser.add_argument('--pos_samp_size', default=500, type=int)
    # parser.add_argument('--valid_feature_path', default="../FeatureBuffer/YAGO3_10.valid.feature.csv", type=str)
    # parser.add_argument('--valid_feature_path', default="../FeatureBuffer/WN18RR.valid.feature.csv", type=str)
    parser.add_argument('--valid_feature_path', default="../FeatureBuffer/fb15k237.valid.feature_10.csv", type=str)
    parser.add_argument('--epoch', default=5000, type=int)
    args = parser.parse_args()
    args.graph_names = ['fb15k237']  # args.graph_names = ["fb15k237", "WN18RR", "YAGO3_10"]
    args.processed_path = Path("../DATA/processed")
    # args.processed_path = Path("../DATA/processed/WN18RR/test_case")  # 做测试代码时使用 # args.processed_path = Path("../DATA/processed")
    args.hold_out_fold = "hold_out_"
    args.files_suffix = ["train", "valid", "test"]
    args.hit_range = [1, 3, 10]
    for name in args.graph_names:
        process_pool = MyPool(processes=args.hold_out_num)
        for hold_out_id in range(args.hold_out_num):
            process_pool.apply_async(func=one_hold_out_experiment,
                                     args=(hold_out_id,
                                           args,
                                           name,),
                                     error_callback=handle_error)
        process_pool.close()
        process_pool.join()


if __name__ == "__main__":
    main()
