import pdb
from pathlib import Path
from mhyao_src.GraphManager import ProcessedGraphManager
from mhyao_src.Experiments import PRATrain


if __name__ == "__main__":
    graph_names = ["fb15k237", "WN18RR", "YAGO3_10"]
    processed_path = Path("../DATA/processed")
    hold_out_fold = "hold_out_"
    files_suffix = ["train", "valid"]
    split_k = 10
    for name in graph_names:
        for hold_out_k in range(split_k):
            hold_out_fold_path = processed_path / name / (hold_out_fold + str(hold_out_k))
            train_file_name = name + "." + str(hold_out_k) + "." + files_suffix[0] + "." + "graph"
            graph_train = ProcessedGraphManager(file_path=hold_out_fold_path / train_file_name,
                                                if_add_reverse_relation=False)
            valid_file_name = name + "." + str(hold_out_k) + "." + files_suffix[1] + "." + "graph"
            graph_valid = ProcessedGraphManager(file_path=hold_out_fold_path / valid_file_name,
                                                if_add_reverse_relation=False)
            if len(graph_valid.entity_set_difference(graph_train)) != 0:
                print(f"There are {len(graph_valid.entity_set_difference(graph_train))}"
                      f"entities that are missing in the training dataset.")
            # 载入可能的超参数
            # alpha_grid = [0.1, 0.2, 0.25, 0.3, 0.35]
            alpha_grid = [0.2]
            for alpha in alpha_grid:
                one_hold_out_experiment = PRATrain(query_graph=graph_train,
                                                   neg_pairs_path=processed_path / name / "fb15k237.neg.graph",
                                                   meta_path_file=hold_out_fold_path / "train.MetaPaths",
                                                   hold_out_path=hold_out_fold_path,
                                                   hyper_param=alpha)
                one_hold_out_experiment.train_this_hold_out()
