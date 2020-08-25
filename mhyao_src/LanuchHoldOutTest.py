import pdb
from pathlib import Path
from mhyao_src.GraphManager import ProcessedGraphManager
from mhyao_src.Experiments import PRATrain, Validation, Test


if __name__ == "__main__":
    graph_names = ["fb15k237", "WN18RR", "YAGO3_10"]
    processed_path = Path("../DATA/processed")
    hold_out_fold = "hold_out_"
    files_suffix = ["train", "valid", "test"]
    split_k = 10
    hit_range = 10
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
            valid_results = []
            for id, alpha in enumerate(alpha_grid):
                one_hold_out_experiment = PRATrain(query_graph=graph_train,
                                                   neg_pairs_path=processed_path / name / "fb15k237.neg.graph",
                                                   meta_path_file=hold_out_fold_path / "train.MetaPaths",
                                                   hold_out_path=hold_out_fold_path,
                                                   hyper_param=alpha)
                one_hold_out_experiment.train_this_hold_out()
                valid_experiment = Validation(model_pt=one_hold_out_experiment.model_pt,
                                              query_graph_pt=graph_train,
                                              predict_graph_pt=graph_valid,
                                              hit_range=hit_range)
                result_tuple = valid_experiment.tail_predict()
                print(f"Validating hyper-parameter [{id/len(alpha_grid)}]: "
                      f"\thit@{hit_range}'s accuracy; MR; MRR: {result_tuple}.")
                valid_results.append((id, result_tuple))
            test_file_name = name + "." + str(hold_out_k) + "." + files_suffix[2] + "." + "graph"
            graph_test = ProcessedGraphManager(file_path=hold_out_fold_path / test_file_name,
                                               if_add_reverse_relation=False)
            test_experiment = Test(query_graph_pt=graph_train,
                                   predict_graph_pt=graph_test,
                                   hit_range=hit_range)
            hyper_param_id, _ = test_experiment.find_best_results(valid_results)
            train_valid_experiment = PRATrain(query_graph=graph_train,
                                              neg_pairs_path=processed_path / name / "fb15k237.neg.graph",
                                              meta_path_file=hold_out_fold_path / "train.MetaPaths",
                                              hold_out_path=hold_out_fold_path,
                                              hyper_param=alpha_grid[hyper_param_id])
            train_valid_experiment.load_model_from_file()