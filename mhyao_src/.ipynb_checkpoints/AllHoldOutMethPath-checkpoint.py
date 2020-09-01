import pdb
from pathlib import Path
from mhyao_src.GraphManager import ProcessedGraphManager


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
            # valid_file_name = name + "." + str(hold_out_k) + "." + files_suffix[1] + "." + "graph"
            # graph_valid = ProcessedGraphManager(file_path=hold_out_fold_path / valid_file_name)
            # if len(graph_valid.entity_set_difference(graph_train)) != 0:
            #     print(f"There are {len(graph_valid.entity_set_difference(graph_train))}"
            #           f"entities that are missing in the training dataset.")
            graph_train.write_down_meta_paths(write_file_path=hold_out_fold_path / "train.MetaPaths",
                                              max_depth=2)
            # graph_train.merge(graph_valid)
            # graph_train.write_down_meta_paths(write_file_path=hold_out_fold_path / "train_valid.MetaPath",
            #                                   max_depth=3)
