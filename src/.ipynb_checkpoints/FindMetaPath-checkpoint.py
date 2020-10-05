# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from ProcessedGraph import ProcessedGraph
from utils import find_and_save_meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_num', default=100, type=int)
    parser.add_argument('--max_depth', default=4, type=int)
    parser.add_argument('--hit_appear_ratio', default=1e-1, type=int)
    args = parser.parse_args()
    # graph_names = ["fb15k237", "WN18RR", "YAGO3_10"]
    graph_names = ["fb15k237"]
    processed_path = Path("../DATA/processed")
    # processed_path = Path("../DATA/processed/fb15k237/test_case")  # 做测试代码时使用
    hold_out_num = 10
    for name in graph_names:
        for hold_out_id in range(hold_out_num):
            hold_out_fold_path = processed_path / name / ("hold_out_" + str(hold_out_id))
            graph_file_path = hold_out_fold_path / (name + "." + str(hold_out_id) + ".train.graph")
            train_graph = ProcessedGraph(file_path=graph_file_path)

            save_path = hold_out_fold_path / ("train.MetaPath."+str(args.max_depth))
            find_and_save_meta_path(query_graph_pt=train_graph,
                                    process_num=args.process_num,
                                    max_depth=args.max_depth,
                                    hit_appear_ratio=args.hit_appear_ratio,
                                    save_path=save_path)


if __name__ == "__main__":
    main()
