from GraphManager import FB15k237RawGraph
from pathlib import Path
from SplitGraph import one_over_k_part, write_down_fact_list
import os


if __name__ == "__main__":
    save_path = Path("../DATA/processed/fb15k237/test_case/fb15k237")
    graph = FB15k237RawGraph(root_dir=save_path,
                             train_val_test="train")
    split_k = 10
    for fold in range(split_k):
        test_fact, valid_train_fact = one_over_k_part(all_fact_list=graph.fact_list,
                                                      total_fold=split_k,
                                                      this_fold=fold,
                                                      mode="test", )
        print(f"\t\t以第{fold + 1}份数据生成test集,共{len(test_fact) / len(graph.fact_list) * 100:.3}%条三元组.")
        hold_out_path = save_path / f"hold_out_{fold}"
        if os.path.exists(hold_out_path) is False:
            os.makedirs(hold_out_path)
        file_name = hold_out_path / ("fb15k237" + f".{fold}.test.graph")
        write_down_fact_list(file_name, test_fact)
        valid_fact, train_fact = one_over_k_part(all_fact_list=valid_train_fact,
                                                 total_fold=split_k - 1,
                                                 this_fold=fold % (split_k - 1),
                                                 mode="valid", )
        print(f"\t\t以第{(fold + 1) % split_k + 1}份数据生成valid集,共{len(valid_fact) / len(graph.fact_list) * 100:.3}%条三元组.")
        file_name = hold_out_path / ("fb15k237" + f".{fold}.valid.graph")
        write_down_fact_list(file_name, valid_fact)
        print(f"\t\t以剩余数据生成train集,共{len(train_fact) / len(graph.fact_list) * 100:.3}%条三元组.")
        file_name = hold_out_path / ("fb15k237" + f".{fold}.train.graph")
        write_down_fact_list(file_name, train_fact)
