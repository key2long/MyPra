from mhyao_src.GraphManager import FB15k237RawGraph, WN18RRRawGraph, YAGO310RawGraph
import os


def get_fact_of_entity_set(factlist: list,
                           entity_set: frozenset):
    left_out_fact_tmp = []
    for fact_tmp in factlist:
        head_tmp = fact_tmp[0]
        tail_tmp = fact_tmp[2]
        if head_tmp in entity_set:
            left_out_fact_tmp.append(fact_tmp)
            continue
        if tail_tmp in entity_set:
            left_out_fact_tmp.append(fact_tmp)
    return left_out_fact_tmp


def get_entity_set(factlist: list):
    mid_count_dict = {}
    for one_fact in factlist:
        head = one_fact[0]
        tail = one_fact[2]
        if head in mid_count_dict:
            mid_count_dict[head] += 1
        else:
            mid_count_dict[head] = 1
        if tail in mid_count_dict:
            mid_count_dict[tail] += 1
        else:
            mid_count_dict[tail] = 1
    return mid_count_dict


def one_over_k_part(all_fact_list: list,
                    total_fold: int,
                    this_fold: int,
                    mode: str,):
    r_fact_dict = {}
    for fact in all_fact_list:
        head_mid = fact[0]
        relation = fact[1]
        tail_mid = fact[2]
        if relation in r_fact_dict:
            r_fact_dict[relation]["count"] += 1
            r_fact_dict[relation]["fact"].append([head_mid, tail_mid])
        else:
            r_fact_dict[relation] = {}
            r_fact_dict[relation]["count"] = 1
            r_fact_dict[relation]["fact"] = [[head_mid, tail_mid]]

    for relation in r_fact_dict.keys():
        r_fact_dict[relation]["fact"].sort()

    split_fold_list = [[] for i in range(total_fold)]
    for relation, count_fact_dict in r_fact_dict.items():
        for fold_j, fact in enumerate(count_fact_dict["fact"]):
            which_fold_tmp = fold_j % len(split_fold_list)
            head_mid = fact[0]
            tail_mid = fact[1]
            split_fold_list[which_fold_tmp].append([head_mid, relation, tail_mid])

    fold_entity_set_list = []
    for fold_j, fact_list in enumerate(split_fold_list):
        entity_dict = get_entity_set(fact_list)
        entity_set = frozenset(entity_dict.keys())
        fold_entity_set_list.append(entity_set)

    unique_e_in_this_e_set = frozenset()
    set_of_all_e_sets = frozenset(fold_entity_set_list)
    this_e_set = fold_entity_set_list[this_fold]
    set_of_this_e_set = frozenset([this_e_set])
    set_of_rest_e_sets = set_of_all_e_sets - set_of_this_e_set
    union_of_rest_e_sets = frozenset()
    for other_e_set in set_of_rest_e_sets:
        union_of_rest_e_sets = union_of_rest_e_sets | other_e_set
    unique_e_in_this_e_set = this_e_set - union_of_rest_e_sets
    if mode == "test":
        print(f"\t第{this_fold + 1}份数据中有{len(unique_e_in_this_e_set)}个遗漏实体；遗漏实体重新分配.")
    elif mode == "valid":
        if this_fold != 0:
            print(f"\t\t第{(this_fold+1) + 1}份数据中有{len(unique_e_in_this_e_set)}个遗漏实体；遗漏实体重新分配.")
        else:
            print(f"\t\t第{1}份数据中有{len(unique_e_in_this_e_set)}个遗漏实体；遗漏实体重新分配.")

    this_fold_fact_list = split_fold_list[this_fold]
    rest_fold_fact_list = []
    left_out_fact_list = get_fact_of_entity_set(this_fold_fact_list, unique_e_in_this_e_set)
    for left_out_fact in left_out_fact_list:
        this_fold_fact_list.remove(left_out_fact)
        rest_fold_fact_list.append(left_out_fact)

    for fold in range(len(split_fold_list)):
        if fold == this_fold:
            continue
        rest_fold_fact_list += split_fold_list[fold]

    return this_fold_fact_list, rest_fold_fact_list


def write_down_fact_list(file_name: str,
                         fact_list: list):
    with open(file_name, "w") as f:
        for triple in fact_list:
            content = ""
            for tmp in triple:
                content += tmp + "\t"
            content += "\n"
            f.write(content)
    f.close()


if __name__ == "__main__":
    data_dir = "../DATA"
    train = "train"
    valid = "valid"
    test = "test"
    fb15k237 = FB15k237RawGraph(root_dir=data_dir,
                                train_val_test=train)
    fb15k237_valid = FB15k237RawGraph(root_dir=data_dir,
                                      train_val_test=valid)
    fb15k237_test = FB15k237RawGraph(root_dir=data_dir,
                                     train_val_test=test)
    wn18rr = WN18RRRawGraph(root_dir=data_dir,
                            train_val_test=train)
    wn18rr_valid = WN18RRRawGraph(root_dir=data_dir,
                                  train_val_test=valid)
    wn18rr_test = WN18RRRawGraph(root_dir=data_dir,
                                 train_val_test=test)
    yago = YAGO310RawGraph(root_dir=data_dir,
                           train_val_test=train)
    yago_valid = YAGO310RawGraph(root_dir=data_dir,
                                 train_val_test=valid)
    yago_test = YAGO310RawGraph(root_dir=data_dir,
                                train_val_test=test)
    fb15k237.fact_list += fb15k237_valid.fact_list + fb15k237_test.fact_list
    wn18rr.fact_list += wn18rr_valid.fact_list + wn18rr_test.fact_list
    yago.fact_list += yago_valid.fact_list + yago_test.fact_list
    graphs = [fb15k237, wn18rr, yago]
    split_k = 10
    for graph in graphs:
        graph_folder_path = graph.processed_dir / graph.name
        if os.path.exists(graph_folder_path) is False:
            os.makedirs(graph_folder_path)
        print(f"正在处理{graph.name};")
        for fold in range(split_k):
            test_fact, valid_train_fact = one_over_k_part(all_fact_list=graph.fact_list,
                                                          total_fold=split_k,
                                                          this_fold=fold,
                                                          mode="test",)
            print(f"\t\t以第{fold + 1}份数据生成test集,共{len(test_fact) / len(graph.fact_list) * 100:.3}%条三元组.")
            hold_out_path = graph_folder_path / f"hold_out_{fold}"
            if os.path.exists(hold_out_path) is False:
                os.makedirs(hold_out_path)
            file_name = hold_out_path / (graph.name + f".{fold}.test.graph")
            write_down_fact_list(file_name, test_fact)
            valid_fact, train_fact = one_over_k_part(all_fact_list=valid_train_fact,
                                                     total_fold=split_k - 1,
                                                     this_fold=fold % (split_k - 1),
                                                     mode="valid",)
            print(f"\t\t以第{(fold+1)%split_k + 1}份数据生成valid集,共{len(valid_fact) / len(graph.fact_list) * 100:.3}%条三元组.")
            file_name = hold_out_path / (graph.name + f".{fold}.valid.graph")
            write_down_fact_list(file_name, valid_fact)
            print(f"\t\t以剩余数据生成train集,共{len(train_fact) / len(graph.fact_list) * 100:.3}%条三元组.")
            file_name = hold_out_path / (graph.name + f".{fold}.train.graph")
            write_down_fact_list(file_name, train_fact)
