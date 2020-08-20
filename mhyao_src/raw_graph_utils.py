from mhyao_src.nxGraph import NxGraph
from pathlib import Path
import re


def parse_fb15k237_data(name, train_val_test, raw_dir):
    """
    用python内置的字典,列表等数据结构存储知识图谱；
    默认该存储该图谱的文件以"train"作为文件后缀.
    :param:
        name: 知识图谱名字
        raw_dir: 存储知识图谱文件所在文件夹(相对路径格式,不需要以"/"结尾).
    :return:
        mid_nid_dict: machineId-->nodeId. 详情见行间注释；
        nid_mid_dict: nodeId-->machineId. 详情见行间注释；
        relation_rid_dict: relation-->{"rid": rid, "count": count}. 详情见行间注释；
        rid_relation_dict: rid-->{"relation": relation, "count": count}. 详情见行间注释;
        fact_list: [{...},{"head_mid": head_mid, "tail_mid": tail_mid, "key": rid, "relation": relation},...,{...}].
    """
    # setup paths
    graph = raw_dir / f"{name}.{train_val_test}"
    print(f"网络名称为:{graph};")

    # domain_dId_dict 以KG中关系所属的/domain作为key;以/domain的编号dId为value.数据格式如下:
    # domain_dId_dict = { "/domain1" : 0,
    #                     "..."      : 1,
    #                      ...
    #                     "..."      : N }
    # 简单记做: domain_dId_dict
    domain_dId_dict = {}

    # type_tId_dict 以KG中关系所属的/type作为key;以/type的编号tId为value.数据格式如下:
    # type_tId_dict = { "/type1" : 0,
    #                   "..."    : 1,
    #                    ...
    #                   "..."    : N }
    # 简单记做: type_tId_dict
    type_tId_dict = {}

    # machineId_nodeId_dict 以KG中实体的机器码/m/*作为key;以实体的编号nodeId为value.数据格式如下:
    # machineId_nodeId_dict = { "/m/049vl7" : 0,
    #                           "..."       : 1,
    #                            ...
    #                           "..."       : N }
    # 简单记做: mid_nid_dict
    mid_nid_dict = {}

    # nodeId_machineId_dict 是 mid_nid_dict的反转.数据格式如下:
    # nodeId_machineId_dict = { 0 : "/m/049vl7",
    #                           1 :       "...",
    #                            ...
    #                           N :       "...",}
    # 简单记做: nid_mid_dict
    nid_mid_dict = {}

    # relation_rid_dict 以KG中关系的名字/domain/type/properties作为key;
    # 以另一个字典{"rid":,"count":}为value.该字典主要给各关系编号,并记录出现频次.数据格式如下:
    # relation_rid_dict = { "/d/t/p0" : {"rid":0, "count":103},
    #                       "..."     : {"rid":1, "count":...},
    #                        ...
    #                       "..."     : {"rid":R, "count":...},}
    relation_rid_dict = {}

    # rid_relation_dict 以KG中关系的编号rid作为key;
    # 以另一个字典{"relation":"/d/t/p","count":}为value.数据格式如下:
    # rid_relation_dict = { 0 : {"relation":"/d/t/p0", "count":103},
    #                       1 : {"relation":"...",     "count":...},
    #                        ...
    #                       R : {"relation":"...",     "count":...},}
    rid_relation_dict = {}

    # 过第一遍,统计上述前五个字典
    nid = 0
    did = 0
    tid = 0
    try:
        with open(graph, "r") as f:
            for factId, fact in enumerate(f.readlines()):
                head_relation_tail = fact.rstrip("\n")
                # re.split支持正则表达式,按多个标识符切割,\t表示tab；\s表示一个空格；\s*表示多个空格
                edge = [item for item in re.split(r"[\t\s]\s*", head_relation_tail)]
                if len(edge) != 3:
                    raise print(f"三元组读取错误:出现多于三个元素的事实；在第{factId+1}行.")
                head_mid = edge[0]
                multi_relation = edge[1].split(".")
                tail_mid = edge[2]
                # 向4个字典存储
                if head_mid not in mid_nid_dict:
                    mid_nid_dict[head_mid] = nid
                    nid_mid_dict[nid] = head_mid
                    nid += 1
                if tail_mid not in mid_nid_dict:
                    mid_nid_dict[tail_mid] = nid
                    nid_mid_dict[nid] = tail_mid
                    nid += 1
                for relation_tmp in multi_relation:
                    r_split = relation_tmp.split("/")  # ['', "/domain", "/type", "/property"]
                    if r_split[1] not in domain_dId_dict:  # 填充domain
                        domain_dId_dict[r_split[1]] = did
                        did += 1
                    if r_split[2] not in type_tId_dict:  # 填充type
                        type_tId_dict[r_split[2]] = tid
                        tid += 1
                    relation = "/" + str(domain_dId_dict[r_split[1]]) + \
                               "/" + str(type_tId_dict[r_split[2]])
                    for tmp in r_split[3:]:
                        relation += "/" + tmp
                    # relation = "/" + r_split[1] + \
                    #            "/" + r_split[2] + \
                    #            "/" + r_split[3]
                    if relation not in relation_rid_dict:
                        relation_rid_dict[relation] = {"count": 1}
                    else:
                        relation_rid_dict[relation]["count"] += 1
        f.close()
    except:
        print(f"上述try模块的某行出现了错误")
    print(f"共有{len(mid_nid_dict)}个实体；\n实体编号最大为nid={nid-1}.")
    print(f"共有{len(domain_dId_dict)}种domain;")
    print(f"共有{len(type_tId_dict)}种type;")
    print(f"共有{len(relation_rid_dict)}种关系;")
    # 填充第4个字典
    reriddict_sort_by_count_list = sorted(relation_rid_dict.items(), key=lambda item: item[1]["count"], reverse=True)
    tmp = 0
    for rid, (relation, count_dict) in enumerate(reriddict_sort_by_count_list):
        relation_rid_dict[relation]["rid"] = rid
        tmp = rid
        rid_relation_dict[rid] = {"relation": relation, "count":count_dict["count"]}
        # print(f"{rid+1}:{rid_relation_dict[rid]}")
    print(f"关系编号最大为rid={tmp}.")

    # 过第二遍, fact_list记录了所有的三元组, 每条三元组的格式如下:
    # fact_list = [ {"head_mid":"/m/fea2r", "tail_mid":"/m/43wgw", "key":rid, "relation":"/d/t/p"},
    #               {...},...,{...}]
    fact_list = []
    try:
        with open(graph, "r") as f:
            for factId, fact in enumerate(f.readlines()):
                head_relation_tail = fact.rstrip("\n")
                # re.split支持正则表达式,按多个标识符切割,\t表示tab；\s表示一个空格；\s*表示多个空格
                edge = [item for item in re.split(r"[\t\s]\s*", head_relation_tail)]
                if len(edge) != 3:
                    raise print(f"三元组读取错误:出现多于三个元素的事实；在第{factId+1}行.")
                head_mid = edge[0]
                multi_relation = edge[1].split(".")
                tail_mid = edge[2]
                for relation_tmp in multi_relation:
                    r_split = relation_tmp.split("/")
                    relation = "/" + str(domain_dId_dict[r_split[1]]) + \
                               "/" + str(type_tId_dict[r_split[2]])
                    for tmp in r_split[3:]:
                        relation += "/" + tmp
                    # relation = "/" + r_split[1] + \
                    #            "/" + r_split[2] + \
                    #            "/" + r_split[3]
                    fact_list.append({"head_mid": head_mid,
                                      "tail_mid": tail_mid,
                                      "key": relation_rid_dict[relation]["rid"],
                                      "relation": relation})
        f.close()
    except:
        print(f"上述try模块的某行出现了错误")

    return mid_nid_dict, nid_mid_dict, relation_rid_dict, rid_relation_dict, fact_list


def py2nxGraphStyle(mid_nid_dict, fact_list):
    """
    将python格式的知识图谱数据转化为networkx格式的数据
    :param mid_nid_dict: machineId-->nodeId.
    :param fact_list: [{...},{"head_mid": head_mid, "tail_mid": tail_mid, "key": rid, "relation": relation},...,{...}]
    :return: nx_graph
    """
    nx_graph = NxGraph()
    # 添加节点,元素为mid
    for mid,_ in mid_nid_dict.items():
        nx_graph.add_node(mid, name=mid)

    # 添加边
    for fact in fact_list:
        nx_graph.add_edge(fact["head_mid"],
                          fact["tail_mid"],
                          key=fact["key"],
                          relation=fact["relation"])
    return nx_graph


def parse_wn18rr_data(name, train_val_test, raw_dir):
    """
    用python内置的字典,列表等数据结构存储知识图谱；
    默认该存储该图谱的文件以"train"作为文件后缀.
    :param:
        name: 知识图谱名字
        raw_dir: 存储知识图谱文件所在文件夹(相对路径格式,不需要以"/"结尾).
    :return:
        fact_list: [{...},{"head_mid": head_mid, "tail_mid": tail_mid, "relation": relation},...,{...}].
    """
    # setup paths
    graph = raw_dir / f"{name}.{train_val_test}"
    print(f"网络名称为:{graph};")
    fact_list = []
    with open(graph, "r") as f:
        for factId, fact in enumerate(f.readlines()):
            head_relation_tail = fact.rstrip("\n")
            # re.split支持正则表达式,按多个标识符切割,\t表示tab；\s表示一个空格；\s*表示多个空格
            edge = [item for item in re.split(r"[\t\s]\s*", head_relation_tail)]
            if len(edge) != 3:
                raise print(f"三元组读取错误:出现多于三个元素的事实；在第{factId + 1}行.")
            head_mid = edge[0]
            relation = edge[1][1:]
            tail_mid = edge[2]
            fact_list.append({
                "head_mid": head_mid,
                "tail_mid": tail_mid,
                "relation": relation
            })
    f.close()
    return fact_list


def parse_yago310_data(name, train_val_test, raw_dir):
    """
        用python内置的字典,列表等数据结构存储知识图谱；
        默认该存储该图谱的文件以"train"作为文件后缀.
        :param:
            name: 知识图谱名字
            raw_dir: 存储知识图谱文件所在文件夹(相对路径格式,不需要以"/"结尾).
        :return:
            fact_list: [{...},{"head_mid": head_mid, "tail_mid": tail_mid, "relation": relation},...,{...}].
        """
    # setup paths
    graph = raw_dir / f"{name}.{train_val_test}"
    print(f"网络名称为:{graph};")
    fact_list = []
    with open(graph, "r") as f:
        for factId, fact in enumerate(f.readlines()):
            head_relation_tail = fact.rstrip("\n")
            # re.split支持正则表达式,按多个标识符切割,\t表示tab；\s表示一个空格；\s*表示多个空格
            edge = [item for item in re.split(r"[\t\s]\s*", head_relation_tail)]
            if len(edge) != 3:
                raise print(f"三元组读取错误:出现多于三个元素的事实；在第{factId + 1}行.")
            head_mid = edge[0]
            relation = edge[1]
            tail_mid = edge[2]
            fact_list.append({
                "head_mid": head_mid,
                "tail_mid": tail_mid,
                "relation": relation
            })
    f.close()
    return fact_list


def py2nxGraph(fact_list):
    """
    将python格式的知识图谱数据转化为networkx格式的数据
    :param mid_nid_dict: machineId-->nodeId.
    :param fact_list: [{...},{"head_mid": head_mid, "tail_mid": tail_mid, "key": rid, "relation": relation},...,{...}]
    :return: nx_graph
    """
    nx_graph = NxGraph()
    for fact in fact_list:
        nx_graph.add_edge(fact["head_mid"],
                          fact["tail_mid"],
                          relation=fact["relation"])
    return nx_graph


if __name__ == "__main__":
    parse_fb15k237_data(name="fb15k237", train_val_test='train', raw_dir=Path("../DATA/raw"))
    parse_wn18rr_data(name="WN18RR", train_val_test="train", raw_dir=Path("../DATA/raw"))
