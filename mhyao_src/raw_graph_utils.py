from nxGraph import NxGraph
from pathlib import Path
import re


def parse_fb15k237_data(name, train_val_test, raw_dir):
    """
    用python内置的字典,列表等数据结构存储知识图谱；
    :param name: 知识图谱名字
    :param train_val_test: raw文件夹中三元组文件的后缀,表示是否为train/valid/test文件.
    :param raw_dir: 存储知识图谱文件所在文件夹(相对路径格式,不需要以"/"结尾).
    :return fact_list: [[...],[head_mid, relation, tail_mid],...,[...]].
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
                raise print(f"三元组读取错误:出现多于三个元素的事实；在第{factId+1}行.")
            head_mid = edge[0]
            multi_relation = edge[1].split(".")
            tail_mid = edge[2]
            for relation in multi_relation:
                fact_list.append([head_mid, relation, tail_mid])
    f.close()
    return fact_list


def parse_wn18rr_data(name, train_val_test, raw_dir):
    """
    用python内置的字典,列表等数据结构存储知识图谱；
    :param name: 知识图谱名字
    :param train_val_test: raw文件夹中三元组文件的后缀,表示是否为train/valid/test文件.
    :param raw_dir: 存储知识图谱文件所在文件夹(相对路径格式,不需要以"/"结尾).
    :return fact_list: [[...],[head_mid, relation, tail_mid],...,[...]].
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
            fact_list.append([head_mid, relation, tail_mid])
    f.close()
    return fact_list


def parse_yago310_data(name, train_val_test, raw_dir):
    """
    用python内置的字典,列表等数据结构存储知识图谱；
    :param name: 知识图谱名字
    :param train_val_test: raw文件夹中三元组文件的后缀,表示是否为train/valid/test文件.
    :param raw_dir: 存储知识图谱文件所在文件夹(相对路径格式,不需要以"/"结尾).
    :return fact_list: [[...],[head_mid, relation, tail_mid],...,[...]].
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
            fact_list.append([head_mid, relation, tail_mid])
    f.close()
    return fact_list


def py2nxGraph(fact_list):
    """
    将python格式的知识图谱数据转化为networkx格式的数据
    :return fact_list: [[...],[head_mid, relation, tail_mid],...,[...]].
    :return: nx_graph
    """
    nx_graph = NxGraph()
    for fact in fact_list:
        nx_graph.add_edge(fact[0],
                          fact[2],
                          relation=fact[1])
    return nx_graph


if __name__ == "__main__":
    parse_fb15k237_data(name="fb15k237", train_val_test='train', raw_dir=Path("../DATA/raw"))
    parse_wn18rr_data(name="WN18RR", train_val_test="train", raw_dir=Path("../DATA/raw"))
