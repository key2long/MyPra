import networkx as nx
from pylab import show


class NxGraph(nx.MultiDiGraph):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def knbrs(self, start, k):
        undigraph = self.to_undirected()
        nbrs = set([start])
        for l in range(k):
            tmp = set((nbr for n in nbrs for nbr in undigraph[n]))
            nbrs = nbrs.union(tmp)
        return nbrs

    def demo_sub_graph(self, mid, k_nbr):
        """
        找出从mid出发,图中所有可达的节点；
        并将这些节点(包括mid节点本身)作为一个子图,画出来.
        :param mid: 出发节点的机器id, 是一个字符串
        :return: 画出一个子图
        """
        nodes = self.knbrs(mid, k_nbr)
        sub_graph = self.subgraph(nodes)
        self._draw_subgraph(sub_graph)

    def _draw_subgraph(self, sub_graph):
        print(f"一共有{len(sub_graph.nodes())}个节点")
        node_labels = nx.get_node_attributes(sub_graph, 'name')
        edge_labels = nx.get_edge_attributes(sub_graph, 'relation')
        pos = nx.spring_layout(sub_graph)
        nx.draw(sub_graph, pos)
        nx.draw_networkx_labels(sub_graph, pos=pos, labels=node_labels)
        for k, v in edge_labels.items():
            print(k, v)
        show()

    def demo_shortest_graph(self, head_mid, tail_mid):
        nodes = nx.shortest_path(self.to_undirected(), head_mid, tail_mid)
        sub_graph = self.subgraph(nodes)
        self._draw_subgraph(sub_graph)
