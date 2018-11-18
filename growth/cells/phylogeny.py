import networkx as nx
import matplotlib.pyplot as plt
from ete3 import Tree, TreeStyle


class Dendrogram:

    def __init__(self, newick):
        self.newick = newick



class Phylogeny:

    def __init__(self, edge_list):
        self.networkx = nx.DiGraph(edge_list)
        self.newick = self.networkx_to_newick(self.networkx)

    @staticmethod
    def bfs_edge_list(graph, root):
        """ Returns breadth search first edge list starting at <root>. """
        return list(nx.bfs_edges(graph, root))

    @classmethod
    def networkx_to_newick(cls, G):
        """ Convert <G> to Newick format. """

        edge_list = cls.bfs_edge_list(G, '')
        tree = cls.tree_from_edges(edge_list)
        return cls.tree_to_newick(tree) + ';'

    @classmethod
    def recursive_search(cls, adict, key):
        """ Recursively search <adict> for <key>. """
        if key in adict:
            return adict[key]
        for k, v in adict.items():
            item = cls.recursive_search(v, key)
            if item is not None:
                return item

    @classmethod
    def tree_from_edges(cls, edges):
        """ Returns tree dictionary from <edges> list. """
        tree = {'': {}}
        for src, dst in edges:
            subt = cls.recursive_search(tree, src)
            subt[dst] = {}
        return tree

    @classmethod
    def tree_to_newick(cls, tree):
        """ Returns Newick formatted string for <tree>. """
        items = []
        for k in tree.keys():
            s = ''
            if len(tree[k].keys()) > 0:
                subt = cls.tree_to_newick(tree[k])
                if subt != '':
                    s += '(' + subt + ')'
            s += k
            items.append(s)
        return ','.join(items)
