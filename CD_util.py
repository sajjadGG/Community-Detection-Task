import numpy as np
import matplotlib.pyplot as plt
import random

from cdlib import algorithms, evaluation
import networkx as nx

from sklearn.metrics import normalized_mutual_info_score


def draw_graph_com(G, coms, fig_size=(8, 8)):
    """draw graph with community depicted in coms

    Args:
        G (networkx.classes.graph.Graph): Graph
        coms (list): list of list of size community and member of each list vertices belonging to the community
        fig_size (tuple, optional): figure size. Defaults to (8, 8).
    """

    color_map = {}  # map vertex to community to be used as color map
    for i, com in enumerate(coms):
        for v in com:
            color_map[v] = i
    plt.figure(figsize=fig_size)
    nx.draw(
        G,
        cmap=plt.get_cmap("viridis"),
        node_color=[color_map[v] for v in G.nodes],
        with_labels=True,
    )
    plt.show()


def to_community_list(G):
    """
    extract community vertex list from graph G have node attribute community

    Args:
        G (networkx.classes.graph.Graph): Graph

    Returns:
        list: list of list of size community and member of each list vertices belonging to the community
    """

    # Assumption: communities are disjoint
    l = []
    visited = set()
    for v in G:
        if v not in visited:
            temp = [node for node in G.nodes[v]["community"]]
            l.append(temp)
            for t in temp:
                visited.add(t)
    return l


def to_vertex_community(coms):
    """from comunity vertex list to vertex community list

    Args:
        coms (list): list of list of size community and member of each list vertices belonging to the community

    Returns:
        list: list of vertex denoting community
    """

    return [
        x[1]
        for x in sorted(
            [(node, nid) for nid, cluster in enumerate(coms) for node in cluster],
            key=lambda x: x[0],
        )
    ]


def normalized_mutual_info_score(coms_pred, coms_true):
    return normalized_mutual_info_score(
        to_vertex_community(coms_true), to_vertex_community(coms_pred)
    )
