import numpy as np
import matplotlib.pyplot as plt
import random

from cdlib import algorithms, evaluation
import networkx as nx

from sklearn.metrics import normalized_mutual_info_score
from networkx.generators.community import LFR_benchmark_graph as lfr

###------------------------###
### utils  ###
###------------------------###

import sys
import threading
from time import sleep

try:
    import thread
except ImportError:
    import _thread as thread


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print("{0} took too long".format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    """
    use as decorator to exit process if
    function takes longer than s seconds
    """

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer


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


def normalized_mutual_info_acc(coms_pred, coms_true):
    return normalized_mutual_info_score(
        to_vertex_community(coms_true), to_vertex_community(coms_pred)
    )


###------------------------###
### generate graph ###
###------------------------###


def generate_graph(n_low=250, n_high=10000):
    n = np.random.randint(n_low, n_high)
    tau1 = 3
    tau2 = 1.5
    mu = np.random.uniform(0.03, 0.75)
    max_degree = int(0.1 * n)
    max_community = int(0.1 * n)
    average_degree = 20
    G = lfr(
        n,
        tau1,
        tau2,
        mu,
        average_degree=average_degree,
        max_community=max_community,
        max_degree=max_degree,
        seed=10,
    )
    return G


@exit_after(5)
def generate_graph_full(n, tau1, tau2, mu, average_degree, min_community):
    return lfr(
        n, tau1, tau2, mu, average_degree=average_degree, min_community=min_community
    )


def generate_random_lfr(
    n_inter,
    tau1_inter,
    tau2_inter,
    mu_inter,
    average_degree_inter,
    min_community_inter,
    max_iter=10,
):
    generated = False
    max_iter = 10
    count = 1
    G = None
    while not generated or count < max_iter:
        count += 1
        try:
            n = np.random.randint(n_inter[0], n_inter[1])
            tau1 = np.random.uniform(tau1_inter[0], tau1_inter[1])
            tau2 = np.random.uniform(tau2_inter[0], tau2_inter[1])
            mu = np.random.uniform(mu_inter[0], mu_inter[1])
            average_degree = np.random.randint(
                average_degree_inter[0], average_degree_inter[1]
            )
            min_community = np.random.randint(
                min_community_inter[0], min_community_inter[1]
            )
            G = generate_graph(82, 3, 1.3, 0.18, average_degree=5, min_community=50)
            generated = True
        except:
            generated = False
            continue
    if not generated:
        raise ("Max iter exceeded")

    return G
