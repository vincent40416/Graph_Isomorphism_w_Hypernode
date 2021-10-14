# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
===========
Erdos Renyi
===========

Create an G{n,m} random graph with n nodes and m edges
and report some properties.

This graph is sometimes called the Erdős-Rényi graph
but is different from G{n,p} or binomial_graph which is also
sometimes called the Erdős-Rényi graph.
"""
# Author: Aric Hagberg (hagberg@lanl.gov)

#    Copyright (C) 2004-2019 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.

import matplotlib.pyplot as plt
from networkx import nx
from igraph import Graph
from random import randint
import numpy as np
import csv

<<<<<<< HEAD:erdos_renyi_generator.py
num = 10000
n = 100  # 10 nodes
p = 2 * np.log(n) / n  # 20 edges
=======
num = 500
n = 500  # nodes num
p = 2 * np.log(n) / n  # 20 edges
m = 45
>>>>>>> vincent_branch:erdos_renyi_generator1.py
np.set_printoptions(threshold=np.inf)

Affinity = np.full((n+1, n+1), -1, dtype=float)
for i in range(n+1):
    Affinity[i][i] = 1


def swap(matrix, i, j):
    """
    Swap row and column i and j in-place.

    Examples
    --------
    >>> matrix = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> swap(matrix, 2, 0)
    array([[8, 7, 6], [0,1,2]
           [5, 4, 3],
           [2, 1, 0]])
    """
    # swap columns
    copy = matrix[:, i].copy()
    matrix[:, i] = matrix[:, j]
    matrix[:, j] = copy
    # swap rows
    copy = matrix[i, :].copy()
    matrix[i, :] = matrix[j, :]
    matrix[j, :] = copy
    return matrix


def aff_swap(matrix, i, j):
    # print('{0}  {1}'.format(i, j))
    copy = matrix[:, i].copy()
    matrix[:, i] = matrix[:, j]
    matrix[:, j] = copy
    # print(matrix)
    return


def convert_to_auto(aff, mapping):
    # print(mapping[0])
    for i in mapping[1:]:
        # print(i)
        for index in range(len(i)):
            # print('{0}  {1}'.format(index, i[index]))
            if(index != i[index]):
                aff[i[index]][index] = 1
                aff[index][i[index]] = 1

    # print(aff)
    return aff


print("start genrating")

with open('Graph_dataset_hypernode_' + str(n) + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['GraphA', 'GraphB', 'Affinity'])
    g_list = []
    for k in range(num):
<<<<<<< HEAD:erdos_renyi_generator.py
        G = Graph.Erdos_Renyi(n=n, p=p)
        A = G.get_adjacency()
=======
        if k > 9 * num / 10:
            G = nx.circulant_graph(n=n, offset = [1, 2])
            # print(networkx_Gmatrix)
        elif k > 3 * num / 10:
            G = nx.erdos_renyi_graph(n=n, p=p)
        elif k > 2 * num / 10:
            G = nx.generators.classic.path_graph(n=n)
            a = randint(0, n - 2)
            G.remove_edge(a, a + 1)
        else:
            G = nx.generators.trees.random_tree(n=n)

        list = [(G.number_of_nodes() + 1, i) for i in range(G.number_of_nodes())]
        G.add_edges_from(list)

        IG = Graph.from_networkx(G)
        A = IG.get_adjacency()
        # print(A)
>>>>>>> vincent_branch:erdos_renyi_generator1.py
        graph_np = np.array(A.data, dtype=float)
        # print(graph_np)
        automor = IG.get_automorphisms_vf2()
        # print(automor)
        for i in range(0, 1):
            Aff = Affinity.copy()
            convert_to_auto(Aff, automor)
            gern_np = graph_np.copy()
<<<<<<< HEAD:erdos_renyi_generator.py
            swaps = 50
=======
            swaps = int(n**2 / 2)
>>>>>>> vincent_branch:erdos_renyi_generator1.py
            for step in range(swaps):
                a = randint(0, n - 1)
                b = a
                while (b == a):
                    b = randint(0, n - 1)
                # print(a, b)
                gern_np = swap(gern_np, a, b)
                aff_swap(Aff, a, b)
            # print(graph_np)
            # print(gern_np)
            # print(Aff)
            writer.writerow([graph_np, gern_np, Aff])
        # for j in range(0, num):
        #     G_rand = nx.gnm_random_graph(n, m, seed=10)
        #     # print(nx.is_isomorphic(G, G_rand))
        #     if not (nx.is_isomorphic(G, G_rand)):
        #         # print(j)
        #         graph_rand_np = nx.to_numpy_matrix(G_rand)
        #         # print(graph_rand_np)
        #         writer.writerow([graph_np, graph_rand_np, nx.is_isomorphic(G, G_rand)])

print("complete")
