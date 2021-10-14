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
from random import randint
import numpy as np
num = 100
n = 100  # 10 nodes
p = 2 * np.log(n) / n  # 20 edges



# some properties
# print("node degree clustering")
# for v in nx.nodes(G):
#     print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

# print the adjacency list
# for line in nx.generate_adjlist(G):
#     print(line)
np.set_printoptions(threshold=np.inf)
import csv

# 開啟輸出的 CSV 檔案
with open('Graph_dataset.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    writer.writerow(['GraphA', 'GraphB', 'ISO'])
    g_list = []
    a = 0
    for k in range(0, 100):
        G = nx.gnp_random_graph(n, p, seed=randint(1, 100))
        graph_np = nx.to_numpy_matrix(G)
        for i in range(0, num):
            x = randint(0, n - 1)
            y = randint(0, n - 1)
            copied_g_np = graph_np.copy()
            copied_g_np[[y, x]] = copied_g_np[[x, y]]
            copied_g_np[:, [y, x]] = copied_g_np[:, [x, y]]
            generated_graph_G = nx.from_numpy_array(copied_g_np)
            writer.writerow([graph_np, copied_g_np, nx.is_isomorphic(G, generated_graph_G)])
        for j in range(0, num):
            G_rand = nx.gnm_random_graph(n, p, seed=10)
            # print(nx.is_isomorphic(G, G_rand))
            if not (nx.is_isomorphic(G, G_rand)):
                # print(j)
                graph_rand_np = nx.to_numpy_matrix(G_rand)
                # print(graph_rand_np)
                writer.writerow([graph_np, graph_rand_np, nx.is_isomorphic(G, G_rand)])
print("complete")
