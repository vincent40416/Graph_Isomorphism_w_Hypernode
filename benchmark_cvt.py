from cnfgen import readGraph
from igraph import *
import numpy as np
import networkx as nx
import glob
import csv
import pandas as pd

path = 'Dataset/'
files = []
np.set_printoptions(threshold=np.inf)
# benchmark = pd.read_csv('./time_benchmark_False.csv')
for filename in glob.iglob(path + 'cfi-*-[r|t]2/*', recursive=False): #'cfi-*-[d|s|z]*/*'
    # print(filename)
    files.append(filename)

    # break
files.sort()
count = 0
# with open('./Dataset/Benchmark_False_sample.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['GraphA', 'GraphB', 'is_ISO'])
for i, f in enumerate(files):
    if i >= 10:
        break
    if i % 2 == 0:

        # print(i)
        # print(files[i])
        # print(files[i+1])
        G1 = readGraph(files[i], graph_type='simple', file_format='dimacs')
        list = [(0, i) for i in range(G1.number_of_nodes()+1)]
        G1.add_edges_from(list)
        # benchmark['Number of Node'][count] = G1.number_of_nodes()
        # print(G1.edges(72))
        print(G1.number_of_nodes())
        # print(G1.edges(0))
        count = count + 1
        npG1 = nx.to_numpy_array(G1)
        # print(npG1[G1.number_of_nodes()-1])
        #
        G2 = readGraph(files[i+1], graph_type='simple', file_format='dimacs')
        list = [(0, i) for i in range(G2.number_of_nodes()+1)]
        G2.add_edges_from(list)
        # print(G2.edges(0))
        # print(G2.edges(0))
        # print(G2.number_of_nodes())
        bool_iso = nx.is_isomorphic(G1, G2)
        print(bool_iso)
        npG2 = nx.to_numpy_array(G2)
        # writer.writerow([npG1, npG2, True])
# benchmark.to_csv('./time_benchmark_False_1.csv')
