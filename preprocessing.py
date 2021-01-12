import torch
import pandas as pd
import numpy as np
# from torch_geometric.data import InMemoryDataset
# from torch_geometric.data import Data
from torch.utils.data import Dataset
import dgl
import random

df = pd.read_csv('./Graph_dataset.csv')
graph_size = 100


def transfrom_str_to_nparray(str, i):
    str = str.replace(']', '').replace('[', '').replace('\n', '')
    str = str.split(".")
    str.pop()
    arr = np.array(str)
    arr = arr.astype(np.int)
    arr = np.reshape(arr, (-1, i))
    return arr


def transform_to_edge_index(matrix):
    edge_index = []
    for i in range(graph_size):
        for j in range(graph_size):
            if(matrix[i][j] == 1):
                edge_index.append((i, j))

    return edge_index


def build_graph(edge_list):
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(graph_size)
    # all 78 edges as a list of tuple
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.ndata['h'] = np.ones((graph_size, graph_size))
    # edges are directional in DGL; make them bi-directional
    # g.add_edges(dst, src)

    return g


# Torch Geometric Graph structure
class GraphDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        new_data = dict()
        featureA = transfrom_str_to_nparray(df.iloc[index]['GraphA'], graph_size)
        featureB = transfrom_str_to_nparray(df.iloc[index]['GraphB'], graph_size)
        feature_aff = transfrom_str_to_nparray(df.iloc[index]['Affinity'], graph_size)
        # print(len(feature_aff))
        graph_A = np.array(transform_to_edge_index(featureA), dtype=np.int64).T
        graph_B = np.array(transform_to_edge_index(featureB), dtype=np.int64).T
        Feature = np.array(np.arange(len(feature_aff)), dtype=np.int64)
        Aff = np.array(feature_aff, dtype=np.int64)
        new_data['GA'] = graph_A
        new_data['GB'] = graph_B
        new_data['Aff'] = Aff
        new_data['Feature'] = Feature
        return new_data

    def __len__(self):
        return len(self.df)


# DGL graph structure
def Self_Design_Graph(df):
    data_list = []
    for index, row in df.iterrows():
        featureA = transfrom_str_to_nparray(row['GraphA'], graph_size)
        featureB = transfrom_str_to_nparray(row['GraphB'], graph_size)
        feature_aff = transfrom_str_to_nparray(row['Affinity'], graph_size)
        graph_A = build_graph(transform_to_edge_index(featureA))
        graph_B = build_graph(transform_to_edge_index(featureB))
        data_list.append((graph_A, graph_B, feature_aff))
    return data_list


dataset = GraphDataset(df)
# print(np.shape(dataset[0]['GA']))
# train_loader = torch.utils.data.DataLoader(GraphDataset(df), batch_size=32)
# print(dataset[0]['Feature'])
