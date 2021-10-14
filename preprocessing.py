import torch
import pandas as pd
import numpy as np
# from torch_geometric.data import InMemoryDataset
# from torch_geometric.data import Data
from torch.utils.data import Dataset
import math
# from Graph_iso import graph_size
import dgl
import random

<<<<<<< HEAD
df = pd.read_csv('./Graph_dataset.csv')
graph_size = 100
=======
graph_size = 10
>>>>>>> vincent_branch


def transfrom_str_to_nparray(str, i):
    str = str.replace(']', '').replace('[', '').replace('\n', '')
    str = str.split(".")
    str.pop()
    arr = np.array(str)
    arr = arr.astype(np.int)
    x = int(math.sqrt(len(arr)))
    arr = np.reshape(arr, (-1, x))
    return arr


def transform_to_edge_index(matrix, graph_size):
    edge_index = []
    graph_size = len(matrix)
    for i in range(graph_size):
        for j in range(graph_size):
            if(matrix[i][j] == 1):
                edge_index.append((i, j))

    return edge_index


# Torch Geometric Graph structure
class GraphDataset(Dataset):
    def __init__(self, df, graph_size):
        self.df = df
        self.graph_size = graph_size

    def __getitem__(self, index):
        new_data = dict()
        featureA = transfrom_str_to_nparray(self.df.iloc[index]['GraphA'], self.graph_size)
        featureB = transfrom_str_to_nparray(self.df.iloc[index]['GraphB'], self.graph_size)
        feature_aff = transfrom_str_to_nparray(self.df.iloc[index]['Affinity'], self.graph_size)
        # print(len(feature_aff))
        graph_A = np.array(transform_to_edge_index(featureA, self.graph_size), dtype=np.int64).T
        graph_B = np.array(transform_to_edge_index(featureB, self.graph_size), dtype=np.int64).T
        # graph_A = np.array(featureA, dtype=np.int64).T
        # graph_B = np.array(featureB, dtype=np.int64).T
        Feature = np.array(np.arange(len(feature_aff)), dtype=np.int64)
        Aff = np.array(feature_aff, dtype=np.int64)
        new_data['GA'] = graph_A
        new_data['GB'] = graph_B
        new_data['Aff'] = Aff
        # print(Aff)
        new_data['Feature'] = Feature
        return new_data

    def __len__(self):
        return len(self.df)


class TestGraphDataset(Dataset):
    def __init__(self, df, graph_size):
        self.df = df
        self.graph_size = graph_size

    def __getitem__(self, index):
        new_data = dict()
        featureA = transfrom_str_to_nparray(self.df.iloc[index]['GraphA'], self.graph_size)
        featureB = transfrom_str_to_nparray(self.df.iloc[index]['GraphB'], self.graph_size)
        graph_size = len(featureA)
        graph_A = np.array(transform_to_edge_index(featureA, self.graph_size), dtype=np.int64).T
        graph_B = np.array(transform_to_edge_index(featureB, self.graph_size), dtype=np.int64).T
        new_data['GA'] = graph_A
        new_data['GB'] = graph_B
        new_data['size'] = graph_size
        # new_data['is_ISO'] = bool(self.df.iloc[index]['is_ISO'])
        # print(Aff)
        return new_data

    def __len__(self):
        return len(self.df)
