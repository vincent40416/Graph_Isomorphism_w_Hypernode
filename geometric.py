import torch
from torch_geometric.data import Data
import csv
import pandas as pd
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
#         self.lin = torch.nn.Linear(in_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]
#
#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#
#         # Step 2: Linearly transform node feature matrix.
#         x = self.lin(x)
#
#         # Step 3: Compute normalization
#         row, col = edge_index
#         deg = degree(row, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#
#         # Step 4-6: Start propagating messages.
#         return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
#                               norm=norm)
#
#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]
#
#         # Step 4: Normalize node features.
#         return norm.view(-1, 1) * x_j
#
#     def update(self, aggr_out):
#         # aggr_out has shape [N, out_channels]
#
#         # Step 6: Return new node embeddings.
#         return aggr_out
# # 開啟輸出的 CSV 檔案
#
# conv = GCNConv(16, 32)
# x = conv(x, edge_index)
df = pd.read_csv('./Graph_dataset.csv')
row = np.bool_(df.iloc[1]['ISO'])
print(row)
y = torch.BoolTensor([row])
print(y)
# print(df.iloc[1])
# a = len(df)
# print(a)
# for i in range(len(df)):
#     print(i)
# print(df['GraphA'])
# print()
# frame = frame.to_numpy()
# # print(frame)
# a = frame[0, 1]
# a = a.replace(']', '').replace('[', '').replace('\n', '')
# a = a.split(".")
# a.pop()
# arr = np.array(a)
# arr = arr.astype(np.int)
# arr = np.reshape(arr, (-1, 5))
# print(arr)
# a_np = np.fromstring(a, sep='\n')
# print(a_np)


# with open('Graph_dataset.csv', newline='') as csvfile:
#
#     rows = csv.reader(csvfile, delimiter=':')
#     for row in rows:
#         print(row['GraphA'])

    # edge_index = torch.tensor([[0, 1],
    #                            [1, 0],
    #                            [1, 2],
    #                            [2, 1]], dtype=torch.long)
    # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    #
    # data = Data(x=x, edge_index=edge_index.t().contiguous())
    # print(data)
