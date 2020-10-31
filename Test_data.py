import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

# from preprocessing import GraphDataset
from preprocessing import Self_Design_Graph
from model import GCN
import torch
print(torch.cuda.is_available())
# import warnings
# warnings.filterwarnings('ignore')
# import dgl
# import csv
# graph_size = 10
# batch_size = 1
# read_model = './Model/Graph_iso-v2_%d.pth' % 3
# save_csv = './Result/Test_result_%d.csv' % 2
#
#
# def collate(samples):
#     # 输入`samples` 是一个列表
#     # 每个元素都是一个二元组 (图, 标签)
#     graphA, graphB, Aff = map(list, zip(*samples))
#     batched_graphA = dgl.batch(graphA)
#     batched_graphB = dgl.batch(graphB)
#     return batched_graphA, batched_graphB, torch.tensor(Aff)
#
#
# df = pd.read_csv('./Graph_dataset.csv')
# # 创建一个训练数据集和测试数据集
# dftest1 = df[:int(len(df) * 0.1)]
# dftest2 = df[int(len(df) * 0.9):]
# testset = Self_Design_Graph(dftest1)
# testset2 = Self_Design_Graph(dftest2)
# # 使用 PyTorch 的 DataLoader 和之前定义的 collate 函数。
# test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True, collate_fn=collate)
# test_dataloader2 = DataLoader(testset2, batch_size=batch_size, shuffle=True, collate_fn=collate)
# input = torch.tensor(np.ones((graph_size, graph_size)))
#
# model = GCN(10)
# model.load_state_dict(torch.load(read_model))
# model.eval()
#
# correct = 0
# with open(save_csv, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Pred', 'Aff'])
#     for iter, (GraphA_, GraphB_, Aff_) in enumerate(test_dataloader):
#         input2 = input.clone()
#         pred = model(GraphA_, GraphB_, input2)
#         # if iter >= 1:
#         #     break
#         # if (iter < 1):
#         #     print(pred)
#         pred = pred.detach().numpy()
#         pred = np.where(pred > 0, 1, -1)
#
#         Aff_ = Aff_.detach().numpy()
#         pred = pred.astype(int)
#         Aff_ = Aff_.astype(int)
#         if (iter < 1):
#             print(pred)
#             print(Aff_)
#         writer.writerow([pred, Aff_])
#         correct += float(np.equal(pred, Aff_).sum().item())
#
#     acc = correct / (len(test_dataloader) * graph_size * graph_size * batch_size)
#     print('Org_data Accuracy: {:.4f}'.format(acc))
#     correct = 0
#     for iter, (GraphA_, GraphB_, Aff_) in enumerate(test_dataloader2):
#         input2 = input.clone()
#         pred = model(GraphA_, GraphB_, input2)
#         pred = pred.detach().numpy()
#         pred = np.where(pred > 0, 1, -1)
#
#         Aff_ = Aff_.detach().numpy()
#         pred = pred.astype(int)
#         Aff_ = Aff_.astype(int)
#         if (iter < 1):
#             print(pred)
#             print(Aff_)
#         writer.writerow([pred, Aff_])
#         correct += float(np.equal(pred, Aff_).sum().item())
#
#     acc = correct / (len(test_dataloader2) * graph_size * graph_size * batch_size)
# print('Test_data Accuracy: {:.4f}'.format(acc))
