import torch
import pandas as pd
import numpy as np
from preprocessing import Self_Design_Graph
from preprocessing import GraphDataset
import csv

df = pd.read_csv('./Graph_dataset.csv')
# 创建一个训练数据集和测试数据集
dftrain = df[:int(len(df) * 0.01)]

trainset = [GraphDataset(dftrain).__getitem__(i) for i in range(3)]


# output
def collate_fn(batch):
    nodes_list = [b['Feature'] for b in batch]
    print(len(batch))
    nodes = np.concatenate(nodes_list, axis=0)
    print(nodes)
    nodes_lens = np.fromiter(map(lambda l: l.shape[0], nodes_list), dtype=np.int64)
    nodes_inds = np.cumsum(nodes_lens)
    nodes_num = nodes_inds[-1]
    nodes_inds = np.insert(nodes_inds, 0, 0)
    # total_nodes = nodes_inds[-1]
    nodes_inds_2 = np.delete(nodes_inds, -1)
    # print(nodes_inds)
    GA_edges_list = [b['GA'] for b in batch]
    GA_edges_list = [e + i for e, i in zip(GA_edges_list, nodes_inds_2)]
    GA_edges = np.concatenate(GA_edges_list, axis=1)
    GB_edges_list = [b['GB'] for b in batch]
    GB_edges_list = [e + i for e, i in zip(GB_edges_list, nodes_inds_2)]
    GB_edges = np.concatenate(GB_edges_list, axis=1)
    Affinity = np.full((nodes_num, nodes_num), -1, dtype=np.int64)
    with open('12345.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for i in range(len(batch)):
            Affinity[np.ix_([i for i in range(nodes_inds[i], nodes_inds[i + 1])], [i for i in range(nodes_inds[i], nodes_inds[i + 1])])] = batch[i]['Aff']
            writer.writerow(batch[i]['Aff'])
        writer.writerow(Affinity)
    Feature = np.full((nodes_num, nodes_num), 1, dtype=np.int64)
    # Feature = [e + i for e, i in zip(nodes_list, nodes_inds_2)]
    # Feature = np.concatenate(Feature, axis=0)
    # print(Feature)
    # labels = [b[2] for b in batch]
    # labels = np.concatenate(labels, axis=0)
    #
    # batch_mask = [np.array([i] * k, dtype=np.int32) for i, k in zip(range(len(batch)), nodes_lens)]
    # batch_mask = np.concatenate(batch_mask, axis=0)
    return [torch.from_numpy(GA_edges).type(torch.LongTensor), torch.from_numpy(GB_edges).type(torch.LongTensor), torch.from_numpy(Affinity).type(torch.FloatTensor), torch.from_numpy(Feature).type(torch.FloatTensor)]


def collate(samples):
    # 输入`samples` 是一个列表
    # 每个元素都是一个二元组 (图, 标签)
    GA = [torch.from_numpy(np.array(item['GA'], dtype=np.int64).T).type(torch.LongTensor) for item in samples]
    GB = [torch.from_numpy(np.array(item['GB'], dtype=np.int64).T).type(torch.LongTensor) for item in samples]
    Aff = [torch.from_numpy(np.array(item['Aff'], dtype=np.int64).T).type(torch.FloatTensor) for item in samples]

    Feature = [torch.from_numpy(np.full((item['Feature'][-1] + 1, item['Feature'][-1] + 1), 1, dtype=np.int64).T).type(torch.LongTensor) for item in samples]
    print(Feature)
    return [GA, GB, Aff, Feature]


# data_loader = DataLoader(GraphDataset(dftrain), batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
collate(trainset)
# print(torch.cuda.is_available())
