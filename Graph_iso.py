import torch
import pandas as pd
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
# from preprocessing import GraphDataset
from preprocessing import Self_Design_Graph
from preprocessing import GraphDataset
from model import GNN_Geo
from utils import collate_fn

import warnings
warnings.filterwarnings('ignore')
import dgl
import csv

graph_size = 10
batch_size = 1
epochs = 200


df = pd.read_csv('./Graph_dataset.csv')
# 创建一个训练数据集和测试数据集
dftrain = df[:int(len(df) * 0.8)]
dftest1 = df[:int(len(df) * 0.1)]
dftest2 = df[int(len(df) * 0.9):]
trainset = Self_Design_Graph(dftrain)
testset = Self_Design_Graph(dftest1)
testset2 = Self_Design_Graph(dftest2)
# 使用 PyTorch 的 DataLoader 和之前定义的 collate 函数。
data_loader = DataLoader(GraphDataset(dftrain), batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
test_dataloader = DataLoader(GraphDataset(dftest1), batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
test_dataloader2 = DataLoader(GraphDataset(dftest2), batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
input = torch.tensor(np.ones((graph_size, graph_size)))

model = GNN_Geo(10, 64).cuda()
loss_func = torch.nn.L1Loss() # L2 LOSS
optimizer = optim.Adam(model.parameters(), lr=0.01)
# lambdaLR = lambda epoch: 0.96**epoch
# schedular = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaLR)
model.train()
epoch_losses = []
with open('Pred_in_training.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['pred', 'Aff'])
    for epoch in range(epochs):
        epoch_loss = 0
        for iter, data in enumerate(data_loader):   # tqdm(enumerate(data_loader), total=len(data_loader), desc="Batches"):
            # print(data)

            data = [i.cuda() for i in data]
            prediction_aff = model(data)
            # if iter > 2:
            #     break
            # print(prediction_aff.type())
            loss = loss_func(prediction_aff, data[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)
        if (epoch + 1) % 4 == 0:
            model.eval()
            correct = 0
            for iter, data in enumerate(test_dataloader):
                data = [i.cuda() for i in data]
                pred = model(data)
                pred = pred.cpu().detach().numpy()
                pred = np.where(pred > 0, 1, -1)
                Aff_ = data[2].cpu().detach().numpy()
                pred = pred.astype(int)
                Aff_ = Aff_.astype(int)
                if(iter == 1):
                    print(pred, Aff_)
                    writer.writerow([np.asarray(pred), np.asarray(Aff_)])
                correct += float(np.equal(pred, Aff_).sum().item())

            acc = correct / (len(test_dataloader) * graph_size * graph_size * batch_size * batch_size)
            print('Org_data Accuracy: {:.4f}'.format(acc))
            correct = 0
            for iter, data in enumerate(test_dataloader2):
                data = [i.cuda() for i in data]
                pred = model(data)
                pred = pred.cpu().detach().numpy()
                pred = np.where(pred > 0, 1, -1)
                Aff_ = data[2].cpu().detach().numpy()
                pred = pred.astype(int)
                Aff_ = Aff_.astype(int)
                correct += float(np.equal(pred, Aff_).sum().item())

            acc = correct / (len(test_dataloader2) * graph_size * graph_size * batch_size * batch_size)
            print('Test_data Accuracy: {:.4f}'.format(acc))

# # Greedy
# Affinity = np.full((graph_size, graph_size), -1, dtype=float)
# State = np.full((graph_size, graph_size), -1, dtype=float)
# for iter, (GraphA_, GraphB_, Aff_) in enumerate(test_dataloader):
#     pred_Aff = Affinity.copy()
#     for i in range(graph_size):
#         # C' = Model(A,B)
#         # C' max ,find A(i) = B(j)
#         # 1. remove A(3) B(4) 2. A(3) B(1)
#         m = model(GraphA_, GraphB_)
#         m = m.numpy()
#         indices = np.concatenate(((m / graph_size).view(-1, 1), (m % graph_size).view(-1, 1)), axis=1)
#         max_node_in_A = indices[0][0]
#         max_node_in_B = indices[0][1]
#         pred_Aff[max_node_in_A][max_node_in_B] = 1
#
#
#
#     pred_Aff.amax[1] =


torch.save(model.state_dict(), './Model/Graph_iso-v3_%d.pth' % epoch)
