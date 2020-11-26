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
from utils import select_data_tensorboard
from utils import select_embedding_tensorboard
import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter
import datetime
# default `log_dir` is "runs" - we'll be more specific here
log_dir = "runs/GIN_exp_01/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_writer = SummaryWriter(log_dir)
warnings.filterwarnings('ignore')
import dgl
import csv

graph_size = 10
batch_size = 1
epochs = 0
num_epochs = 100
parser = argparse.ArgumentParser()

parser.add_argument('--LR', type=float, default=0.001)

args = parser.parse_args()

df = pd.read_csv('./Graph_dataset.csv')
# 创建一个训练数据集和测试数据集
dftrain = df[:int(len(df) * 0.7)]
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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") , batch_size
device = torch.device('cuda:3')
model = GNN_Geo(graph_size, 64, batch_size).to(device)

# data = next(iter(data_loader))
# data = [i.to(device) for i in data]
# tb_writer.add_graph(model, data[0], data[1], data[3])
if epochs != 0:
    model.load_state_dict(torch.load('./Model/Graph_iso-v3_%d.pth' % epochs))

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = torch.nn.DataParallel(model)
# model.to(device)
loss_func = torch.nn.L1Loss()  # L2 LOSS
optimizer = optim.Adam(model.parameters(), lr=args.LR)
# lambdaLR = lambda epoch: 0.96**epoch
# schedular = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaLR)
model.train()
epoch_losses = []
with open('Pred_in_training.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['pred', 'Aff'])
    for epoch in range(epochs, num_epochs):
        epoch_loss = 0
        for iter, data in enumerate(data_loader):   # tqdm(enumerate(data_loader), total=len(data_loader), desc="Batches"):
            # print(data)
            # print(data[0].size())
            # print(data[2].size())
            data = [i.to(device) for i in data]
            FA, FB, prediction_aff = model(data[0], data[1], data[3])
            loss = loss_func(prediction_aff, data[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

        epoch_loss /= (iter + 1)
        tb_writer.add_scalar('loss', epoch_loss, epoch)
        print('Epoch {}, loss {:.8f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)
        if (epoch + 1) % 4 == 0:
            model.eval()
            correct = 0
            for iter, data in enumerate(test_dataloader):
                data = [i.to(device) for i in data]
                FA, FB, pred = model(data[0], data[1], data[3])

                pred = pred.cpu().detach().numpy()
                pred = np.where(pred > 0, 1, -1)
                Aff_ = data[2].cpu().detach().numpy()
                pred = pred.astype(int)
                Aff_ = Aff_.astype(int)
                if(iter == 1):
                    data[3] = data[3].cpu().detach().numpy()
                    FA = FA.cpu().detach().numpy()
                    FB = FB.cpu().detach().numpy()
                    torch.set_printoptions(threshold=5000, precision=2)
                    print(data[3])
                    print(FA, FB)
                    print(pred, Aff_)
                    select_data_tensorboard(data, tb_writer, epoch)
                    select_embedding_tensorboard(FA, tb_writer, epoch, "FA")
                    select_embedding_tensorboard(FB, tb_writer, epoch, "FB")
                    select_embedding_tensorboard(pred, tb_writer, epoch, "pred")
                    select_embedding_tensorboard(Aff_, tb_writer, epoch, "Aff")
                    writer.writerow([np.asarray(pred), np.asarray(Aff_)])
                    torch.set_printoptions(profile="default")
                correct += float(np.equal(pred, Aff_).sum().item())

            acc = correct / (len(test_dataloader) * graph_size * graph_size * batch_size * batch_size)
            tb_writer.add_scalar('Org_acc', acc, epoch)
            print('Org_data Accuracy: {:.8f}'.format(acc))
            correct = 0
            for iter, data in enumerate(test_dataloader2):
                data = [i.to(device) for i in data]
                FA, FB, pred = model(data[0], data[1], data[3])

                FA = FA.cpu().detach().numpy()
                FB = FB.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()
                pred = np.where(pred > 0, 1, -1)
                Aff_ = data[2].cpu().detach().numpy()
                pred = pred.astype(int)
                Aff_ = Aff_.astype(int)
                correct += float(np.equal(pred, Aff_).sum().item())

            acc = correct / (len(test_dataloader2) * graph_size * graph_size * batch_size * batch_size)
            tb_writer.add_scalar('Test_acc', acc, epoch)
            print('Test_data Accuracy: {:.8f}'.format(acc))

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


torch.save(model.state_dict(), './Model/Graph_iso-v4_%d.pth' % num_epochs)
