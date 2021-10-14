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
from utils import select_matrices
from utils import compare_matrix
import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter
import datetime
# default `log_dir` is "runs" - we'll be more specific here
log_dir = "runs/GCN_exp_02/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_writer = SummaryWriter(log_dir)
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)
import dgl
import csv

graph_size = 100
batch_size = 1
epochs = 0
num_epochs = 1
parser = argparse.ArgumentParser()

parser.add_argument('--LR', type=float, default=0.001)

args = parser.parse_args()

df = pd.read_csv('./Graph_dataset.csv')
# 创建一个训练数据集和测试数据集
dftrain = df[:int(len(df) * 0.9)]
dftest1 = df[:int(len(df) * 0.01)]
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
    model.load_state_dict(torch.load('./Model/Graph_iso-GCNCONV_%d.pth' % epochs))

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = torch.nn.DataParallel(model)
# model.to(device)
loss_func = torch.nn.MSELoss()  # L2 LOSS
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
        # for iter, data in enumerate(data_loader):   # tqdm(enumerate(data_loader), total=len(data_loader), desc="Batches"):
            # print(data)
            # print(data[0].size())
            # print(data[2].size())
            # data = [i.to(device) for i in data]
            # FA, FB, prediction_aff = model(data[0], data[1], data[3])
            # loss = loss_func(prediction_aff, data[2])
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # epoch_loss += loss.detach().item()

        # epoch_loss /= (iter + 1)
        # tb_writer.add_scalar('loss', epoch_loss, epoch)
        # print('Epoch {}, loss {:.8f}'.format(epoch, epoch_loss))
        # epoch_losses.append(epoch_loss)
        if (epoch + 1) % 1 == 0:
            model.eval()
            correct = 0
            perfect_match = 0
            for iter, data in enumerate(test_dataloader):
                perfect_match = perfect_match + 1
                data = [i.to(device) for i in data]
                FA, FB, pred = model(data[0], data[1], data[3])
                FA = FA.cpu().detach().numpy()
                FB = FB.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()
                pred_b = compare_matrix(FA, FB)
                pred = np.where(pred_b > 0, 1, -1)
                Aff_ = data[2].cpu().detach().numpy()
                pred_b = pred_b.astype(int)
                Aff_ = Aff_.astype(int)
                print(np.equal(pred_b, Aff_).sum().item())
                if(np.equal(pred_b, Aff_).sum().item() != graph_size * graph_size):
                    perfect_match = perfect_match - 1
                    data[3] = data[3].cpu().detach().numpy()
                    torch.set_printoptions(threshold=5000, precision=2)
                    # print(data[3])
                    # print(FA, FB)
                    # print(pred_b, Aff_)
                    select_data_tensorboard(data, tb_writer, epoch)
                    select_matrices(FA, FB, pred_b, pred, Aff_, tb_writer, epoch)
                    writer.writerow([np.asarray(pred_b), np.asarray(Aff_)])
                    torch.set_printoptions(profile="default")
                # print()
                correct += float(np.equal(pred_b, Aff_).sum().item())
            acc = correct / (len(test_dataloader) * graph_size * graph_size * batch_size * batch_size)
            print(perfect_match)
            perfect_acc = perfect_match / (len(test_dataloader) * batch_size)
            tb_writer.add_scalar('Org_acc', acc, epoch)
            print('Org_data Accuracy: {:.8f}'.format(acc))
            print('Org_data_perfect Accuracy: {:.8f}'.format(perfect_acc))
            correct = 0
            perfect_match = 0
            for iter, data in enumerate(test_dataloader2):
                perfect_match = perfect_match + 1
                data = [i.to(device) for i in data]
                FA, FB, pred = model(data[0], data[1], data[3])
                FA = FA.cpu().detach().numpy()
                FB = FB.cpu().detach().numpy()
                pred_b = compare_matrix(FA, FB)
                pred = pred.cpu().detach().numpy()
                pred = np.where(pred > 0, 1, -1)
                Aff_ = data[2].cpu().detach().numpy()
                pred_b = pred_b.astype(int)
                Aff_ = Aff_.astype(int)
                correct += float(np.equal(pred_b, Aff_).sum().item())
                if(int(np.equal(pred_b, Aff_).sum().item()) != graph_size * graph_size):
                    perfect_match = perfect_match - 1

            acc = correct / (len(test_dataloader2) * graph_size * graph_size * batch_size * batch_size)
            perfect_acc = perfect_match / (len(test_dataloader2) * batch_size)
            tb_writer.add_scalar('Test_acc', acc, epoch)
            print('Test_data Accuracy: {:.8f}'.format(acc))
            print('Test_data_perfect Accuracy: {:.8f}'.format(perfect_acc))

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


torch.save(model.state_dict(), './Model/Graph_iso-GCNCONV_%d.pth' % num_epochs)
