import torch
import pandas as pd
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
# from preprocessing import GraphDataset
from preprocessing import GraphDataset, TestGraphDataset
from model import GNN_Geo, RNN, GCN,GIN
from model import customLoss
from utils import collate_fn, collate
from utils import select_data_tensorboard
from utils import select_embedding_tensorboard
from utils import select_matrices
from utils import compare_matrix, compare_test_matrix,remove_node
import torch_geometric
import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter
import datetime
import timeit

# default `log_dir` is "runs" - we'll be more specific here
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import csv

graph_size = 100 + 1
batch_size = 1
epochs = 0
num_epochs = 1
depth = 10
embedding_dim = 50
dataset_type = ""
# v1?
# v2 10000 data with tree and non tree
# v3 non trainable except pooling
# v4 100 dataset
version = "v2"
parser = argparse.ArgumentParser()

parser.add_argument('--LR', type=float, default=0.001)

args = parser.parse_args()
log_dir = "runs/exp_03_Gsize_embeddingdim_" + str(embedding_dim) + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_writer = SummaryWriter(log_dir)
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)
# df = pd.read_csv('Dataset/Graph_dataset_hypernode_' + dataset_type + str(100) + '.csv')
# path = 'Dataset/Graph_dataset_Tree_' + dataset_type + str(100) + '.csv'
# df_test = pd.read_csv(path)
df_test = pd.read_csv('./Graph_dataset_circulant_11.csv')
# df_test = pd.read_csv('./Dataset/Graph_dataset_hypernode_5.csv')
# df_test = pd.read_csv('./Dataset/Benchmark_True_sample.csv')
# Dataset
# dftrain = df[:int(len(df) * 0.1)]
# dftest1 = df[int(len(df) * 0.01):int(len(df) * 0.02)]
dftest2 = df_test[:int(len(df_test))]
# dftest2 = df_test[2:5]
# DataLoader
# data_loader = DataLoader(GraphDataset(dftrain, graph_size), batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
# test_dataloader = DataLoader(GraphDataset(dftest1, graph_size), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
test_dataloader2 = DataLoader(TestGraphDataset(dftest2, graph_size), batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True)

device = torch.device('cuda:2')
batch = 1
# model = GNN_Geo(depth=embedding_dim, node_sum=graph_size, embedding_dim=embedding_dim, batch=batch).to(device).double()
# model = GraphUNet(in_channels=graph_size, hidden_channels=graph_size, out_channels=graph_size, depth=embedding_dim).to(device).double()
# model = UNet(in_channels=graph_size, hidden_channels=graph_size, out_channels=graph_size, depth=50).to(device).double()
model = GCN(depth=depth, node_sum=graph_size, embedding_dim=embedding_dim, batch=batch).to(device).double()
# model = GIN(depth=depth, node_sum=graph_size, embedding_dim=embedding_dim, batch=batch).to(device).double()
if epochs != 0:
    model.load_state_dict(torch.load('./Model/Graph_isov2_' + version + '_embeddingdim_' + str(embedding_dim) + '_e' + str(epochs) + '.pth'))

loss_func = customLoss().to(device)  # L2 LOSS

trainable = ['pools', 'up_convs']
print([kv[0] for kv in model.named_parameters() if any([kv[0].startswith(nt) for nt in trainable])])
# optimizer = optim.Adam(
#     [kv[1] for kv in model.named_parameters() if any([kv[0].startswith(nt) for nt in trainable])],
#     lr=args.LR
# )
optimizer = optim.Adam(model.parameters(), lr=args.LR)


# print('start training')
# model.train()
# epoch_losses = []
# with open('Pred_in_training.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['pred', 'Aff'])
#     for epoch in range(epochs, epochs + num_epochs):
#         print('epoch : ' + str(epoch))
#         epoch_loss = 0
#         if num_epochs != 1:
#             for iter, data in enumerate(data_loader):   # tqdm(enumerate(data_loader), total=len(data_loader), desc="Batches"):
#
#                 # input data
#                 data = [i.to(device) for i in data]
#                 Feature = torch.from_numpy(np.full((graph_size, embedding_dim), 1, dtype=np.int64)).type(torch.FloatTensor).to(device)
#                 # model
#                 FA = model(data[0], Feature.double())
#                 FB = model(data[1], Feature.double())
#
#                 # train
#                 loss = loss_func(FA, FB, data[2])
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.detach().item()
#
#             epoch_loss /= (iter + 1)
#             tb_writer.add_scalar('loss', epoch_loss, epoch)
#             print('Epoch {}, loss {:.8f}'.format(epoch, epoch_loss))
#             epoch_losses.append(epoch_loss)
#
#         # validate
#         if (epoch + 1) % num_epochs == 0:
#             model.eval()
#             correct = 0
#             perfect_match = 0
#             hidden_A = torch.zeros(graph_size, graph_size).to(device)
#             hidden_B = torch.zeros(graph_size, graph_size).to(device)
#             for iter, data in enumerate(test_dataloader):
#                 perfect_match = perfect_match + 1
#                 data = [i.to(device) for i in data]
#                 Feature = torch.from_numpy(np.full((graph_size, embedding_dim), 1, dtype=np.int64)).type(torch.FloatTensor).to(device)
#                 # model
#                 FA = model(data[0], Feature.double())
#                 FB = model(data[1], Feature.double())
#
#                 Aff_ = data[2].cpu().detach().numpy().astype(int)
#                 FA = FA.cpu().detach().numpy()
#                 FB = FB.cpu().detach().numpy()
#                 pred_b = compare_matrix(FA, FB).astype(int)
#                 print(np.equal(pred_b, Aff_).sum().item())
#                 if(int(np.equal(pred_b, Aff_).sum().item()) != graph_size * graph_size):
#                     arr = np.not_equal(pred_b, Aff_)
#                 #     print(pred_b)
#                 #     # print(np.where(arr))
#                     print(Aff_[np.where(arr)])
#                 #     # print(FA[np.where(arr)[0][1], :])
#                 #     # print(FB[np.where(arr)[0][1], :])
#                 #     # # perfect_match = perfect_match - 1
#                 #     # # data[3] = data[3].cpu().detach().numpy()
#                 #     torch.set_printoptions(threshold=5000, precision=2)
#                 #     # # print(data[3])
#                 #     # # print(FA, FB)
#                 #     # # print(pred_b, Aff_)
#                 #     # select_data_tensorboard(data, tb_writer, iter)
#                 #     # select_matrices(FA, FB, pred_b, Aff_, tb_writer, iter)
#                 #     # writer.writerow([np.asarray(pred_b), np.asarray(Aff_)])
#                 #     # torch.set_printoptions(profile="default")
#                 # print()
#                 if(int(np.equal(pred_b, Aff_).sum().item()) != graph_size * graph_size):
#                     perfect_match = perfect_match - 1
#                 correct += float(np.equal(pred_b, Aff_).sum().item())
#             acc = correct / (len(test_dataloader) * graph_size * graph_size * batch_size * batch_size)
#             # print(perfect_match)
#             perfect_acc = perfect_match / (len(test_dataloader) * batch_size)
#             tb_writer.add_scalar('Org_acc', acc, epoch)
#             print('Org_data Accuracy: {:.8f}'.format(acc))
#             print('Org_data_perfect Accuracy: {:.8f}'.format(perfect_acc))
# print('train end')
# if num_epochs != 1:
#     torch.save(model.state_dict(), "./Model/Graph_iso_" + version + '_' + str(graph_size) + "_embeddingdim_" + str(embedding_dim) + "_e" + str(epochs + num_epochs-1) + ".pth")

model.eval()
time_list = []
# only check isomorphic or not
print("start testing")
correct = 0
perfect_match = 0
for iter, data in enumerate(test_dataloader2):
    flag = True
    perfect_match = perfect_match + 1
    g_size = torch.max(data[0]).item() + 1
    start = timeit.default_timer()
    for index in range(g_size):
        # print(type(data[0]))
        # print(index)
        data = [i.to(device) for i in data]

        Feature = torch.from_numpy(np.full((g_size - index, embedding_dim), 1, dtype=np.int64)).type(torch.FloatTensor).to(device)
        # print(Feature.size())
        FA = model(data[0], Feature.double())
        # print(FA.size())
        FB = model(data[1], Feature.double())
        # print(FB.size())
        # Aff_ = data[2].cpu().detach().numpy().astype(bool)
        FA = FA.cpu().detach().numpy()
        FB = FB.cpu().detach().numpy()
        # compare_matrix(FA, FB)
        # pred_b = compare_test_matrix(FA, FB)
        data, pred = remove_node(FA, FB, data)
        if (pred == False):
            flag = False
            break


    # data = [i.to(device) for i in data]
    # Feature = torch.from_numpy(np.full((g_size, embedding_dim), 1, dtype=np.int64)).type(torch.FloatTensor).to(device)
    # # print(Feature.size())
    # FA = model(data[0], Feature.double())
    # # print(FA.size())
    # FB = model(data[1], Feature.double())
    # # print(FB.size())
    # # Aff_ = data[2].cpu().detach().numpy().astype(bool)
    # FA = FA.cpu().detach().numpy()
    # FB = FB.cpu().detach().numpy()
    # pred = compare_test_matrix(FA, FB)
    # # data, pred = remove_node(FA, FB, data)
    # # if (pred == False):
    # #     flag = False
    # #     break
    # # print(np.equal(pred_b, Aff_).item())


    correct += int(pred == True)
    # data = remove_node(data)
    print(correct)
    stop = timeit.default_timer()
    time_list.append([g_size, stop-start])
# with open("time_benchmark_False.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(time_list)
acc = correct / (len(test_dataloader2))
# perfect_acc = perfect_match / (len(test_dataloader2) * batch_size)
print('Test_data Accuracy: {:.8f}'.format(acc))
# print('Test_data_perfect Accuracy: {:.8f}'.format(perfect_acc))
