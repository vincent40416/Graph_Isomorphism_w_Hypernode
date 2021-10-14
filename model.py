import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
<<<<<<< HEAD
from torch_geometric.nn import SGConv
from torch_geometric.nn import GINConv
=======
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GINConv
# from torch_geometric.nn import
>>>>>>> vincent_branch
batch_size = 1

# DGL data structure
def gcn_message(edges):
    """
    compute a batch of message called 'msg' using the source nodes' feature 'h'
    :param edges:
    :return:
    """
    return {'msg': edges.src['h']}


def gcn_reduce(nodes):
    """
    compute the new 'h' features by summing received 'msg' in each node's mailbox.
    :param nodes:
    :return:
    """
    return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}


class GIN(nn.Module):
    def __init__(self, depth, node_sum, embedding_dim, batch):
        super(GIN, self).__init__()
        self.batch = batch
        self.layers = []
        nn1 = nn.Sequential(nn.Linear(embedding_dim, embedding_dim, bias=False))
        for i in range(int(depth)):

            self.layers += [GINConv(nn1)]
        self.modulelist = nn.ModuleList(self.layers)

    def forward(self, g, feature):
        for layer in self.modulelist[:-1]:
            feature = layer(feature, g)
        return feature


class GCN(nn.Module):
    def __init__(self, depth, node_sum, embedding_dim, batch):
        super(GCN, self).__init__()
        self.batch = batch
        self.layers = []
        for i in range(int(depth)):
            self.layers += [GCNConv(embedding_dim, embedding_dim,improved= True, normalize=True, bias=False)]
        self.modulelist = nn.ModuleList(self.layers)

    def forward(self, g, feature):
        for layer in self.modulelist[:-1]:
            feature = layer(feature, g)
            # feature = nn.functional.leaky_relu(feature)
        return feature



<<<<<<< HEAD
class GCN_01(nn.Module):
    """
    Define a 2-layer GCN model.
    """
    def __init__(self, in_feats):
        super(GCN_01, self).__init__()
        self.gnn_layers = nn.ModuleList()
        self.gcn1 = GCNLayer(in_feats, in_feats)
        self.gcn2 = GCNLayer(in_feats, in_feats)
        self.gcn3 = GCNLayer(in_feats, in_feats)
        self.norm = torch.nn.BatchNorm1d(in_feats)

    def forward(self, g1, g2, feature):

        h1 = self.gcn1(g1, feature)
        h1 = self.norm(h1)
        h1 = self.gcn2(g1, h1)
        h1 = self.gcn3(g1, h1) # model_v3
        h2 = self.gcn1(g2, feature)
        h2 = self.norm(h2)
        h2 = self.gcn2(g2, h2)
        h2 = self.gcn3(g2, h2) # model_v3
        ha = torch.transpose(h1, 1, 0)
        h3 = torch.matmul(ha, h2)
        result = h3
        return result
=======
>>>>>>> vincent_branch


# Pytorch Geo structure
class GNN_Geo(torch.nn.Module):
    def __init__(self, node_sum, embedding_dim, batch_size):  # add batch_size into model
        super(GNN_Geo, self).__init__()
<<<<<<< HEAD
        # self.node_embedding = nn.Embedding(node_sum, node_sum)
        self.layer1 = nn.Sequential(
            torch.nn.Linear(batch_size * node_sum, embedding_dim),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            torch.nn.Linear(embedding_dim, batch_size * node_sum),
            nn.LeakyReLU(),
        )
        self.convolution_1 = GCNConv(batch_size * node_sum, embedding_dim, normalize=True)
        self.convolution_2 = GCNConv(embedding_dim, embedding_dim, normalize=True)
        self.convolution_3 = GCNConv(embedding_dim, embedding_dim, normalize=True)
        self.convolution_4 = GCNConv(embedding_dim, embedding_dim, normalize=True)
        self.convolution_5 = GCNConv(embedding_dim, embedding_dim, normalize=True)
        self.convolution_6 = GCNConv(embedding_dim, batch_size * node_sum, normalize=False)
=======
        self.layers = []
        self.convolution = GCNConv(batch_size * node_sum, batch_size * node_sum, normalize=True, bias=False)
        for i in range(int(embedding_dim)):
            self.layers += [self.convolution, nn.LeakyReLU()]
        self.layers += [self.convolution]
        self.modulelist = nn.ModuleList(self.layers)

        # sequential = nn.Sequential(*modules)
        self.convolution_1 = GCNConv(batch_size * node_sum, batch_size * node_sum, normalize=True)
        self.convolution_2 = GCNConv(batch_size * node_sum, batch_size * node_sum, normalize=True)
        self.convolution_3 = GCNConv(batch_size * node_sum, batch_size * node_sum, normalize=True)
        self.convolution_4 = GCNConv(batch_size * node_sum, batch_size * node_sum, normalize=True)
        self.convolution_5 = GCNConv(batch_size * node_sum, batch_size * node_sum, normalize=True)
        self.convolution_6 = GCNConv(batch_size * node_sum, batch_size * node_sum, normalize=True)
>>>>>>> vincent_branch
        # self.convolution_1 = GATConv(batch_size * node_sum, embedding_dim)
        # self.convolution_2 = GATConv(embedding_dim, embedding_dim)
        # self.convolution_3 = GATConv(embedding_dim, embedding_dim)
        # self.convolution_4 = GATConv(embedding_dim, embedding_dim)
        # self.convolution_5 = GATConv(embedding_dim, embedding_dim)
        # self.convolution_6 = GATConv(embedding_dim, batch_size * node_sum)
        self.leakyrelu = nn.LeakyReLU()
        self.Linear1 = torch.nn.Bilinear(batch_size * node_sum, batch_size * node_sum, batch_size * node_sum, bias=False)
        self.Linear2 = torch.nn.Linear(batch_size * node_sum, batch_size * node_sum, bias=False)

    def convolutional_pass(self, features, edge_index):
        features = self.convolution_1(features, edge_index)
        # features = self.leakyrelu(features)
        # non linearity (leaky relu)
        features = self.convolution_2(features, edge_index)
        features = self.leakyrelu(features)
        features = self.convolution_3(features, edge_index)
        features = self.leakyrelu(features)
        features = self.convolution_4(features, edge_index)
        features = self.leakyrelu(features)
        features = self.convolution_5(features, edge_index)
        features = self.leakyrelu(features)
        features = self.convolution_6(features, edge_index)
        # features = self.leakyrelu(features)

        return features

    def forward(self, edge_index_1, edge_index_2, feature):
        # edge_index_1 = data[0]  # Graph A edge index 2,e
        # edge_index_2 = data[1]
<<<<<<< HEAD
        # # print(data[3].type())
=======
        # print(feature.type())
>>>>>>> vincent_branch
        # feature = data[3]  # feature dim (N, d) dim 固定
        # feature = self.node_embedding(data[3])
        # print(edge_index_1.size())
        # print(feature.size())
        # print(feature)
<<<<<<< HEAD
        FA_b = self.convolutional_pass(feature, edge_index_1)
        FB = self.convolutional_pass(feature, edge_index_2)
=======
        FA_b = feature
        FB = feature
        # print(self.modulelist)
        for i, layer in enumerate(self.modulelist):
            if i % 2 == 0:
                FA_b = layer(FA_b, edge_index_1)
                FB = layer(FB, edge_index_2)
            else:
                FA_b = layer(FA_b)
                FB = layer(FB)
        # FA_b = self.convolutional_pass(feature, edge_index_1)
        # FB = self.convolutional_pass(feature, edge_index_2)
>>>>>>> vincent_branch
        FA = torch.transpose(FA_b, 1, 0)
        prediction_aff = torch.matmul(FA, FB)
        return FA_b, FB, prediction_aff


<<<<<<< HEAD
"""
Implement of GCN module.
"""


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.):
        super(GCN, self).__init__()
        self.trans_msg = nn.Linear(in_dim, out_dim)
        self.nonlinear = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    """
    Input :
        x : (N, in_dim)
        m : (N, N)
    Output :
        out : (N, out_dim)
    """
    def forward(self, x: torch.Tensor, m: torch.Tensor):
        x_msg = self.trans_msg(x)
        x_msg = self.nonlinear(x_msg)
        x_msg = self.dropout(x_msg)

        row_degree = torch.sum(m, dim=1, keepdim=True)   # (N, 1)
        col_degree = torch.sum(m, dim=0, keepdim=True)   # (1, N)
        degree = torch.mm(torch.sqrt(row_degree), torch.sqrt(col_degree))  # (N, N)
        out = torch.mm(m / degree, x_msg)

        return out
=======
class RNN(nn.Module):

    # you can also accept arguments in your model constructor
    def __init__(self, node_sum):
        super(RNN, self).__init__()

        self.convolution_1 = GCNConv(batch_size * node_sum, node_sum, normalize=True, bias=True)
        nn1 = nn.Sequential(nn.Linear(node_sum, node_sum), nn.LeakyReLU(), nn.Linear(node_sum, node_sum))
        self.isomorphic = GINConv(nn1)
        self.SAGE = SAGEConv(batch_size * node_sum, node_sum, normalize=True, bias=False)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, edge_index_1, edge_index_2, feature_A, feature_B, hidden_A, hidden_B):

        FA = self.convolution_1(feature_A + 0.8 * hidden_A, edge_index_1)
        FA = self.leakyrelu(FA)
        FB = self.convolution_1(feature_B + 0.8 * hidden_B, edge_index_2)
        FB = self.leakyrelu(FB)
        # prediction_aff = torch.matmul(FA, FB)
        return FA, FB, hidden_A, hidden_B


class customLoss(nn.Module):
    def __init__(self):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(customLoss, self).__init__()

    def forward(self, FA, FB, Aff):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        # Transform targets to one-hot vector
        loss_sum = 0
        for i, x in enumerate(Aff):
            for j, y in enumerate(x):
                # print(y)
                if y == 1:
                    loss = torch.sum(torch.abs(FA[i, :] - FB[j, :]))
                    # loss = diff)
                    # print(loss)
                else:
                    loss = 1 / torch.sum(torch.abs(FA[i, :] - FB[j, :]))
                    # loss = diff)

                    # print(loss)
                loss_sum += loss

        return loss_sum
>>>>>>> vincent_branch
