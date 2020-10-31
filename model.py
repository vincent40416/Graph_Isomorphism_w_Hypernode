import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
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


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.

        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_message, gcn_reduce)
            g.update_all(gcn_message, gcn_reduce)
            h = g.ndata['h']
            h = self.linear(h.float())
            return h


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
        h3 = torch.mul(ha, h2)
        result = h3
        return result


# Pytorch Geo structure
class GNN_Geo(torch.nn.Module):
    def __init__(self, node_sum, embedding_dim):
        super(GNN_Geo, self).__init__()
        # self.node_embedding = nn.Embedding(node_sum, node_sum)
        self.convolution_1 = GCNConv(node_sum, embedding_dim, normalize=True)
        self.convolution_2 = GCNConv(embedding_dim, embedding_dim, normalize=True)
        self.convolution_3 = GCNConv(embedding_dim, embedding_dim, normalize=True)
        self.convolution_4 = GCNConv(embedding_dim, embedding_dim, normalize=True)
        self.convolution_5 = GCNConv(embedding_dim, embedding_dim, normalize=True)
        self.convolution_6 = GCNConv(embedding_dim, node_sum, normalize=True)
        self.leakyrelu = torch.nn.LeakyReLU()

    def convolutional_pass(self, edge_index, features):
        features = self.convolution_1(features, edge_index)
        features = self.leakyrelu(features)
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
        features = self.leakyrelu(features)

        return features

    def forward(self, data):
        edge_index_1 = data[0]  # Graph A edge index 2,e
        edge_index_2 = data[1]
        # print(data[3].type())
        feature = data[3]  # feature dim (N, d) dim 固定
        # feature = self.node_embedding(data[3])
        # print(edge_index_1.size())
        # print(feature)
        FA = self.convolutional_pass(edge_index_1, feature)
        FB = self.convolutional_pass(edge_index_2, feature)
        torch.set_printoptions(threshold=5000)
        # print(data[2])
        # print(FA)
        # print(FB)
        torch.set_printoptions(profile="default")
        # FA = self.convolutional_pass(edge_index_1, FA)
        # FB = self.convolutional_pass(edge_index_2, FB)
        # cos difference
        FB = torch.transpose(FB, 1, 0)
        combined_feature = torch.matmul(FA, FB)
        return combined_feature


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


"""
Implement of GCN network.
"""


class Net(nn.Module):
    def __init__(self, nodes_num, embedding_dim, hidden_dims, num_classes, dropout=0.):
        super(Net, self).__init__()

        self.node_embedding = nn.Embedding(nodes_num, embedding_dim)
        gcns = []
        in_dim = embedding_dim
        for d in hidden_dims:
            gcns.append(GCN(in_dim, d, dropout))
            in_dim = d
        self.gcns = nn.ModuleList(gcns)

        self.classifier = nn.Linear(in_dim, num_classes)

    """
    Input :
        x : (N, out_dim)
        bm : (N, )
    Output :
        out : (batch_size, out_dim)
    """
    def gcn_maxpooling(self, x, bm):
        batch_size = torch.max(bm) + 1
        out = []
        for i in range(batch_size):
            inds = (bm == i).nonzero()[:, 0]
            x_ind = torch.index_select(x, dim=0, index=inds)
            out.append(torch.max(x_ind, dim=0, keepdim=False)[0])
        out = torch.stack(out, dim=0)

        return out

    def gcn_meanpooling(self, x, lens):
        batch_size = torch.max(bm) + 1
        out = []
        for i in range(batch_size):
            inds = (bm == i).nonzero()[:, 0]
            x_ind = torch.index_select(x, dim=0, index=inds)
            out.append(torch.mean(x_ind, dim=0, keepdim=False))
        out = torch.stack(out, dim=0)

        return out

    def gcn_sumpooling(self, x, lens):
        batch_size = torch.max(bm) + 1
        out = []
        for i in range(batch_size):
            inds = (bm == i).nonzero()[:, 0]
            x_ind = torch.index_select(x, dim=0, index=inds)
            out.append(torch.sum(x_ind, dim=0, keepdim=False))
        out = torch.stack(out, dim=0)

        return out

    """
    Input :
        x : (N, )
        m : (N, N)
        bm : (N, )
    Output :
        output : (batch_size, num_classes)
    """
    def forward(self, x, m, bm):
        x_emb = self.node_embedding(x)  # (N, embedding_dim)
        out = x_emb
        for ml in self.gcns:
            out = ml(out, m)
        output = self.gcn_maxpooling(out, bm)   # (batch_size, out_dim)

        logits = self.classifier(output)  # (batch_size, num_classes)

        return logits
