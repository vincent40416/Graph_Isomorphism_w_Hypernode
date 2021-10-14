import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
<<<<<<< HEAD

def collate(samples):
    # 输入`samples` 是一个列表
    # 每个元素都是一个二元组 (图, 标签)
    List = []
    for item in samples:
        GA = [torch.from_numpy(np.array(item['GA'], dtype=np.int64).T).type(torch.LongTensor)]
        GB = [torch.from_numpy(np.array(item['GB'], dtype=np.int64).T).type(torch.LongTensor)]
        Aff = [torch.from_numpy(np.array(item['Aff'], dtype=np.int64).T).type(torch.FloatTensor)]

        Feature = [torch.from_numpy(np.full((item['Feature'][-1] + 1, item['Feature'][-1] + 1), 1, dtype=np.int64).T).type(torch.FloatTensor)]
        List.append([GA.cuda(), GB.cuda(), Aff.cuda(), Feature.cuda()])
    # print(Feature)
    return List
=======


def collate(batch):
    nodes_list = [b['GA'] for b in batch]
    nodes = np.concatenate(nodes_list, axis=0)
    nodes_lens = np.fromiter(map(lambda l: l.shape[0], nodes_list), dtype=np.int64)
    nodes_inds = np.cumsum(nodes_lens)
    nodes_num = nodes_inds[-1]
    nodes_inds = np.insert(nodes_inds, 0, 0)
    nodes_inds_2 = np.delete(nodes_inds, -1)
    GA_edges_list = [b['GA'] for b in batch]
    GA_edges_list = [e + i for e, i in zip(GA_edges_list, nodes_inds_2)]
    GA_edges = np.concatenate(GA_edges_list, axis=1)
    GB_edges_list = [b['GB'] for b in batch]
    GB_edges_list = [e + i for e, i in zip(GB_edges_list, nodes_inds_2)]
    GB_edges = np.concatenate(GB_edges_list, axis=1)
    size = [b['size'] for b in batch]
    size = np.array(size)
    # iso = [b['is_ISO']=='True' for b in batch]
    # iso = np.array(iso)
    # print(Feature)
    return [torch.from_numpy(GA_edges).type(torch.LongTensor), torch.from_numpy(GB_edges).type(torch.LongTensor), torch.from_numpy(size).type(torch.LongTensor)]
>>>>>>> vincent_branch


def collate_fn(batch):
    nodes_list = [b['Feature'] for b in batch]
    nodes = np.concatenate(nodes_list, axis=0)
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
    for i in range(len(batch)):
        Affinity[np.ix_([i for i in range(nodes_inds[i], nodes_inds[i + 1])], [i for i in range(nodes_inds[i], nodes_inds[i + 1])])] = batch[i]['Aff']

    Feature = np.full((nodes_num, nodes_num), 1, dtype=np.int64)
    # Feature = [e + i for e, i in zip(nodes_list, nodes_inds_2)]
    # Feature = np.concatenate(Feature, axis=0)

    return [torch.from_numpy(GA_edges).type(torch.LongTensor), torch.from_numpy(GB_edges).type(torch.LongTensor), torch.from_numpy(Affinity).type(torch.FloatTensor), torch.from_numpy(Feature).type(torch.FloatTensor)]


def select_data_tensorboard(data, writer, epoch):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    list = []
    data_A = data[0].cpu().detach().numpy()
    for row in range(len(data_A[1])):
        list.append((data_A[0][row], data_A[1][row]))
    G = nx.Graph(list)
    fig = plt.figure()
    nx.draw(G, pos=nx.spring_layout(G), with_labels=True)
    plt.draw()
    # writer.add_figure('epoch_%d_A' % epoch, fig)

    list_B = []
    data_B = data[1].cpu().detach().numpy()
    for row in range(len(data_B[1])):
        list_B.append((data_B[0][row], data_B[1][row]))
    G_B = nx.Graph(list_B)
    fig_B = plt.figure()
    nx.draw(G_B, pos=nx.spring_layout(G_B), with_labels=True)
    plt.draw()
    figs = [fig, fig_B]

<<<<<<< HEAD
    writer.add_figure('epoch_%d_GA_and_GB' % epoch, figs)
=======
    writer.add_figure('iter_%d_GA_and_GB' % epoch, figs)
>>>>>>> vincent_branch
    # writer.close()


def select_embedding_tensorboard(data, writer, epoch, identity):
    # data = data.numpy()
<<<<<<< HEAD
    fig, ax = plt.subplots(figsize=(8, 8))
=======
    fig, ax = plt.subplots(figsize=(12, 12))
>>>>>>> vincent_branch
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(data, cmap='cool')

    for (i, j), z in np.ndenumerate(data):
<<<<<<< HEAD
        ax.text(j, i, '{:^5.4f}'.format(z), ha='center', va='center')
=======
        ax.text(j, i, '{:.9f}'.format(z), ha='center', va='center')
>>>>>>> vincent_branch
    # fig.figure(figsize=(4, 4))
    # writer.add_figure('epoch_%d_GA_and_GB' % epoch, figs)
    # writer.close()
    return fig


<<<<<<< HEAD
def select_matrices(FA, FB, pred_b, pred, Aff, writer, epoch):
    fig1 = select_embedding_tensorboard(FA, writer, epoch, "FA")
    fig2 = select_embedding_tensorboard(FB, writer, epoch, "FB")
    fig3 = select_embedding_tensorboard(pred_b, writer, epoch, "pred_b")
    fig4 = select_embedding_tensorboard(pred, writer, epoch, "pred")
    fig5 = select_embedding_tensorboard(Aff, writer, epoch, "Aff")
    figs = [fig1, fig2, fig3, fig4, fig5]
    writer.add_figure('epoch_%d_FA_FB_pred_Aff' % epoch, figs)


def compare_matrix(FA, FB):
    index = np.where((np.isclose(FA, FB[:, None])).all(-1))
    Aff = np.full((np.size(FA, 0), np.size(FA, 1)), -1)
    for i in range(len(index[0])):
        Aff[index[1][i]][index[0][i]] = 1
    return Aff
=======
def select_matrices(FA, FB, pred_b, Aff, writer, epoch):
    fig1 = select_embedding_tensorboard(FA, writer, epoch, "FA")
    fig2 = select_embedding_tensorboard(FB, writer, epoch, "FB")
    fig3 = select_embedding_tensorboard(pred_b, writer, epoch, "pred_b")
    # fig4 = select_embedding_tensorboard(pred, writer, epoch, "pred")
    fig5 = select_embedding_tensorboard(Aff, writer, epoch, "Aff")
    figs = [fig1, fig2, fig3, fig5]
    writer.add_figure('image_%d_FA_FB_pred_Aff' % epoch, figs)


def compare_matrix(FA, FB):
    # print(type(FA[1][1]))
    # FA = np.round(FA, 9)
    # FB = np.round(FB, 9)
    # gsize10: rtol = 1e-10, atol= 1e-7 acc = 1.0
    # gsize100: rtol=1e-15 atol=1e-13 acc = 1.0
    # gsize100: rtol=1e-15 atol=1e-13 acc = 1.0
    index = np.where((np.isclose(FA, FB[:, None], rtol=1e-15, atol=1e-13)).all(-1))
    print(index)
    Aff = np.full((np.size(FA, 0), np.size(FA, 0)), -1)
    for i in range(len(index[0])):
        Aff[index[1][i]][index[0][i]] = 1
    return Aff


def compare_test_matrix(FA, FB):
    flag = 0
    # print(FA)
    # print(FB)
    # gsize10: rtol = 1e-10, atol= 1e-7 acc = 1.0
    # gsize100: rtol=1e-15 atol=1e-13 acc = 1.0
    # gsize100: rtol=1e-15 atol=1e-13 acc = 1.0
    # print(np.where((np.isclose(FA, FB[:, None], rtol=1e-15, atol=1e-13)).all(-1)))
    index = np.where((np.isclose(FA, FB[:, None], rtol=1e-15, atol=1e-13)).all(-1))
    # print(index)
    Aff = np.full((np.size(FA, 0), np.size(FA, 0)), 0)
    for i in range(len(index[0])):
        Aff[index[1][i]][index[0][i]] = 1
    # print(Aff)
    print(Aff.any(axis=1).all())
    return Aff.any(axis=1).all()


def remove_and_reorder(edge_list, index):
    print('removed nodes:{}'.format(index))
    # print(edge_list[0, :] == int(index))
    edge_list = edge_list[:, edge_list[0, :] != int(index)]
    edge_list = edge_list[:, edge_list[1, :] != int(index)]
    # print(edge_list.size())
    edge_list[edge_list[:, :] >= int(index)] -= 1
    # print(edge_list)
    return edge_list


def remove_node(FA, FB, data):
    # print(FA)
    # print(FB)
    index = np.where((np.isclose(FA, FB[:, None], rtol=1e-15, atol=1e-13)).all(-1))
    # print(index[0][1])
    Aff = np.full((np.size(FA, 0), np.size(FA, 0)), 0)
    # print(index[1][0])
    if(len(index[0]) == 0):
        print("!!!!")
        return data, False
    for i in range(len(index[0])):
        Aff[index[1][i]][index[0][i]] = 1
    # print(Aff)
    # if (Aff.any(axis=0).all() == False)
    #     print(Aff[Aff.any(axis=1)])
    # print(Aff.any(axis=0).all())

    data[0] = remove_and_reorder(data[0], index[1][0])
    data[1] = remove_and_reorder(data[1], index[0][0])
    print('num of edges {}'.format(data[0].size()))
    print('num of edges {}'.format(data[1].size()))
    return data, Aff.any(axis=1).all()
>>>>>>> vincent_branch
