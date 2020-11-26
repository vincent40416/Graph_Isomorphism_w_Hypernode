import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from preprocessing import GraphDataset
from utils import collate_fn
from utils import select_data_tensorboard
from utils import select_embedding_tensorboard
from model import GCN
import networkx as nx
import matplotlib.pyplot as plt
import datetime
log_dir = "runs/Test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)
writer.close()
df = pd.read_csv('./Graph_dataset.csv')
dftrain = df[:int(len(df) * 0.01)]
data_loader = DataLoader(GraphDataset(dftrain), batch_size=1, shuffle=True, collate_fn=collate_fn, drop_last=True)

for epoch in range(0, 10):
    epoch_loss = 0
    for iter, data in enumerate(data_loader):
        if(iter < 10):
            continue
        # print(data.size())
        # print(data[0].size())
        select_data_tensorboard(data, writer, epoch)

        select_embedding_tensorboard(data[2], writer, epoch, 'Aff')
        break
    break
writer.close()
