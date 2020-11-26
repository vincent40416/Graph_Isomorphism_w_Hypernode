import torch
import torch.nn as nn
from model import GNN_Geo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GNN_Geo(10, 64).cuda()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
model.to(device)
print(device)
