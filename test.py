import numpy as np
from utils import compare_matrix
import matplotlib.pyplot as plt
from networkx import nx
from random import randint
import numpy as np


n = 100  # 10 nodes
p = 2 * np.log(n) / n  # 20 edges
np.set_printoptions(threshold=np.inf)
G = nx.gnp_random_graph(n, p, seed=randint(1, 100))
graph_np = nx.to_numpy_matrix(G)
print(graph_np)
