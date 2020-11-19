
import torch
import torch_geometric as tg

import networkx as nx
import numpy as np

from torch_geometric.data import Data, Batch
from src.model.springmodel import SpringModel
from src.data_generater.spring import SpringSim

import matplotlib.pyplot as plt

def batch(batch_size, n_balls=10):
    model = SpringSim(n_balls=n_balls)
    pos, vel, adj = model.sample_trajectory((1+batch_size)*100)

    G = nx.from_numpy_array(adj)
    edge_index = torch.LongTensor(np.array(G.edges()).T)
    edge_index = tg.utils.to_undirected(edge_index)
    
    data_list = []
    for b in range(batch_size):
        pos_f = torch.Tensor(pos[b].T)
        pos_n = torch.Tensor(pos[b+1].T)
        vel_f = torch.Tensor(vel[b].T)
        vel_n = torch.Tensor(vel[b+1].T)

        data = Data(num_nodes=n_balls,
                    edge_index=edge_index, 
                    pos_f=pos_f,
                    pos_n=pos_n,
                    vel_f=vel_f,
                    vel_n=vel_n)
        data_list.append(data)
    batch = Batch.from_data_list(data_list)
    return batch

if __name__ == "__main__":
    cuda_id = "cuda:0"
    batch_size = 64
    pos_in_dim = 2
    vel_in_dim = 2
    edge_in_dim = 4
    hid_dim = 64
    num_epoch = 5000
    lr = 0.001
    n_balls = 10

    
    device = torch.device(cuda_id)
    model = SpringModel(pos_in_dim, edge_in_dim, vel_in_dim, hid_dim)
    model = model.to(device)
    model.load_state_dict(torch.load("./checkpoints/spring_model.pt"))

    val_batch = batch(64, n_balls=10).to(device)
    pos_hat, vel_hat = model(val_batch)
    print(pos_hat - val_batch.pos_n)
    print(val_batch.pos_f - val_batch.pos_n)

    print(pos_hat.shape)
    print(vel_hat.shape)

    
