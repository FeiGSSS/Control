import torch
import torch.optim as optim
import torch_geometric as tg

import networkx as nx
import numpy as np

from torch.optim.lr_scheduler import StepLR
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
    # args
    cuda_id = "cuda:0"
    batch_size = 128
    pos_in_dim = 2
    vel_in_dim = 2
    edge_in_dim = 4
    hid_dim = 64
    num_epoch = 10000
    lr = 0.001
    n_balls = 20

    flag = "base"

    
    device = torch.device(cuda_id)
    model = SpringModel(pos_in_dim, edge_in_dim, vel_in_dim, hid_dim)
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=100, gamma=0.9, verbose=False)

    val_batch = batch(batch_size, n_balls=n_balls).to(device)

    train_loss = []
    val_loss = []
    for epoch in range(num_epoch):
        opt.zero_grad()
        model.train()
        train_batch = batch(batch_size, n_balls=n_balls).to(device)
        model.edge_index = train_batch.edge_index
        node_f = torch.cat((train_batch.pos_f, train_batch.vel_f), dim=1)
        node_n = torch.cat((train_batch.pos_n, train_batch.vel_n), dim=1)
        pred_node_n = model(node_f)
        loss = model.loss_fn(pred_node_n, node_n)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 50 == 0:
            train_loss.append(loss.item())
            model.eval()
            model.edge_index = val_batch.edge_index
            node_f = torch.cat((val_batch.pos_f, val_batch.vel_f), dim=1)
            node_n = torch.cat((val_batch.pos_n, val_batch.vel_n), dim=1)
            pred_node_n = model(node_f)
            loss = model.loss_fn(pred_node_n, node_n)
            val_loss.append(loss.item())
            print("Epoch = {:<3}: train = {:.10f}, val = {:.10f}".format(
                epoch, train_loss[-1], val_loss[-1]))
            torch.save(model.state_dict(), "./checkpoints/spring_{}_model.pt".format(flag))
            np.save("./checkpoints/train_loss_{}.npy".format(flag), train_loss)
            np.save("./checkpoints/val_loos_{}.npy".format(flag), val_loss)

    torch.save(model.state_dict(), "./checkpoints/spring_{}_model.pt".format(flag))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(["train", "val"])
    plt.savefig("./figs/loss_{}.pdf".format(flag))
