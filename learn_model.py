import warnings
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import time

from tqdm import tqdm
from  multiprocessing import Pool

import torch
import torch.optim as optim
import torch_geometric as tg
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data, Batch
from src.model.springmodel import SpringModel
from src.data_generater.spring import SpringSim

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore")

def batch(batch_size, n_balls=10, _delta_T=0.001):

    sample_freq = 100
    sample_t0 = np.random.choice(range(50, 500))
    T = (sample_t0+batch_size)*sample_freq

    model = SpringSim(n_balls=n_balls, _delta_T=_delta_T)
    pos, vel, adj = model.sample_trajectory(T, sample_freq)
    # 模拟初始态的速度为0， 不宜采用作为训练输入
    # 故取中间（t=10）作为初始态
    pos = pos[sample_t0:]
    vel = vel[sample_t0:]
    adj = adj

    G = nx.from_numpy_array(adj)
    edge_index = torch.LongTensor(np.array(G.edges()).T)
    edge_index = tg.utils.to_undirected(edge_index)

    pos_0 = torch.Tensor(pos[0])
    vel_0 = torch.Tensor(vel[0])

    pos_res = torch.Tensor(pos[1:])
    vel_res = torch.Tensor(vel[1:])

    delta_t = torch.arange(batch_size)*(_delta_T*sample_freq)
    data = Data(num_nodes=n_balls,
                edge_index=edge_index, 
                pos_0=pos_0.transpose(0,1),
                pos_res=pos_res.transpose(1,2),
                vel_0=vel_0.transpose(0,1),
                vel_res=vel_res.transpose(1,2),
                delta_t=delta_t)
    return data

def generate_data(datasize, new_data=False):
    dataset = []
    print("--> Generating Dataset")
    t0 = time.time()
    if new_data:
        pool = Pool(30)
        process = []
        for _ in range(datasize):
            process.append(pool.apply_async(batch, (batch_size, n_balls)))
        pool.close()
        pool.join()
        for res in process:
            dataset.append(res.get())
        
        with open("./checkpoints/dataset_ode.pkl", "wb") as f:
            pkl.dump(dataset, f)
    else:
        with open("./checkpoints/dataset_ode.pkl", "rb") as f:
            dataset = pkl.load(f)
    print("-->DataSize = {}, Time = {:.1f}s".format(datasize, time.time()-t0))
    return dataset

if __name__ == "__main__":
    # args
    cuda_id = "cuda:0"
    batch_size = 30
    pos_in_dim = 2
    vel_in_dim = 2
    edge_in_dim = 4
    hid_dim = 64
    num_epoch = 60000
    lr = 0.0001
    n_balls = 20
    #random_n_balls = True
    datasize = 3000
    new_data = False
    noise = None
    flag = "base"

    
    device = torch.device(cuda_id)
    model = SpringModel(pos_in_dim, edge_in_dim, vel_in_dim, hid_dim)
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=200, gamma=0.97, verbose=False)
    
    train_set = generate_data(datasize, new_data)
    val_batch = batch(batch_size, n_balls=n_balls).to(device)

    train_loss = []
    val_loss = []
    for epoch in range(num_epoch):
        t0 = time.time()
        opt.zero_grad()
        model.train()
        train_batch = train_set[np.random.choice(range(datasize))].to(device)
        model.edge_index = train_batch.edge_index
        node_f = torch.cat((train_batch.pos_0, train_batch.vel_0), dim=1)
        node_n = torch.cat((train_batch.pos_res, train_batch.vel_res), dim=2)
        pred_node_n = odeint(func=model, y0=node_f, t=train_batch.delta_t, method="rk4")
        loss = model.loss_fn(pred_node_n[1:], node_n)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 50 == 0:
            train_loss.append(loss.item())
            model.eval()
            model.edge_index = val_batch.edge_index
            node_f = torch.cat((val_batch.pos_0, val_batch.vel_0), dim=1)
            node_n = torch.cat((val_batch.pos_res, val_batch.vel_res), dim=2)
            pred_node_n = odeint(func=model, y0=node_f, t=val_batch.delta_t, method="rk4")
            loss = model.loss_fn(pred_node_n[1:], node_n)
            val_loss.append(loss.item())
            print("Epoch = {:<5}: train = {:.10f}, val = {:.10f}".format(
                epoch, train_loss[-1], val_loss[-1]))
            torch.save(model.state_dict(), "./checkpoints/spring_{}_model.pt".format(flag))
            np.save("./checkpoints/train_loss_{}.npy".format(flag), train_loss)
            np.save("./checkpoints/val_loos_{}.npy".format(flag), val_loss)

    torch.save(model.state_dict(), "./checkpoints/spring_{}_model.pt".format(flag))
