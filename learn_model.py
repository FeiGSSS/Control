import warnings
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import time

from tqdm import tqdm

import torch
import torch.optim as optim
import torch_geometric as tg
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data, Batch
from src.model.springmodel import SpringModel
from src.data_generater.spring import SpringSim
from src.utils import generate_data, sample
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint


torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")



if __name__ == "__main__":
    # args
    cuda_id = "cuda:1"
    prediction_step = 40
    pos_in_dim = 2
    vel_in_dim = 2
    edge_in_dim = 4
    hid_dim = 64
    num_epoch = 60000
    lr = 0.001
    n_balls = 30
    datasize = 3000
    new_data = True
    noise = None
    flag = "base"

    data_save_path = "./data/dataset_nb_{:d}_step_{:d}.pkl".format(n_balls, prediction_step)
    model_save_path = "./checkpoints/spring_{}_model.pt".format(flag)
    train_loss_save = "./res/train_loss_{}.npy".format(flag)
    val_loss_save   = "./res/val_loss_{}.npy".format(flag)
    
    device = torch.device(cuda_id)
    model = SpringModel(pos_in_dim, edge_in_dim, vel_in_dim, hid_dim)
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=100, gamma=0.95, verbose=False)
    
    train_set = generate_data(new_data,
                              data_save_path,
                              datasize,
                              prediction_step=prediction_step,
                              n_balls=n_balls)
    val_batch = sample(prediction_step, n_balls=n_balls).to(device)

    train_loss = []
    val_loss = []
    t0 = time.time()
    for epoch in range(num_epoch):
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
            torch.save(model.state_dict(), model_save_path)
            np.save(train_loss_save, train_loss)
            np.save(val_loss_save, val_loss)

            print("Epoch = {:<5}: train = {:.10f}, val = {:.10f}, Time = {:.1f}s".format(
                epoch, train_loss[-1], val_loss[-1], time.time()-t0))
            t0 = time.time()

    torch.save(model.state_dict(), model_save_path)
