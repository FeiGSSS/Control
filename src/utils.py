# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/11/28 17:23:14
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import sys
sys.path.append("..")

import time
import numpy as np
import networkx as nx
import pickle as pkl
from  multiprocessing import Pool
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch_geometric as tg
from torch_geometric.data import Data, Batch

from src.data_generater.spring import SpringSim


def sample(prediction_step , n_balls, _delta_T=0.001, sample_freq = 100):
    """
    This function generate training data for IVP(init value problem) prediction.
    Sampling time intervel is (_delta_T * sample_freq), i.e. regular intervel.
    Input:
        --prediction_step: int
        --n_balls
        --_delta_T: minimum time intervel of simulation
        --sample_freq: 
    Notices:
    (_delta_T*sample_freq) is the time intervel of output data
    """
    # 开始采样的时间点随机设置，增加样本的多样性
    sample_t0 = np.random.choice(range(10, 500))

    T = (sample_t0+prediction_step)*sample_freq
    model = SpringSim(n_balls=n_balls, _delta_T=_delta_T)
    pos, vel, adj = model.sample_trajectory(T, sample_freq)
    pos = pos[sample_t0:]
    vel = vel[sample_t0:]

    G = nx.from_numpy_array(adj)
    edge_index = torch.LongTensor(np.array(G.edges()).T)
    edge_index = tg.utils.to_undirected(edge_index)

    pos_0 = torch.Tensor(pos[0])
    vel_0 = torch.Tensor(vel[0])

    pos_res = torch.Tensor(pos[1:])
    vel_res = torch.Tensor(vel[1:])

    delta_t = torch.arange(prediction_step)*(_delta_T*sample_freq)
    data = Data(num_nodes=n_balls,
                edge_index=edge_index, 
                pos_0=pos_0.transpose(0,1),
                pos_res=pos_res.transpose(1,2),
                vel_0=vel_0.transpose(0,1),
                vel_res=vel_res.transpose(1,2),
                delta_t=delta_t)
    return data

def generate_data(new_data, data_save_path, data_size, **kargs):
    """
    generate or load dataset for training with config **kargs
    """
    dataset = []
    print("--> Generating Dataset")
    t0 = time.time()
    if new_data:
        pool = Pool(60)
        process = []
        for _ in range(data_size):
            process.append(pool.apply_async(sample, tuple(kargs.values())))
        pool.close()
        pool.join()
        for res in tqdm(process):
            dataset.append(res.get())
        
        with open(data_save_path, "wb") as f:
            pkl.dump(dataset, f)
    else:
        with open(data_save_path, "rb") as f:
            dataset = pkl.load(f)
    print("-->DataSize = {}, Time = {:.1f}s".format(data_size, time.time()-t0))
    return dataset
 
class mydataset(Dataset):
    """
    This is a custom class of dataset
    """
    def __init__(self, data_list):
        super(mydataset, self).__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

def collate_fn(graph_list):
    """
    batch_list contains pyg.Data like:
    Data(delta_t=[501], edge_index=[2, 162], pos_0=[30, 2], pos_res=[500, 30, 2], vel_0=[30, 2], vel_res=[500, 30, 2])
    delta_t is the same for every sample, i.e. this version of dataloader does not support irregular sampling.
    500 or 501 is the No. of time steps
    162 is the Np. of edges
    30 if the #of nodes
    """
    assert len(graph_list) >= 1
    if len(graph_list) == 1:
        return graph_list[0]
    batch_delta_t = graph_list[0].delta_t
    batch_pos_res = []
    batch_vel_res = []
    batch = Batch.from_data_list(graph_list)
    for g in graph_list:
        batch_pos_res.append(g.pos_res)
        batch_vel_res.append(g.vel_res)
    batch.delta_t = batch_delta_t
    batch.pos_res = torch.cat(batch_pos_res, dim=1)
    batch.vel_res = torch.cat(batch_vel_res, dim=1)

    return batch
