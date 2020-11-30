# -*- encoding: utf-8 -*-
'''
@File    :   learn_control.py
@Time    :   2020/11/29 21:29:31
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import warnings


import warnings

import torch
from torch.utils import data
import torch_geometric as tg

from src.model.springmodel import SpringModel
from src.data_generater.spring import SpringSim
from src.utils import sample
from src.model.control import control

from torchdiffeq import odeint

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    cuda_id = "cuda:1"
    device = torch.device(cuda_id)
    pos_in_dim = 2
    vel_in_dim = 2
    edge_in_dim = 4
    hid_dim = 64

    # data
    control_time_steps = 400
    n_balls = 30
    data = sample(control_time_steps, n_balls).to(device)
    print(data)
    # expected final state
    xf = None
    # load trained model
    trained_model = SpringModel(pos_in_dim, edge_in_dim, vel_in_dim, hid_dim)
    trained_model.load_state_dict(torch.load("./checkpoints/spring_batch_model.pt"))   
    trained_model = trained_model.to(device)
    trained_model.edge_index = data.edge_index

    # define control part
    control_model = control(trained_model, n_balls, 4, xf, data.delta_t, hid_dim)
    print(control_model)

