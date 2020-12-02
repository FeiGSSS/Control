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
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
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
    lr = 5e-5

    # data
    control_time_steps = 100
    n_balls = 30
    data = sample(control_time_steps, n_balls).to(device)
    # expected final state
    xf = data.vel_res[-1, :, :].squeeze()
    scale = torch.arange(control_time_steps).to(device)/control_time_steps
    # load trained model
    trained_model = SpringModel(pos_in_dim, edge_in_dim, vel_in_dim, hid_dim)
    trained_model.load_state_dict(torch.load("./checkpoints/spring_batch_model.pt"))   
    trained_model = trained_model.to(device)
    trained_model.edge_index = data.edge_index

    # define control part
    control_model = control(trained_model, n_balls, 4, xf, data.delta_t, hid_dim)
    control_model.U.edge_index = data.edge_index
    control_model = control_model.to(device)

    # optimization
    opt = optim.SGD(control_model.U.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=10, gamma=0.5, verbose=False)

    state_0 = torch.cat((data.pos_0, data.vel_0), dim=1)
    for epoch in range(500):
        states = odeint(control_model, y0=state_0, t=data.delta_t, method="rk4")
        vel = states[-1, :, -2:].squeeze()
        loss = F.mse_loss(vel, xf)
        loss.backward()
        opt.step()
        scheduler.step()
        print("Epoch = {:<3}, Loss = {:.3f}".format(epoch, loss.item()))
        if epoch % 10 == 0:
            print(vel-xf)
