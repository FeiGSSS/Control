# -*- encoding: utf-8 -*-
'''
@File    :   control.py
@Time    :   2020/11/29 21:30:32
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from src.model.encoder import NodeEncoder, EdgeEncoder
from src.model.decoder import NodeDecoder


class control(nn.Module):
    def __init__(self, ode_model, model_size, model_dim, xf, T, hid_dim):
        r"""
        --ode_model: nn.Module, a pre-trained ode model
        --model_size: int, the system's size, i.e. the number of variables
        --model_dim: int, the dim of each variable (e.g. 4 for spring model)
        --xf, tensor, [2, model_dim]. expected states of system
        --T, list, the allowed time points to control
        """
        super(control, self).__init__()
        self.ode_model = ode_model
        self.model_size = model_size
        self.model_dim = model_dim
        self.xf = xf
        self.T = T
        self.hid_dim = hid_dim

        self.edge_index = None

        self.node_encoder = NodeEncoder(in_dim=2+model_dim, hid_dim=hid_dim)
        self.edge_encoder = EdgeEncoder(in_dim=2*hid_dim, hid_dim=hid_dim)
        self.node_decoder = NodeDecoder(in_dim=2*hid_dim, hid_dim=2)

    def forward(self, t, node_f):
        with torch.no_grad(): 
            node_n_model = self.ode_model(t, node_f)
        vel_control = self.controled_model(t, node_f)
        node_n_model[:, -2:] += vel_control
        return node_n_model

    def controled_model(self, t, node_f):
        assert self.edge_index is not None
        node_feature = torch.cat((self.xf, node_f), dim=1)
        node_emb = self.node_encoder(node_feature)
        edge_feature = torch.cat((node_emb[self.edge_index[0]], 
                                  node_emb[self.edge_index[1]]), dim=1) # >> E*4
        edge_emb = self.edge_encoder(edge_feature)
        node_aggregate = scatter_add(edge_emb, self.edge_index[1])
        node_emb = torch.cat((node_emb, node_aggregate), dim=1)

        vel_control = self.node_decoder(node_emb)
        return vel_control
