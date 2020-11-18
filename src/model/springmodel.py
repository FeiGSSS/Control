import torch
import torch.nn as nn
import torch.nn.functional as F 



from src.model.encoder import PositionEncoder, VelocEncoder, NodeEncoder
from src.model.processer import Processer
from src.model.decoder import PositionDecoder, VelocDecoder

from torchdiffeq import odeint_adjoint as odeint

class SpringModel(nn.Module):
    def __init__(self, pos_in_dim=2,
                 edge_in_dim=4, vel_in_dim=2, hid_dim=64):
        super(SpringModel, self).__init__()

        self.pos_encoder = PositionEncoder(pos_in_dim, hid_dim)
        self.vel_encoder = VelocEncoder(vel_in_dim, hid_dim)
        self.node_encoder = NodeEncoder(2*hid_dim, hid_dim)


        self.process = Processer(hid_dim)

        self.pos_decoder = PositionDecoder(hid_dim, pos_in_dim)
        self.vel_decoder = VelocDecoder(hid_dim, vel_in_dim)

        self.loss_fn = nn.MSELoss()

    def forward(self, graph):
        r"""graph should be an undirected graph"""
        pos = graph.pos_f # >> N*2
        vel = graph.vel_f # >> N*2
        pos_hid  = self.pos_encoder(pos)
        vel_hid  = self.vel_encoder(vel)
        node_hidden = self.node_encoder(torch.cat((pos_hid, vel_hid), dim=1)) # >> N*(hid_dim)

        self.process.edge_index = graph.edge_index
        node_hidden = self.process(node_hidden)        

        pos_hat = self.pos_decoder(node_hidden)
        vel_hat = self.vel_decoder(node_hidden)

        return pos_hat, vel_hat

    def loss(self, graph):
        pos_true = graph.pos_n
        vel_true = graph.vel_n

        pos_hat, vel_hat = self.forward(graph)

        loss_pos = self.loss_fn(pos_hat, pos_true)
        loss_vel = self.loss_fn(vel_hat, vel_true)

        return loss_pos + loss_vel


        
        











