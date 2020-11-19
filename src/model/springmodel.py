import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch_scatter import scatter_add

from src.model.encoder import PositionEncoder, EdgeEncoder, VelocEncoder
from src.model.processer import Processer
from src.model.decoder import PositionDecoder, VelocDecoder

from torchdiffeq import odeint_adjoint as odeint

class SpringModel(nn.Module):
    def __init__(self, pos_in_dim=2,
                 edge_in_dim=4, vel_in_dim=2, hid_dim=64):
        super(SpringModel, self).__init__()
        self.edge_index = []

        self.pos_encoder = PositionEncoder(pos_in_dim, hid_dim)
        self.edge_encoder = EdgeEncoder(edge_in_dim, hid_dim)
        self.vel_encoder = VelocEncoder(vel_in_dim, hid_dim)

        self.process = Processer(3*hid_dim, hid_dim)

        self.pos_decoder = PositionDecoder(hid_dim, pos_in_dim)
        self.vel_decoder = VelocDecoder(hid_dim, vel_in_dim)

        self.loss_fn = nn.MSELoss()

    def forward(self, node_f):
        r"""node_f has size of N*4, and it's the concatenane of nodes position and veloc"""
        assert self.edge_index is not []
        pos = node_f[:,:2] # >> N*2
        vel = node_f[:,2:] # >> N*2
        edge_index = self.edge_index # 2*E, undirected
        edge_in_feat = torch.cat((pos[edge_index[0]], 
                                  pos[edge_index[1]]), dim=1) # >> E*4
        pos_hid  = self.pos_encoder(pos)
        vel_hid  = self.vel_encoder(vel)
        edge_hid = self.edge_encoder(edge_in_feat)
        neighbor_aggs_hid = scatter_add(src=edge_hid, index=edge_index[1], dim=0)
        
        node_hidden_status = self.process(torch.cat(
            (pos_hid, vel_hid, neighbor_aggs_hid), dim=1))
        # node_hidden_status = odeint(self.process, 
        #     torch.cat((pos_hid, vel_hid, neighbor_aggs_hid), dim=1), 
        #     torch.Tensor([0, 0.001]))

        pos_hat = self.pos_decoder(node_hidden_status)
        vel_hat = self.vel_decoder(node_hidden_status)

        return torch.cat((pos_hat, vel_hat), dim=1)


        
        











