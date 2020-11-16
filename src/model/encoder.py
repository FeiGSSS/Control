import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(PositionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, positions):
        r"""positions has size of [N, in_dim],
        representing N nodes' postions"""

        return self.encoder(positions)

class VelocEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(VelocEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim)
        )
    def forward(self, velocs):
        r"""velocs has size of [N, in_dim],
        representing N nodes' velocities"""
        return self.encoder(velocs)

class EdgeEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(EdgeEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, edges):
        r"""velocs has size of [E, in_dim],
        representing N nodes' edge features"""
        return self.encoder(edges)