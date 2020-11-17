import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(PositionDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, hid_dim)
        )

    def forward(self, positions):
        return self.encoder(positions)

class VelocDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(VelocDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, hid_dim)
        )

    def forward(self, positions):
        return self.encoder(positions)

