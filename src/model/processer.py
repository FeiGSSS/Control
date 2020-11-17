import torch
import torch.nn as nn
import torch.nn.functional as F

class Processer(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(Processer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, hidden):
        r"""positions has size of [N, in_dim],
        representing N nodes' hidden status"""
        return self.encoder(hidden)