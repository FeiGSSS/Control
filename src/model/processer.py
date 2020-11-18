import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder import EdgeEncoder
from torch_scatter import scatter_add

class Processer(nn.Module):
    def __init__(self, hid_dim):
        super(Processer, self).__init__()
        self.edge_index = []
        self.edge_encoder = EdgeEncoder(2*hid_dim, hid_dim)
        self.update = nn.Sequential(
            nn.Linear(2*hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, hidden):
        r"""hidden has size of [N, hid_dim],
        representing N nodes' hidden status"""
        edge_feat = torch.cat((hidden[self.edge_index[0]], 
                               hidden[self.edge_index[1]]), dim=1) # >> E*(2*hid_dim)
        edge_feat = self.edge_encoder(edge_feat) # >> E*(hid_dim)
        neighbor_agg_hiddden = scatter_add(src=edge_feat, 
                                            index=self.edge_index[1], dim=0) # N*hid_dim
        cat_feat = torch.cat((hidden, neighbor_agg_hiddden), dim=1) # N*(2*hid_dim)
        return self.update(cat_feat)