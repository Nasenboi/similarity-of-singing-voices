"""
Copied and adapted from:
https://github.com/difra100/GATSY-Music_Artist_Similarity/blob/main/reduced_data_experiments/architectures.py

For more information see model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATSY(nn.Module):
    """This architecture is one of the 2 GAT networks, its aim is to be improved in order to see how it performs on the artist similarity task"""

    def __init__(self, n_heads, n_layers, input_shape: int, layer_size: int = 256):
        super(GATSY, self).__init__()

        """ The Batch normalization layers have been introduced to speed the training up, and indeed to obtain better results. """

        self.batch1 = torch.nn.BatchNorm1d(layer_size)
        self.batch2 = torch.nn.BatchNorm1d(layer_size)
        self.batch3 = torch.nn.BatchNorm1d(layer_size)
        self.batch4 = torch.nn.BatchNorm1d(n_heads * layer_size)

        self.n_layer = n_layers
        GAT_l = []
        self.GAT1 = GATConv(layer_size, layer_size, heads=n_heads, bias=True)
        for n in range(n_layers - 1):
            GAT_l.append(GATConv(n_heads * layer_size, layer_size, heads=n_heads, bias=True))

        self.GAT_l = nn.Sequential(*GAT_l)

        batch_l = []

        for n in range(n_layers - 1):
            batch_l.append(torch.nn.BatchNorm1d(n_heads * layer_size))

        self.batch_l = nn.Sequential(*batch_l)

        self.linear1 = nn.Linear(input_shape, layer_size)
        self.linear2 = nn.Linear(layer_size, layer_size)
        self.linear3 = nn.Linear(layer_size, layer_size)

    def forward(self, x, edges):

        x = self.linear1(x)
        x = self.batch1(x)
        x = F.elu(x)
        x = self.linear2(x)
        x = self.batch2(x)
        x = F.elu(x)
        x = self.linear3(x)
        x = self.batch3(x)
        x = F.elu(x)

        x = self.GAT1(x, edges)
        x = self.batch4(x)
        x = F.elu(x)

        for i, layer in enumerate(self.GAT_l):
            x = layer(x, edges)  # + x
            x = self.batch_l[i](x)
            x = F.elu(x)

        return x
