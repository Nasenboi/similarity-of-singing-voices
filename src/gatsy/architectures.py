"""
Copied and adapted from:
https://github.com/difra100/GATSY-Music_Artist_Similarity/blob/main/full_data_experiments/src/architectures.py

For more information see model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATSY(nn.Module):
    """
    Graph Attention Network (GATSY) for artist similarity task.
    - Customizable GAT layers: number of GAT layers is controlled by `n_layers`.
    - Fixed back-end: two linear layers with batch normalization and ELU activations.
    - Batch normalization included after each layer for faster training and better convergence.
    """

    def __init__(self, n_heads, n_layers, input_dim=2613, hidden_dim=256, output_dim=256):
        """
        Args:
            n_heads (int): Number of attention heads in GAT layers.
            n_layers (int): Number of GAT layers.
            input_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden feature dimensionality.
            output_dim (int): Output feature dimensionality.
        """
        super(GATSY, self).__init__()

        # GAT layers
        self.gat_layers = nn.ModuleList()
        # self.batch_norms = nn.ModuleList()

        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=n_heads, concat=True))
        # self.batch_norms.append(nn.BatchNorm1d(n_heads * hidden_dim))

        for _ in range(n_layers - 1):
            self.gat_layers.append(GATConv(n_heads * hidden_dim, hidden_dim, heads=n_heads, concat=True))
            # self.batch_norms.append(nn.BatchNorm1d(n_heads * hidden_dim))

        # Back-end: Two fixed linear layers
        self.linear1 = nn.Linear(n_heads * hidden_dim, hidden_dim)
        # self.batch1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edges):
        """
        Args:
            x (Tensor): Input node features of shape [num_nodes, input_dim].
            edges (Tensor): Edge indices of shape [2, num_edges].
        Returns:
            Tensor: Node embeddings of shape [num_nodes, output_dim].
        """
        # GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = F.elu(gat_layer(x, edges))
            x = F.normalize(x, p=2, dim=1)

        # Back-end linear layers
        x = F.elu(self.linear1(x))
        x = self.linear2(x)

        return x
