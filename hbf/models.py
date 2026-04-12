"""HBF and standard NN layer/network builders in PyTorch.

Ported from hbf_nn_builder.py (TensorFlow 1.x) to PyTorch nn.Module.
"""

import torch
import torch.nn as nn
import numpy as np


class HBFLayer(nn.Module):
    """Single Hierarchical Basis Function layer.

    Computes: A = exp(beta * (2*X@W - (||X||^2 + ||W||^2)))
    where beta = 1/S^2, W are learned templates, S is learned bandwidth.
    """

    def __init__(self, dim_in: int, dim_out: int, S_init: float = 0.5,
                 use_bn: bool = False, trainable_bn: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim_in, dim_out, dtype=torch.float64) * 0.1)
        self.S = nn.Parameter(torch.full((dim_out,), S_init, dtype=torch.float64))
        self.bn = nn.BatchNorm1d(dim_out, dtype=torch.float64, affine=trainable_bn) if use_bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # beta = 1/S^2
        beta = (1.0 / self.S) ** 2  # (D_out,)

        # Efficient squared distance: -||x - w||^2 = 2<x,w> - ||x||^2 - ||w||^2
        WW = (self.W * self.W).sum(dim=0, keepdim=True)  # (1, D_out)
        XX = (x * x).sum(dim=1, keepdim=True)             # (M, 1)
        Delta_tilde = 2.0 * x @ self.W - (WW + XX)        # (M, D_out)

        Z = beta * Delta_tilde  # (M, D_out)

        if self.bn is not None:
            Z = self.bn(Z)

        A = torch.exp(Z)  # (M, D_out)
        return A


class StandardNNLayer(nn.Module):
    """Standard neural network layer with optional batch normalization."""

    def __init__(self, dim_in: int, dim_out: int, use_bn: bool = False,
                 trainable_bn: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, dtype=torch.float64)
        self.bn = nn.BatchNorm1d(dim_out, dtype=torch.float64, affine=trainable_bn) if use_bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        if self.bn is not None:
            z = self.bn(z)
        return torch.relu(z)


class HBFNetwork(nn.Module):
    """Multi-layer HBF network with final linear output layer.

    Architecture: input -> HBFLayer_1 -> ... -> HBFLayer_{L-1} -> Linear -> output
    """

    def __init__(self, dims: list, S_init: list = None,
                 use_bn: bool = False, trainable_bn: bool = True):
        super().__init__()
        self.dims = dims
        nb_hidden = len(dims) - 2  # exclude input and output dims

        if S_init is None:
            S_init = [0.5] * (nb_hidden + 1)

        layers = []
        for l in range(nb_hidden):
            layers.append(HBFLayer(dims[l], dims[l + 1], S_init=S_init[l + 1],
                                   use_bn=use_bn, trainable_bn=trainable_bn))
        self.hidden_layers = nn.ModuleList(layers)

        # Final linear (summation) layer: maps last hidden dim to output dim
        self.output_layer = nn.Linear(dims[-2], dims[-1], bias=False, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class StandardNNNetwork(nn.Module):
    """Multi-layer standard NN with final linear output layer.

    Architecture: input -> NNLayer_1 -> ... -> NNLayer_{L-1} -> Linear -> output
    """

    def __init__(self, dims: list, use_bn: bool = False, trainable_bn: bool = True):
        super().__init__()
        self.dims = dims
        nb_hidden = len(dims) - 2

        layers = []
        for l in range(nb_hidden):
            layers.append(StandardNNLayer(dims[l], dims[l + 1],
                                          use_bn=use_bn, trainable_bn=trainable_bn))
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(dims[-2], dims[-1], bias=False, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
