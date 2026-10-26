"""Weight initialization strategies for HBF and standard NN networks.

Ported from initializations.py (TensorFlow) to work with PyTorch tensors.
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans


def _get_centers_from_data(X_train: np.ndarray, n_centers: int) -> np.ndarray:
    """Subsample data points to use as RBF centers."""
    N = X_train.shape[0]
    indices = np.random.choice(N, size=n_centers, replace=False)
    return X_train[indices, :]  # (n_centers, D)


def _get_kpp_centers(X: np.ndarray, n_centers: int) -> np.ndarray:
    """K-means++ initialization for RBF centers."""
    km = KMeans(n_clusters=n_centers, init='k-means++', n_init=1, max_iter=1)
    km.fit(X)
    return km.cluster_centers_  # (n_centers, D)


def _solve_output_weights(X_train, Y_train, centers, stddev):
    """Solve for output layer weights C via kernel least squares: KC = Y."""
    beta = (1.0 / stddev) ** 2
    Kern = np.exp(-beta * euclidean_distances(X=X_train, Y=centers, squared=True))
    C, _, _, _ = np.linalg.lstsq(Kern, Y_train, rcond=None)
    return C, Kern


# -- HBF initializations --

def init_hbf_truncated_normal(model, dims, mu=0.0, std=0.9, S_init=0.5):
    """Initialize HBF network with truncated normal weights."""
    with torch.no_grad():
        for i, layer in enumerate(model.hidden_layers):
            layer.W.data = torch.clamp(
                torch.randn_like(layer.W) * std + mu, -2 * std, 2 * std
            )
            layer.S.data.fill_(S_init)
        model.output_layer.weight.data = torch.clamp(
            torch.randn_like(model.output_layer.weight) * std + mu, -2 * std, 2 * std
        )


def init_hbf_data(model, X_train, Y_train, dims, S_init=0.5, mu=0.0, std=0.9):
    """Initialize first HBF layer templates from data samples."""
    with torch.no_grad():
        centers = _get_centers_from_data(X_train, dims[1])
        W = centers.T  # (D, n_centers)
        model.hidden_layers[0].W.data = torch.from_numpy(W).double()
        for layer in model.hidden_layers:
            layer.S.data.fill_(S_init)
        model.output_layer.weight.data = torch.clamp(
            torch.randn_like(model.output_layer.weight) * std + mu, -2 * std, 2 * std
        )


def init_hbf_kernel(model, X_train, Y_train, dims, S_init=0.5):
    """Initialize HBF with data centers and solve output layer via kernel LS."""
    with torch.no_grad():
        centers = _get_centers_from_data(X_train, dims[1])
        W = centers.T
        model.hidden_layers[0].W.data = torch.from_numpy(W).double()
        for layer in model.hidden_layers:
            layer.S.data.fill_(S_init)
        C, _ = _solve_output_weights(X_train, Y_train, centers, S_init)
        model.output_layer.weight.data = torch.from_numpy(C.T).double()


def init_hbf_kpp(model, X_train, Y_train, dims, S_init=0.5):
    """Initialize HBF with K-means++ centers and kernel LS output."""
    with torch.no_grad():
        centers = _get_kpp_centers(X_train, dims[1])
        W = centers.T
        model.hidden_layers[0].W.data = torch.from_numpy(W).double()
        for layer in model.hidden_layers:
            layer.S.data.fill_(S_init)
        C, _ = _solve_output_weights(X_train, Y_train, centers, S_init)
        model.output_layer.weight.data = torch.from_numpy(C.T).double()


# -- Standard NN initializations --

def init_nn_truncated_normal(model, mu=0.0, std=0.9, b_init=0.5):
    """Initialize standard NN with truncated normal weights."""
    with torch.no_grad():
        for layer in model.hidden_layers:
            layer.linear.weight.data = torch.clamp(
                torch.randn_like(layer.linear.weight) * std + mu, -2 * std, 2 * std
            )
            layer.linear.bias.data.fill_(b_init)
        model.output_layer.weight.data = torch.clamp(
            torch.randn_like(model.output_layer.weight) * std + mu, -2 * std, 2 * std
        )
