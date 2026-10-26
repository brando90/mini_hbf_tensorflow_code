"""Data generation for 1D and 2D function approximation tasks.

Ported from f_1D_data.py and f_2D_data.py.
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


# -- Target functions --

def f1D_task1():
    """1D target: f(x) = 2*(2*(cos(x)^2) - 1)^2 - 1"""
    return lambda x: 2 * np.power(2 * np.power(np.cos(x), 2) - 1, 2) - 1


def f2D_func_task2_2(x_1, x_2, nb_recursive_layers=2):
    """2D recursive target function: xsinlog composition."""
    c_h = 1.2 * np.pi
    h = np.cos(c_h * (x_1 * x_2 ** 2))
    c_a1 = 1.2
    A = c_a1 * h
    for _ in range(nb_recursive_layers):
        A = 0.99 * A * np.sin(0.1 * np.pi * np.log(np.abs(A) + 1e-12))
    return 60 * A


# -- Data generation --

def generate_1d_data(N_train=60000, N_cv=60000, N_test=60000,
                     low_x=-2 * np.pi, high_x=2 * np.pi):
    """Generate 1D function approximation data."""
    f = f1D_task1()

    X_train = low_x + (high_x - low_x) * np.random.rand(N_train, 1)
    Y_train = f(X_train)

    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv, 1)
    Y_cv = f(X_cv)

    X_test = low_x + (high_x - low_x) * np.random.rand(N_test, 1)
    Y_test = f(X_test)

    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test


def _generate_meshgrid(N, start_val, end_val, random=False):
    """Generate a 2D meshgrid (uniform or random)."""
    sqrtN = int(np.ceil(N ** 0.5))
    if random:
        x_range = np.sort(np.random.uniform(low=start_val, high=end_val, size=sqrtN))
        y_range = np.sort(np.random.uniform(low=start_val, high=end_val, size=sqrtN))
    else:
        x_range = np.linspace(start_val, end_val, sqrtN)
        y_range = np.linspace(start_val, end_val, sqrtN)
    return np.meshgrid(x_range, y_range)


def _meshgrid_to_dataset(X, Y, Z):
    """Convert meshgrid arrays to (N, 2) input and (N, 1) output arrays."""
    dim_x, dim_y = X.shape
    N = dim_x * dim_y
    X_data = np.zeros((N, 2))
    Y_data = np.zeros((N, 1))
    i = 0
    for dx in range(dim_x):
        for dy in range(dim_y):
            X_data[i, :] = [X[dx, dy], Y[dx, dy]]
            Y_data[i, :] = Z[dx, dy]
            i += 1
    return X_data, Y_data


def generate_2d_data(func=None, nb_recursive_layers=2, N_train=60000,
                     N_cv=60000, N_test=60000, start_val=-1, end_val=1):
    """Generate 2D function approximation data."""
    if func is None:
        func = f2D_func_task2_2

    def _gen_split(N):
        X_mesh, Y_mesh = _generate_meshgrid(N, start_val, end_val, random=True)
        dim_x, dim_y = X_mesh.shape
        Z_mesh = np.zeros(X_mesh.shape)
        for dx in range(dim_x):
            for dy in range(dim_y):
                Z_mesh[dx, dy] = func(X_mesh[dx, dy], Y_mesh[dx, dy],
                                      nb_recursive_layers=nb_recursive_layers)
        X_data, Y_data = _meshgrid_to_dataset(X_mesh, Y_mesh, Z_mesh)
        return X_data, Y_data

    X_train, Y_train = _gen_split(N_train)
    X_cv, Y_cv = _gen_split(N_cv)
    X_test, Y_test = _gen_split(N_test)
    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test


def get_data(task_name, N_train=5000, N_cv=1000, N_test=1000):
    """Load data by task name.

    Args:
        task_name: One of 'f1d_task1', 'f2d_task2_2', 'f2d_task2_2_small'.
        N_train, N_cv, N_test: Dataset sizes.
    """
    if task_name == 'f1d_task1':
        return generate_1d_data(N_train=N_train, N_cv=N_cv, N_test=N_test)
    elif task_name in ('f2d_task2_2', 'f2d_task2_2_small'):
        n = 500 if 'small' in task_name else N_train
        return generate_2d_data(N_train=n, N_cv=min(N_cv, n), N_test=min(N_test, n))
    else:
        raise ValueError(f"Unknown task_name: {task_name}")


# -- PyTorch Dataset wrapper --

class FunctionDataset(Dataset):
    """Wraps numpy arrays as a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).double()
        self.Y = torch.from_numpy(Y).double()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
