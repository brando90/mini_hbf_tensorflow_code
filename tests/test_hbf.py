"""Unit tests for the HBF PyTorch port."""

import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hbf.models import HBFLayer, HBFNetwork, StandardNNLayer, StandardNNNetwork
from hbf.data import generate_1d_data, generate_2d_data, f1D_task1, f2D_func_task2_2, get_data
from hbf.initializations import (
    init_hbf_truncated_normal, init_hbf_data, init_hbf_kernel, init_hbf_kpp,
    init_nn_truncated_normal,
)
from hbf.krls import get_kernel_matrix, get_krls_coeffs, evaluate_models


# -- Model tests --

class TestHBFLayer:
    def test_output_shape(self):
        layer = HBFLayer(dim_in=5, dim_out=10)
        x = torch.randn(32, 5, dtype=torch.float64)
        out = layer(x)
        assert out.shape == (32, 10)

    def test_output_positive(self):
        """HBF output is exp(...) so must be positive."""
        layer = HBFLayer(dim_in=3, dim_out=7)
        x = torch.randn(16, 3, dtype=torch.float64)
        out = layer(x)
        assert (out > 0).all()

    def test_output_bounded(self):
        """exp(negative) <= 1, so HBF output should be <= 1 (no BN)."""
        layer = HBFLayer(dim_in=3, dim_out=7)
        x = torch.randn(16, 3, dtype=torch.float64)
        out = layer(x)
        # Z = beta * Delta_tilde, Delta_tilde <= 0, so Z <= 0, exp(Z) <= 1
        assert (out <= 1.0 + 1e-10).all()

    def test_rbf_matches_numpy(self):
        """Verify HBF layer matches manual numpy RBF computation."""
        np.random.seed(0)
        torch.manual_seed(0)
        D_in, D_out, M = 4, 3, 10

        layer = HBFLayer(dim_in=D_in, dim_out=D_out)
        W_np = layer.W.detach().numpy()
        S_np = layer.S.detach().numpy()

        x_np = np.random.randn(M, D_in)
        x_torch = torch.from_numpy(x_np).double()

        # Numpy reference
        beta = (1.0 / S_np) ** 2
        WW = (W_np ** 2).sum(axis=0, keepdims=True)
        XX = (x_np ** 2).sum(axis=1, keepdims=True)
        Delta = 2.0 * x_np @ W_np - (WW + XX)
        Z = beta * Delta
        A_np = np.exp(Z)

        # PyTorch
        A_torch = layer(x_torch).detach().numpy()

        np.testing.assert_allclose(A_torch, A_np, rtol=1e-10)

    def test_with_batch_norm(self):
        layer = HBFLayer(dim_in=5, dim_out=10, use_bn=True)
        x = torch.randn(32, 5, dtype=torch.float64)
        layer.train()
        out = layer(x)
        assert out.shape == (32, 10)

    def test_gradients_flow(self):
        layer = HBFLayer(dim_in=3, dim_out=5)
        x = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.W.grad is not None
        assert layer.S.grad is not None


class TestHBFNetwork:
    def test_forward(self):
        model = HBFNetwork(dims=[5, 10, 3])
        x = torch.randn(16, 5, dtype=torch.float64)
        out = model(x)
        assert out.shape == (16, 3)

    def test_deep_forward(self):
        model = HBFNetwork(dims=[5, 10, 8, 3])
        x = torch.randn(16, 5, dtype=torch.float64)
        out = model(x)
        assert out.shape == (16, 3)

    def test_1d_regression(self):
        """HBF can overfit a small 1D dataset."""
        torch.manual_seed(42)
        np.random.seed(42)
        X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
        f = f1D_task1()
        Y = f(X)

        model = HBFNetwork(dims=[1, 30, 1])
        init_hbf_truncated_normal(model, [1, 30, 1], std=0.3, S_init=0.5)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        X_t = torch.from_numpy(X).double()
        Y_t = torch.from_numpy(Y).double()

        initial_loss = None
        for step in range(500):
            optimizer.zero_grad()
            loss = ((model(X_t) - Y_t) ** 2).mean()
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, "Training should reduce loss"


class TestStandardNN:
    def test_forward(self):
        model = StandardNNNetwork(dims=[5, 10, 3])
        x = torch.randn(16, 5, dtype=torch.float64)
        out = model(x)
        assert out.shape == (16, 3)

    def test_with_bn(self):
        model = StandardNNNetwork(dims=[5, 10, 3], use_bn=True)
        model.train()
        x = torch.randn(16, 5, dtype=torch.float64)
        out = model(x)
        assert out.shape == (16, 3)


# -- Data tests --

class TestData:
    def test_1d_data_shapes(self):
        X_tr, Y_tr, X_cv, Y_cv, X_te, Y_te = generate_1d_data(
            N_train=100, N_cv=20, N_test=20
        )
        assert X_tr.shape == (100, 1)
        assert Y_tr.shape == (100, 1)
        assert X_cv.shape == (20, 1)
        assert X_te.shape == (20, 1)

    def test_1d_function_values(self):
        # f(x) = 2*(2*cos(x)^2 - 1)^2 - 1
        f = f1D_task1()
        assert abs(f(0.0) - 1.0) < 1e-10  # cos(0)=1, 2*(2*1-1)^2-1 = 2*1-1 = 1
        assert abs(f(np.pi / 2) - 1.0) < 1e-10  # cos(pi/2)~0, 2*(0-1)^2-1 = 2-1 = 1
        assert abs(f(np.pi / 4) - (-1.0)) < 1e-10  # cos(pi/4)=1/sqrt(2), 2*(2*0.5-1)^2-1 = -1

    def test_2d_data_shapes(self):
        X_tr, Y_tr, X_cv, Y_cv, X_te, Y_te = generate_2d_data(
            N_train=100, N_cv=50, N_test=50
        )
        assert X_tr.shape[1] == 2
        assert Y_tr.shape[1] == 1

    def test_get_data_f1d(self):
        X_tr, Y_tr, X_cv, Y_cv, X_te, Y_te = get_data('f1d_task1', N_train=100, N_cv=20, N_test=20)
        assert X_tr.shape == (100, 1)

    def test_get_data_invalid(self):
        with pytest.raises(ValueError):
            get_data('nonexistent_task')


# -- Initialization tests --

class TestInitializations:
    def test_truncated_normal(self):
        model = HBFNetwork(dims=[3, 10, 1])
        init_hbf_truncated_normal(model, [3, 10, 1])
        assert model.hidden_layers[0].W.shape == (3, 10)

    def test_data_init(self):
        X = np.random.randn(50, 3)
        Y = np.random.randn(50, 1)
        model = HBFNetwork(dims=[3, 10, 1])
        init_hbf_data(model, X, Y, [3, 10, 1])
        # Check templates come from data (W is transposed centers)
        W = model.hidden_layers[0].W.detach().numpy()
        assert W.shape == (3, 10)

    def test_kernel_init(self):
        X = np.random.randn(50, 3)
        Y = np.random.randn(50, 1)
        model = HBFNetwork(dims=[3, 10, 1])
        init_hbf_kernel(model, X, Y, [3, 10, 1], S_init=0.5)
        # Output layer should be initialized from kernel LS
        assert model.output_layer.weight.shape == (1, 10)

    def test_kpp_init(self):
        X = np.random.randn(100, 3)
        Y = np.random.randn(100, 1)
        model = HBFNetwork(dims=[3, 10, 1])
        init_hbf_kpp(model, X, Y, [3, 10, 1], S_init=0.5)
        assert model.output_layer.weight.shape == (1, 10)

    def test_nn_init(self):
        model = StandardNNNetwork(dims=[3, 10, 1])
        init_nn_truncated_normal(model, std=0.5, b_init=0.1)
        assert model.hidden_layers[0].linear.weight.shape == (10, 3)


# -- KRLS tests --

class TestKRLS:
    def test_kernel_matrix_shape(self):
        X = np.random.randn(20, 3)
        centers = np.random.randn(5, 3)
        K = get_kernel_matrix(X, centers, stddev=0.5)
        assert K.shape == (20, 5)

    def test_kernel_matrix_positive(self):
        X = np.random.randn(20, 3)
        centers = np.random.randn(5, 3)
        K = get_kernel_matrix(X, centers, stddev=0.5)
        assert (K > 0).all()
        assert (K <= 1.0 + 1e-10).all()

    def test_krls_coeffs(self):
        np.random.seed(0)
        X = np.random.randn(50, 2)
        centers = X[:10, :]
        K = get_kernel_matrix(X, centers, stddev=1.0)
        Y = np.random.randn(50, 1)
        C = get_krls_coeffs(K, Y)
        assert C.shape == (10, 1)

    def test_evaluate_models(self):
        np.random.seed(0)
        data = generate_1d_data(N_train=200, N_cv=50, N_test=50)
        results = evaluate_models(data, stddevs=[0.3, 0.5, 1.0], nb_centers_list=[10, 20])
        assert 10 in results
        assert 20 in results
        assert 'errors_best' in results[10]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
