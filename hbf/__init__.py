"""HBF (Hierarchical Basis Function) neural networks in PyTorch.

Ported from the original TensorFlow implementation by Brando Miranda (MIT MEng, 2016).
"""

from hbf.models import HBFLayer, HBFNetwork, StandardNNLayer, StandardNNNetwork
from hbf.data import (
    f1D_task1,
    f2D_func_task2_2,
    generate_1d_data,
    generate_2d_data,
    get_data,
)
from hbf.initializations import (
    init_hbf_truncated_normal,
    init_hbf_data,
    init_hbf_kernel,
    init_hbf_kpp,
    init_nn_truncated_normal,
)
from hbf.krls import (
    get_kernel_matrix,
    get_krls_coeffs,
    evaluate_models,
    get_best_shape_and_mdl,
)
