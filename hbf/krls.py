"""Kernel Recursive Least Squares (KRLS) and cross-validation utilities.

Ported from krls_and_cv.py. These are numpy-based (framework-agnostic).
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances


def get_kernel_matrix(X, centers, stddev):
    """Compute Gaussian/RBF kernel matrix.

    Args:
        X: Data matrix (N, D).
        centers: RBF centers (K, D).
        stddev: Bandwidth parameter.

    Returns:
        Kernel matrix (N, K).
    """
    beta = (1.0 / stddev) ** 2
    return np.exp(-beta * euclidean_distances(X=X, Y=centers, squared=True))


def get_krls_coeffs(Kern, Y):
    """Solve KC = Y by least squares."""
    C, _, _, _ = np.linalg.lstsq(Kern, Y, rcond=None)
    return C


def _get_subsampled_points(X, K, replace=False):
    """Randomly subsample K data points as centers."""
    N = X.shape[0]
    indices = np.random.choice(N, size=K, replace=replace)
    return X[indices, :], indices


def get_best_shape_and_mdl(K, data, stddevs, nb_inits=1):
    """Find best bandwidth and centers for fixed K via cross-validation.

    Returns:
        Tuple of (best_params, errors_best, errors_stats).
    """
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = data

    cv_errors = []
    train_errors = []
    test_errors = []
    C_hats = []
    centers_tried = []
    stddevs_for_runs = []

    for stddev in stddevs:
        for _ in range(nb_inits):
            centers, _ = _get_subsampled_points(X_train, K)
            centers_tried.append(centers)

            Kern_train = get_kernel_matrix(X_train, centers, stddev)
            Kern_cv = get_kernel_matrix(X_cv, centers, stddev)
            Kern_test = get_kernel_matrix(X_test, centers, stddev)

            C_hat = get_krls_coeffs(Kern_train, Y_train)
            C_hats.append(C_hat)

            train_errors.append(mean_squared_error(Y_train, Kern_train @ C_hat))
            cv_errors.append(mean_squared_error(Y_cv, Kern_cv @ C_hat))
            test_errors.append(mean_squared_error(Y_test, Kern_test @ C_hat))
            stddevs_for_runs.append(stddev)

    min_idx = int(np.argmin(cv_errors))
    best_params = {
        'C_hat': C_hats[min_idx],
        'centers': centers_tried[min_idx],
        'stddev': stddevs_for_runs[min_idx],
    }
    errors_best = {
        'train': train_errors[min_idx],
        'cv': cv_errors[min_idx],
        'test': test_errors[min_idx],
    }
    errors_stats = {
        'train_mean': np.mean(train_errors),
        'cv_mean': np.mean(cv_errors),
        'test_mean': np.mean(test_errors),
        'train_std': np.std(train_errors),
        'cv_std': np.std(cv_errors),
        'test_std': np.std(test_errors),
    }
    return best_params, errors_best, errors_stats


def evaluate_models(data, stddevs, nb_centers_list, nb_inits=1):
    """Evaluate KRLS models across multiple center counts and bandwidths.

    Args:
        data: Tuple of (X_train, Y_train, X_cv, Y_cv, X_test, Y_test).
        stddevs: List of bandwidth values to try.
        nb_centers_list: List of center counts to try.
        nb_inits: Number of random initializations per (K, stddev) pair.

    Returns:
        Dict with results per center count.
    """
    results = {}
    for K in nb_centers_list:
        best_params, errors_best, errors_stats = get_best_shape_and_mdl(
            K, data, stddevs, nb_inits=nb_inits
        )
        results[K] = {
            'best_params': best_params,
            'errors_best': errors_best,
            'errors_stats': errors_stats,
        }
    return results
