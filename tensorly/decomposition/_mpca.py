# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================
import tensorly as tl
from ..base import unfold
from ..tenalg import multi_mode_dot

# License: BSD 3 clause


def mpca(tensor, var_ratio=0.95, max_iter=1):
    """Multilinear Principal Component Analysis (MPCA)

    Parameters
    ----------
    tensor : Tensor
        Input data, shape (n_samples, I_1, I_2, ..., I_N), where n_samples
        is the number of samples, I_1, I_2, ..., I_N are the dimensions of 
        corresponding mode (1, 2, ..., N), respectively.
    var_ratio : float, optional
        Percentage of variance explained to keeep (between 0 and 1), by default 0.95
    max_iter : int, optional
        Maximum number of iteration, by default 1.

    Returns
    -------
    factors: Tensor list
        A list of transposed projection matrices list of transposed projection matrices, shapes 
        (P_1, I_1), ..., (P_N, I_N), where P_1, ..., P_N are output tensor shape for each sample.
    idx_order: Tensor
        The ordering index of projected (and vectorised) features in descending variance.
    tensor_mean: Tensor
        Per-feature empirical mean, estimated from the training set, shape (I_1, I_2, ..., I_N).
    
    References
    ----------
    ..  [1]Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos, "MPCA: Multilinear
        Principal Component Analysis of Tensor Objects", IEEE Transactions on Neural 
        Networks, Vol. 19, No. 1, Page: 18-39, January 2008. For initial Matlab
        implementation, please go to https://uk.mathworks.com/matlabcentral/fileexchange/26168.

    Examples:
    ----------
        >>> import numpy as np
        >>> import tensorly as tl
        >>> from tensorly.base import unfold
        >>> from tensorly.tenalg import multi_mode_dot
        >>> from tensorly.decomposition import mpca
        >>> tensor = tl.tensor(np.random.random((40, 20, 25, 20)))
        >>> tensor.shape
        (40, 20, 25, 20)
        >>> factors, idx_order, tensor_mean = mpca(tensor)
        >>> tensor_pc = multi_mode_dot(tensor - tensor_mean, factors, modes=[1, 2, 3])
        >>> tensor_pc.shape
        (40, 18, 23, 18)
        >>> tensor_pc_vec = unfold(tensor_pc, mode=0)
        >>> tensor_pc_vec_sorted = tensor_pc_vec[:, idx_order]
        >>> tensor_rec = multi_mode_dot(tensor_pc, factors, modes=[1, 2, 3], transpose=True) + tensor_mean
        >>> tensor_rec.shape
        (40, 20, 25, 20)
    """
    if var_ratio <= 0 or var_ratio > 1:
        raise ValueError('var_ratio value should be in range (0, 1], but given %s.' % var_ratio)

    n_dims = tl.ndim(tensor)
    if not n_dims >= 2:
        raise ValueError('Input tensor should be at least a 2D matrix, but given a vector.')

    # tensor_ = tl.zeros(tensor.shape)  # normalised tensor (mean removal)
    tensor_mean = tl.mean(tensor, axis=0)
    tensor_ = tensor - tensor_mean

    # init
    phi = dict()
    factors = []
    shape_out = ()

    for i in range(1, n_dims):
        phi[i] = unfold(tensor_, mode=i)
        u, s, v = tl.partial_svd(phi[i])
        idx_sorted = (-1 * s).argsort()
        cum = s[idx_sorted]
        var_tot = tl.sum(cum)

        for j in range(1, cum.shape[0] + 1):
            if tl.sum(cum[: j]) / var_tot > var_ratio:
                shape_out += (j,)
                break
        factors.append(u[:, idx_sorted][:, : shape_out[i - 1]].T)

    for i_iter in range(max_iter):
        for i in range(1, n_dims):  # ith mode
            tensor_features = multi_mode_dot(
                tensor_,
                [factors[m] for m in range(n_dims - 1) if m != i - 1],
                modes=[m for m in range(1, n_dims) if m != i]
            )
            phi[i] = unfold(tensor_features, i)
            u, s, v = tl.partial_svd(phi[i])
            idx_sorted = (-1 * s).argsort()
            # lambdas[i] = s[idx_sorted]
            factors[i - 1] = (u[:, idx_sorted][:, :shape_out[i - 1]]).T

    # tensor features/projections
    tensor_features = multi_mode_dot(tensor_, factors, modes=[m for m in range(1, n_dims)])
    tf_unfold = unfold(tensor_features, mode=0)  # vectorise tensor features
    # diagonal of the covariance of tensor features
    tpc_cov_diag = tl.diag(tl.dot(tf_unfold.T, tf_unfold))
    idx_order = (-1 * tpc_cov_diag).argsort()

    return factors, idx_order, tensor_mean
