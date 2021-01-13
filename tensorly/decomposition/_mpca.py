# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================
import tensorly as tl
from ..base import unfold
from ..tenalg import multi_mode_dot

# License: BSD 3 clause


def mpca(tensor, var_ratio=0.95, max_iter=1, mean_removal=True):
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
    mean_removal: bool, optional
        Whether remove the mean from training data, by default True.

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

    n_dim = tl.ndim(tensor)
    if not n_dim >= 2:
        raise ValueError('Input tensor should be at least a 2D matrix, but given a vector.')
    n_spl = tensor.shape[0]

    # tensor_ = tl.zeros(tensor.shape)  # normalised tensor (mean removal)
    if mean_removal:
        tensor_mean = tl.mean(tensor, axis=0)
        tensor_ = tensor - tensor_mean  # remove_mode_mean(tensor, mode=-1)
    else:
        tensor_ = tensor
        tensor_mean = tl.zeros(shape=tensor.shape[1:])
    # init
    phi = dict()
    eig_vecs_sorted = dict()  # dictionary of eigenvectors for all modes
    # lambdas = dict()  # dictionary of eigenvalues for all modes
    # cums = dict()  # cumulative distribution of eigenvalues for all modes
    factors = []
    shape_out = ()
    for i in range(1, n_dim):
        for j in range(n_spl):
            tensor_j = unfold(tensor_[j, :], mode=i - 1)
            if j not in phi:
                phi[i] = 0
            phi[i] = phi[i] + tl.dot(tensor_j, tensor_j.T)

    for i in range(1, n_dim):
        u, s, v = tl.partial_svd(phi[i])
        idx_sorted = (-s).argsort()
        eig_vecs_sorted[i] = u[:, idx_sorted]
        cum = s[idx_sorted]
        var_tot = tl.sum(cum)

        for j in range(1, cum.shape[0] + 1):
            if tl.sum(cum[:j]) / var_tot > var_ratio:
                shape_out += (j,)
                break
        factors.append(eig_vecs_sorted[i][:, :shape_out[i - 1]].T)

    for i_iter in range(max_iter):
        phi = dict()
        for i in range(1, n_dim):  # ith mode
            if i not in phi:
                phi[i] = 0
            for j in range(n_spl):
                tensor_j = tensor_[j, :]  # jth tensor/sample
                # principal component of tensor j
                tpc_j = multi_mode_dot(tensor_j, [factors[m] for m in range(n_dim - 1) if m != i - 1],
                                       modes=[m for m in range(n_dim - 1) if m != i - 1])
                tpc_j_unfold = unfold(tpc_j, i - 1)
                phi[i] = tl.dot(tpc_j_unfold, tpc_j_unfold.T) + phi[i]

            u, s, v = tl.partial_svd(phi[i])
            idx_sorted = (-s).argsort()
            # lambdas[i] = s[idx_sorted]
            factors[i - 1] = u[:, idx_sorted]
            factors[i - 1] = (factors[i - 1][:, :shape_out[i - 1]]).T

    # tensor principal components
    tpc = multi_mode_dot(tensor_, factors, modes=[m for m in range(1, n_dim)])
    tpc_unfold = unfold(tpc, mode=0)  # vectorise tensor principal components
    # diagonal of the covariance of tensor principal components
    tpc_cov_diag = tl.diag(tl.dot(tpc_unfold.T, tpc_unfold))
    idx_order = (-tpc_cov_diag).argsort()

    return factors, idx_order, tensor_mean
