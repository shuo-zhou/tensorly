# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================
# import numpy as np
import tensorly as tl
from ..base import unfold, fold
from ..tenalg import multi_mode_dot

# License: BSD 3 clause


def _remove_mode_mean(tensor, mode=0):
    """Remove mean of a tensor along the given mode

        Parameters
        ----------
        tensor : Tensor
            Input data to remove mean, shape (I_1, I_2, ..., I_N), where I_1,
            I_2, ..., I_N are the dimensions of corresponding mode (1, 2, ...,
            N), respectively.
        mode : int
            Along which mode to compute the mean, by default 0.

        Returns
        -------
        tensor_scaled: Tensor
            Tensor data which mean has been removed, (I_1, I_2, ..., I_N).
        tensor_mean: Tensor
            Mean tensor estimated from the training set, shape (I_1, I_2, ...,
            I_mode-1, I_mode+1, ..., I_N).
        """
    tensor_unfold = unfold(tensor, mode=mode)
    tensor_mean = tl.mean(tensor_unfold, axis=0)
    tensor_scaled = tensor_unfold - tensor_mean
    tensor_scaled = fold(tensor_scaled, mode=mode, shape=tensor.shape)

    if mode < 0:
        mode = tl.ndim(tensor) + mode
    mean_shape = ()
    for i in range(len(tensor.shape)):
        if i != mode:
            mean_shape += (tensor.shape[i],)
    mean_shape += (1,)
    tensor_mean = fold(tensor_mean, -1, mean_shape)
    tensor_mean = tensor_mean[..., 0]

    return tensor_scaled, tensor_mean


def _reduce_mean(tensor):
    """Remove mean of a tensor along the last dimension

    Parameters
    ----------
    tensor : Tensor
        Input data to remove mean, shape (I_1, I_2, ..., I_N, n_samples),
        where n_samples is the number of samples, I_1, I_2, ..., I_N are 
        the dimensions of corresponding mode (1, 2, ..., N), respectively.

    Returns
    -------
    tensor_scaled: Tensor
        Tensor data which mean has been removed, (I_1, I_2, ..., I_N, n_samples).
    tensor_mean: Tensor
        Mean tensor estimated from the training set, shape (I_1, I_2, ..., I_N).
    """

    tensor_mean = tl.mean(tensor, axis=-1)
    n_sample = tensor.shape[-1]
    tensor_scaled = tensor.copy()
    for i in range(n_sample):
        tensor_scaled[..., i] = tensor[..., i] - tensor_mean
        
    return tensor_scaled, tensor_mean


def mpca(tensor, var_ratio=0.95, max_iter=1):
    """Multilinear Principal Component Analysis (MPCA)

    Parameters
    ----------
    tensor : Tensor
        Input data, shape (I_1, I_2, ..., I_N, n_samples), where n_samples
        is the number of samples, I_1, I_2, ..., I_N are the dimensions of 
        corresponding mode (1, 2, ..., N), respectively.
    var_ratio : float, optional
        Percentage of variance explained to keeep (between 0 and 1), by default 0.95
    max_iter : int, optional
        Maximum number of iteration, by default 1

    Returns
    -------
    proj_mats: Tensor list
        A list of transposed projection matrices list of transposed projection matrices.
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
        >>> tensor = tl.tensor(np.random.random((20, 25, 20, 40)))
        >>> tensor.shape
        (20, 25, 20, 40)
        >>> proj_mats, idx_order, tensor_mean = mpca(tensor)
        >>> n_sample = tensor.shape[-1]
        >>> for i in range(n_sample):
        >>>     tensor[..., i] = tensor[..., i] - tensor_mean
        >>> tensor_pc = multi_mode_dot(tensor, proj_mats, modes=[0, 1, 2])
        >>> tensor_pc.shape
        (18, 23, 18, 40)
        >>> tensor_rec = multi_mode_dot(tensor_pc, proj_mats, modes=[0, 1, 2], transpose=True)
        >>> tensor_rec.shape
        (20, 25, 20, 40)
        >>> tensor_unfold = unfold(tensor_pc, mode=-1)
        >>> tensor_unfold_sorted = tensor_unfold[:, idx_order]
    """
    if var_ratio <= 0 or var_ratio > 1:
        raise ValueError('var_ratio value should be in range (0, 1], but given %s.' % var_ratio)

    n_dim = tl.ndim(tensor)
    if not n_dim >= 2:
        raise ValueError('Input tensor should be at least a 2D matrix, but given a vector.')
    dim_in = tensor.shape
    n_spl = dim_in[-1]

    # tensor_ = tl.zeros(tensor.shape)  # normalised tensor (mean removal)
    tensor_, tensor_mean = _remove_mode_mean(tensor, mode=-1)
    # init
    phi = dict()
    eig_vecs_sorted = dict()  # dictionary of eigenvectors for all modes
    # lambdas = dict()  # dictionary of eigenvalues for all modes
    # cums = dict()  # cumulative distribution of eigenvalues for all modes
    proj_mats = []
    shape_out = ()
    for i in range(n_spl):
        # tensor_[..., i] = tensor[..., i] - tensor_mean
        for j in range(n_dim - 1):
            tensor_i = unfold(tensor_[..., i], mode=j)
            if j not in phi:
                phi[j] = 0
            phi[j] = phi[j] + tl.dot(tensor_i, tensor_i.T)

    for i in range(n_dim - 1):
        u, s, v = tl.partial_svd(phi[i])
        idx_sorted = (-s).argsort()
        eig_vecs_sorted[i] = u[:, idx_sorted]
        cum = s[idx_sorted]
        var_tot = tl.sum(cum)

        for j in range(1, cum.shape[0] + 1):
            if tl.sum(cum[:j]) / var_tot > var_ratio:
                shape_out += (j,)
                break
        proj_mats.append(eig_vecs_sorted[i][:, :shape_out[i]].T)

    for i_iter in range(max_iter):
        phi = dict()
        for i in range(n_dim - 1):  # ith mode
            if i not in phi:
                phi[i] = 0
            for j in range(n_spl):
                tensor_j = tensor_[..., j]  # jth tensor/sample
                # principal component of tensor j
                tpc_j = multi_mode_dot(tensor_j, [proj_mats[m] for m in range(n_dim - 1) if m != i],
                                       modes=[m for m in range(n_dim - 1) if m != i])
                tpc_j_unfold = unfold(tpc_j, i)
                phi[i] = tl.dot(tpc_j_unfold, tpc_j_unfold.T) + phi[i]

            u, s, v = tl.partial_svd(phi[i])
            idx_sorted = (-s).argsort()
            # lambdas[i] = s[idx_sorted]
            proj_mats[i] = u[:, idx_sorted]
            proj_mats[i] = (proj_mats[i][:, :shape_out[i]]).T

    # tensor principal components
    tpc = multi_mode_dot(tensor_, proj_mats, modes=[m for m in range(n_dim - 1)])
    tpc_unfold = unfold(tpc, mode=-1)  # vectorise the transformed features
    # diagonal of the covariance of vectorised features
    tpc_cov_diag = tl.diag(tl.dot(tpc_unfold, tpc_unfold.T))
    idx_order = (-tpc_cov_diag).argsort()

    return proj_mats, idx_order, tensor_mean
