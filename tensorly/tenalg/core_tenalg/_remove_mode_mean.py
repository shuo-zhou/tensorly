import tensorly as tl
from ...base import unfold, fold


def remove_mode_mean(tensor, mode=0):
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
