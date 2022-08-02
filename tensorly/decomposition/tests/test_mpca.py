# import numpy as np
# import pytest
# import tensorly as tl
# from ...base import unfold
# from ...tenalg import multi_mode_dot
# from .._mpca import mpca
# from ...random import check_random_state, random_tensor
#
# # n_sample = 20
# # tensor = []
# # for i in range(n_sample):
# #     tensor.append(random_tensor(shape=(20, 25, 20, 1)))
# # tensor = tl.concatenate(tensor, axis=-1)
# tl.set_backend('pytorch')
# rng = check_random_state(144)
# tensor = random_tensor(shape=(40, 20, 25, 20), random_state=rng)
# # tensor1, tensor_mean1 = _reduce_mean(tensor)
# # tensor2, tensor_mean2 = remove_mode_mean(tensor, mode=-1)
# # tensor = np.random.random(size=(30, 35, 30, 20))
# proj_mats, idx_order, tensor_mean = mpca(tensor, var_ratio=0.9)
# tensor_pc = multi_mode_dot(tensor - tensor_mean,
#                            proj_mats, modes=[1, 2, 3])
# tensor_pc_vec = unfold(tensor_pc, mode=-1)

import tensorly as tl
import pytest

from .._mpca import mpca
from ...tenalg import multi_mode_dot
from ...random import check_random_state, random_tensor
from ...testing import assert_array_equal, assert_


VAR_RATIOS = [0.7, 0.8, 0.9]
BACKENDS = ["numpy", "pytorch"]


@pytest.mark.parametrize("var_ratio", VAR_RATIOS)
@pytest.mark.parametrize("max_iter", [1, 5, 10])
@pytest.mark.parametrize("backend", BACKENDS)
def test_mpca(var_ratio, max_iter, backend):
    tl.set_backend(backend)
    rng = check_random_state(144)
    tensor = random_tensor(shape=(40, 20, 25, 20), random_state=rng)
    factors, idx_order, tensor_mean = mpca(tensor, var_ratio=var_ratio, max_iter=max_iter)
    tensor_ = multi_mode_dot(tensor - tensor_mean, factors, modes=[1, 2, 3])
    assert_array_equal(tensor_.ndim, tensor.ndim)
    assert_array_equal(tensor_.shape[0], tensor.shape[0])

    for i in range(1, tensor.ndim):
        assert_(tensor_.shape[i] <= tensor.shape[i])
        assert_array_equal(factors[i - 1].shape[1], tensor.shape[i])

    tensor_rec = multi_mode_dot(tensor_, factors, modes=[1, 2, 3], transpose=True) + tensor_mean
    assert_array_equal(tensor_rec.shape, tensor.shape)

    # basic mpca test, return tensor

    # x_proj = mpca.fit(x).transform(x)
    #
    # testing.assert_equal(x_proj.ndim, x.ndim)
    # testing.assert_equal(x_proj.shape[0], x.shape[0])
    # assert mpca.n_components <= np.prod(x.shape[1:])
    # assert n_components < mpca.n_components
    # for i in range(1, x.ndim):
    #     assert x_proj.shape[i] <= x.shape[i]
    #     testing.assert_equal(mpca.proj_mats[i - 1].shape[1], x.shape[i])
    #
    # x_rec = mpca.inverse_transform(x_proj)
    # testing.assert_equal(x_rec.shape, x.shape)
    # # tol = 10 ** (-10 * var_ratio + 3)
    # # testing.assert_allclose(x_rec, x, rtol=tol)
    #
    # # test return vector
    # mpca.set_params(**{"return_vector": True, "n_components": n_components})
    #
    # x_proj = mpca.transform(x)
    # testing.assert_equal(x_proj.ndim, 2)
    # testing.assert_equal(x_proj.shape[0], x.shape[0])
    # testing.assert_equal(x_proj.shape[1], n_components)
    # x_rec = mpca.inverse_transform(x_proj)
    # testing.assert_equal(x_rec.shape, x.shape)
    #
    # # test n_samples = 1
    # x0_proj = mpca.transform(x[0])
    # testing.assert_equal(x0_proj.ndim, 2)
    # testing.assert_equal(x0_proj.shape[0], 1)
    # testing.assert_equal(x0_proj.shape[1], n_components)
    # x0_rec = mpca.inverse_transform(x0_proj)
    # testing.assert_equal(x0_rec.shape[1:], x[0].shape)
    #
    # # test n_components exceeds upper limit
    # mpca.set_params(**{"return_vector": True, "n_components": np.prod(x.shape[1:]) + 1})
    # x_proj = mpca.transform(x)
    # testing.assert_equal(x_proj.shape[1], np.prod(mpca.shape_out))

