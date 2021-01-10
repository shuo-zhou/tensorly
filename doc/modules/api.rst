=============
API reference
=============

:mod:`tensorly`: Manipulating the backend with a unified interface
==================================================================

For each backend, tensorly provides the following uniform functions:

.. automodule:: tensorly
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    set_backend
    get_backend
    context
    tensor
    is_tensor
    shape
    ndim
    to_numpy
    copy
    concatenate
    reshape
    transpose
    moveaxis
    arange
    ones
    zeros
    zeros_like
    eye
    diag
    where
    clip
    max
    min
    all
    mean
    sum
    prod
    sign
    abs
    sqrt
    norm
    dot
    kron
    solve
    qr
    kr
    partial_svd


:mod:`tensorly.base`: Core tensor functions
============================================

.. automodule:: tensorly.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.base

.. autosummary::
    :toctree: generated/
    :template: function.rst

    unfold
    fold
    tensor_to_vec
    vec_to_tensor
    partial_unfold
    partial_fold
    partial_tensor_to_vec
    partial_vec_to_tensor


:mod:`tensorly.cp_tensor`: Tensors in the CP format
=============================================================

.. automodule:: tensorly.cp_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.cp_tensor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    cp_to_tensor
    cp_to_unfolded
    cp_to_vec
    cp_normalize
    cp_norm
    cp_mode_dot
    unfolding_dot_khatri_rao


:mod:`tensorly.tucker_tensor`: Tensors in Tucker format
=======================================================

.. automodule:: tensorly.tucker_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.tucker_tensor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    tucker_to_tensor
    tucker_to_unfolded
    tucker_to_vec
    tucker_mode_dot


:mod:`tensorly.tt_tensor`: Tensors in Matrix-Product-State format
==================================================================

.. automodule:: tensorly.tt_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.tt_tensor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    tt_to_tensor
    tt_to_unfolded
    tt_to_vec


:mod:`tensorly.tt_matrix`: Matrices in TT format
================================================

.. automodule:: tensorly.tt_matrix
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.tt_matrix

.. autosummary::
    :toctree: generated/
    :template: function.rst

    tt_matrix_to_tensor
    tt_matrix_to_unfolded
    tt_matrix_to_vec


:mod:`tensorly.parafac2_tensor`: Tensors in PARAFAC2 format
===========================================================

.. automodule:: tensorly.parafac2_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.parafac2_tensor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    parafac2_to_tensor
    parafac2_to_slice
    parafac2_to_slices
    parafac2_to_unfolded
    parafac2_to_vec


:mod:`tensorly.tenalg`: Tensor algebra
======================================

.. automodule:: tensorly.tenalg
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.tenalg

.. autosummary::
    :toctree: generated/
    :template: function.rst

    khatri_rao
    kronecker
    mode_dot
    multi_mode_dot
    proximal.soft_thresholding
    proximal.svd_thresholding
    proximal.procrustes
    inner
    contract
    tensor_dot
    batched_tensor_dot
    higher_order_moment


:mod:`tensorly.decomposition`: Tensor Decomposition
====================================================

.. automodule:: tensorly.decomposition
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.decomposition

Classes
-------

Note that these are currently experimental and may change in the future.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    CP
    RandomizedCP
    CPPower
    Tucker
    TensorTrain
    Parafac2
    SymmetricCP

Functions
---------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    parafac
    non_negative_parafac
    sample_khatri_rao
    randomised_parafac
    tucker
    partial_tucker
    non_negative_tucker
    robust_pca
    tensor_train
    tensor_train_matrix
    parafac2
    symmetric_power_iteration
    symmetric_parafac_power_iteration


:mod:`tensorly.regression`: Tensor Regression
==============================================

.. automodule:: tensorly.regression
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.regression

.. autosummary::
    :toctree: generated/
    :template: class.rst

    tucker_regression.TuckerRegressor
    cp_regression.CPRegressor


:mod:`tensorly.metrics`: Performance measures
==============================================

.. automodule:: tensorly.metrics
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.metrics

.. autosummary::
    :toctree: generated/
    :template: function.rst

    regression.MSE
    regression.RMSE


:mod:`tensorly.random`: Sampling random tensors
===============================================

.. automodule:: tensorly.random
   :no-members:
   :no-inherited-members:

.. currentmodule:: tensorly.random

.. autosummary::
   :toctree: generated/
   :template: function.rst

   random_cp
   random_tucker
   random_tt
   random_tt_matrix
   random_parafac2
   check_random_state



:mod:`tensorly.datasets`: Creating and loading data
====================================================

.. automodule:: tensorly.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.datasets

.. autosummary::
    :toctree: generated/
    :template: function.rst

    synthetic.gen_image


:mod:`tensorly.contrib`: Experimental features
==============================================

.. automodule:: tensorly.contrib
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.contrib

.. autosummary::
    :toctree: generated/
    :template: function.rst

    decomposition.tensor_train_cross

Sparse tensor operations
------------------------

Enables tensor operations on sparse tensors.
Currently, the following decomposition methods are supported (for the NumPy backend, using Sparse):

.. automodule:: tensorly.contrib.sparse

.. currentmodule:: tensorly.contrib

.. autosummary::
    :toctree: generated/

   sparse.decomposition.tucker
   sparse.decomposition.partial_tucker
   sparse.decomposition.non_negative_tucker
   sparse.decomposition.robust_pca
   sparse.decomposition.parafac
   sparse.decomposition.non_negative_parafac
   sparse.decomposition.symmetric_parafac_power_iteration


