""" 
The :mod:`tensorly.tenalg` module contains utilities for Tensor Algebra 
operations such as khatri-rao or kronecker product, n-mode product, etc.
"""

from .core_tenalg import mode_dot, multi_mode_dot
from .core_tenalg import kronecker
from .core_tenalg import khatri_rao
from .core_tenalg import inner
from .core_tenalg import outer, batched_outer
from .core_tenalg import higher_order_moment
from .core_tenalg import _tt_matrix_to_tensor
from .core_tenalg import remove_mode_mean
from .core_tenalg import tensordot

from . import core_tenalg as core
from . import einsum_tenalg

from ..backend import _LOCAL_STATE

_DEFAULT_TENALG_BACKEND = 'core'

_TENALG_BACKENDS = {'core':core,
                    'einsum':einsum_tenalg}

def initialize_tenalg_backend():
    """Initialises the backend

    1) retrieve the default backend name from the `TENSORLY_TENALG_BACKEND` environment variable
        if not found, use _DEFAULT_TENALG_BACKEND
    2) sets the backend to the retrieved backend name
    """
    tenalg_backend_name = os.environ.get('TENSORLY_TENALG_BACKEND', _DEFAULT_TENALG_BACKEND)
    if tenalg_backend_name not in _TENALG_BACKENDS:
        msg = ("TENSORLY_BACKEND should be one of {}, got {}. Defaulting to {}'").format(
            ', '.join(map(repr, _TENALG_BACKENDS)),
            tenalg_backend_name, _DEFAULT_TENALG_BACKEND)
        warnings.warn(msg, UserWarning)
        tenalg_backend_name = _DEFAULT_TENALG_BACKEND

    set_tenalg_backend(tenalg_backend_name, local_threadsafe=False)

def get_tenalg_backend():
    """Returns the current backend
    """
    return _LOCAL_STATE.tenalg_backend


def set_tenalg_backend(backend='core', local_threadsafe=False):
    """Set the current tenalg backend

    Parameters
    ----------
    backend : {'core', 'einsum'}
        * if 'core', our manually optimized implementations are used 
        * if 'einsum', all operations are dispatched to ``einsum``

    If True, the backend will not become the default backend for all threads.
        Note that this only affects threads where the backend hasn't already
        been explicitly set. If False (default) the backend is set for the
        entire session.
    """
    if backend in _TENALG_BACKENDS:
        _LOCAL_STATE.tenalg_backend = backend
        if local_threadsafe == False:
            global _DEFAULT_TENALG_BACKEND
            _DEFAULT_TENALG_BACKEND = backend
    else:
        raise ValueError(f'Unknown tenalg backend {backend}')

@contextmanager
def tenalg_backend_context(backend, local_threadsafe=False):
    """Context manager to set the backend for TensorLy.

    Parameters
    ----------
    backend : {'core', 'einsum'}
        * if 'core', our manually optimized implementations are used 
        * if 'einsum', all operations are dispatched to ``einsum``
    local_threadsafe : bool, optional
        If True, the backend will not become the default backend for all threads.
        Note that this only affects threads where the backend hasn't already
        been explicitly set. If False (default) the backend is set for the
        entire session.

    Examples
    --------
    Set the backend to numpy globally for this thread:

    >>> import tensorly.tenalg as tlg
    >>> tlg.set_tenalg_backend('core')
    >>> with tlg.set_tenalg_backend('einsum'):
    ...     pass
    """
    _old_backend = get_tenalg_backend()
    set_tenalg_backend(backend, local_threadsafe=local_threadsafe)
    try:
        yield
    finally:
        set_tenalg_backend(_old_backend)


def dynamically_dispatch_tenalg(function):
    name = function.__name__

    @wraps(function)
    def dynamically_dispatched_fun(*args, **kwargs):
        #print('hello')
        try:
            current_backend = _TENALG_BACKENDS[_LOCAL_STATE.tenalg_backend]
        except AttributeError:
            current_backend = _TENALG_BACKENDS[_DEFAULT_TENALG_BACKEND]
            
        if hasattr(current_backend, name):
            fun = getattr(current_backend, name)(*args, **kwargs)
        else:
            warnings.warn(f'tenalg: defaulting to core tenalg backend, {name}'
                          f'not yet implemented in {_LOCAL_STATE.tenalg_backend} backend.')
            fun = getattr(core, name)(*args, **kwargs)
        return fun
    return dynamically_dispatched_fun


mode_dot = dynamically_dispatch_tenalg(mode_dot)
multi_mode_dot = dynamically_dispatch_tenalg(multi_mode_dot)
kronecker = dynamically_dispatch_tenalg(kronecker)
khatri_rao = dynamically_dispatch_tenalg(khatri_rao)
inner = dynamically_dispatch_tenalg(inner)
outer = dynamically_dispatch_tenalg(outer)
batched_outer = dynamically_dispatch_tenalg(batched_outer)
tensordot = dynamically_dispatch_tenalg(tensordot)
higher_order_moment = dynamically_dispatch_tenalg(higher_order_moment)
remove_mode_mean = dynamically_dispatch_tenalg(remove_mode_mean)

    @classmethod
    def use_dynamic_dispatch(cls):
        # Define class methods and attributes that dynamically dispatch to the backend
        for name in cls._functions:
            try:
                delattr(cls, name)
            except AttributeError:
                pass
            setattr(
                cls,
                name,
                staticmethod(
                    cls.dispatch_backend_method(
                        name, getattr(cls.current_backend(), name)
                    )
                ),
            )
        for name in cls._attributes:
            try:
                delattr(cls, name)
            except AttributeError:
                pass
            setattr(cls, name, dynamically_dispatched_class_attribute(name))

    @classmethod
    def load_backend(cls, backend_name):
        """Registers a new backend by importing the corresponding module
            and adding the correspond `Backend` class in Backend._LOADED_BACKEND
            under the key `backend_name`

        Parameters
        ----------
        backend_name : str, name of the backend to load

        Raises
        ------
        ValueError
            If `backend_name` does not correspond to one listed
                in `_KNOWN_BACKEND`
        """
        if backend_name not in cls.available_backend_names:
            msg = f"Unknown backend name {backend_name!r}, known backends are {cls.available_backend_names}"
            raise ValueError(msg)
        if backend_name not in TenalgBackend._available_tenalg_backends:
            importlib.import_module(f"tensorly.tenalg.{backend_name}_tenalg")
        if backend_name in TenalgBackend._available_tenalg_backends:
            backend = TenalgBackend._available_tenalg_backends[backend_name]()
            # backend = getattr(module, )()
            cls._loaded_backends[backend_name] = backend

        return backend


# Initialise the backend to the default one
TenalgBackendManager.initialize_backend()
TenalgBackendManager.use_dynamic_dispatch()

sys.modules[__name__].__class__ = TenalgBackendManager
