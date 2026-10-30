"""Process-local LP64 SciPy DPOSV binding with a relocatable Numba cache.

The cached object refers to a *symbol name*, never a previous process' pointer.
Each importing process resolves and registers that symbol before cache loading.
The pointer and SciPy module stay alive for the life of the process. Only the
LP64 Cython capsule ABI is currently supported; never guess integer widths.
No ridge, altered tolerance, float32 or fastmath is introduced.
"""
from __future__ import annotations
import ctypes
import numpy as np
from numba import njit,types
from numba.extending import get_cython_function_address
from llvmlite import binding as llvm
from scipy.linalg import cython_lapack as _lapack

_SYMBOL='certfix3_rc6_dposv_lp64_v1'
_get_name=ctypes.pythonapi.PyCapsule_GetName
_get_name.argtypes=[ctypes.py_object];_get_name.restype=ctypes.c_char_p
CAPSULE_SIGNATURE=_get_name(_lapack.__pyx_capi__['dposv']).decode('ascii')

def _validate_signature(signature):
    # Exactly 8 args: char*, int*, int*, double*, int*, double*, int*, int*.
    # Cython uses a typedef rather than spelling out 'double'.
    if not signature.startswith('void (') or not signature.endswith(')'):
        raise RuntimeError('Unsupported SciPy DPOSV capsule ABI; use kernel_backend="lu" or the tested environment')
    args=[v.strip() for v in signature[6:-1].split(',')]
    valid=(len(args)==8 and args[0]=='char *' and all(args[i]=='int *' for i in (1,2,4,6,7))
           and args[3]==args[5] and (args[3]=='double *' or args[3].endswith('cython_lapack_d *'))
           and ctypes.sizeof(ctypes.c_int)==4)
    if not valid:
        raise RuntimeError('Unsupported SciPy DPOSV integer/scalar ABI: '+signature+'; select kernel_backend="lu" explicitly')

_validate_signature(CAPSULE_SIGNATURE)
_ADDRESS=get_cython_function_address('scipy.linalg.cython_lapack','dposv')
llvm.add_symbol(_SYMBOL,_ADDRESS)
_DPOSV=types.ExternalFunction(_SYMBOL,types.void(*([types.voidptr]*8)))

@njit(cache=True)
def solve_spd(A,b):
    """SPD attempt then original LU. Internal input is a symmetric C-order system.

    Symmetric C-order storage is also the transpose seen by Fortran LAPACK.
    F and x are private scratch; failed Cholesky cannot corrupt the LU input.
    A singular matrix remains singular: np.linalg.solve may still raise.
    """
    # np.linalg.solve in the parent Numba kernel checked every input entry.
    # Keep that contract even though raw LAPACK itself does not perform it.
    for value in np.nditer(A):
        if not np.isfinite(value.item()):
            raise np.linalg.LinAlgError("Array must not contain infs or NaNs.")
    for value in np.nditer(b):
        if not np.isfinite(value.item()):
            raise np.linalg.LinAlgError("Array must not contain infs or NaNs.")
    if A.shape[0]!=len(b) or A.shape[1]!=len(b):
        raise np.linalg.LinAlgError("Incompatible array sizes, system is not dimensionally valid.")
    if len(b)==0:return b.copy()
    n=np.array([len(b)],dtype=np.int32);one=np.array([1],dtype=np.int32)
    info=np.zeros(1,dtype=np.int32);uplo=np.array([ord('L')],dtype=np.uint8)
    F=A.copy();x=b.copy()
    _DPOSV(uplo.ctypes,n.ctypes,one.ctypes,F.ctypes,n.ctypes,x.ctypes,n.ctypes,info.ctypes)
    if info[0]==0:return x
    if info[0]<0:raise ValueError('LAPACK DPOSV argument error')
    return np.linalg.solve(A,b)
