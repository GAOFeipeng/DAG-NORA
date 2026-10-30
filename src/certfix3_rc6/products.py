"""Equivalent compiled extended-precision products used only by difference screens."""
from pathlib import Path
import ctypes
import numpy as np
L=ctypes.CDLL(str(Path(__file__).resolve().parent/'native/stable_products.so'))
D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
E=np.ctypeslib.ndpointer(dtype=np.longdouble,flags='C_CONTIGUOUS')
I=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
L.rc6_linear_terms.argtypes=[ctypes.c_int,ctypes.c_int,E,D,D,I,E,ctypes.c_double]
L.rc6_ld_matmul_t.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,E,E,E,ctypes.c_double]
L.rc6_linear_terms.restype=L.rc6_ld_matmul_t.restype=ctypes.c_int

def matmul(a,b,deadline=0.):
    a=np.ascontiguousarray(a,dtype=np.longdouble);bt=np.ascontiguousarray(np.asarray(b,dtype=np.longdouble).T)
    if a.ndim!=2 or bt.ndim!=2 or a.shape[1]!=bt.shape[1]:raise ValueError('matrix shape mismatch')
    out=np.empty((len(a),len(bt)),dtype=np.longdouble)
    code=L.rc6_ld_matmul_t(len(a),len(bt),a.shape[1],a,bt,out,deadline)
    if code:raise TimeoutError('hot-fit cap in stable matrix product')
    return out

def linear_delta(G,old,new,mu,lam,g_extended=None,deadline=0.):
    n=len(old);cols=np.flatnonzero(np.any(old!=new,axis=0)).astype(np.int64)
    if len(cols)==0:return np.longdouble(0),np.longdouble(0)
    g=np.ascontiguousarray(G,dtype=np.longdouble) if g_extended is None else g_extended
    p=np.ascontiguousarray(old,dtype=np.float64);q=np.ascontiguousarray(new,dtype=np.float64)
    terms=np.empty((n,len(cols)),dtype=np.longdouble)
    code=L.rc6_linear_terms(n,len(cols),g,p,q,cols,terms,deadline)
    if code:raise TimeoutError('hot-fit cap in stable update difference')
    pl=p[:,cols].astype(np.longdouble);ql=q[:,cols].astype(np.longdouble);v=ql-pl
    d1=np.sum((np.abs(ql)-np.abs(pl))[v!=0],dtype=np.longdouble)
    dl=np.sum(terms,dtype=np.longdouble)
    dj=np.longdouble(mu)*(dl+np.longdouble(lam)*d1)
    ab=np.longdouble(mu)*(np.sum(np.abs(terms))+abs(np.longdouble(lam)*d1))
    return dj,ab
