"""Inherited finite inverse/gradient gates; serialized per process.

Only full refreshes run these gates; rank steps use periodic residual probes.
"""
from __future__ import annotations

from pathlib import Path

import ctypes,time

import numpy as np

import scipy.linalg as sla

LIB=ctypes.CDLL(str(Path(__file__).resolve().parent/'native/quad_inverse.so'))

PTR=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')

LIB.quad_inverse.argtypes=[ctypes.c_int,PTR,ctypes.c_double,PTR,PTR,PTR,PTR,PTR]

LIB.quad_inverse.restype=ctypes.c_int

LIB.quad_witness.argtypes=[ctypes.c_int,PTR,ctypes.c_double,PTR,PTR,PTR]

U=np.finfo(float).eps/2

REL_INVERSE_TOL=1e-8

GRAD_SCALED_TOL=1e-8

STATS={}

def reset_stats():
 STATS.clear();STATS.update(checks=0,fast_pass=0,quad_fallbacks=0,quad_seconds=0.,quad_witnesses=0,
  positive_fallbacks=0,reliable_refreshes=0,rejected_unresolved=0,max_fast_backward=0.,max_accepted_gradient_error=0.,events=[],
  certfix1_attempts=0,certfix1_successes=0,certfix1_unresolved=0,certfix1_seconds=0.,
  certfix1_construction_seconds=0.,certfix1_exact_verification_seconds=0.,certfix1_events=[])

reset_stats()

class NumericalUnresolved(FloatingPointError):pass

def domain_certificate(W,K,s):
 d=len(W);x=K@np.ones(d);A=W*W
 if not np.all(np.isfinite(x)) or np.min(x)<=0:
  return dict(legal=False,reason='no_positive_witness',lower=-float('inf'),old_abs_residual=float('inf'))
 y=s*x-A@x;gamma=(d+8)*U/(1-(d+8)*U)
 rad=gamma*(s*np.abs(x)+A@np.abs(x))
 lower=float(np.min(y-rad));method='binary64_outward_error_bound'
 if lower<=0:
  lo=np.empty(d);hi=np.empty(d)
  LIB.quad_witness(d,np.ascontiguousarray(W),s,np.ascontiguousarray(x),lo,hi)
  lower=float(lo.min());method='binary128_dot_enclosure';STATS['quad_witnesses']+=1
 return dict(legal=bool(lower>0),reason='certified' if lower>0 else 'unresolved',lower=lower,
  max_x=float(x.max()),old_abs_residual=float(np.max(np.abs(y-1.))),method=method)

def forward_bound(K,E):
 d=len(K);infl=1.+64*(d+4)*U
 e=float(np.max(np.sum(E,axis=1)))*infl
 if not np.isfinite(e) or e>=.1:return None,float(e)
 P=(np.abs(K)@E)*infl;Q=(P@E)*infl
 tail=np.sum(Q,axis=1)*infl*(e/(1-e))
 B=(P+Q+tail[:,None])*infl
 return B,float(e)

def accuracy(W,K,mu,E=None,B=None):
 if B is None:
  B,e=forward_bound(K,E)
  if B is None:return dict(reliable=False,inverse_relative_bound=float('inf'),gradient_scaled_bound=float('inf'),residual_bound=e),None
 else:e=None
 inv=float(np.max(B/(1.+np.abs(K))))
 bg=2*np.abs(W)*B.T;gscaled=float(np.max(bg/(mu+2*np.abs(W*K.T))))
 good=np.isfinite(inv) and inv<=REL_INVERSE_TOL and gscaled<=GRAD_SCALED_TOL
 return dict(reliable=bool(good),inverse_relative_bound=inv,gradient_scaled_bound=gscaled,residual_bound=e),B

def check_existing(W,K,s,mu):
 d=len(W);I=np.eye(d);M=s*I-W*W;dom=domain_certificate(W,K,s)
 AK=np.abs(M)@np.abs(K);residual=I-M@K;gamma=(d+8)*U/(1-(d+8)*U)
 E=np.abs(residual)+gamma*(AK+I)
 berr=float(np.max(np.divide(np.abs(residual),AK+I,out=np.zeros_like(M),where=(AK+I)>0)))
 acc,B=accuracy(W,K,mu,E=E)
 STATS['max_fast_backward']=max(STATS['max_fast_backward'],berr)
 return dict(domain=dom,accuracy=acc,componentwise_backward_error=berr),B

def quad(W,s):
 d=len(W);K=np.empty_like(W);E=np.empty_like(W);cast=np.empty_like(W);ld=np.empty(1);rad=np.empty(1)
 t=time.perf_counter();status=LIB.quad_inverse(d,np.ascontiguousarray(W),float(s),K,E,cast,ld,rad)
 elapsed=time.perf_counter()-t;STATS['quad_seconds']+=elapsed;STATS['quad_fallbacks']+=1
 if status:raise NumericalUnresolved(f'binary128 solve unresolved/nonpositive elimination pivot: {status}')
 B,e=forward_bound(np.abs(K)+cast,E)
 if B is None:raise NumericalUnresolved('binary128 inverse residual enclosure failed')
 B+=cast
 return K,float(ld[0]),B,dict(binary128_residual_bound=e,logdet_rounding_radius=float(rad[0]),seconds=elapsed)
