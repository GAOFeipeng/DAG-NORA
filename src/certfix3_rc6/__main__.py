"""python -m certfix3_rc6 --x X.npy --out run --budget 600 --cpu 0"""
import argparse,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Single CRC using a frozen conditional-column solver')
    p.add_argument('--x',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--budget',type=float,default=600.)
    p.add_argument('--cpu',type=int)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--column-only',action='store_true')
    p.add_argument('--w',type=Path,help='raw endpoint .npy for RC only; requires --mu and --s')
    p.add_argument('--mu',type=float);p.add_argument('--s',type=float)
    a=p.parse_args()
    if a.cpu is not None:
        if a.cpu not in os.sched_getaffinity(0):p.error('requested CPU is not available')
        os.sched_setaffinity(0,{a.cpu})
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]='1'
    import numpy as np
    from . import fit,refine
    from .io import save_result
    X=np.load(a.x,allow_pickle=False)
    mode='audit' if a.audit else 'production'
    kw=dict(hot_cap_seconds=a.budget,mode=mode,checkpoints=a.out/'audit' if a.audit else None)
    if a.w is None:
        result=fit(X,column_only=a.column_only,**kw)
    else:
        if a.mu is None or a.s is None:p.error('--w requires the actual saved --mu and --s')
        if a.column_only:p.error('--column-only cannot be combined with --w')
        result=refine(X,np.load(a.w,allow_pickle=False),mu=a.mu,s=a.s,**kw)
    path=save_result(result,a.out)
    print(f"{result['method_id']}: {result['status']}; returned={result['final_returned']}; independent_qualified={result['numerically_qualified']}; {result['seconds']:.6f}s")
    print(path)
    return 0 if result['final_returned'] else 2

if __name__=='__main__':raise SystemExit(main())
