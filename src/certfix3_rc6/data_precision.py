"""Shared fixed-Gram data difference with explicit binary-tree reduction.

Extracted from the existing terminal implementation. Products and differences
are formed after long-double conversion. Its matching envelope uses
k=2*d+ceil(log2(d*changed_columns))+12, plus final float64 conversion error.
This is the existing engineering error model, not a new precision backend.
The platform gate is supplied by the same cached capability check at both
entry points. There is no state cache here.
"""
import numpy as np


def _pair_sum(a):
    """Explicit binary tree: every summand traverses at most ceil(log2(m))."""
    v=np.asarray(a,dtype=np.longdouble).ravel().copy()
    if not len(v):return np.longdouble(0)
    while len(v)>1:
        n=len(v)//2
        w=v[:2*n:2]+v[1:2*n:2]
        v=np.concatenate((w,v[-1:])) if len(v)%2 else w
    return v[0]


def pairwise_data_delta(G,old,new,mu,*,precision_available):
    if not precision_available:
        return dict(reliable=False,reason='no_extra_precision')
    if (not np.isfinite(mu) or mu<=0 or any(a.dtype!=np.dtype('float64') or
            not np.isfinite(a).all() for a in (G,old,new))):
        return dict(reliable=False,reason='invalid_stored_inputs')
    ld=np.longdouble;d=len(G);cols=np.flatnonzero(np.any(new!=old,axis=0))
    if not len(cols):return dict(reliable=True,loss_delta_F=0.,data_error_F=0.,rounding_steps=0)
    g=G.astype(ld);o=old[:,cols].astype(ld);n=new[:,cols].astype(ld)
    p=n-o;r=o-np.eye(d,dtype=ld)[:,cols]
    gr=g@r;gp=g@p
    if gr.dtype!=np.dtype(ld) or gp.dtype!=np.dtype(ld):
        return dict(reliable=False,reason='dtype_not_preserved')
    terms=p*(gr+ld(.5)*gp)
    scale_terms=abs(p)*(abs(g)@abs(r)+ld(.5)*(abs(g)@abs(p)))
    value=ld(mu)*_pair_sum(terms);scale=ld(mu)*_pair_sum(scale_terms)
    depth=(p.size-1).bit_length();k=2*d+depth+12;u=np.finfo(ld).eps/ld(2)
    ku=ld(k)*u
    if not all(np.isfinite(x) for x in (value,scale,ku)) or ku>=ld(.25):
        return dict(reliable=False,reason='data_roundoff_model_unavailable')
    gamma=ku/(1-ku);rounded=float(value)
    conversion=abs(ld(rounded)-value)
    error=gamma/(1-gamma)*scale+conversion
    return dict(reliable=np.isfinite(rounded),loss_delta_F=rounded,
        data_error_F=float(np.nextafter(float(error),np.inf)),absolute_scale_F=float(scale),
        rounding_steps=k,reduction_depth=depth,conversion_error_F=float(conversion),
        intermediate_dtype=str(gr.dtype),unit_roundoff=str(u))

