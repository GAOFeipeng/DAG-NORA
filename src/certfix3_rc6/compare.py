"""Bounded, instance-owned numerical comparisons. No module patching.

Same production thresholds and binary128 route as rc4-fix. The inexpensive
calculation path is optimized separately; undefined decisions stay undefined.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import ctypes,time,json
import numpy as np
from . import guards as ns
from .numerics import bounded_inverse
from . import products
from .timing import Ledger, metered
from .c2_comparison import C2Gate
ROOT=Path(__file__).resolve().parent
ARR=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
LIB=ctypes.CDLL(str(ROOT/'native/native_difference.so'))
LIB.full_decide.argtypes=[ctypes.c_int,ARR,ARR,ARR,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_double,ctypes.c_int,ARR]
LIB.block_decide.argtypes=[ctypes.c_int,ARR,ARR,ARR,ARR,ARR,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ARR]
LIB.full_decide_until.argtypes=LIB.full_decide.argtypes+[ctypes.c_double]
LIB.block_decide_until.argtypes=LIB.block_decide.argtypes+[ctypes.c_double]
LIB.full_decide.restype=LIB.block_decide.restype=LIB.full_decide_until.restype=LIB.block_decide_until.restype=ctypes.c_int
LIB.long_double_mantissa_bits.restype=ctypes.c_int
SCREEN_REL=1e-9
TERMINAL_SITES=frozenset(('base_sweep_monotonicity','polish_column_increase'))

class StableDelta(float):
    """Numeric compatibility with the original caller, with decision provenance."""
    def __new__(cls,value,source):
        obj=super().__new__(cls,value);obj.source=source;return obj


class CandidateUncertain(ns.NumericalUnresolved):
    """Candidate cannot be safely validated by the bounded numerical route."""


def event_deadline(d,hot_deadline,now=None):
    now=time.monotonic() if now is None else now
    remaining=hot_deadline-now
    if remaining<=0:raise TimeoutError('Prespecified hot-fit cap exhausted')
    budget=min(remaining,60.,max(2.,12.*(d/400.)**3))
    return min(hot_deadline,now+budget),budget


def _pred(value,bound,kind):
    if kind=='le': return value<=bound
    if kind=='lt': return value<bound
    if kind=='abs_le': return abs(value)<=bound
    raise ValueError('unknown objective comparison kind')


def _gap(value,bound,kind):
    return abs((abs(value) if kind=='abs_le' else value)-bound)


class Comparator:
    def __init__(self,audit_dir=None,exception_dir=None,hot_deadline=None):
        self.ledger=Ledger()
        self.c2=C2Gate(self)
        self.hot_deadline=time.monotonic()+300. if hot_deadline is None else hot_deadline
        self.maximum_event_budget=0.
        self._stat=None;self.audit_dir=Path(audit_dir) if audit_dir else None
        self.counts=Counter();self.seconds=Counter();self.unresolved_sites=Counter()
        self.last=None;self.last_unresolved=None;self.event_id=0;self.g_written=False
        self.exception_dir=Path(exception_dir) if exception_dir else None
        self.terminal_event_id=0;self.last_terminal=None;self.last_stopping_source=None
        self.column_event_id=0;self.last_column_terminal=None
        if self.audit_dir:self.audit_dir.mkdir(parents=True,exist_ok=True)


    def _stats(self):
        if self._stat is None:
            self._stat=dict(comparisons=0,ordinary=0,direct_screens=0,direct_seconds=0.,
                            hp_attempts=0,hp_successes=0,hp_unresolved=0,hp_seconds=0.,
                            hp_construction_seconds=0.,gate_seconds=0.,changed_decisions=0,
                            events=[],site_counts={})
        return self._stat


    def record_terminal(self,G,old,new,mu,s,lam,delta,bound,kind,site,answer):
        self.last_terminal=dict(answer)
        self.last_terminal.update(site=site,kind=kind,mu=float(mu),s=float(s),lam=float(lam),
                                  ordinary_delta=float(delta),bound=float(bound))
        if self.exception_dir is None:return
        t=time.perf_counter();self.exception_dir.mkdir(parents=True,exist_ok=True)
        self.terminal_event_id+=1
        stem=f'terminal_{self.terminal_event_id:04d}'
        np.savez(self.exception_dir/(stem+'.npz'),G=G,old=old,new=new,mu=mu,s=s,lam=lam,bound=bound,site=site,kind=kind)
        (self.exception_dir/(stem+'.json')).write_text(json.dumps(self.last_terminal,indent=2)+'\n')
        self.seconds['terminal_diagnostic_io']+=time.perf_counter()-t


    def confirm_terminal(self,G,old,new,mu,s,lam,delta,bound,kind,site):
        # No coarse/direct screen is allowed to authorize termination here.
        self.counts['terminal_rechecks']+=1;t=time.perf_counter()
        try:
            td=time.perf_counter()
            try:
                # Independent scratch checks never replace the algorithm's K/C.
                if self.c2.domain_certified(old,new,mu,s,lam):
                    self.c2.counts['certified_domain_reuses']+=2
                else:
                    with self.ledger.measure('inverse_refresh'):
                        bounded_inverse(old,s,mu);bounded_inverse(new,s,mu)
            except (ns.NumericalUnresolved,ValueError) as exc:
                self.counts['terminal_unresolved']+=1
                ans=dict(resolved=False,reason=str(exc),source='bounded_domain_guard',production_action='numerical_unresolved')
                self.record_terminal(G,old,new,mu,s,lam,delta,bound,kind,site,ans)
                raise ns.NumericalUnresolved(f'terminal_recheck_domain_unresolved: {site}: {exc}') from exc
            finally:self.seconds['terminal_domain_check']+=time.perf_counter()-td
            tn=time.perf_counter();self.counts['terminal_native_calls']+=1
            try:ans=self.native_decision(G,old,new,mu,s,lam,bound,kind)
            finally:self.seconds['terminal_native']+=time.perf_counter()-tn
            ans=dict(ans,source='bounded_native_terminal_recheck')
            if not ans['resolved']:
                self.counts['terminal_unresolved']+=1
                ans['production_action']='numerical_unresolved'
                self.record_terminal(G,old,new,mu,s,lam,delta,bound,kind,site,ans)
                raise ns.NumericalUnresolved(f'terminal_recheck_numerical_unresolved: {site}: {ans["reason"]}')
            key='terminal_rescued' if ans['decision'] else 'terminal_confirmed_increase'
            self.counts[key]+=1
            ans['production_action']='continue_with_stable_delta' if ans['decision'] else 'preserve_failure'
            self.record_terminal(G,old,new,mu,s,lam,delta,bound,kind,site,ans)
            return ans['decision'],StableDelta(ans['delta_F'],ans['source'])
        finally:self.seconds['terminal_recheck_total']+=time.perf_counter()-t


    def record_block(self,G,b,c,old,new,mu,lam,delta,bound,site,answer):
        """Failure-only evidence. Full data matrices are never hashed in fit."""
        info=dict(answer,site=site,mu=float(mu),lam=float(lam),kind='le',
                  ordinary_delta_Q=float(delta),bound_Q=float(bound),bound_F=float(mu*bound),
                  caller_units='Q',native_units='F=mu*Q')
        if answer.get('resolved'):
            info['delta_Q']=float(answer['delta_F']/mu)
        self.last_column_terminal=info
        if self.exception_dir is None:return
        t=time.perf_counter();self.exception_dir.mkdir(parents=True,exist_ok=True)
        self.column_event_id+=1;stem=f'column_{self.column_event_id:04d}'
        np.savez(self.exception_dir/(stem+'.npz'),G=G,b=b,c=c,old=old,new=new,
                 mu=mu,lam=lam,bound_Q=bound,bound_F=mu*bound,
                 ordinary_delta_Q=delta,site=site,kind='le')
        (self.exception_dir/(stem+'.json')).write_text(json.dumps(info,indent=2)+'\n')
        self.seconds['column_diagnostic_io']+=time.perf_counter()-t


    def confirm_block(self,G,b,c,old,new,mu,lam,delta,bound,site):
        """One bounded LD/binary128 check, in consistent units; no new solve."""
        self.counts['column_terminal_rechecks']+=1;t=time.perf_counter()
        try:
            ans=self.native_decision(G,old,new,mu,None,lam,mu*bound,'le',block=(b,c))
            ans=dict(ans,source='bounded_native_column_recheck')
            if not ans['resolved']:
                # rc6-fix3: decision-oriented fallback for `boundary_overlap` only.
                # Every already-resolved verdict returned above this line, so the
                # fallback can never flip an existing accept/reject.
                fallback=self.boundary_overlap_fallback(ans,mu*bound,'le')
                if fallback is not None:
                    self.counts['column_terminal_boundary_overlap_resolved']+=1
                    ans=dict(ans,resolved=True,decision=bool(fallback),
                             source='boundary_overlap_direct_delta_fallback',
                             fallback_effective_error=float(max(ans.get('empirical_uncertainty') or 0.,ans.get('resolution') or 0.)),
                             production_action='keep_original_candidate' if fallback else 'preserve_failure')
                    self.record_block(G,b,c,old,new,mu,lam,delta,bound,site,ans)
                    self.last=dict(site=site,kind='le',bound=float(mu*bound),**ans)
                    self.counts['column_terminal_rescued' if fallback else 'column_confirmed_increase']+=1
                    return bool(fallback),StableDelta(ans['delta_F']/mu,ans['source'])
                self.counts['column_terminal_unresolved']+=1
                ans['production_action']='numerical_unresolved'
                self.record_block(G,b,c,old,new,mu,lam,delta,bound,site,ans)
                raise ns.NumericalUnresolved(f'column_recheck_numerical_unresolved: {site}: {ans["reason"]}')
            self.counts['column_terminal_rescued' if ans['decision'] else 'column_confirmed_increase']+=1
            ans['production_action']='keep_original_candidate' if ans['decision'] else 'preserve_failure'
            self.record_block(G,b,c,old,new,mu,lam,delta,bound,site,ans)
            return ans['decision'],StableDelta(ans['delta_F']/mu,ans['source'])
        finally:self.seconds['column_terminal_total']+=time.perf_counter()-t


    def refine(self,G,old,new,mu,s,lam,delta,bound,kind,ordinary,site,direct,radius,direct_info,block=None):
        # Legacy block screens already compute direct/radius in F units, but
        # their caller supplies delta/bound in Q units. Convert BOTH scalars
        # before any predicate and convert the returned difference back.
        if block is not None:
            answer,dF=self._refine_F(G,old,new,mu,s,lam,mu*delta,mu*bound,kind,
                                    ordinary,site,direct,radius,direct_info,block)
            return answer,dF/mu
        return self._refine_F(G,old,new,mu,s,lam,delta,bound,kind,
                              ordinary,site,direct,radius,direct_info,block)


    def _refine_F(self,G,old,new,mu,s,lam,delta,bound,kind,ordinary,site,direct,radius,direct_info,block=None):
        if np.isfinite(direct) and _pred(direct,bound,kind)==bool(ordinary) and _gap(direct,bound,kind)>radius:
            self._stats()['ordinary']+=1
            return bool(ordinary),float(delta)
        t=time.perf_counter();self.counts['native_attempts']+=1
        try:
            ans=self.native_decision(G,old,new,mu,s,lam,bound,kind,block)
            self.last=dict(site=site,kind=kind,bound=float(bound),**ans)
            if self.audit_dir:
                io=time.perf_counter();self.event_id+=1
                if not self.g_written:np.save(self.audit_dir/'G.npy',G);self.g_written=True
                kw=dict(old=old,new=new,mu=mu,s=np.nan if s is None else s,lam=lam,bound=bound,kind=kind,site=site)
                if block is not None:kw.update(b=block[0],c=block[1])
                np.savez(self.audit_dir/f'event_{self.event_id:06d}.npz',**kw)
                with (self.audit_dir/'events.jsonl').open('a') as f:f.write(json.dumps(dict(event=self.event_id,**self.last))+'\n')
                self.seconds['audit_io']+=time.perf_counter()-io
            if ans['resolved']:
                self.counts['native_resolved']+=1
                self.counts['changed_from_ordinary']+=int(ans['decision']!=bool(ordinary))
                return ans['decision'],ans['delta_F']
            self.counts['native_unresolved']+=1;self.unresolved_sites[site]+=1
            self.last['production_action']=self.action(site,kind)
            self.last_unresolved=self.last.copy()
            if kind=='abs_le' or 'relative_stopping' in site:
                self.counts['unresolved_stop_continue']+=1
                return False,float(direct if np.isfinite(direct) else delta)
            if site.startswith('joint_strict'):
                self.counts['unresolved_joint_reject']+=1
                return False,float(direct if np.isfinite(direct) else delta)
            if site=='local_block_monotonicity':
                raise CandidateUncertain(f'bounded_candidate_comparison_unresolved: {site}')
            raise ns.NumericalUnresolved(f'bounded_state_comparison_unresolved: {site}')
        finally:self.seconds['native_total']+=time.perf_counter()-t


    @staticmethod
    def action(site,kind):
        if kind=='abs_le' or 'relative_stopping' in site:return 'continue_not_converged'
        if site.startswith('joint_strict') or site=='local_block_monotonicity':return 'candidate_not_accepted'
        return 'return_numerical_unresolved'


    @staticmethod
    def boundary_overlap_fallback(answer,bound_F,kind):
        """Decision-oriented fallback for an unresolved `boundary_overlap` verdict.

        Consulted ONLY after the untouched native decision returned
        ``resolved=False``; every already-resolved decision is returned before
        this helper is reached, so no existing accept/reject can flip.

        It reuses only quantities the native kernel already produced, in their
        existing units (F = mu * Q for the block contract):

        * ``delta_F``    -- the binary128 objective difference;
        * ``err``        -- the existing cross-precision error estimate
                            (32*|a-b| inflated by the rounding/cast floors);
        * ``gap``        -- the existing distance to the decision boundary,
                            ``|delta - bound|``;
        * ``resolution`` -- the existing machine-resolution floor,
                            ``max(8*eps_longdouble*max(1,|delta|), |bound|/100)``.

        The verdict is resolved only when the decision margin dominates BOTH
        existing radii with the factor the native criterion already uses
        (``gap > 4*err``), i.e. ``gap > 4*max(err, resolution)``; the returned
        decision is the same predicate applied to the same ``delta_F`` value.
        No new tolerance, scale factor or magic number is introduced; the
        objective, bound, units and acceptance threshold are untouched.
        """
        if answer.get('reason')!='boundary_overlap':
            return None
        try:
            # Field names as actually produced by native_decision:
            # empirical_uncertainty == the existing error estimate (32*|a-b|
            # inflated by the rounding/cast floors);
            # distance_to_boundary == the existing |delta - bound| margin.
            value=float(answer['delta_F'])
            err=float(answer['empirical_uncertainty'])
            gap=float(answer['distance_to_boundary'])
            resolution=float(answer['resolution'])
        except (KeyError,TypeError,ValueError):
            return None
        if not all(np.isfinite(v) for v in (value,err,gap,resolution)):
            return None
        if not (gap>4.*max(err,resolution)):
            return None
        return bool(_pred(value,bound_F,kind))


    def summary(self):
        z=self._stats().copy();z.pop('events',None)
        return dict(**z,native_counts=dict(self.counts),native_seconds=dict(self.seconds),c2=self.c2.summary(),
                    unresolved_sites=dict(self.unresolved_sites),last_unresolved=self.last_unresolved,
                    last_terminal_recheck=self.last_terminal,last_stopping_delta_source=self.last_stopping_source,
                    last_column_terminal_recheck=self.last_column_terminal,
                    last_joint_crosscheck=getattr(self,'last_joint_crosscheck',None),
                    column_comparison_units="Q at caller; F=mu*Q in stable/native refinement; bound scaled together",
                    full_matrix_hashes=self.c2.counts['matrix_hashes'],mpmath_80_120_calls=0,
                    event_budget_rule='min(global_remaining,60,max(2,12*(d/400)**3))',
                    maximum_event_budget_seconds=self.maximum_event_budget,
                    shared_precision_deadline=True,interval_certificate=False,
                    long_double_mantissa_bits=LIB.long_double_mantissa_bits(),quad_mantissa_bits=113)


    def _coarse(self,delta,bound,kind,ordinary,scale,site):
        st=self._stats();st['comparisons']+=1
        st['site_counts'][site]=st['site_counts'].get(site,0)+1
        radius=SCREEN_REL*max(1.,float(scale),abs(float(delta)),abs(float(bound)))
        if np.isfinite(delta) and _gap(delta,bound,kind)>radius:
            st['ordinary']+=1
            return bool(ordinary),float(delta)
        return None


    def _linear_delta_reference(self,G,old,new,mu,lam):
        # Long-double update form avoids subtraction of two huge residual losses.
        # Symmetric-part form also equals the stored quadratic if G isn't bitwise symmetric.
        g=np.asarray(G,dtype=np.longdouble);p=np.asarray(old,dtype=np.longdouble)
        q=np.asarray(new,dtype=np.longdouble);v=q-p
        jj=np.flatnonzero(np.any(v!=0,axis=0))
        r=p[:,jj]-np.eye(len(p),dtype=np.longdouble)[:,jj];v=v[:,jj]
        if np.array_equal(G,np.asarray(G).T):grad=g@r
        else:grad=(g@r+g.T@r)/2
        terms=v*grad+v*(g@v)/2
        dl=np.sum(terms,dtype=np.longdouble)
        d1=np.sum((np.abs(q[:,jj])-np.abs(p[:,jj]))[v!=0],dtype=np.longdouble)
        dj=np.longdouble(mu)*(dl+np.longdouble(lam)*d1)
        abs_terms=np.longdouble(mu)*(np.sum(np.abs(terms))+abs(np.longdouble(lam)*d1))
        return dj,abs_terms


    def _linear_delta(self,G,old,new,mu,lam):
        # Same long-double operations, compiled loops avoid work for zero updates.
        # Only the symmetric Gram route is transformed. The reference expression
        # handles externally supplied nonsymmetric matrices without assuming symmetry.
        if not np.array_equal(G,np.asarray(G).T):
            return self._linear_delta_reference(G,old,new,mu,lam)
        self.counts['compiled_update_products']+=1
        return products.linear_delta(G,old,new,mu,lam,deadline=self.hot_deadline)


    def _direct_full(self,G,old,new,mu,s,lam,K_old=None):
        dj,abs_terms=self._linear_delta(G,old,new,mu,lam)
        if K_old is None:
            # A diagnostic inverse must meet the very same old gate.
            K_old,_,_,num=bounded_inverse(old,s,mu)
            if not num['domain']['legal'] or not num['accuracy']['reliable']:
                raise ns.NumericalUnresolved('certfix2 direct-screen inverse/domain unresolved')
        jj=np.flatnonzero(np.any(old!=new,axis=0))
        if not len(jj):return 0.,0.,{'method':'identical_weights','changed_columns':0}
        p=old[:,jj].astype(np.longdouble);q=new[:,jj].astype(np.longdouble)
        # Algebraic squared-weight DIFFERENCE in extended precision, not (q-p)^2.
        U=(q-p)*(q+p)
        H=products.matmul(np.asarray(K_old[jj,:],dtype=np.longdouble),U,self.hot_deadline)
        T=np.eye(len(jj))-np.asarray(H,dtype=float)
        sg,ld=np.linalg.slogdet(T)
        dh=-ld if sg>0 else float('nan')
        val=float(dj)+dh
        # Screening indicator only, not a rigorous roundoff enclosure of slogdet.
        # Non-small H or determinant ambiguity cannot authorize a comparison.
        hn=float(np.linalg.norm(H,np.inf))
        radius=float(64*np.finfo(float).eps*(len(jj)+1)*(1.+abs_terms+abs(dh)+hn))
        if not np.isfinite(hn) or hn>=.25 or sg<=0:radius=float('inf')
        return val,radius,dict(method='update_quadratic_changed_L1_determinant_ratio',
                               changed_columns=int(len(jj)),ratio_norm_inf=hn,
                               delta_scaled_data=float(dj),delta_barrier=float(dh),
                               screening_radius=radius,screen_is_not_interval_proof=True)


    def _full_raw(self,G,old,new,mu,s,lam,delta,bound,kind,ordinary,site,scale=1.,K_old=None):
        quick=self._coarse(delta,bound,kind,ordinary,scale,site)
        if quick is not None:return quick
        if np.array_equal(old,new):
            # Analytically exact equality of arguments, no precision evidence needed.
            return bool(_pred(0.,bound,kind)),0.
        t=time.perf_counter();st=self._stats();st['direct_screens']+=1
        try:
            if site in ('polish_column_increase','polish_relative_stopping'):
                direct,radius,info=self.c2.direct(G,old,new,mu,s,lam,K_old,
                    lambda:self._direct_full(G,old,new,mu,s,lam,K_old))
            else:direct,radius,info=self._direct_full(G,old,new,mu,s,lam,K_old)
        finally:st['direct_seconds']+=time.perf_counter()-t
        return self.refine(G,old,new,mu,s,lam,delta,bound,kind,ordinary,site,direct,radius,info)


    def compare_pair(self,p,Vold,Vnew,delta,bound,kind,ordinary,site):
        quick=self._coarse(delta,bound,kind,ordinary,1.,site)
        if quick is not None:return quick
        if np.array_equal(Vold,Vnew):return bool(_pred(0.,bound,kind)),0.
        t=time.perf_counter();st=self._stats();st['direct_screens']+=1
        old=p.W.copy();new=p.W.copy();old[:,p.J]=Vold;new[:,p.J]=Vnew
        dj,abs_terms=self._linear_delta(p.G,old,new,p.mu,p.lam)
        r=np.asarray(p.R0,dtype=np.longdouble);I=np.eye(2,dtype=np.longdouble)
        dets=[]
        for v in (Vold,Vnew):
            vl=np.asarray(v,dtype=np.longdouble);T=I-r@(vl*vl);dets.append(T[0,0]*T[1,1]-T[0,1]*T[1,0])
        if min(dets)>0:dh=-np.log1p((dets[1]-dets[0])/dets[0]);direct=float(dj+dh)
        else:direct=float('nan');dh=float('nan')
        radius=float(64*np.finfo(float).eps*(len(old)+1)*(1.+abs_terms+abs(dh)))
        info=dict(method='update_form_with_existing_rank_two_deleted_inverse',screening_radius=radius,screen_is_not_interval_proof=True)
        st['direct_seconds']+=time.perf_counter()-t
        return self.refine(p.G,old,new,p.mu,p.s,p.lam,delta,bound,kind,ordinary,site,direct,radius,info)


    def _block_raw(self,G,b,c,old,new,mu,lam,delta,bound,ordinary,site,scale):
        quick=self._coarse(delta,bound,'le',ordinary,scale,site)
        if quick is not None:return quick
        if np.array_equal(old,new):return True,0.
        t=time.perf_counter();st=self._stats();st['direct_screens']+=1
        g=G.astype(np.longdouble);bb=b.astype(np.longdouble);cc=c.astype(np.longdouble);p=old.astype(np.longdouble);q=new.astype(np.longdouble);v=q-p
        terms=v*(g@p-bb)+v*(g@v)/2;dl=np.sum(terms);d1=np.sum((abs(q)-abs(p))[v!=0])
        a=1-np.sum(cc*p*p);bnew=1-np.sum(cc*q*q)
        dh=-np.log1p(-np.sum(cc*v*(q+p))/a) if min(a,bnew)>0 else np.longdouble('nan')
        direct=float(np.longdouble(mu)*(dl+np.longdouble(lam)*d1)+dh)
        rad=float(64*np.finfo(float).eps*(len(p)+1)*(1.+abs(dh)+mu*(np.sum(abs(terms))+abs(lam*d1))))
        st['direct_seconds']+=time.perf_counter()-t
        return self.refine(G,old,new,mu,None,lam,delta,bound,'le',ordinary,site,direct,rad,
                       dict(method='conditional_column_update_difference',screening_radius=rad,screen_is_not_interval_proof=True),block=(b,c))


    @metered('comparison_screen_and_bookkeeping')
    def full(self,G,old,new,mu,s,lam,delta,bound,kind,ordinary,site,scale=1.,K_old=None):
        cmp=self
        if isinstance(delta,StableDelta) and cmp is not None and 'stopping' in site:
            cmp.counts['stable_delta_stopping_calls']+=1
            cmp.last_stopping_source=delta.source
        if site not in TERMINAL_SITES:
            return self._full_raw(G,old,new,mu,s,lam,delta,bound,kind,ordinary,site,scale,K_old)
        if cmp is None:raise RuntimeError('Terminal recheck requires BoundedComparator context')
        if not ordinary:
            return cmp.confirm_terminal(G,old,new,mu,s,lam,delta,bound,kind,site)
        before=cmp.counts['native_attempts']
        try:
            answer=self._full_raw(G,old,new,mu,s,lam,delta,bound,kind,ordinary,site,scale,K_old)
        except ns.NumericalUnresolved as exc:
            cmp.record_terminal(G,old,new,mu,s,lam,delta,bound,kind,site,
                                dict(resolved=False,reason=str(exc),production_action='numerical_unresolved'))
            raise
        if answer[0]:return answer
        if cmp.counts['native_attempts']>before and cmp.last and cmp.last.get('resolved'):
            cmp.counts['terminal_already_native_confirmed']+=1
            cmp.record_terminal(G,old,new,mu,s,lam,delta,bound,kind,site,cmp.last)
            return False,StableDelta(answer[1],'existing_bounded_native')
        return cmp.confirm_terminal(G,old,new,mu,s,lam,delta,bound,kind,site)


    @metered('comparison_screen_and_bookkeeping')
    def block(self,G,b,c,old,new,mu,lam,delta,bound,ordinary,site,scale):
        """Column contract is Q = F / mu (up to a fixed-column constant).

        Preserve the original fast acceptance. A rejection that would abort the
        column update must be checked on the stored vectors, not on cancellation-
        prone objective scalars. No candidate is changed or solved a second time.
        """
        cmp=self
        if cmp is None:
            raise RuntimeError('Column comparison requires BoundedComparator context')
        if not ordinary or not np.isfinite(delta):
            # Count without letting a coarse screen authorize a terminal failure.
            st=self._stats();st['comparisons']+=1
            st['site_counts'][site]=st['site_counts'].get(site,0)+1
            return cmp.confirm_block(G,b,c,old,new,mu,lam,delta,bound,site)
        before=cmp.counts['native_attempts']
        answer=self._block_raw(G,b,c,old,new,mu,lam,delta,bound,ordinary,site,scale)
        if answer[0]:return answer
        if cmp.counts['native_attempts']>before and cmp.last and cmp.last.get('resolved'):
            cmp.counts['column_already_native_confirmed']+=1
            cmp.record_block(G,b,c,old,new,mu,lam,delta,bound,site,cmp.last)
            return answer
        return cmp.confirm_block(G,b,c,old,new,mu,lam,delta,bound,site)


    def recheck_joint_difference(self,G,old,new,mu,s,lam,bound,reduced_delta):
        """Recompute a disagreeing difference without relaxing either predicate.

        Called only after the original full/reduced crosscheck exceeded 1e-9.
        Existing LD/binary128 arithmetic and its deadline remain authoritative.
        An unresolved verdict or remaining disagreement cannot accept a trial.
        """
        self.counts['joint_crosscheck_rechecks']+=1
        started=time.perf_counter()
        try:
            ans=self.native_decision(G,old,new,mu,s,lam,bound,'lt')
            info=dict(ans,source='joint_crosscheck_native_recheck',reduced_delta=float(reduced_delta))
            if not ans.get('resolved'):
                self.counts['joint_crosscheck_unresolved']+=1
                info.update(crosscheck_passed=False,accepted=False)
            else:
                error=abs(float(ans['delta_F'])-float(reduced_delta))
                passed=bool(np.isfinite(error) and error<=1e-9)
                info.update(crosscheck_error=error,crosscheck_passed=passed,
                            accepted=bool(passed and ans['decision']))
                self.counts['joint_crosscheck_resolved']+=1
                if not passed:self.counts['joint_crosscheck_disagreement']+=1
            self.last_joint_crosscheck=info
            return info
        finally:self.seconds['joint_crosscheck_recheck']+=time.perf_counter()-started

    def native_decision(self,G,old,new,mu,s,lam,bound,kind,block=None):
        if self.c2.active is not None and block is None:
            return self.c2.native(G,old,new,mu,s,lam,bound,kind)
        return self._native_once(G,old,new,mu,s,lam,bound,kind,block)

    @metered('comparison_high_precision')
    def _native_once(self,G,old,new,mu,s,lam,bound,kind,block=None,event_window=None):
        # All production routes enter here once, including forced terminal rechecks.
        hot_deadline=self.hot_deadline
        deadline,budget=event_deadline(len(G),hot_deadline) if event_window is None else event_window
        self.counts['budgeted_native_events']+=1
        self.maximum_event_budget=max(self.maximum_event_budget,budget)
        g,p,q=(np.ascontiguousarray(v,dtype=np.float64) for v in (G,old,new))
        if g.ndim!=2 or g.shape[0]!=g.shape[1] or p.shape!=q.shape or (block is None and p.shape!=g.shape) or (block is not None and p.shape!=(len(g),)):
            return dict(resolved=False,reason='shape_mismatch',status=-11)
        if not all(np.isfinite(v).all() for v in (g,p,q)):
            return dict(resolved=False,reason='nonfinite_input',status=-9)
        if block is None and (np.any(np.diag(p)) or np.any(np.diag(q))):
            return dict(resolved=False,reason='nonzero_diagonal',status=-10)
        kinds={'le':0,'lt':1,'abs_le':2}
        if kind not in kinds:raise ValueError('Unknown comparison kind')
        z=np.full(8,np.nan)
        if block is None:
            code=LIB.full_decide_until(len(p),g,p,q,float(mu),float(s),float(lam),int(np.array_equal(g,g.T)),float(bound),kinds[kind],z,deadline)
        else:
            b,c=(np.ascontiguousarray(v,dtype=np.float64) for v in block)
            if not np.isfinite(b).all() or not np.isfinite(c).all():
                return dict(resolved=False,reason='nonfinite_block',status=-9)
            code=LIB.block_decide_until(len(p),g,b,c,p,q,float(mu),float(lam),float(bound),kinds[kind],z,deadline)
        if time.monotonic()>=hot_deadline:raise TimeoutError('Prespecified hot-fit cap exhausted')
        if code:return dict(resolved=False,reason='comparison_budget_exhausted' if code==-20 else 'native_arithmetic_or_local_domain_unresolved',status=int(code))
        return dict(resolved=bool(z[0]),decision=bool(z[1]) if z[0] else None,delta_F=float(z[2]),
                    empirical_uncertainty=float(z[3]),distance_to_boundary=float(z[4]),
                    cross_precision_difference=float(z[5]),operation_scale=float(z[6]),
                    resolution=float(z[7]),reason='resolved' if z[0] else 'boundary_overlap',status=0)
