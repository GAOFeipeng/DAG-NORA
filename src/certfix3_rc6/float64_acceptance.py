"""Exploratory C1/C2 float64 transactions; exact column/R kernels unchanged.

The tolerance is an engineering rule, NOT a rigorous forward error bound.
No full-matrix high precision objective comparison is consulted. A data-only
long-double difference is available once, on demand, after an undecided endpoint.
"""
from collections import Counter
from functools import lru_cache
import json
import math
import time
import numpy as np
from . import data_precision
import scipy.linalg as sla
from .compare import Comparator as OriginalComparator, StableDelta
from .c2_comparison import ComparisonUnresolved
from .guards import NumericalUnresolved
from .recovery import Recovery, RECOVERABLE, _finite
from .soft_events import HardNumericalFailure, require_finite, require_certified
from .finite_policy import (decide as finite_decide, objective_scale, stop_change,
                            state_reliable, STAGE_POSITIVE_CAP, SINGLE_RELATIVE_F)

POLICY = dict(absolute_F=1e-12, eps_multiplier=64., dimension_offset=8,
              single_relative_F=SINGLE_RELATIVE_F, stage_positive_normalized_cap=STAGE_POSITIVE_CAP, stagnant_tie_window=20,
              stagnant_required_relative_progress=.01, A_C2_residual=1e-4,
              B_C2_residual=1e-3, max_float64_rechecks=1,
              stagnant_tie_action='diagnostic_only')


def tolerance(d, *scales):
    return POLICY['absolute_F'] + 64*np.finfo(float).eps*(d+8)*max(1., *(abs(float(x)) for x in scales))


def classify(delta, tau, reliable=True):
    if not reliable or not np.isfinite(delta) or not np.isfinite(tau):return 'unresolved'
    if delta < -tau:return 'descent'
    if delta > tau:return 'rise'
    return 'tie'


def classify_evidence(delta,tau,error):
    """Decision margin must cover a separate numerical error estimate.

    This engineering envelope is not a rigorous interval certificate. In
    particular, increasing error can only make a verdict unresolved; it never
    increases the acceptance tolerance or turns an uncertain rise into a tie.
    """
    if not all(np.isfinite(x) for x in (delta,tau,error)) or error<0:return 'unresolved'
    if delta+error < -tau:return 'descent'
    if delta-error > tau:return 'rise'
    if abs(delta)+error <= tau:return 'tie'
    return 'unresolved'


def endpoint_evidence(G,anchor,trial):
    """Exceptional fixed-G difference using certified endpoint log-determinants.

    No new inverse, LU, high precision call or sample-space objective. Absolute
    Gram products measure cancellation BEFORE the dot products have cancelled.
    Only the original frozen numerical objective is compared.
    """
    if anchor.params!=trial.params or not np.array_equal(G,G.T):
        return dict(delta_F=float('nan'),error_F=float('inf'),reliable=False,reason='endpoint_identity')
    old,new=anchor.W,trial.W;mu,s,lam=anchor.params;d=len(G)
    if any(not np.isfinite(v).all() for v in (G,old,new)):
        return dict(delta_F=float('nan'),error_F=float('inf'),reliable=False,reason='nonfinite_endpoint')
    if any(ev.metrics.get('domain_lower',0.)<=0 or
           ev.metrics.get('inverse_error_bound',float('inf'))>1e-8 or
           ev.metrics.get('gradient_error_bound',float('inf'))>1e-8 for ev in (anchor,trial)):
        return dict(delta_F=float('nan'),error_F=float('inf'),reliable=False,reason='endpoint_state_gate')
    p=new-old;cols=np.flatnonzero(np.any(p!=0,axis=0))
    if not len(cols):return dict(delta_F=0.,error_F=0.,reliable=True,reason='identical_weights')
    v=p[:,cols];r=old[:,cols]-np.eye(d)[:,cols]
    gr=G@r;gp=G@v;av=abs(v);ar=abs(r);ag=abs(G)
    agr=ag@ar;agp=ag@av
    loss=mu*math.fsum(map(float,(v*(gr+.5*gp)).ravel()))
    l1=mu*lam*math.fsum(map(float,(abs(new[:,cols])-abs(old[:,cols])).ravel()))
    barrier=trial.metrics['h']-anchor.metrics['h']
    delta=math.fsum((loss,l1,barrier))
    product_scale=mu*float(np.sum(av*(agr+.5*agp)))
    l1_scale=mu*lam*float(np.sum(abs(old[:,cols])+abs(new[:,cols])))
    factor=64*np.finfo(float).eps*(d+8)
    barrier_error=anchor.metrics.get('barrier_error_estimate_F',float('inf'))+trial.metrics.get('barrier_error_estimate_F',float('inf'))
    error=factor*(1.+product_scale+l1_scale+abs(delta))+barrier_error
    # Diagnose agreement with the ordinary endpoint on its own PRE-cancellation
    # scale. This consistency radius is separate from both tau and error_F.
    endpoint_scale=.5*mu*float(np.sum(ar*agr+(ar+av)*(agr+agp)))
    # Unchanged columns cancel in theory, but the two scalar sums round them too.
    endpoint_scale+=abs(anchor.metrics['J']*mu)+abs(trial.metrics['J']*mu)
    ordinary=trial.metrics['F']-anchor.metrics['F']
    consistency=factor*(1.+endpoint_scale+l1_scale+abs(ordinary))+error
    reliable=bool(np.isfinite(error) and np.isfinite(consistency) and
                  abs(delta-ordinary)<=consistency)
    return dict(delta_F=delta,error_F=error,reliable=reliable,reason='certified_endpoint_components',
                loss_delta_F=loss,l1_delta_F=l1,barrier_delta_F=barrier,
                data_error_F=factor*product_scale,l1_error_F=factor*l1_scale,
                arithmetic_error_F=factor*(1.+abs(delta)),
                data_absolute_product_scale_F=product_scale,
                ordinary_difference_error_F=abs(delta-ordinary),
                ordinary_consistency_radius_F=consistency,barrier_error_F=barrier_error)


@lru_cache(maxsize=1)
def longdouble_capability():
    """Test arithmetic, not the platform-dependent dtype name or storage size."""
    ld=np.longdouble;fi=np.finfo(ld)
    x=np.array([[1e16,1.,-1e16]],dtype=np.float64)
    product=x.astype(ld)@np.ones((3,1),dtype=ld)
    values=np.array([1.,np.nextafter(1.,2.),-1e100,np.finfo(float).tiny,
                     np.nextafter(0.,1.)],dtype=np.float64)
    supported=bool(fi.nmant>np.finfo(float).nmant and product.dtype==np.dtype(ld)
        and product[0,0]==ld(1) and ld(1)+ld(2)**-60!=ld(1)
        and np.array_equal(values,values.astype(ld).astype(np.float64)))
    return dict(supported=supported,nmant=int(fi.nmant),eps=str(fi.eps),
                dtype=str(np.dtype(ld)),storage_bytes=np.dtype(ld).itemsize)


def evidence_within_budget(evidence,scale,remaining):
    decision=finite_decide(evidence['delta_F'],evidence['error_F'],
                           evidence.get('reliable'),scale)
    return bool(np.isfinite(remaining) and remaining>=0 and decision['accepted']
                and decision['normalized_positive_charge']<=remaining)


def data_refinement_reason(evidence,tau,*,scale=None,remaining=None):
    """Refine only if removing data uncertainty could settle the full verdict."""
    fields=('data_error_F','l1_error_F','arithmetic_error_F','barrier_error_F')
    if not evidence.get('reliable') or not all(k in evidence for k in fields):return 'missing_or_unreliable_evidence'
    if scale is not None:
        if remaining is None or not np.isfinite(remaining) or remaining<0:return 'invalid_remaining_credit'
        if evidence_within_budget(evidence,scale,remaining):return 'already_decided'
        data=evidence['data_error_F']
        other=max(evidence['error_F']-data,math.fsum(evidence[k] for k in fields[1:]))
        if not all(np.isfinite(x) and x>=0 for x in (data,other)) or data==0:return 'nonfinite_or_no_data_error'
        # Eligibility only: a data correction within the EXISTING envelope
        # could help. This optimistic pair is never used to accept or charge.
        possible=dict(evidence,delta_F=evidence['delta_F']-data,error_F=other)
        if not evidence_within_budget(possible,scale,remaining):return 'other_components_or_credit_limit'
        return 'data_blocks_decision'
    # Retain the old standalone diagnostic contract; CRC supplies its policy.
    if classify_evidence(evidence['delta_F'],tau,evidence['error_F'])!='unresolved':return 'already_decided'
    data=evidence['data_error_F'];other=math.fsum(evidence[k] for k in fields[1:])
    if not all(np.isfinite(x) and x>=0 for x in (data,other)):return 'nonfinite_error'
    if classify_evidence(evidence['delta_F'],tau,other)=='unresolved':return 'other_components_limit'
    return 'data_blocks_decision'


def _gamma(k):
    u=np.finfo(float).eps/2
    return k*u/(1-k*u)


def _upper_norm(a):
    """Inflate a nonnegative infinity-norm reduction for binary64 rounding."""
    return float(np.nextafter(np.max(np.sum(abs(a),axis=1))/(1-_gamma(a.shape[1]+2)),np.inf))


def _sum_error(values):
    # Three already computed components; no common objective constant remains.
    return float(np.nextafter(_gamma(8)*math.fsum(abs(float(x)) for x in values),np.inf))


def direct_barrier_delta(anchor,trial,tau,check_deadline=lambda:None):
    """General nonsymmetric M0 E=D, followed by a bounded trace-log series.

    Only checked endpoints may enter. No new inverse or domain recovery. The
    inherited absolute inverse envelope B is enlarged for M's formation error.
    Solve, formation, power arithmetic, trace summation and truncation errors
    are explicit. At most eight terms; a failed bound stays unresolved.
    """
    old,new=anchor.W,trial.W;d=len(old);s=anchor.params[1];u=np.finfo(float).eps/2
    if (anchor.params!=trial.params or not np.isfinite(tau) or tau<=0 or
        any(not np.isfinite(a).all() for a in (old,new,anchor.K,anchor.B)) or
        any(ev.metrics.get('domain_lower',0.)<=0 or ev.metrics.get('inverse_error_bound',np.inf)>1e-8 or
            ev.metrics.get('gradient_error_bound',np.inf)>1e-8 for ev in (anchor,trial))):
        return dict(reliable=False,reason='direct_barrier_state_gate')
    if np.array_equal(old,new):
        return dict(reliable=True,delta_h=0.,error_h=0.,terms=0,dense_products=0,solves=0,
                    norm_E=0.,solve_error=0.,truncation_error=0.,rounding_error=0.,reason='identical_weights')
    check_deadline()
    p=new-old;z=new+old;D=-p*z;A=old*old;M=s*np.eye(d)-A
    if not np.isfinite(D).all() or not np.isfinite(M).all():return dict(reliable=False,reason='direct_nonfinite_formation')
    # Relative roundoff estimates are not used on an underflowed formation.
    tiny=np.finfo(float).tiny
    if np.any((p!=0)&(z!=0)&(abs(D)<tiny)) or np.any((old!=0)&(abs(A)<tiny)):
        return dict(reliable=False,reason='direct_formation_underflow')
    try:E=sla.solve(M,D,assume_a='gen',check_finite=False)
    except (sla.LinAlgError,ValueError,FloatingPointError) as exc:
        return dict(reliable=False,reason='direct_solve_unresolved',exception=str(exc),solves=1)
    check_deadline()
    residual=D-M@E;absolute_product=abs(M)@abs(E)
    gamma=_gamma(d+8);dform=_gamma(3)*abs(D);mform=(u/(1-u))*A
    formation_effect=mform@abs(E)
    # R bounds the exact stored-W equation residual, not just the LAPACK solve.
    R=(abs(residual)+gamma*(abs(D)+absolute_product)+dform+formation_effect)/(1-gamma)
    H=(abs(anchor.K)+anchor.B)/(1-_gamma(2))
    formation_inverse_radius=_upper_norm(H)*_upper_norm(mform)
    if not np.isfinite(formation_inverse_radius) or formation_inverse_radius>=1:
        return dict(reliable=False,reason='direct_inverse_envelope_unresolved',solves=1,dense_products=3)
    eta=_upper_norm(H@R)/(1-formation_inverse_radius)/(1-gamma)
    radius=_upper_norm(E)
    if not all(np.isfinite(x) for x in (eta,radius)) or radius+eta>=1:
        return dict(reliable=False,reason='trace_log_radius_unresolved',solves=1,dense_products=4,norm_E=radius,E_error=eta)
    solve_error=d*eta/(1-radius-eta)
    Y=E.copy();power_error=0.;terms=[];term_errors=[];power_products=0
    for k in range(1,9):
        check_deadline()
        terms.append((-1.)**k*math.fsum(map(float,np.diag(Y)))/k)
        term_errors.append(d*power_error/k+_gamma(d+4)*math.fsum(map(float,abs(np.diag(Y))))/k)
        tail=d*radius**(k+1)/((k+1)*(1-radius))
        if tail<=tau/16 or k==8:break
        ynorm=_upper_norm(Y)
        Y=Y@E;power_products+=1
        power_error=power_error*radius+gamma*ynorm*radius
    delta=math.fsum(terms)
    rounding=math.fsum(term_errors)+_gamma(len(terms)+4)*math.fsum(abs(x) for x in terms)
    error=float(np.nextafter(math.fsum((solve_error,rounding,tail)),np.inf))
    reliable=bool(np.isfinite(delta) and np.isfinite(error))
    return dict(reliable=reliable,reason='direct_trace_log' if reliable else 'direct_nonfinite_series',
        delta_h=delta,error_h=error,norm_E=radius,E_error=eta,formation_inverse_radius=formation_inverse_radius,
        formation_D_error=_upper_norm(dform),formation_M_error=_upper_norm(mform),
        measured_solve_residual=_upper_norm(residual),exact_equation_residual_bound=_upper_norm(R),
        solve_error=solve_error,truncation_error=tail,rounding_error=rounding,
        terms=len(terms),solves=1,dense_products=4+power_products)


def direct_difference_evidence(G,anchor,trial,evidence,tau,check_deadline=lambda:None):
    """One change-scaled review of the complete difference; no extra precision."""
    required=('loss_delta_F','data_error_F','ordinary_consistency_radius_F')
    if not evidence.get('reliable') or not all(k in evidence for k in required):
        return evidence,dict(attempted=False,reason='missing_or_unreliable_evidence')
    if anchor.params!=trial.params or not np.array_equal(G,G.T):
        return dict(evidence,reliable=False),dict(attempted=False,reason='endpoint_identity')
    barrier=direct_barrier_delta(anchor,trial,tau,check_deadline)
    info=dict(attempted=True,before=evidence.copy(),barrier=barrier)
    if not barrier['reliable']:
        info['reason']=barrier['reason'];return dict(evidence,reliable=False),info
    mu,s,lam=anchor.params;diff=abs(trial.W)-abs(anchor.W)
    l1=mu*lam*math.fsum(map(float,diff.ravel()))
    l1_scale=abs(mu*lam)*math.fsum(map(float,abs(diff).ravel()))
    l1_error=_gamma(diff.size+6)*l1_scale
    pieces=(evidence['loss_delta_F'],l1,barrier['delta_h'])
    delta=math.fsum(pieces);arithmetic=_sum_error(pieces)
    error=float(np.nextafter(math.fsum((evidence['data_error_F'],l1_error,barrier['error_h'],arithmetic)),np.inf))
    ordinary=trial.metrics['F']-anchor.metrics['F']
    reliable=bool(np.isfinite(error) and abs(delta-ordinary)<=evidence['ordinary_consistency_radius_F'])
    revised=dict(evidence,delta_F=delta,error_F=error,l1_delta_F=l1,l1_error_F=l1_error,
        l1_difference_scale_F=l1_scale,barrier_delta_F=barrier['delta_h'],barrier_error_F=barrier['error_h'],
        arithmetic_error_F=arithmetic,difference_scaled_error=True,reliable=reliable,
        ordinary_difference_error_F=abs(delta-ordinary),reason='direct_difference_components')
    info.update(reason='direct_difference_reviewed',after=revised.copy());return revised,info


def longdouble_loss_delta(G,old,new,mu):
    """Ordinary entry; use the same computation and envelope as terminal review."""
    return data_precision.pairwise_data_delta(G,old,new,mu,
        precision_available=longdouble_capability()['supported'])


def refine_data_evidence(G,anchor,trial,evidence):
    """Exactly one data-only pass; retain every non-data error allocation."""
    # The caller has already checked path identity and evaluated both states.
    if anchor.params!=trial.params or not np.array_equal(G,G.T):
        return evidence,dict(attempted=False,reason='endpoint_identity')
    computed=longdouble_loss_delta(G,anchor.W,trial.W,anchor.params[0])
    info=dict(attempted=True,backend=longdouble_capability(),before=evidence.copy(),data=computed)
    if not computed['reliable'] or computed['data_error_F']>=evidence['data_error_F']:
        info['reason']='data_estimate_not_improved';return evidence,info
    delta=math.fsum((computed['loss_delta_F'],evidence['l1_delta_F'],evidence['barrier_delta_F']))
    factor=64*np.finfo(float).eps*(len(G)+8)
    sum_error=(_sum_error((computed['loss_delta_F'],evidence['l1_delta_F'],evidence['barrier_delta_F']))
               if evidence.get('difference_scaled_error') else factor*(1.+abs(delta)))
    arithmetic=max(evidence['arithmetic_error_F'],sum_error)
    nondata=max(evidence['error_F']-evidence['data_error_F'],
                math.fsum((evidence['l1_error_F'],evidence['barrier_error_F'],arithmetic)))
    error=float(np.nextafter(math.fsum((computed['data_error_F'],nondata)),np.inf))
    ordinary=trial.metrics['F']-anchor.metrics['F']
    # Retain the existing ordinary-endpoint consistency check and radius.
    reliable=bool(evidence['reliable'] and abs(delta-ordinary)<=evidence['ordinary_consistency_radius_F'])
    revised=dict(evidence,delta_F=delta,error_F=error,loss_delta_F=computed['loss_delta_F'],
                 data_error_F=computed['data_error_F'],arithmetic_error_F=arithmetic,
                 reliable=reliable,ordinary_difference_error_F=abs(delta-ordinary),
                 reason='same_endpoints_longdouble_data_only')
    info.update(reason='data_refined',after=revised.copy());return revised,info


def column_delta(G,b,c,old,new,mu,lam):
    """Compute delta F=mu*delta Q using only nonzero difference coordinates.

    Exact zeros skip arithmetic, never prune a coefficient/candidate. G is the
    symmetric centered Gram used by the unchanged solver. Two products were not
    exposed by that solver, so their additional cost is measured explicitly.
    """
    if not np.isfinite(new).all() or not np.isfinite(old).all():
        raise NumericalUnresolved('nonfinite_conditional_weights')
    p=new-old; ids=np.flatnonzero(p); active=np.flatnonzero(old)
    if not len(ids):return 0.,0.
    v=p[ids]
    grad=G[np.ix_(ids,active)]@old[active]-b[ids]
    gp=G[np.ix_(ids,ids)]@v
    quadratic=v*(grad+.5*gp)
    dl=math.fsum(map(float,quadratic))
    l1terms=np.abs(new[ids])-np.abs(old[ids])
    d1=math.fsum(map(float,l1terms))
    q0=math.fsum(map(float,c*old*old))
    dq=math.fsum(map(float,c[ids]*v*(new[ids]+old[ids])))
    q1=math.fsum(map(float,c*new*new))
    if not all(np.isfinite(x) for x in (q0,q1,dq,dl,d1)) or min(1-q0,1-q1)<=0:
        raise ComparisonUnresolved('conditional_difference_nonfinite_or_domain_unresolved')
    ratio=-dq/(1-q0)
    if ratio<=-1:return float('nan'),float('inf')
    dh=-math.log1p(ratio)
    delta=math.fsum((mu*dl,mu*lam*d1,dh))
    magnitude=mu*(math.fsum(map(float,np.abs(quadratic)))+lam*math.fsum(map(float,np.abs(l1terms))))+abs(dh)
    return delta,magnitude


from .objective_stopping import within_bound, relative_change, POLICY_ID as STOPPING_POLICY_ID


class FloatComparator(OriginalComparator):
    def full(self,G,old,new,mu,s,lam,delta,bound,kind,ordinary,site,scale=1.,K_old=None):
        if site in ('base_sweep_monotonicity','polish_column_increase'):
            raise ComparisonUnresolved('sweep_comparison_requires_recorded_path')
        if site in ('base_relative_stopping','polish_relative_stopping'):
            gate=getattr(self,'float_gate',None)
            if gate is None or not gate.cache_matches(G,new,(mu,s,lam),delta):
                raise ComparisonUnresolved('stale_stopping_delta')
            self.counts['float64_stopping_cache_reuses']+=1
            c=gate.cached_path
            if not c.get('accepted_full_sweep'):
                raise ComparisonUnresolved('stopping_requires_accepted_full_sweep')
            c['stopping_bound_F']=float(bound)
            if not c.get('stop_eligible',True):
                return False,delta
            accepted=within_bound(delta,bound)
            return accepted,delta
        if site=='terminal_polish_monotonicity':
            return self.float_gate.tail_decision(G,old,new,(mu,s,lam))
        return super().full(G,old,new,mu,s,lam,delta,bound,kind,ordinary,site,scale,K_old)

    def summary(self):
        out=super().summary()
        out['float64']=self.float_gate.summary() if hasattr(self,'float_gate') else None
        return out


class FloatRecovery(Recovery):
    def __init__(self, context, comparator):
        super().__init__(context,comparator)
        comparator.float_gate=self
        self.path=None;self.cached_path=None;self.phase=None;self.phase_terms=[]
        self.phase_positive=0.;self.phase_min=0.;self.phase_initial=None
        self.fcounts=Counter();self.records=[];self.tie_residuals=[]
        self.total_positive=0.;self.max_drift=0.
        self.whole_phase_path_valid=True;self.was_stagnant=False
        self.phase_prefix_terms=[];self.phase_endpoint=None
        self.recovery_evidence=None;self.recovery_connected=False
        self.phase_bias_used=0.;self.bias_history=[];self.recovery_charge=0.
        self.recovery_comparison_records=[]
        self.soft_events=[]

    def start_phase(self,G,W,params,name):
        if self.phase is not None:
            self.bias_history.append(dict(phase=self.phase[2],params=list(self.phase[1]),normalized_positive=self.phase_bias_used))
        self.phase_bias_used=0.
        self.phase=(id(G),tuple(params),name);self.phase_initial=W.copy();self.G_basis=G.copy();self.G_reference=G
        self.phase_terms=[];self.phase_positive=0.;self.phase_min=0.;self.tie_residuals=[]
        self.phase_tau=tolerance(len(W),self.context.cached[1]['F'])
        self.path=None;self.cached_path=None
        self.whole_phase_path_valid=True;self.was_stagnant=False
        self.phase_prefix_terms=[];self.phase_endpoint=self.phase_initial

    def begin(self,G,anchor,location):
        owned=self.context.checkpoint()
        if (owned.params!=anchor.params or not np.array_equal(owned.W,anchor.W)
            or owned.metrics!=anchor.metrics):
            raise ComparisonUnresolved('sweep_anchor_not_current_certified_state')
        key=(id(G),anchor.params,location['phase'])
        if self.phase!=key:self.start_phase(G,anchor.W,anchor.params,location['phase'])
        elif self.phase_initial is None:
            valid=self.whole_phase_path_valid
            debit=self.phase_bias_used;terms=list(self.phase_terms);positive=self.phase_positive;minimum=self.phase_min
            self.start_phase(G,anchor.W,anchor.params,location['phase'])
            self.phase_bias_used=debit;self.phase_terms=terms;self.phase_positive=positive;self.phase_min=minimum
            self.whole_phase_path_valid=valid
            self.fcounts['recovery_baselines_rebuilt']+=1
            self.emit(dict(event='recovery_baseline_rebuilt',**location,state_version=self.context.state_version))
        elif not np.array_equal(G,self.G_basis):raise ComparisonUnresolved('unrecorded_Gram_mutation')
        elif self.phase_endpoint is not None and not np.array_equal(anchor.W,self.phase_endpoint):
            raise ComparisonUnresolved('unrecorded_phase_weight_change')
        self.anchor_stopping_cache=self.cached_path
        self.path=dict(G_id=id(G),params=anchor.params,terms=[],magnitudes=[],
                       expected=anchor.W.copy(),version=self.context.state_version,
                       anchor_owned=owned,anchor_W=owned.W,anchor_metrics=owned.metrics,
                       location=dict(location),columns=0,soft_rejections=0,accepted_changes=0)
        self.cached_path=None
        self.emit(dict(event='sweep_start',**location,residual=anchor.metrics['residual_normalized']))

    def add_column(self,G,b,c,old,new,j,mu,lam):
        p=self.path
        if p is None or p['G_id']!=id(G) or p['params'][0]!=mu or p['params'][2]!=lam:
            raise HardNumericalFailure('column_path_identity_changed')
        if not np.array_equal(p['expected'][:,j],old):
            raise HardNumericalFailure('unrecorded_column_mutation')
        with self.ledger.measure('float64_comparison'):
            value,mag=column_delta(G,b,c,old,new,mu,lam)
        if not np.isfinite(value) or not np.isfinite(mag):
            raise ComparisonUnresolved('unreliable_column_path_difference')
        p['terms'].append(value);p['magnitudes'].append(mag)
        p['accepted_changes']+=int(not np.array_equal(old,new))
        p['expected'][:,j]=new;p['columns']+=1
        # rank_one subsequently increments the context version exactly once.
        p['version']=self.context.state_version+1

    def record_soft(self,event,location,**details):
        row=dict(event=event,**location,**details,accepted=False,
                 action='reject_retain_certified_state_continue')
        self.soft_events.append(_finite(row));self.fcounts[event]+=1
        self.emit(row)

    def checked_column_incumbent(self,G,j,old):
        """Exceptional, detached check; normal columns incur no full refresh."""
        ctx=self.context;p=self.path
        require_finite(G,ctx.W,ctx.K,old)
        if (p is None or not np.array_equal(G,self.G_basis)
                or not np.array_equal(ctx.W,p['expected'])
                or not np.array_equal(ctx.W[:,j],old)):
            raise HardNumericalFailure('column_incumbent_not_recorded_accepted_state')
        try:
            state=require_certified(ctx.evaluate(G,ctx.W.copy(),*p['params'],commit=False))
            from .guards import check_existing
            diagnostic,_=check_existing(ctx.W,ctx.K,ctx.s,ctx.mu)
            if not diagnostic['domain']['legal'] or not diagnostic['accuracy']['reliable']:
                # Existing bounded refresh, only if the incumbent inverse needs it.
                ctx.refresh(ctx.K);p['version']=ctx.state_version
                diagnostic,_=check_existing(ctx.W,ctx.K,ctx.s,ctx.mu)
                if not diagnostic['domain']['legal'] or not diagnostic['accuracy']['reliable']:
                    raise HardNumericalFailure('incumbent_inverse_cannot_be_rebuilt')
        except RECOVERABLE as exc:
            raise HardNumericalFailure('incumbent_cannot_be_certified: '+str(exc)) from exc
        return state

    def reject_column_exception(self,G,old,j,exc):
        """A solver produced no certifiable proposal; retain the same column."""
        if isinstance(exc,HardNumericalFailure):raise exc
        incumbent=self.checked_column_incumbent(G,j,old);p=self.path
        p['terms'].append(0.);p['magnitudes'].append(0.);p['columns']+=1
        p['soft_rejections']+=1;p['version']=self.context.state_version
        self.record_soft('column_update_numerically_uncertain',dict(p['location'],column=int(j)),
            mu=incumbent.params[0],s=incumbent.params[1],reason=str(exc),
            objective_before_F=incumbent.metrics['F'],proposal_objective_F=None,
            stable_difference_F=None,error_budget_F=None,rollback_W_exact=True,
            rollback_domain_valid=True,rollback_inverse_reliable=True)

    def certify_or_reject_column(self,G,b,c,old,new,j,mu,lam,old_block,inc,bound,site):
        """One existing column predicate for C1/C2; no commit on rejection.

        Q/F units, native recheck, and the inherited conditional allowance are
        unchanged. A whole-endpoint review is only a fallback for an undecided
        conditional comparison, and must also satisfy the original bound.
        """
        require_finite(old,b,c)
        finite_proposal=bool(np.isfinite(new).all())
        uncertain=not finite_proposal;reason='certified_column_increase' if finite_proposal else 'nonfinite_uncommitted_proposal'
        evidence=None;review=None;accepted=False
        try:
            if finite_proposal:
                accepted,delta=self.comparator.block(G,b,c,old,new,mu,lam,inc,bound,
                    not(inc>bound),site,abs(old_block))
                if accepted:
                    self.add_column(G,b,c,old,new,j,mu,lam)
                    return True
        except HardNumericalFailure:raise
        except NumericalUnresolved as exc:
            uncertain=True;accepted=False;reason=str(exc)
        incumbent=self.checked_column_incumbent(G,j,old)
        ctx=self.context;proposal=incumbent.W.copy();proposal[:,j]=new
        before_W=ctx.W.copy();before_K=ctx.K.copy();proposal_state=None
        try:
            if finite_proposal:proposal_state=ctx.evaluate(G,proposal,*incumbent.params,commit=False)
            if uncertain and proposal_state is not None and state_reliable(proposal_state) and np.isfinite(bound):
                scale=objective_scale(incumbent,proposal_state)
                verdict,delta,source,evidence,refinement,direct,terminal=self.review_difference(
                    G,incumbent,proposal_state,mu*bound,scale,'unresolved',mu*inc,'column_endpoint_review')
                decision=finite_decide(evidence['delta_F'],evidence['error_F'],evidence.get('reliable'),scale)
                review=dict(source=source,data_refinement=refinement,difference_review=direct,terminal_review=terminal)
                if decision['accepted'] and decision['upper_F']<=mu*bound:
                    try:self.add_column(G,b,c,old,new,j,mu,lam)
                    except HardNumericalFailure:raise
                    except NumericalUnresolved as exc:reason+='; '+str(exc)
                    else:
                        self.fcounts['column_endpoint_review_accepted']+=1
                        return True
        except HardNumericalFailure:raise
        except RECOVERABLE as exc:
            reason+='; proposal_state_unavailable: '+str(exc)
        # The proposal has never entered W or K. Count a visited column with
        # exactly zero accepted change, not its uncertain estimate.
        assert np.array_equal(ctx.W,before_W) and np.array_equal(ctx.K,before_K)
        p=self.path;p['terms'].append(0.);p['magnitudes'].append(0.)
        p['columns']+=1;p['soft_rejections']+=1;p['version']=ctx.state_version
        native=self.comparator.last_column_terminal or self.comparator.last_unresolved
        self.record_soft('column_update_numerically_uncertain' if uncertain else 'column_update_rejected',
            dict(p['location'],column=int(j)),mu=mu,s=ctx.s,reason=reason,
            objective_before_F=incumbent.metrics['F'],
            proposal_objective_F=proposal_state.metrics['F'] if proposal_state is not None else None,
            stable_difference_F=evidence['delta_F'] if evidence else (native or {}).get('delta_F'),
            error_budget_F=evidence['error_F'] if evidence else (native or {}).get('empirical_uncertainty'),
            acceptance_bound_F=mu*bound,endpoint_evidence=evidence,review=review,native_column_review=native,
            rollback_W_exact=True,rollback_K_exact=True,rollback_domain_valid=True,rollback_inverse_reliable=True)
        return False

    def cache_matches(self,G,W,params,delta):
        c=self.cached_path
        good=(c is not None and c['G_id']==id(G) and c['params']==tuple(params)
                and c['version']==self.context.state_version and W is self.context.W
                and isinstance(delta,StableDelta) and delta is c['delta']
                and np.array_equal(G,self.G_basis) and np.array_equal(W,c['W_owned']))
        if not good and c is not None:
            self.cached_path=None;self.fcounts['stopping_cache_invalidations']+=1
        return good

    def emit(self,row):
        row=dict(row,monotonic=time.monotonic(),counts=dict(self.fcounts),
                 exclusive_seconds=dict(self.ledger.seconds),
                 exclusive_calls=dict(self.ledger.calls))
        path=getattr(self.context,'float64_progress_path',None)
        if path:
            with open(path,'a') as handle:handle.write(json.dumps(_finite(row),allow_nan=False)+'\n')

    def invalidate(self,reason):
        self.path=None;self.cached_path=None;self.fcounts['path_invalidations']+=1
        self.fcounts['invalidate_'+reason]+=1

    def rollback(self,anchor,reason):
        self.context.adopt(anchor);self.invalidate(reason);self.fcounts['rollbacks']+=1
        self.emit(dict(event='rollback',reason=reason,residual=anchor.metrics['residual_normalized']))

    def rollback_comparison(self,W,anchor,location,exc,start):
        # Preserve the candidate before adoption, using the existing evidence API.
        if self.path is not None and 'anchor_owned' in self.path:
            anchor=self.path['anchor_owned']
        super().rollback_comparison(W,anchor,location,exc,start)
        self.invalidate('unresolved_or_timeout');self.fcounts['rollbacks']+=1
        self.emit(dict(event='rollback',**location,reason=str(exc),
                       residual=anchor.metrics['residual_normalized']))

    def evidence_label(self,evidence,tau,scale):
        label=(classify_evidence(evidence['delta_F'],tau,evidence['error_F'])
               if evidence.get('reliable') else 'unresolved')
        decision=finite_decide(evidence['delta_F'],evidence['error_F'],evidence.get('reliable'),scale)
        if decision['accepted'] and label not in ('descent','tie'):
            return 'finite_precision'
        return label

    def stopping_ok(self,delta,bound):
        c=self.cached_path or {}
        return bool(c.get('accepted_full_sweep') and c.get('stop_eligible',True) and within_bound(delta,bound))

    def rejected_sweep(self,G,anchor,location,exc,pending,tol,bound,can_stop=True):
        """Reject an undecided sweep; subsequent scans consume the native cap."""
        if isinstance(exc,HardNumericalFailure):raise exc
        if not isinstance(exc,(ComparisonUnresolved,NumericalUnresolved)):raise exc
        require_finite(G,anchor.W,anchor.K,anchor.B)
        if not np.array_equal(G,self.G_basis):raise HardNumericalFailure('unrecorded_Gram_mutation')
        require_certified(anchor)
        # Transactional restoration cannot erase elapsed time, events or credit.
        self.context.adopt(anchor);self.invalidate('rejected_sweep')
        previous=getattr(self,'anchor_stopping_cache',None)
        same=(previous is not None and previous['G_id']==id(G)
              and previous['params']==anchor.params
              and np.array_equal(previous['W_owned'],anchor.W))
        if same:
            self.cached_path=dict(previous,version=self.context.state_version)
        if (can_stop and same and 'stopping_bound_F' in previous
                and self.stopping_ok(previous['delta'],previous['stopping_bound_F'])):
            self.fcounts['rejected_anchor_completed']+=1
            self.emit(dict(event='rejected_anchor_completed',**location,reason=str(exc)))
            return 'stop',anchor,previous['delta']
        self.record_soft('sweep_update_numerically_uncertain',location,reason=str(exc),
            objective_before_F=anchor.metrics['F'],rollback_domain_valid=True,rollback_inverse_reliable=True,
            rollback_W_exact=True,stop_inferred_from_rejection=False)
        self.context.check_deadline()
        if pending:
            # No new refresh budget. Retain the certified anchor and consume
            # the next existing sweep; do not manufacture a stopping delta.
            self.phase_endpoint=anchor.W.copy()
            if self.cached_path is None:
                self.cached_path=dict(G_id=id(G),params=anchor.params,version=self.context.state_version,
                    W_owned=anchor.W.copy(),delta=StableDelta(0.,'rejected_sweep_no_change'),
                    accepted_full_sweep=False,stop_eligible=False)
            return 'retry',anchor,StableDelta(0.,'rejected_sweep_no_change')
        # The only additional work is the existing float64 state refresh.
        try:state=require_certified(self.context.evaluate(G,anchor.W.copy(),*anchor.params,commit=False))
        except RECOVERABLE as failure:
            raise HardNumericalFailure('certified_anchor_refresh_failed: '+str(failure)) from failure
        self.context.adopt(state);self.invalidate('bounded_retry_refresh')
        self.phase_endpoint=state.W.copy()
        self.fcounts['rejection_extra_refreshes']+=1
        self.emit(dict(event='rejection_retry',**location,reason=str(exc),
                       residual=state.metrics['residual_normalized'],normalized_bias=self.phase_bias_used))
        self.cached_path=dict(G_id=id(G),params=anchor.params,version=self.context.state_version,
            W_owned=state.W.copy(),delta=StableDelta(0.,'rejected_sweep_no_change'),
            accepted_full_sweep=False,stop_eligible=False)
        return 'retry',state,StableDelta(0.,'rejected_sweep_no_change')

    def review_difference(self,G,anchor,trial,tau,scale,verdict,delta,source):
        """Shared exceptional endpoint chain; formulas and budgets are unchanged."""
        ctx=self.context;evidence=None;refinement=None;review=None
        if verdict in ('rise','unresolved'):
            # The state was already refreshed above. Repeating the same
            # refresh or an unguarded determinant ratio adds no evidence.
            # A credible endpoint can replace an inaccurate path estimate;
            # signs alone and a large consistency radius cannot authorize it.
            self.fcounts['endpoint_evidence_checks']+=1
            ctx.check_deadline()
            with self.ledger.measure('float64_comparison'):
                evidence=endpoint_evidence(G,anchor,trial)
                verdict=self.evidence_label(evidence,tau,scale)
            ctx.check_deadline()
            if verdict=='unresolved' and evidence['reliable']:
                self.fcounts['difference_review_calls']+=1
                with self.ledger.measure('direct_difference_review'):
                    evidence,review=direct_difference_evidence(G,anchor,trial,evidence,tau,ctx.check_deadline)
                self.fcounts['direct_barrier_solves']+=review.get('barrier',{}).get('solves',0)
                self.fcounts['direct_barrier_dense_products']+=review.get('barrier',{}).get('dense_products',0)
                verdict=self.evidence_label(evidence,tau,scale)
                review.update(verdict_before='unresolved',verdict_after=verdict)
                self.fcounts['difference_review_'+verdict]+=1
            if evidence['reliable'] and not evidence_within_budget(evidence,scale,STAGE_POSITIVE_CAP-self.phase_bias_used):
                reason=data_refinement_reason(evidence,tau,scale=scale,remaining=STAGE_POSITIVE_CAP-self.phase_bias_used)
                refinement=dict(attempted=False,reason=reason)
                if reason=='data_blocks_decision':
                    capability=longdouble_capability()
                    if capability['supported']:
                        self.fcounts['data_precision_calls']+=1
                        with self.ledger.measure('longdouble_data_refinement'):
                            evidence,refinement=refine_data_evidence(G,anchor,trial,evidence)
                        ctx.check_deadline()
                        verdict=self.evidence_label(evidence,tau,scale)
                        refinement.update(verdict_before='unresolved',verdict_after=verdict)
                        self.fcounts['data_precision_'+verdict]+=1
                    else:
                        refinement.update(reason='no_extra_precision',backend=capability)
                        self.fcounts['data_precision_unavailable']+=1
            if verdict!='unresolved':
                delta=evidence['delta_F'];source=('endpoint_longdouble_data' if refinement and refinement.get('reason')=='data_refined'
                        else 'direct_difference_endpoint' if review and review.get('reason')=='direct_difference_reviewed'
                        else 'float64_certified_endpoint')
        terminal=None
        if verdict in ('rise','unresolved') and evidence is not None:
            # This gate is AFTER the complete a085 chain, including its
            # successful late data refinement. Accepted old paths do not
            # call or adopt any result from the new handler.
            from .terminal_review import terminal_evidence
            old_verdict=verdict
            base=review['before'] if review and 'before' in review else evidence
            self.fcounts['terminal_review_calls']+=1
            with self.ledger.measure('terminal_failed_chain_review'):
                evidence,terminal=terminal_evidence(G,anchor,trial,base,evidence,tau,ctx.check_deadline,
                    scale=scale,remaining=STAGE_POSITIVE_CAP-self.phase_bias_used,
                    data_attempted=bool(refinement and refinement.get('attempted')),
                    data_result=None if not refinement else refinement.get('data'))
            verdict=self.evidence_label(evidence,tau,scale)
            terminal.update(verdict_before=old_verdict,verdict_after=verdict)
            self.fcounts['terminal_review_'+verdict]+=1
            self.fcounts['terminal_data_calls']+=terminal.get('data_calls',0)
            self.fcounts['terminal_barrier_calls']+=terminal.get('barrier_calls',0)
            if verdict!='unresolved':
                delta=evidence['delta_F'];source='terminal_failed_chain_endpoint'
        return verdict,delta,source,evidence,refinement,review,terminal

    def finish_sweep(self,G,W,K,anchor,limit,site,location):
        ctx=self.context;p=self.path;start=time.perf_counter();restoring=False
        try:
            require_finite(G,W,K)
            if (p is None or p['G_id']!=id(G) or p['params']!=anchor.params
                or p['version']!=ctx.state_version or p['columns']!=len(W)
                or not np.array_equal(W,ctx.W) or (ctx.mu,ctx.s)!=anchor.params[:2]
                or not np.array_equal(p['anchor_W'],anchor.W) or p['anchor_metrics']!=anchor.metrics
                or not np.array_equal(p['expected'],W) or not np.array_equal(G,self.G_basis)):
                raise HardNumericalFailure('incomplete_or_stale_sweep_path')
            if not all(np.isfinite(x) for x in p['terms']+p['magnitudes']):
                raise HardNumericalFailure('nonfinite_recorded_sweep_path')
            try:
                trial=ctx.evaluate(G,W,*anchor.params,commit=False)
                if not state_reliable(anchor) or not state_reliable(trial):
                    raise NumericalUnresolved('unreliable_finite_precision_endpoint')
            except RECOVERABLE as exc:
                # Genuine state gate failure retains the inherited bounded recovery.
                restoring=True
                return self.restore(G,W,anchor,limit,site,location,reason=str(exc),already_tried=True)
            with self.ledger.measure('float64_comparison'):
                delta=math.fsum(p['terms']);mag=math.fsum(p['magnitudes'])
                tau=tolerance(len(W),anchor.metrics['F'],trial.metrics['F'],mag)
                endpoint=trial.metrics['F']-anchor.metrics['F']
                endpoint_tau=tolerance(len(W),anchor.metrics['objective_abs_scale'],trial.metrics['objective_abs_scale'])
                reliable=abs(delta-endpoint)<=endpoint_tau
                verdict=classify(delta,tau,reliable)
                path_delta=delta;evidence=None;refinement=None;review=None;source='float64_recorded_column_path'
                scale=objective_scale(anchor,trial)
            terminal=None
            if verdict in ('rise','unresolved'):
                verdict,delta,source,evidence,refinement,review,terminal=self.review_difference(
                    G,anchor,trial,tau,scale,verdict,delta,source)
            decision=(finite_decide(evidence['delta_F'],evidence['error_F'],evidence.get('reliable'),scale)
                      if evidence is not None else None)
            if decision is not None and not decision['accepted']:
                verdict='rise' if decision['action']=='reject' else 'unresolved'
            error=evidence['error_F'] if evidence is not None and evidence.get('reliable') else None
            charge=(decision['normalized_positive_charge'] if decision and decision['accepted']
                    else max(delta,0.)/scale)
            drift=float(np.linalg.norm(K-trial.K,np.inf)/max(1.,np.linalg.norm(trial.K,np.inf)))
            rec=dict(**location,classification=verdict,delta_F=delta,tau_F=tau,
                     path_delta_F=path_delta,path_endpoint_consistent=bool(reliable),
                     decision_source=source,endpoint_evidence=evidence,data_refinement=refinement,difference_review=review,
                     endpoint_delta_F=endpoint,endpoint_consistency_tolerance=endpoint_tau,
                     sum_abs_column_deltas=mag,inverse_drift=drift,
                     residual=trial.metrics['residual_normalized'],params=list(anchor.params))
            rec.update(finite_policy_decision=decision,objective_scale_F=scale,normalized_positive_charge=charge)
            if terminal is not None:rec['terminal_review']=terminal
            self.records.append(rec);self.fcounts[verdict]+=1
            if verdict not in ('tie','descent','finite_precision'):
                if verdict=='unresolved':raise ComparisonUnresolved('float64_sweep_unresolved')
                self.rollback(anchor,verdict)
                raise NumericalUnresolved('confirmed_float64_sweep_increase_rejected')
            proposed_positive=self.phase_positive+max(delta,0.)
            net=math.fsum(self.phase_terms+[delta])
            cap=STAGE_POSITIVE_CAP
            proposed_bias=self.phase_bias_used+charge
            if proposed_bias>cap:
                self.fcounts['cumulative_drift_rejections']+=1
                raise ComparisonUnresolved('cumulative_numerical_tie_drift_exceeded')
            ctx.adopt(trial)
            value=StableDelta(delta,source)
            self.cached_path=dict(G_id=id(G),params=anchor.params,version=ctx.state_version,delta=value,W_owned=trial.W.copy(),scale_F=scale,error_F=error,
                                  accepted_full_sweep=True,objective_before_F=float(anchor.metrics['F']),
                                  stop_eligible=not(p['soft_rejections'] and not p['accepted_changes']))
            self.phase_endpoint=self.cached_path['W_owned']
            self.path=None;self.phase_terms.append(delta);self.phase_positive=proposed_positive
            self.phase_min=min(self.phase_min,net);self.total_positive+=max(delta,0.)
            self.phase_bias_used=proposed_bias
            self.max_drift=max(self.max_drift,net-self.phase_min)
            if source in ('float64_certified_endpoint','endpoint_longdouble_data','direct_difference_endpoint','terminal_failed_chain_endpoint'):
                # Replace this scan's increment exactly once. The adopted W/K
                # and stopping cache are the new baseline; prior positive drift
                # and the phase-entry link are retained, not granted a new cap.
                self.fcounts['endpoint_rebases']+=1
            rec.update(accepted=True,cumulative_delta_F=net,cumulative_positive_F=proposed_positive,
                       cumulative_positive_cap_F=None,normalized_positive_used=proposed_bias,normalized_positive_cap=cap)
            rec.update(objective_relative_change=relative_change(delta,anchor.metrics['F']),
                       objective_before_F=float(anchor.metrics['F']),stopping_policy=STOPPING_POLICY_ID)
            self.emit(dict(event='sweep_accepted',**rec))
            ctx.last_accepted_location=dict(location)
            hook=getattr(ctx,'float64_checkpoint_hook',None)
            if hook is not None:hook(ctx,location,rec)
            if verdict=='tie':self.tie_residuals.append(rec['residual'])
            else:self.tie_residuals=[]
            window=POLICY['stagnant_tie_window']
            stagnant=False
            if len(self.tie_residuals)>=window:
                recent=self.tie_residuals[-window:]
                tol=getattr(ctx,'float64_c2_tolerance',1e-4) if location['phase']=='polish' else 1e-4
                if min(recent)>tol and min(recent)>=recent[0]*(1-POLICY['stagnant_required_relative_progress']):
                    stagnant=True
                    if not self.was_stagnant:
                        self.fcounts['stagnation_diagnostics']+=1
                        self.emit(dict(event='stagnation_diagnostic',**location,residual=rec['residual'],
                                       action='continue_native_stopping',window=window))
            self.was_stagnant=stagnant
            return trial,value,drift,False
        except (HardNumericalFailure,ComparisonUnresolved,TimeoutError) as exc:
            # A nested restore already records/rolls back its own failure.
            if not restoring:
                self.rollback_comparison(W,anchor,location,exc,start)
            raise

    def endpoint_delta(self,G,anchor,trial):
        """Float64 counterpart of inherited update/ratio screen, fallback only."""
        old,new=anchor.W,trial.W;mu,s,lam=anchor.params
        p=new-old;cols=np.flatnonzero(np.any(p!=0,axis=0))
        if not len(cols):return 0.
        v=p[:,cols];r=old[:,cols]-np.eye(len(old))[:,cols]
        terms=v*(G@r+.5*(G@v))
        dj=mu*math.fsum((math.fsum(map(float,terms.ravel())),lam*math.fsum(map(float,(abs(new[:,cols])-abs(old[:,cols])).ravel()))))
        H=anchor.K[cols,:]@((new[:,cols]-old[:,cols])*(new[:,cols]+old[:,cols]))
        sign,ld=np.linalg.slogdet(np.eye(len(cols))-H)
        return dj-ld if sign>0 and np.isfinite(ld) else float('nan')

    def compare(self,G,anchor,trial,limit,site):
        # Used only after genuine invalid-state/column recovery; the path was reset.
        with self.ledger.measure('float64_comparison'):
            delta=self.endpoint_delta(G,anchor,trial)
            tau=tolerance(len(G),anchor.metrics['F'],trial.metrics['F'])
            endpoint=trial.metrics['F']-anchor.metrics['F']
            endpoint_tau=tolerance(len(G),anchor.metrics['objective_abs_scale'],trial.metrics['objective_abs_scale'])
            reliable=(anchor.params==trial.params and np.isfinite(endpoint)
                      and abs(delta-endpoint)<=endpoint_tau)
            verdict=classify(delta,tau,reliable)
        source='float64_recovery_endpoint';decision=None
        if verdict=='unresolved':
            # Keep every resolved fast-path decision and its arithmetic intact.
            # Recovery compares the same fixed endpoints as a complete sweep;
            # it must use that existing evidence chain before declaring failure.
            if not state_reliable(anchor) or not state_reliable(trial):
                raise ComparisonUnresolved('unreliable_recovery_endpoint')
            self.fcounts['recovery_stable_reviews']+=1
            original_delta=delta
            scale=objective_scale(anchor,trial)
            verdict,delta,source,evidence,refinement,review,terminal=self.review_difference(
                G,anchor,trial,tau,scale,verdict,delta,source)
            decision=finite_decide(evidence['delta_F'],evidence['error_F'],
                                   evidence.get('reliable'),scale)
            # Reuse the evidence and accounting, not a larger acceptance region:
            # the certified upper difference must satisfy recovery's original tau.
            if not decision['accepted'] or decision.get('upper_F',float('inf'))>tau:
                verdict=('rise' if evidence.get('reliable') and
                         decision.get('lower_F',-float('inf'))>tau else 'unresolved')
            else:
                verdict='descent' if decision['upper_F'] < -tau else 'tie'
                delta=evidence['delta_F']
            self.recovery_comparison_records.append(dict(site=site,params=list(anchor.params),
                original_delta_F=original_delta,endpoint_delta_F=endpoint,
                endpoint_consistency_tolerance=endpoint_tau,tau_F=tau,
                classification=verdict,decision_source=source,endpoint_evidence=evidence,
                data_refinement=refinement,difference_review=review,terminal_review=terminal,
                finite_policy_decision=decision))
        self.fcounts['recovery_'+verdict]+=1
        if verdict=='unresolved':raise ComparisonUnresolved('float64_recovery_comparison_unresolved')
        value=StableDelta(delta,source)
        if verdict in ('descent','tie') and self.recovery_connected:
            positive=self.phase_positive+max(delta,0.)
            net=math.fsum(self.phase_terms+[delta])
            self.recovery_charge=(decision['normalized_positive_charge'] if decision is not None
                                  else max(delta,0.)/objective_scale(anchor,trial))
            if self.phase_bias_used+self.recovery_charge>STAGE_POSITIVE_CAP:
                self.fcounts['cumulative_drift_rejections']+=1
                raise ComparisonUnresolved('cumulative_numerical_tie_drift_exceeded_at_recovery')
        self.recovery_evidence=(G,anchor,trial,value)
        return verdict in ('descent','tie'),value

    def restore(self,G,W,anchor,limit,site,location,**kwargs):
        connected=(self.whole_phase_path_valid and self.phase is not None
            and self.phase[:2]==(id(G),anchor.params) and self.phase_initial is not None
            and self.phase_endpoint is not None and np.array_equal(anchor.W,self.phase_endpoint)
            and np.array_equal(G,self.G_basis))
        self.recovery_connected=connected;self.recovery_evidence=None
        self.invalidate('recovery')
        try:
            state,delta,drift,scaled=super().restore(G,W,anchor,limit,site,location,**kwargs)
        finally:self.recovery_connected=False
        e=self.recovery_evidence
        linked=(connected and e is not None and e[0] is G and e[1] is anchor
                and e[2] is state and e[3] is delta and np.isfinite(float(delta)))
        changed=not np.array_equal(state.W,anchor.W)
        if linked:
            if changed:
                # The accepted delta is anchor -> restored state, so partial
                # uncommitted column deltas are not added a second time.
                # Recovery bridges the original stage; neither ledger is reset.
                self.phase_bias_used+=self.recovery_charge
                net=math.fsum(self.phase_terms+[float(delta)])
                self.max_drift=max(self.max_drift,net-min(self.phase_min,net))
                self.total_positive+=max(float(delta),0.)
                self.phase_terms.append(float(delta))
                self.phase_positive+=max(float(delta),0.)
                self.phase_min=min(self.phase_min,net)
                self.fcounts['recovery_baselines_rebuilt']+=1
            self.cached_path=dict(G_id=id(G),params=state.params,version=self.context.state_version,
                                  delta=delta,W_owned=state.W.copy(),scale_F=objective_scale(anchor,state),error_F=None,
                                  accepted_full_sweep=False)
            self.phase_endpoint=self.cached_path['W_owned']
            self.fcounts['recovery_bridges' if changed else 'inverse_only_path_preserved']+=1
        else:
            # A detached restore may be legal yet lack the link to C2 entry.
            self.phase_initial=None;self.phase_endpoint=None
            # Keep all debits even when the recovery link is unavailable.
            self.cached_path=None;self.whole_phase_path_valid=False
            self.fcounts['unlinked_recovery_commits']+=1
        self.tie_residuals=[];self.fcounts['recovery_commits']+=1;self.was_stagnant=False
        self.emit(dict(event='recovery_commit',**location,state_version=self.context.state_version,
                       delta_F=float(delta),scaled=bool(scaled),weight_changed_from_anchor=changed,
                       whole_phase_connected=bool(linked),reused_recovery_comparison=bool(linked)))
        return state,delta,drift,scaled

    def tail_decision(self,G,old,new,params):
        if (self.phase is None or self.phase[:2]!=(id(G),tuple(params))
            or self.context.cached is None or self.context.cached[0]!=self.context.state_version):
            raise ComparisonUnresolved('stale_C2_tail_path')
        if not self.whole_phase_path_valid or self.phase_initial is None or not np.array_equal(old,self.phase_initial):
            raise ComparisonUnresolved('C2_tail_path_invalidated_by_recovery')
        if (not np.array_equal(new,self.context.W) or self.cached_path is None
            or self.cached_path['version']!=self.context.state_version
            or not np.array_equal(new,self.cached_path['W_owned'])
            or not np.array_equal(G,self.G_basis)):
            raise ComparisonUnresolved('C2_tail_endpoint_or_data_changed')
        delta=math.fsum(self.phase_prefix_terms+self.phase_terms);tau=tolerance(len(G),self.context.cached[1]['F'])
        self.fcounts['tail_path_reuses']+=1
        return self.phase_bias_used<=STAGE_POSITIVE_CAP,StableDelta(delta,'float64_C2_tail_path')

    def summary(self):
        old=super().summary()
        return dict(**old,float64_policy=POLICY,float64_counts=dict(self.fcounts),
                    soft_events=self.soft_events,
                    recovery_comparison_records=self.recovery_comparison_records,
                    sweep_records=self.records,total_positive_delta_F=self.total_positive,
                    maximum_net_drift_F=self.max_drift,
                    whole_phase_path_valid=self.whole_phase_path_valid,
                    recovery_prefix_delta_F=math.fsum(self.phase_prefix_terms),
                    normal_full_precision_calls=0,engineering_tolerance_not_proof=True,
                    finite_policy_id='practical_finite_precision_v1',phase_normalized_positive=self.phase_bias_used,
                    phase_positive_cap=STAGE_POSITIVE_CAP,phase_bias_history=self.bias_history)


def set_phase(ctx,name):
    now=time.perf_counter()
    if not hasattr(ctx,'float64_phase_seconds'):
        ctx.float64_phase_seconds=Counter();ctx.float64_phase=None;ctx.float64_phase_started=now
    if ctx.float64_phase is not None:ctx.float64_phase_seconds[ctx.float64_phase]+=now-ctx.float64_phase_started
    ctx.float64_phase=name;ctx.float64_phase_started=now
