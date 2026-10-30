"""Bounded review ONLY after the a085 sweep chain finally fails.

Same objective, tolerance and eight-term limit. No inverse refresh or new
precision backend. Error allocations are engineering envelopes with explicit
operation models, not machine-checked interval certificates.
"""
import math
import time
import numpy as np
import scipy.linalg as sla
from . import float64_acceptance as fa
from . import data_precision
from .float64_acceptance import (_gamma, _upper_norm, _sum_error,
    classify_evidence, data_refinement_reason)


def pairwise_data_delta(G,old,new,mu):
    """Compatibility entry; the data computation is shared with ordinary review."""
    return data_precision.pairwise_data_delta(G,old,new,mu,
        precision_available=fa.longdouble_capability()['supported'])


def balanced_barrier_delta(anchor,trial,tau,check_deadline=lambda:None):
    """Same nonsymmetric M0 E=D, with an exact binary diagonal similarity.

    A component envelope for E's solve/formation error is propagated through
    the trace of the log-det derivative. Positive products are rounded upward
    by gamma inflation; both series are capped at eight terms. Failure of any
    applicability check yields unresolved, never an increased allowance.
    """
    W,V=anchor.W,trial.W;d=len(W);s=anchor.params[1];u=np.finfo(float).eps/2
    out=dict(reliable=False,solves=0,dense_products=0,terms=0,error_trace_terms=0)
    def fail(reason):return dict(out,reason=reason)
    if (anchor.params!=trial.params or tau<=0 or not np.isfinite(tau) or
        any(not np.isfinite(x).all() for x in (W,V,anchor.K,anchor.B)) or
        any(e.metrics.get('domain_lower',0)<=0 or e.metrics.get('inverse_error_bound',np.inf)>1e-8
            or e.metrics.get('gradient_error_bound',np.inf)>1e-8 for e in (anchor,trial))):
        return fail('terminal_state_gate')
    if np.array_equal(W,V):return dict(out,reliable=True,reason='identical_weights',delta_h=0.,error_h=0.)
    check_deadline();p=V-W;z=V+W;D=-p*z;A=W*W;M=s*np.eye(d)-A;tiny=np.finfo(float).tiny
    if not np.isfinite(D).all() or not np.isfinite(M).all():return fail('nonfinite_formation')
    if np.any((p!=0)&(z!=0)&(abs(D)<tiny)) or np.any((W!=0)&(abs(A)<tiny)):
        return fail('formation_underflow')
    try:
        out['solves']=1;E=sla.solve(M,D,assume_a='gen',check_finite=False)
        Eb,(sc,_)=sla.matrix_balance(E,permute=False,separate=True)
    except (sla.LinAlgError,ValueError,FloatingPointError) as exc:
        return dict(fail('solve_or_balance_unresolved'),exception=str(exc))
    check_deadline()
    if not np.isfinite(E).all() or not np.isfinite(sc).all() or np.any(sc<=0):return fail('invalid_similarity')
    mantissa,exponent=np.frexp(sc)
    if not np.all(mantissa==.5):return fail('nonbinary_similarity')
    shift=exponent[None,:]-exponent[:,None]
    def scale(Z):
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            T=np.ldexp(Z,shift);back=np.ldexp(T,-shift)
        if (not np.isfinite(T).all() or np.any((Z!=0)&(abs(T)<tiny)) or
                not np.array_equal(back,Z)):
            raise ArithmeticError('similarity_not_exactly_representable')
        return T
    gamma=_gamma(d+8)
    def up(Z):return np.where(Z==0,0.,np.nextafter(Z/(1-_gamma(4)),np.inf))
    def product(X,Y):
        check_deadline();out['dense_products']+=1
        xp=X[X>0];yp=Y[Y>0]
        if len(xp) and len(yp) and float(xp.min())<tiny/float(yp.min()):
            raise ArithmeticError('positive_product_underflow_model')
        Z=X@Y
        return np.where(Z==0,0.,np.nextafter(Z/(1-gamma),np.inf))
    try:
        if not np.array_equal(Eb,scale(E)):return fail('similarity_mismatch')
        # Solve residual and fixed-W formation errors, in original coordinates.
        out['dense_products']+=1;residual=D-M@E
        if np.any((A>0)&(A<tiny/(u/(1-u)))) or np.any((D!=0)&(abs(D)<tiny/_gamma(3))):
            return fail('formation_error_underflow')
        absolute=product(abs(M),abs(E));mform=up((u/(1-u))*A);dform=up(_gamma(3)*abs(D))
        R=up((abs(residual)+gamma*(abs(D)+absolute)+dform+product(mform,abs(E)))/(1-gamma))
        H=up((abs(anchor.K)+anchor.B)/(1-_gamma(2)))
        C=scale(product(H,mform));er=scale(product(H,R));chi=_upper_norm(C)
        out.update(raw_norm_E=_upper_norm(E),norm_E=_upper_norm(Eb),formation_inverse_radius=chi,
            similarity_exponent_min=int(exponent.min()-1),similarity_exponent_max=int(exponent.max()-1),
            measured_solve_residual=_upper_norm(residual),exact_equation_residual_bound=_upper_norm(R))
        if not np.isfinite(chi) or chi>=1:return fail('inverse_formation_envelope_unresolved')
        # (I-C)^-1 er <= er+C er+uniform bound for the remaining terms.
        rest=np.nextafter(chi**2/(1-chi)*_upper_norm(er)/(1-_gamma(8)),np.inf)
        component=up(er+product(C,er)+rest)
        P=up(abs(Eb)+component);beta=_upper_norm(P);rho=_upper_norm(Eb);en=_upper_norm(component)
        out.update(E_error=en,majorant_norm=beta)
        if not all(np.isfinite(x) for x in (beta,rho,en)) or beta>=1:
            return fail('balanced_trace_radius_unresolved')
        Y=Eb.copy();power_error=0.;terms=[];errors=[]
        for k in range(1,9):
            check_deadline();terms.append((-1.)**k*math.fsum(map(float,np.diag(Y)))/k)
            errors.append(d*power_error/k+_gamma(d+4)*math.fsum(map(float,abs(np.diag(Y))))/k)
            tail=np.nextafter(d*rho**(k+1)/((k+1)*(1-rho))/(1-_gamma(32)),np.inf)
            if tail<=tau/16 or k==8:break
            ynorm=_upper_norm(Y);Y=Y@Eb;out['dense_products']+=1
            power_error=np.nextafter((power_error*rho+gamma*ynorm*rho)/(1-_gamma(8)),np.inf)
        # Integrate |tr((I+E+t*dE)^-1 dE)| <= sum tr(P^j component).
        U=component.copy();traces=[]
        for k in range(8):
            traces.append(np.nextafter(math.fsum(map(float,np.diag(U)))/(1-_gamma(d+4)),np.inf))
            remainder=np.nextafter(d*en*beta**(k+1)/(1-beta)/(1-_gamma(32)),np.inf)
            if remainder<=tau/64 or k==7:break
            U=product(P,U)
        solve_error=np.nextafter(math.fsum(traces+[remainder])/(1-_gamma(16)),np.inf)
        rounding=math.fsum(errors)+_gamma(len(terms)+4)*math.fsum(abs(x) for x in terms)
        error=np.nextafter(math.fsum((solve_error,rounding,tail))/(1-_gamma(8)),np.inf)
        delta=math.fsum(terms)
        out.update(reliable=bool(np.isfinite(delta) and np.isfinite(error)),reason='balanced_component_trace_log',
            delta_h=delta,error_h=float(error),terms=len(terms),error_trace_terms=len(traces),
            solve_error=float(solve_error),rounding_error=rounding,truncation_error=float(tail))
        return out
    except ArithmeticError as exc:return dict(fail('similarity_range_unresolved'),exception=str(exc))


def terminal_evidence(G,anchor,trial,base,current,tau,check_deadline=lambda:None,
                      *,scale=None,remaining=None,data_attempted=False,data_result=None):
    """At most one data pass and one balanced barrier pass; no recursive retry."""
    info=dict(attempted=True,data_calls=0,barrier_calls=0,data_seconds=0.,barrier_seconds=0.)
    required=('loss_delta_F','data_error_F','l1_delta_F','l1_error_F','barrier_delta_F',
              'barrier_error_F','arithmetic_error_F','ordinary_consistency_radius_F')
    # Original identity/domain/reliability requirements remain necessary.
    if (anchor.params!=trial.params or not np.array_equal(G,G.T) or
        not base.get('reliable') or not all(k in base for k in required) or not np.isfinite(G).all() or
        any(e.metrics.get('domain_lower',0)<=0 or e.metrics.get('inverse_error_bound',np.inf)>1e-8 or
            e.metrics.get('gradient_error_bound',np.inf)>1e-8 for e in (anchor,trial))):
        return current,dict(info,reason='terminal_identity_or_state_gate')
    if np.array_equal(anchor.W,trial.W):return base,dict(info,reason='identical_weights')
    def acceptable(e):
        if scale is not None:return fa.evidence_within_budget(e,scale,remaining)
        return e.get('reliable') and classify_evidence(e['delta_F'],tau,e['error_F']) in ('descent','tie')
    def needs_data(e):
        return data_refinement_reason(e,tau,scale=scale,remaining=remaining)=='data_blocks_decision'
    if scale is not None and acceptable(current):return current,dict(info,reason='current_finite_evidence_sufficient')
    data=data_result if data_attempted else None
    if data_attempted and data is None:data=dict(reliable=False,reason='prior_data_attempt_unavailable')
    info['prior_data_attempted']=bool(data_attempted)
    def get_data():
        nonlocal data
        if data is None:
            check_deadline();started=time.perf_counter()
            data=pairwise_data_delta(G,anchor.W,trial.W,anchor.params[0])
            info.update(data_calls=1,data_seconds=time.perf_counter()-started,data=data)
            check_deadline()
        elif data_attempted:info['data_reused']=True
        return data
    def combine(e,newdata):
        if not newdata.get('reliable') or newdata['data_error_F']>=e['data_error_F']:return e
        pieces=(newdata['loss_delta_F'],e['l1_delta_F'],e['barrier_delta_F'])
        delta=math.fsum(pieces);arithmetic=max(e['arithmetic_error_F'],_sum_error(pieces))
        error=float(np.nextafter(math.fsum((newdata['data_error_F'],e['l1_error_F'],e['barrier_error_F'],arithmetic)),np.inf))
        reliable=bool(e['reliable'] and abs(delta-(trial.metrics['F']-anchor.metrics['F']))<=base['ordinary_consistency_radius_F'])
        return dict(e,delta_F=delta,error_F=error,loss_delta_F=newdata['loss_delta_F'],data_error_F=newdata['data_error_F'],
            arithmetic_error_F=arithmetic,reliable=reliable,reason='terminal_reused_data' if data_attempted else 'terminal_pairwise_data')
    # An inapplicable or worse trace expansion does not invalidate the original
    # endpoint's independent evidence. Keep its non-data error unchanged.
    if needs_data(base):
        endpoint=combine(base,get_data());info['retained_endpoint']=endpoint
        if acceptable(endpoint):
            return endpoint,dict(info,reason='retained_endpoint_data_review')
    check_deadline();started=time.perf_counter()
    barrier=balanced_barrier_delta(anchor,trial,tau,check_deadline)
    info.update(barrier_calls=1,barrier_seconds=time.perf_counter()-started,barrier=barrier)
    if not barrier['reliable']:return current,dict(info,reason=barrier['reason'])
    mu,_,lam=anchor.params;diff=abs(trial.W)-abs(anchor.W)
    l1=mu*lam*math.fsum(map(float,diff.ravel()))
    l1_scale=abs(mu*lam)*math.fsum(map(float,abs(diff).ravel()))
    l1_error=_gamma(diff.size+6)*l1_scale
    # Reuse an already-completed old data refinement if it was better.
    best=current if current.get('reliable') and current.get('data_error_F',np.inf)<base['data_error_F'] else base
    pieces=(best['loss_delta_F'],l1,barrier['delta_h']);delta=math.fsum(pieces)
    arithmetic=_sum_error(pieces);error=float(np.nextafter(math.fsum((best['data_error_F'],l1_error,barrier['error_h'],arithmetic)),np.inf))
    e=dict(base,loss_delta_F=best['loss_delta_F'],data_error_F=best['data_error_F'],
        l1_delta_F=l1,l1_error_F=l1_error,barrier_delta_F=barrier['delta_h'],barrier_error_F=barrier['error_h'],
        arithmetic_error_F=arithmetic,delta_F=delta,error_F=error,difference_scaled_error=True,
        reliable=bool(abs(delta-(trial.metrics['F']-anchor.metrics['F']))<=base['ordinary_consistency_radius_F']),reason='terminal_balanced_barrier')
    if not acceptable(e) and needs_data(e):e=combine(e,get_data())
    info['after']=e.copy()
    return e,dict(info,reason='bounded_terminal_review')
