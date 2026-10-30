"""Single immutable source of default CRC configuration."""
from dataclasses import dataclass,asdict
from math import isfinite
import numbers
from .recovery import DEFAULT_ALPHAS

@dataclass(frozen=True)
class Config:
    lambda1: float = 0.03
    threshold: float = 0.3
    initial_s: tuple = (1.,0.9,0.8,0.7,0.6)
    mu_init: float = 1.
    mu_factor: float = 0.1
    T: int = 5
    max_sweeps: int = 400
    max_inner_pass: int = 2000
    inner_kkt_tol: float = 1e-6
    outer_prox_resid_tol: float = 1e-4
    relative_tol: float = 1e-6
    max_pairs_per_trigger: int = 8
    edge_threshold: float = 0.05
    max_local_sweeps: int = 100
    local_prox_tol: float = 1e-6
    local_inner_kkt_tol: float = 1e-8
    min_absolute_decrease: float = 1e-11
    min_relative_decrease: float = 1e-8
    hot_cap_seconds: float = 600.
    evaluation_residual_threshold: float = 1e-4
    recovery_alphas: tuple = DEFAULT_ALPHAS
    recovery_no_progress_retries: int = 1

    def __post_init__(self):
        integer_fields=('T','max_sweeps','max_inner_pass','max_pairs_per_trigger','max_local_sweeps','recovery_no_progress_retries')
        for k in integer_fields:
            v=getattr(self,k)
            if isinstance(v,bool) or not isinstance(v,numbers.Integral) or v<(0 if k=='max_corrections' else 1):
                raise ValueError(k+' must be an integer in its valid range')
        for k,v in asdict(self).items():
            if k in integer_fields or k in ('initial_s','recovery_alphas'):continue
            if isinstance(v,bool) or not isinstance(v,numbers.Real) or not isfinite(v) or (v<0 if k in ('lambda1','threshold','edge_threshold') else v<=0):
                raise ValueError(k+' must be finite and nonnegative/positive as appropriate')
        if len(self.initial_s)!=self.T or any(not isfinite(x) or x<=0 for x in self.initial_s):
            raise ValueError('one positive s is required per continuation stage')
        alphas=tuple(self.recovery_alphas)
        if not alphas or any(isinstance(a,bool) or not isfinite(a) or not 0<a<1 for a in alphas) or any(a<=b for a,b in zip(alphas,alphas[1:])):
            raise ValueError('recovery_alphas must be finite, positive and strictly decreasing below one')
        object.__setattr__(self,'recovery_alphas',alphas)

    def algorithm(self):
        return dict(T=self.T,initial_s=list(self.initial_s),mu_init=self.mu_init,mu_factor=self.mu_factor,
            lambda1=self.lambda1,threshold=self.threshold,hot_cap_seconds=self.hot_cap_seconds,
            warm_iter=30000,max_iter=60000,lr=0.0003,checkpoint=1000,beta_1=0.99,beta_2=0.999,
            evaluation_residual_threshold=self.evaluation_residual_threshold,
            column=dict(max_sweeps=self.max_sweeps,max_inner_pass=self.max_inner_pass,
                inner_kkt_tol=self.inner_kkt_tol,outer_prox_resid_tol=self.outer_prox_resid_tol,
                relative_tol=self.relative_tol,inverse_refresh_sweeps=1))

    def search(self):
        names=('max_pairs_per_trigger','edge_threshold','max_local_sweeps',
               'local_prox_tol','local_inner_kkt_tol','min_absolute_decrease','min_relative_decrease')
        return {k:getattr(self,k) for k in names}
