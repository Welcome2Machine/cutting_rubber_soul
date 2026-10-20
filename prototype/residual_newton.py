"""Opt-in residual Newton phase before the original IPC PTC fallback.

This solves the SAME fresh, generally nonconservative force equations. Armijo
acts on ||r||^2/2, never on an invented friction energy. Neither constitutive
history, load interval, physical acceptance nor the combined iteration budget
changes. No production factory enables this diagnostic automatically.
"""
import time
import warnings

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .ipc_reference import BASELINE, LoadStepFailure


def residual_armijo(norm, trial_norm, slope, alpha):
    return (np.isfinite([norm,trial_norm,slope,alpha]).all() and norm>0
            and trial_norm>=0 and slope<0 and 0<alpha<=1
            and .5*trial_norm**2 <= .5*norm**2+1e-4*alpha*slope)


class ResidualNewtonMixin:
    """At most 24 unshifted Newton directions, then original bounded PTC."""
    newton_phase_limit=24
    newton_backtracks=10

    def solve_friction(self,x,depth,deadline):
        if self.nodal_volume is not None:
            raise ValueError('Residual Newton diagnostic does not condense pressure blocks')
        used=0
        for iteration in range(min(self.newton_phase_limit,self.max_iterations)):
            if time.monotonic()>deadline:raise TimeoutError('Residual Newton wall budget')
            self.trial=x.copy();start=time.perf_counter();lag=self.lag(x)
            _,g,H,_=self.evaluate(x,lag,projected=False)
            r=g[self.free];norm=float(np.linalg.norm(r))
            print(f'NEWTON_RESIDUAL depth_mm={depth:.9g} iteration={iteration} residual_N={norm:.9g} '
                  f'assembly_s={time.perf_counter()-start:.6f}',flush=True)
            if norm<=BASELINE['force_tolerance_n']:return x,used
            A=self.coupled_friction_tangent(x,lag,H)[self.free][:,self.free]
            start=time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('error',MatrixRankWarning)
                try:du=spsolve(A,-r)
                except MatrixRankWarning:break
            linear_s=time.perf_counter()-start
            used+=1
            action=A@du;slope=float(r@action)
            relative=float(np.linalg.norm(action+r)/max(norm,1e-30))
            maximum=float(np.max(np.abs(du)))
            if not np.isfinite(du).all() or maximum==0 or relative>1e-6 or slope>=0:
                print(f'NEWTON_FALLBACK invalid_direction linear_relative={relative:.9g}',flush=True);break
            direction=np.zeros(self.total);direction[self.free]=du;direction=direction.reshape(x.shape)
            try:trial,alpha,_=self._ptc_trial(x,direction,maximum)
            except LoadStepFailure as exc:
                print(f'NEWTON_FALLBACK {exc}',flush=True);break
            accepted=False
            for backtrack in range(self.newton_backtracks):
                if time.monotonic()>deadline:raise TimeoutError('Residual Newton line search wall budget')
                if backtrack:
                    alpha*=.5;trial=x+alpha*direction
                    if self.enrichment is not None:
                        minj=self.enrichment.minimum_j(trial[:self.n]-self.mesh.reference_xy)
                        if not np.isfinite(minj) or minj<=0:continue
                if np.array_equal(trial,x):break
                start=time.perf_counter()
                rt=self.evaluate(trial,self.lag(trial),tangent=False)[1][self.free]
                nt=float(np.linalg.norm(rt))
                accept=residual_armijo(norm,nt,slope,alpha)
                print(f'NEWTON_TRIAL iteration={iteration} backtrack={backtrack} alpha={alpha:.9g} '
                      f'residual_N={nt:.9g} accepted={accept} linear_s={linear_s:.6f} '
                      f'trial_s={time.perf_counter()-start:.6f} linear_relative={relative:.9g}',flush=True)
                if accept:
                    x=trial;self.trial=x.copy();accepted=True;break
            if not accepted:break
        print(f'NEWTON_FALLBACK original_PTC used_directions={used}',flush=True)
        original=self.max_iterations
        try:
            self.max_iterations=max(0,original-used)
            value,count=super().solve_friction(x,depth,deadline)
            return value,used+count
        finally:self.max_iterations=original
