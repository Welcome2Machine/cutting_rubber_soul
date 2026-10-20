"""Opt-in nonlinear contact corrections with a locally linearized bulk.

Only the inner algebraic model freezes the material tangent. Contact geometry,
normal pressure and friction are rebuilt at every inner trial. Every outer
iterate uses the original finite-strain residual; acceptance is always fresh.
No material/friction history is advanced here. This is a nonlinear solver
preconditioner, not a linear-elastic replacement of the rubber constitutive law.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from .compact_p2_assembly import compact_matrices
from .ipc_reference import BASELINE, LoadStepFailure
from .residual_newton import residual_armijo


def contact_terms(p,x,*,tangent=False):
    c=p.collisions(x);sx=x[p.surface]
    lag=p.ipc.TangentialCollisions()
    lag.build(p.cmesh,sx,c,p.barrier,p.config.friction_coefficient)
    delta=(x-p.previous)[p.surface]
    g=np.zeros(p.total)
    g[p.surface_dofs]=p.barrier.gradient(c,p.cmesh,sx)+p.friction.gradient(lag,p.cmesh,delta)
    if not tangent:return g
    def matrix(clamp):
        projection=p.ipc.PSDProjectionMethod.CLAMP if clamp else p.ipc.PSDProjectionMethod.NONE
        h=p.barrier.hessian(c,p.cmesh,sx,projection)+p.friction.hessian(lag,p.cmesh,delta,projection)
        h=h.tocoo()
        return coo_matrix((h.data,(p.surface_dofs[h.row],p.surface_dofs[h.col])),shape=(p.total,p.total)).tocsc()
    h=matrix(False)
    return g,p.coupled_friction_tangent(x,lag,h),matrix(True)


class ContactRefreshMixin:
    """Bounded inner PTC contact corrections, safeguarded by full force balance."""
    # Long inner paths can outgrow the material Taylor model even with a small
    # displacement. Refresh before that drift consumes the entire load budget.
    contact_inner_limit=8
    contact_outer_limit=64
    contact_inner_merit=True

    def solve_friction(self,x,depth,deadline):
        if self.element!='p2-local-visco' or self.nodal_volume is not None or self.config.friction_coefficient<=0:
            raise ValueError('Contact refresh requires explicit frictional local P2 candidate')
        used=0
        for outer in range(self.contact_outer_limit):
            if time.monotonic()>deadline:raise TimeoutError('Contact refresh wall budget')
            self.trial=x.copy();tick=time.perf_counter()
            r,k,dofs,_,_=self.material_bulk((x[:self.n]-self.mesh.reference_xy).ravel(),with_tangent=True)
            gb=np.zeros(self.total);np.add.at(gb,dofs.ravel(),r.ravel())
            kb,mb=compact_matrices(k,dofs,self.total,self.linear_elements,self.active_elements,with_metric=True)
            assembly=time.perf_counter()-tick
            anchor=x.copy();gc=contact_terms(self,x)
            g0=gb+gc;norm=float(np.linalg.norm(g0[self.free]))
            print(f'REFRESH_OUTER depth_mm={depth:.9g} outer={outer} fresh_residual_N={norm:.9g} '
                  f'bulk_assembly_s={assembly:.6f} total_inner={used}',flush=True)
            # This is the FULL material/contact force, never the frozen inner residual.
            if norm<=BASELINE['force_tolerance_n']:return x,used
            goal=max(.1*BASELINE['force_tolerance_n'],.1*norm)
            shift=.1;value=x.copy();inner_norm=norm
            for inner in range(min(self.contact_inner_limit,self.max_iterations-used)):
                if time.monotonic()>deadline:raise TimeoutError('Contact refresh inner wall budget')
                tick=time.perf_counter();gc,jc,mc=contact_terms(self,value,tangent=True)
                rr=(gb+kb@(value-anchor).ravel()+gc)[self.free]
                inner_norm=float(np.linalg.norm(rr))
                if inner_norm<=goal:break
                a=(kb+jc)[self.free][:,self.free]
                m=(mb+mc)[self.free][:,self.free]+diags(np.full(len(self.free),BASELINE['hessian_regularization_n_mm']))
                for retry in range(18):
                    if time.monotonic()>deadline:raise TimeoutError('Contact refresh retry wall budget')
                    du=spsolve(a+shift*m,-rr)
                    maximum=float(np.max(np.abs(du)))
                    if not np.isfinite(du).all() or maximum==0:
                        shift=max(.1,2*shift);continue
                    direction=np.zeros(self.total);direction[self.free]=du;direction=direction.reshape(x.shape)
                    trial,alpha,safe=self._ptc_trial(value,direction,maximum)
                    rt=(gb+kb@(trial-anchor).ravel()+contact_terms(self,trial))[self.free]
                    nt=float(np.linalg.norm(rt));action=a@du
                    error=float(np.linalg.norm(rt-rr-alpha*action))
                    stalled=(safe<1e-3 and nt>.99*inner_norm and nt>BASELINE['force_tolerance_n'])
                    model=error<=.25*max(inner_norm,nt)
                    merit=(self.contact_inner_merit and residual_armijo(inner_norm,nt,float(rr@action),alpha))
                    if np.isfinite(nt) and not stalled and not np.array_equal(trial,value) and (model or merit):
                        value=trial;used+=1
                        shift=(max(.1,shift) if (not model or safe<.25) else
                               max(1e-10,min(1e6,shift*nt/max(inner_norm,1e-15)/1.2)))
                        print(f'REFRESH_INNER outer={outer} inner={inner} model_residual_N={nt:.9g} '
                              f'model_error_N={error:.9g} alpha={alpha:.9g} retries={retry} '
                              f'seconds={time.perf_counter()-tick:.6f}',flush=True)
                        inner_norm=nt;break
                    shift=max(.1,2*shift)
                else:raise LoadStepFailure('Contact refresh inner model mismatch')
                # Limit the validity region of a bulk Taylor approximation.
                if nt<=goal or np.max(np.abs(value-anchor))>.5*self.options.tip_radius_mm:break
            if np.array_equal(value,anchor):raise LoadStepFailure('Contact refresh has no inner progress')
            direction=value-anchor
            # Inner CCD segments do not automatically prove the straight outer chord.
            # Explicitly recheck that chord before any full-residual line search.
            safe=self.safe_step(anchor,direction);alpha=min(1.,safe)
            for backtrack in range(10):
                if time.monotonic()>deadline:raise TimeoutError('Contact refresh outer wall budget')
                trial=anchor+alpha*direction
                if self.enrichment.minimum_j(trial[:self.n]-self.mesh.reference_xy)<=0:
                    alpha*=.5;continue
                gt=self.evaluate(trial,self.lag(trial),tangent=False)[1][self.free]
                nt=float(np.linalg.norm(gt))
                model=(gb+kb@(trial-anchor).ravel()+contact_terms(self,trial))[self.free]
                print(f'REFRESH_FRESH outer={outer} backtrack={backtrack} alpha={alpha:.9g} '
                      f'fresh_residual_N={nt:.9g} model_residual_N={np.linalg.norm(model):.9g} '
                      f'bulk_model_error_N={np.linalg.norm(gt-model):.9g}',flush=True)
                if np.isfinite(nt) and (nt<=BASELINE['force_tolerance_n'] or nt<norm*(1.-1e-4*alpha)):
                    x=trial;self.trial=x.copy();break
                alpha*=.5
            else:raise LoadStepFailure('Contact refresh full-residual safeguard failed')
            if used>=self.max_iterations:break
        # Never infer physical balance from an inner correction or small update.
        fresh=float(np.linalg.norm(self.evaluate(x,self.lag(x),tangent=False)[1][self.free]))
        if fresh<=BASELINE['force_tolerance_n']:return x,used
        raise LoadStepFailure(f'Contact refresh iteration budget: residual={fresh:.9g} N, inner={used}')
