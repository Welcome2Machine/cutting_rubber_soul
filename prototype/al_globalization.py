"""Optional residual-globalized AL Newton, with unchanged physical equations.

The diagonal shift changes only the algebraic direction. Excessively long or
inaccurate Newton directions are replaced, not repeatedly clipped to vanishing
line-search steps. Every trial is checked against the original mixed residual.
This is a bounded regularized Newton method, not PETSc's trust-region algorithm.
"""
import time
import numpy as np
from scipy.sparse import bmat, diags, eye
from .al_performance import FastAugmentedBladeProblem
from .ipc_reference import BASELINE, LoadStepFailure
from .augmented_blade import contact_constraints_satisfied,coulomb_projection
from .curve_contact import blocks
from .separated_self_contact import SeparatedContactDomainError


class WarmStartedAugmentedBladeProblem(FastAugmentedBladeProblem):
    """Seed algebraic multipliers by reference-edge identity, never by position.

    This is not committed friction history. Failed load attempts restore the
    last input guess so that adaptive retry cannot inherit a rejected iterate.
    """
    def prepare_quadrature(self,x):
        previous={int(e):v.copy() for e,v in zip(self.contact_edges,self.multipliers)}
        super().prepare_quadrature(x)
        for i,e in enumerate(self.contact_edges):
            if int(e) in previous:self.multipliers[i]=previous[int(e)]

    def step(self,x,old,depth,deadline):
        saved=(self.qedges.copy(),self.qfraction.copy(),self.qweights.copy(),self.multipliers.copy())
        self.trial=x.copy()
        try:return super().step(x,old,depth,deadline)
        except Exception:
            # Preserve the actual rejected iterate BEFORE restoring the guess.
            # It is diagnostic evidence, never an accepted/resumable history.
            data=dict(positions=getattr(self,'trial',x),previous_positions=self.previous,
                      attempted_depth_mm=depth,**self.checkpoint_data())
            self.rejected_al_state={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in data.items()}
            self._set_quadrature(*saved[:3]);self.multipliers=saved[3]
            raise


class GlobalizedAugmentedBladeProblem(FastAugmentedBladeProblem):
    # Existing direction selection stays available for controlled comparisons.
    # "adaptive" is opt-in and changes only algebraic directions, never roots.
    direction_policy='first'
    nested_contact_model=False
    nonmonotone_window=1

    def contact_model_direction(self,h,contact,g,weights,shift_diagonal,deadline=float('inf')):
        """Solve the piecewise-affine contact model at one frozen bulk tangent.

        Inner primal-dual active sets refresh the exact Coulomb projection of
        LINEARIZED gap/slip. No FE history, geometry or accepted state changes.
        Cycles/nonconvergence return None; original Newton remains fallback.
        Final nonlinear merit and physical acceptance remain independent.
        """
        nf=len(self.free);k=contact['kinematic'][:,self.free]
        H=h[self.free][:,self.free]+diags(shift_diagonal[:nf]);B=contact['force_lambda'][self.free]
        step=np.zeros(nf+len(weights));seen=set();initial=None
        for iteration in range(10):
            if time.monotonic()>deadline:raise TimeoutError('AL contact model direction wall budget')
            motion=(k@step[:nf]).reshape(-1,2)
            lm=self.multipliers+step[nf:].reshape(-1,2)
            pn,pt,dk,dl,stick=coulomb_projection(contact['gap_average']+motion[:,0],
                contact['slip_average']+motion[:,1],lm,self.penalty,self.config.friction_coefficient)
            residual=np.r_[g[self.free]+H@step[:nf]+B@step[nf:],weights*(lm-np.column_stack((pn,pt))).ravel()]
            norm=float(np.linalg.norm(residual))
            if initial is None:initial=norm
            if norm<=max(1e-10,1e-8*initial):return step,iteration,norm
            key=(dk.tobytes(),dl.tobytes())
            if key in seen:return None
            seen.add(key)
            M=bmat([[H,B],[diags(weights)@(-blocks(dk)@k),
                diags(weights)@(eye(len(weights),format='csc')-blocks(dl))]],format='csc')
            change=self.linear_direction(M,-residual)
            if not np.isfinite(change).all() or np.linalg.norm(M@change+residual)>max(1e-8*norm,1e-10):return None
            step+=change
        return None

    def _equilibrate(self,x,deadline):
        nf=len(self.free);radius=.01;merit_history=[]
        adaptive=self.direction_policy=='adaptive'
        if self.direction_policy not in ('first','adaptive'):
            raise ValueError('Unknown AL direction policy')
        shifts=(0.,1e-5,1e-3,.01,.1,1.,10.,100.)
        preferred=0
        for iteration in range(60):
            # Positions and multipliers always describe the SAME iterate,
            # including deadline/iteration-cap exits after an accepted update.
            self.trial=x.copy()
            if time.monotonic()>deadline:raise TimeoutError('AL globalized solve wall budget')
            tick=time.perf_counter();_,g,h,j=self.evaluate(x)
            contact,checks=self.constraint_audit(x);weights=np.repeat(self.edge_area,2)
            r=np.r_[g[self.free],weights*contact['constraint'].ravel()]
            norm=float(np.linalg.norm(r));physical=float(np.linalg.norm(g[self.free]));self.trial=x.copy()
            merit_history.append(norm*norm)
            # Optional bounded nonmonotone globalization for active-set
            # transitions. The reference forgets old values after THREE
            # iterations and permits at most 2x current residual magnitude.
            reference_merit=min(max(merit_history[-self.nonmonotone_window:]),4*norm*norm)
            print(f'AL_GLOBAL iteration={iteration} residual_N={physical:.9g} contact_defect_N={checks["integrated_projection_defect_n"]:.9g} minJ={j:.9g} assembly_s={time.perf_counter()-tick:.6f}',flush=True)
            active=self.multipliers[:,0]-self.penalty*contact['gap_average']>0
            print(f'AL_BRANCH active={np.count_nonzero(active)} stick={np.count_nonzero(contact["stick"])} slip={np.count_nonzero(active & ~contact["stick"])}',flush=True)
            print(f'AL_CONE violation_MPa={checks["maximum_cone_violation_mpa"]:.9g} min_normal_MPa={self.multipliers[:,0].min(initial=0):.9g}',flush=True)
            if physical<=BASELINE['force_tolerance_n']*.25 and contact_constraints_satisfied(checks,self.multipliers):
                return x,iteration
            A=bmat([[h[self.free][:,self.free],contact['force_lambda'][self.free]],
                    [diags(weights)@contact['constraint_u'][:,self.free],diags(weights)@contact['constraint_lambda']]],format='csc')
            scale=np.r_[np.maximum(abs(h.diagonal()[self.free]),1.),np.zeros(len(weights))]
            before=self.multipliers.copy();accepted=False;best=None
            order=list(range(preferred,len(shifts)))+list(range(preferred)) if adaptive else list(range(len(shifts)))
            for index in order:
                shift=shifts[index]
                if time.monotonic()>deadline:raise TimeoutError('AL globalized direction wall budget')
                M=A+diags(shift*scale+np.r_[np.full(nf,1e-9),np.zeros(len(weights))])
                model=(self.contact_model_direction(h,contact,g,weights,
                    shift*scale+np.r_[np.full(nf,1e-9),np.zeros(len(weights))],deadline)
                    if self.nested_contact_model else None)
                du=model[0] if model is not None else self.linear_direction(M,-r)
                error=model[2] if model is not None else float(np.linalg.norm(M@du+r))
                if model is not None:print(f'AL_CONTACT_MODEL iterations={model[1]} residual_N={model[2]:.9g}',flush=True)
                extent=float(abs(du[:nf]).max())
                if not np.isfinite(du).all() or error>max(1e-8*norm,1e-10):
                    print(f'AL_DIRECTION_REJECT shift={shift:g} reason=linear_accuracy error_N={error:.9g}',flush=True)
                    continue
                if extent>radius:
                    print(f'AL_DIRECTION_REJECT shift={shift:g} reason=extent extent_mm={extent:.9g}',flush=True)
                    continue
                # A branch-crossing model direction need not be a descent
                # direction for the starting branch. Use its full piecewise
                # affine predicted residual, then require ACTUAL decrease.
                if model is not None:
                    motion=(contact['kinematic'][:,self.free]@du[:nf]).reshape(-1,2)
                    lm=before+du[nf:].reshape(-1,2)
                    pn,pt,*_=coulomb_projection(contact['gap_average']+motion[:,0],
                        contact['slip_average']+motion[:,1],lm,self.penalty,self.config.friction_coefficient)
                    predicted=np.r_[g[self.free]+h[self.free][:,self.free]@du[:nf]+contact['force_lambda'][self.free]@du[nf:],
                        weights*(lm-np.column_stack((pn,pt))).ravel()]
                    Ad=predicted-r
                else:Ad=A@du
                slope=float(r@Ad)
                if not np.isfinite(slope) or slope>=0:continue
                direction=np.zeros(self.total);direction[self.free]=du[:nf];direction=direction.reshape(-1,2)
                delta_lambda=du[nf:].reshape(-1,2)
                alpha=min(1.,self.safe_step(x,direction))
                initial_alpha=alpha
                bounds=getattr(self,'last_safe_bounds',{})
                print(f'AL_BOUND shift={shift:g} initial_alpha={initial_alpha:.9g} J_bound={bounds.get("J",float("nan")):.9g} self_ccd_bound={bounds.get("self_ccd",float("nan")):.9g}',flush=True)
                for back in range(9):
                    if time.monotonic()>deadline:raise TimeoutError('AL globalized trial wall budget')
                    if alpha<1/256:break
                    trial=x+alpha*direction
                    # The exact-arithmetic safe-step bound is not an interval
                    # proof for the represented endpoint. Reject a rounded map
                    # before material evaluation, without swallowing unrelated
                    # constitutive exceptions or weakening positive-J acceptance.
                    represented_j=self.enrichment.minimum_j(trial[:self.n]-self.mesh.reference_xy)
                    if not np.isfinite(represented_j) or represented_j<=0:
                        print(f'AL_GEOMETRY_REJECT alpha={alpha:.9g} represented_minJ={represented_j:.9g}',flush=True)
                        alpha*=.5
                        continue
                    self.multipliers=before+alpha*delta_lambda
                    try:
                        gt=self.evaluate(trial,tangent=False)[1]
                        ct=self.contact_data(trial)['constraint']
                        rt=np.r_[gt[self.free],weights*ct.ravel()]
                    except SeparatedContactDomainError:
                        print(f'AL_DOMAIN_REJECT alpha={alpha:.9g} reason=nonadjacent_proximity',flush=True)
                        alpha*=.5
                        continue
                    finally:
                        self.multipliers=before.copy()
                    actual=float((reference_merit-rt@rt)/2)
                    model=float(-alpha*slope-.5*alpha*alpha*(Ad@Ad))
                    ratio=actual/model if model>0 else -np.inf
                    if (np.isfinite(actual) and actual>max(1e-24,1e-6*norm*norm)
                            and ratio>=.1 and actual>=-1e-4*alpha*slope):
                        candidate=(actual,trial,before+alpha*delta_lambda,index,alpha,ratio,extent,float(np.linalg.norm(rt)))
                        if best is None or actual>best[0]:best=candidate
                        # Do not settle for a tiny first decrease when a more
                        # regularized direction can be evaluated on this SAME
                        # assembly. Retain the best original-merit-safe trial.
                        accepted=(not adaptive or (alpha>=.5 and actual>=.05*norm*norm))
                        break
                    if adaptive and best is not None and back>=1:
                        # Once a safe candidate exists, spend at most two
                        # evaluations on each alternative. Without a safe
                        # candidate retain the ORIGINAL nine-trial fallback.
                        break
                    alpha*=.5
                if accepted:break
            if best is None:
                self.multipliers=before
                raise LoadStepFailure('AL globalized Newton has no reliable residual-decreasing direction')
            _,x,self.multipliers,index,alpha,ratio,extent,nt=best
            self.trial=x.copy()
            if adaptive:
                preferred=max(0,index-1) if alpha==1. and ratio>.9 else index
            print(f'AL_GLOBAL_UPDATE alpha={alpha:.9g} shift={shifts[index]:g} ratio={ratio:.9g} extent_mm={extent:.9g} joint_residual_N={nt:.9g} policy={self.direction_policy} nonmonotone={nt>norm}',flush=True)
        raise LoadStepFailure('AL globalized Newton iteration limit')


class WarmGlobalizedAugmentedBladeProblem(WarmStartedAugmentedBladeProblem,GlobalizedAugmentedBladeProblem):
    """Reference-edge warm start plus bounded residual-globalized directions."""
