"""Finite-increment stress/energy gate followed by dynamic cohesive opening.

Paired virtual equilibria hold the blade, supports and viscous metric fixed.
The virtual finite slit is traction free; existing damage is frozen elastic.
Native Coulomb work on the straight admissible virtual path is subtracted.
This is a frozen-history, prescribed-path CC/CZM hybrid, not a reproduction
of elastic brittle FFM nor a claim of path-independent frictional release.
Virtual fields NEVER replace the physical position/velocity/material history.
Gc is paid once, by the physical cohesive law, not additionally at insertion.
"""
import hashlib
import time
import numpy as np
from scipy.sparse import diags
from mechanics.discrete_gradient import frozen_response
from .ipc_reference import IPCProblem, LoadStepFailure
from .zero_gap_cut import ZeroGapCutProblem
from .energy_cut import EnergyCutProblem
from .force_globalization import solve


def finite_patches(traction, released, y, area, strength):
    """All contiguous virgin tensile nodal tributaries, smallest area first.

    Pointwise discrete stress condition, NOT a positive-part average. A zero
    or compressed node splits a candidate; no leap across intact compression.
    The physical support of P1 opening is reported separately from lumped area.
    """
    good = (~released) & (y > 0) & (traction >= strength)
    patches = []
    for first in np.flatnonzero(good):
        for last in range(first, len(y)):
            if not good[last]:
                break
            ids = np.arange(first, last+1)
            patches.append(ids)
    # Geometrically equal tributaries differ by floating subtraction roundoff.
    # Do not turn that noise into an arbitrary deep-before-shallow crack choice.
    return sorted(patches, key=lambda ids: (round(float(area[ids].sum()), 12), -float(y[ids].max()), int(ids[0])))


def coupled_margin(stress, strength, released_energy, friction_work, fracture_work, uncertainty=0.):
    if (not np.isfinite(np.r_[stress, strength, released_energy, friction_work, fracture_work, uncertainty]).all()
            or strength <= 0 or fracture_work <= 0 or friction_work < -1e-12 or uncertainty < 0):
        raise ValueError('Invalid finite-crack evidence')
    margin = released_energy-friction_work-uncertainty-fracture_work
    return bool(len(stress) and np.min(stress) >= strength and margin >= 0), float(margin)


class FrozenVirtualCrack(ZeroGapCutProblem):
    """Independent zero-time virtual equilibrium; constructor copies no histories back."""
    def __init__(self, host, anchor, metric, history, released, removed, epsilon):
        self.__dict__ = {key: value.copy() if isinstance(value, np.ndarray) else value
                         for key, value in host.__dict__.items()}
        for key in ('_primary_positions', '_primary_increment', 'ffm_pending', 'ffm_checks'):
            self.__dict__.pop(key, None)
        self.previous = anchor.copy()
        # A deadline before iteration zero must not retain the host's trial.
        self.trial = anchor.copy()
        self.previous_material_metric = metric.copy()
        self.previous_history = history.copy()
        # A virgin extrinsic graph has no finite unloading spring. Keep its
        # zero-opening ties in the virtual frozen-damage problem instead of
        # accidentally treating them as already traction free.
        self.released = (released & (history > 0)) | removed
        self.active_contact = np.zeros(self.n, bool)
        self.virtual_removed = removed.copy()
        self.virtual_solver_load = np.zeros(self.total)
        self.virtual_friction_epsilon = epsilon
        self.virtual_solver_phases = []
        end = 2*self.toughness/self.strength
        self.virtual_secant = np.divide(self.strength*np.maximum(1-history/end, 0.), history,
            out=np.zeros_like(history), where=history > 0)*released*(~removed)
        self.friction = self.ipc.FrictionPotential(epsilon)
        self.rebuild_contact()
        self.set_free()

    def material_bulk(self, u, *, with_tangent=True):
        areas, gradients, dofs = self.mesh.bulk_data
        f = np.eye(2)+np.einsum('eai,eaJ->eiJ', u.reshape(-1, 2)[self.mesh.triangles], gradients)
        q = frozen_response(f, self.previous_material_metric, self.config.shear_modulus_n_mm2,
            self.material.mu_ov[0], self.config.bulk_modulus_n_mm2, tangent=with_tangent)
        volume = areas*self.config.width_mm
        r = np.einsum('e,eiJ,eaJ->eai', volume, q['P'], gradients).reshape(-1, 6)
        k = None if not with_tangent else np.einsum('e,eaJ,eiJkL,ebL->eaibk', volume, gradients,
            q['tangent'].reshape(-1, 2, 2, 2, 2), gradients, optimize=True).reshape(-1, 6, 6)
        return r, k, dofs, float(volume@q['energy']), float(q['J'].min())

    def lag(self, x):
        return IPCProblem.lag(self, x)

    def coupled_friction_tangent(self, x, lag, hessian):
        return IPCProblem.coupled_friction_tangent(self, x, lag, hessian)

    def evaluate(self, x, lag=None, tangent=True, projected=True, *, with_metric=False):
        out = list(IPCProblem.evaluate(self, x, lag, tangent, projected, with_metric=with_metric))
        opening = 2*(x[self.seam, 0]-self.config.drop_s_mm)
        ids = 2*self.seam
        out[0] += .25*float(self.seam_area@(self.virtual_secant*opening**2))
        out[1][ids] += self.seam_area*self.virtual_secant*opening
        if tangent:
            extra = np.zeros(self.total)
            extra[ids] = 2*self.seam_area*self.virtual_secant
            out[2] += diags(extra)
            if with_metric:
                out[4] += diags(extra)
        # Temporary force homotopy is used ONLY for finding a virtual root.
        # Its load is exactly zero before any stored-energy/force certification.
        out[0] -= float(self.virtual_solver_load@(x-self.previous).ravel())
        out[1] -= self.virtual_solver_load
        return tuple(out)

    def stored(self, x):
        if np.any(self.virtual_solver_load):
            raise RuntimeError('Artificial virtual solver load cannot enter crack energy')
        return 2*self.evaluate(x, None, tangent=False)[0]

    def _ptc_trial(self, x, direction, maximum):
        # Every numerical segment can be safe while the final virtual chord
        # intersects the blade. Keep the assumed straight friction path in the
        # admissible search domain rather than discovering this only at a root.
        trial,alpha,geometric=super()._ptc_trial(x,direction,maximum)
        for _ in range(24):
            if self.accepted_chord_is_safe(trial):
                return trial,alpha,geometric
            alpha*=.5
            trial=x+alpha*direction
        return x.copy(),0.,geometric

    def equilibrium(self, seed, depth, deadline):
        # Fresh-force PTC crosses nonzero force-merit minima that trapped LM
        # in the actual blade's virtual fracture problem. Its shift is solely
        # a linear solver device; final certification uses the original force.
        self.set_primary_increment(seed, seed-self.previous)
        try:
            value, _ = self.solve_friction(seed.copy(), depth, min(deadline, time.monotonic()+4.))
            self.virtual_solver_method = 'fresh_force_ptc'
        except (LoadStepFailure, TimeoutError):
            if getattr(self, 'ffm_virtual_solver', 'legacy') == 'friction_continuation':
                value = self.friction_continued_equilibrium(seed, depth, deadline)
            else:
                value = self.continued_equilibrium(seed, depth, deadline)
        _, g, _, min_j = self.evaluate(value, self.lag(value), tangent=False)
        self.update_constraint_free(value, g)
        norm = np.sqrt(2)*float(np.linalg.norm(g[self.free]))
        if (not np.isfinite([norm, min_j]).all() or norm > 1e-4 or min_j <= 0
                or not self.accepted_chord_is_safe(value)):
            raise LoadStepFailure('Finite-crack virtual force/CCD gate')
        if np.any(value[self.seam[~self.released], 0] != self.config.drop_s_mm):
            raise LoadStepFailure('Virtual intact tie lost')
        return value, g, norm

    def friction_continued_equilibrium(self, seed, depth, deadline):
        """Zero-time solver homotopy, ending at the original virtual equation.

        Only the direction search uses broader friction smoothing. The anchor,
        frozen A/kappa, blade, full CCD and straight energy path stay fixed.
        No intermediate energy is certified or added to the physical ledger.
        """
        friction, maximum = self.friction, self.max_iterations
        value = seed.copy()
        self.virtual_solver_method = 'friction_continuation'
        self.virtual_solver_phases = []
        try:
            for index, factor in enumerate((1000., 100., 10., 1.)):
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError('Virtual friction continuation budget')
                stage_deadline = now+(deadline-now)/(4-index)
                self.friction = (friction if factor == 1. else
                    self.ipc.FrictionPotential(self.virtual_friction_epsilon*factor))
                self.set_primary_increment(value, value-self.previous)
                stage = value.copy()
                phase = dict(factor=factor, accepted=False)
                self.virtual_solver_phases.append(phase)
                try:
                    self.max_iterations = min(maximum, 60)
                    value, iterations = solve(self, stage.copy(), depth,
                        min(stage_deadline, now+1.))
                    phase['method'] = 'newton_lm'
                except (LoadStepFailure, TimeoutError):
                    self.max_iterations = maximum
                    self.set_primary_increment(stage, stage-self.previous)
                    value, iterations = self.solve_friction(stage.copy(), depth, stage_deadline)
                    phase['method'] = 'fresh_force_ptc'
                phase.update(accepted=True, iterations=iterations,
                    residual_n=float(np.sqrt(2)*np.linalg.norm(
                        self.evaluate(value, self.lag(value), tangent=False)[1][self.free])))
        finally:
            self.friction = friction
            self.max_iterations = maximum
        return value

    def continued_equilibrium(self, seed, depth, deadline):
        # A force-merit iteration can stall at a nonzero minimum. Retry by
        # removing a fictitious balancing load in zero-time stages. Neither
        # physical mass nor material history is changed by this continuation.
        maximum = self.max_iterations
        try:
            self.max_iterations = min(maximum, 35)
            self.set_primary_increment(seed, seed-self.previous)
            value, _ = solve(self, seed.copy(), depth, min(deadline, time.monotonic()+2.))
            self.virtual_solver_method = 'direct'
        except (LoadStepFailure, TimeoutError):
            self.virtual_solver_load[:] = 0.
            self.set_primary_increment(seed, seed-self.previous)
            initial = self.evaluate(seed, self.lag(seed), tangent=False)[1]
            load = np.zeros(self.total)
            load[self.base_free] = initial[self.base_free]
            value, fraction, step = seed.copy(), 1., .25
            self.virtual_solver_method = 'load_continuation'
            while fraction > 0.:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Finite-crack virtual continuation budget')
                target = max(0., fraction-step)
                self.virtual_solver_load = target*load
                self.set_primary_increment(value, value-self.previous)
                try:
                    trial, _ = solve(self, value.copy(), depth, min(deadline, time.monotonic()+2.))
                except (LoadStepFailure, TimeoutError):
                    step *= .5
                    if step < 1e-5:
                        raise LoadStepFailure('Finite-crack virtual continuation stalled')
                    continue
                value, fraction = trial, target
                step = min(.5, step*1.5)
        finally:
            self.max_iterations = maximum
            self.virtual_solver_load[:] = 0.
        return value

    def friction_path_work(self, start, end, order):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        delta = end-start
        work = 0.
        for s, weight in zip(.5*(nodes+1), .5*weights):
            point = start+s*delta
            lag = self.lag(point)
            value = 2*float(self.friction.gradient(lag, self.cmesh, delta[self.surface])@delta[self.surface].ravel())
            if value < -1e-12:
                raise LoadStepFailure('Negative virtual Coulomb work')
            work += weight*value
        return float(work)


class FiniteIncrementMixin:
    def __init__(self, *, ffm_virtual_seconds=10., ffm_friction_epsilon_mm=1e-6,
                 ffm_max_candidates=256, ffm_virtual_solver='legacy', **kwargs):
        super().__init__(**kwargs)
        if (not np.isfinite([ffm_virtual_seconds, ffm_friction_epsilon_mm]).all()
                or min(ffm_virtual_seconds, ffm_friction_epsilon_mm) <= 0
                or type(ffm_max_candidates) is not int or ffm_max_candidates < 1):
            raise ValueError('Explicit positive finite virtual budgets and resolution required')
        self.ffm_virtual_seconds = ffm_virtual_seconds
        self.ffm_epsilon = ffm_friction_epsilon_mm
        self.ffm_max_candidates = ffm_max_candidates
        if ffm_virtual_solver not in ('legacy', 'friction_continuation'):
            raise ValueError('Unknown finite virtual solver')
        self.ffm_virtual_solver = ffm_virtual_solver
        self.ffm_pending, self.ffm_checks = [], []
        self.ffm_failure = None

    def step(self, x, old, depth, deadline):
        self.ffm_pending, self.ffm_checks = [], []
        self.ffm_failure = None
        try:
            value, rec = super().step(x, old, depth, deadline)
            rec['finite_increment_checks'] = self.ffm_checks
            rec['fracture_criterion'] = 'frozen_history_finite_increment_CC_CZM'
            return value, rec
        except BaseException:
            self.ffm_pending = []
            raise

    def propose_insertion(self, x, traction, depth, deadline):
        empty = np.zeros_like(self.released)
        # Cheap necessary screening only; eligibility is recomputed in the
        # paired virtual pre-crack equilibrium before ANY release is accepted.
        if not np.any((~self.released) & (traction >= self.strength)):
            return empty
        virtual_deadline = min(deadline, time.monotonic()+self.ffm_virtual_seconds)
        metric = self.material_response(x)['metric'].copy()
        history = self.cohesive_response(x)['history'].copy()
        base = FrozenVirtualCrack(self, x, metric, history, self.released, empty, self.ffm_epsilon)
        try:
            before, gradient, norm = base.equilibrium(x, depth, virtual_deadline)
        except (LoadStepFailure, TimeoutError) as exc:
            self.capture_virtual_failure(base, x, metric, history, empty, depth, 'before', exc)
            raise
        virtual_stress = -gradient.reshape(-1, 2)[self.seam, 0]/self.seam_area
        # Both the physical pre-insertion state and the frozen equilibrated
        # comparator must satisfy strength; a virtual readjustment alone
        # cannot invent a strength crossing in the actual loading state.
        stress = np.minimum(traction, virtual_stress)
        patches = finite_patches(stress, self.released, self.rest[self.seam, 1], self.seam_area, self.strength)
        if len(patches) > self.ffm_max_candidates:
            raise LoadStepFailure('Finite-crack candidate budget incomplete; no intact certification')
        u0 = base.stored(before)
        for ids in patches:
            if time.monotonic() >= virtual_deadline:
                raise TimeoutError('Finite-crack virtual candidate budget')
            removed = empty.copy()
            removed[ids] = True
            post = FrozenVirtualCrack(self, before, metric, history, self.released, removed, self.ffm_epsilon)
            try:
                after, _, postnorm = post.equilibrium(before, depth, virtual_deadline)
            except (LoadStepFailure, TimeoutError) as exc:
                self.capture_virtual_failure(post, x, metric, history, removed, depth, 'after', exc)
                raise
            released_energy = u0-post.stored(after)
            d5 = post.friction_path_work(before, after, 5)
            d9 = post.friction_path_work(before, after, 9)
            uncertainty = 2*abs(d9-d5)
            area = float(self.seam_area[ids].sum())
            accepted, margin = coupled_margin(stress[ids], self.strength, released_energy, d9,
                                               self.toughness*area, uncertainty)
            opened = np.r_[removed[:-1] | removed[1:]]
            existing = self.released[:-1] | self.released[1:]
            support = self.config.width_mm*float(np.diff(self.rest[self.seam, 1])[opened & ~existing].sum())
            check = dict(nodes=self.seam[ids].tolist(), seam_indices=ids.tolist(),
                stress_min_mpa=float(stress[ids].min()), strength_mpa=self.strength,
                physical_stress_min_mpa=float(traction[ids].min()),
                virtual_stress_min_mpa=float(virtual_stress[ids].min()),
                reference_tributary_area_mm2=area, newly_exposed_support_area_mm2=support,
                virtual_stored_before_n_mm=u0, virtual_stored_after_n_mm=post.stored(after),
                virtual_friction_work_n_mm=d9, friction_quadrature_uncertainty_n_mm=uncertainty,
                required_fracture_work_n_mm=self.toughness*area, energy_margin_n_mm=margin,
                virtual_pre_residual_n=norm, virtual_post_residual_n=postnorm, accepted=accepted,
                virtual_pre_solver=base.virtual_solver_method, virtual_post_solver=post.virtual_solver_method,
                virtual_pre_solver_phases=base.virtual_solver_phases,
                virtual_post_solver_phases=post.virtual_solver_phases,
                viscous_metric_frozen=True, virtual_physical_time_advance_s=0.,
                virtual_dissipation_is_physical_ledger_input=False)
            self.ffm_checks.append(check)
            if accepted:
                self.ffm_pending.append(dict(seed_positions=x.copy(), before_positions=before.copy(),
                    seed_increment=self.step_increment(x),
                    after_positions=after.copy(), frozen_metric=metric, frozen_history=history,
                    previous_released=self.released.copy(), removed=removed, depth_mm=depth,
                    pre_anchor=x.copy(), physical_traction=traction.copy(), stress=stress,
                    evidence_index=len(self.ffm_checks)-1))
                return removed
        return empty

    def capture_virtual_failure(self, virtual, seed, metric, history, removed, depth, stage, exc):
        # Capture before the host rolls back the attempted physical release set.
        # The trial is unaccepted; its original equation must be recomputed.
        self.ffm_failure = dict(metadata=dict(stage=stage, error=str(exc),
            seam_indices=np.flatnonzero(removed).tolist(), depth_mm=float(depth),
            solver=getattr(virtual, 'virtual_solver_method', 'fresh_force_ptc'),
            solver_phases=virtual.virtual_solver_phases, candidate_status='unknown'),
            fields=dict(seed_positions=seed.copy(), seed_increment=self.step_increment(seed),
                virtual_anchor=virtual.previous.copy(), virtual_trial=virtual.trial.copy(),
                frozen_metric=metric.copy(), frozen_history=history.copy(),
                previous_released=self.released.copy(), removed=removed.copy(), depth_mm=depth))

    def write_failure_evidence(self, out, attempt_index):
        if self.ffm_failure is None:
            return None
        path = out/f'failed_virtual_{attempt_index:05d}.npz'
        np.savez_compressed(path, **self.ffm_failure['fields'])
        return dict(self.ffm_failure['metadata'], evidence_file=path.name,
            evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def write_step_evidence(self, out, step_index, rec):
        for index, fields in enumerate(self.ffm_pending):
            path = out/f'finite_increment_{step_index:05d}_{index:03d}.npz'
            np.savez_compressed(path, **fields)
            rec['finite_increment_checks'][fields['evidence_index']].update(
                evidence_file=path.name, evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class FiniteIncrementEnergyCutProblem(FiniteIncrementMixin, EnergyCutProblem):
    pass


class FiniteIncrementBackwardCutProblem(FiniteIncrementMixin, ZeroGapCutProblem):
    pass
