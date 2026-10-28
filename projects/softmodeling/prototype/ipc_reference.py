"""Small quasi-static IPC experiment; official contact kernels, local P1 host.

This is NOT PolyFEM and does not restore the retired multiplier/IPM solver.
The host uses energy Newton for frictionless contact and coupled pseudo-transient
force balance with friction, with CCD and a determinant step bound. No fracture
or mesh adaptation.
"""

from pathlib import Path
from dataclasses import replace
import argparse
import contextlib
import importlib
import hashlib
import json
import os
import subprocess
import sys
import time
import shutil
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve, splu, eigsh, LinearOperator, ArpackNoConvergence
from threadpoolctl import threadpool_limits
from run_support import write_json, save_npz, source_hashes, compute_process_policy
from .model_config import FEMConfig, local_to_lab, _rotation_vectors
from .geometry import GeometryOptions, intact_mesh, blade_boundary_vertices,subdivide_blade
from .material import bulk

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).with_name("ipc_baseline.json")
BASELINE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
# Current specimen/blade task; separate from the immutable historical baseline.
# This is an input preset, not a claim that the revised material is validated.
CURRENT_PRESET = dict(case="oblique", depth=2., increment=.01, friction=.35,
    shear_modulus=FEMConfig().shear_modulus_n_mm2, ogden_alpha=2., dhat=.0025,
    threads=2, seconds=0., max_iterations=500, local_spacing=.05,
    surface_spacing=.005, nose_facets=64, inclination=60., tip_radius=.02,
    fine_halfwidth=3., hessian_mode="regularized", predictor="secant",
    adaptive_load=True, track_tip_contact=True)


class LoadStepFailure(RuntimeError):
    """A rejected nonlinear solve that may be retried from accepted history."""


def residual_merit_decrease(norm, trial_norm, directional_slope, alpha):
    """Sufficient decrease of .5*||fresh residual||^2, not of friction energy.

    A shifted nonsymmetric Newton direction need not descend this merit. Check
    its actual derivative r.T @ A @ du and require a meaningful residual drop.
    This is only an additional trial-step rule; it never accepts a load state.
    """
    return (all(np.isfinite(v) for v in (norm, trial_norm, directional_slope, alpha))
            and norm > 0 and 0 <= trial_norm <= .9*norm and alpha > 0
            and directional_slope < 0
            and trial_norm**2 <= norm**2 + 2e-4*alpha*directional_slope)


def residual_armijo_decrease(norm, trial_norm, directional_slope, alpha):
    """Strict fresh-residual Armijo guard without a fixed percentage drop.

    Small CCD-safe steps may decrease less than ten percent. Requiring an
    actual representable decrease also prevents roundoff from accepting an
    unchanged merit when the Armijo right-hand side rounds back to norm**2.
    This is an iteration safeguard, never a replacement equilibrium test.
    """
    return (all(np.isfinite(v) for v in (norm, trial_norm, directional_slope, alpha))
            and norm > 0 and 0 <= trial_norm < norm and alpha > 0
            and directional_slope < 0
            and trial_norm**2 <= norm**2 + 2e-4*alpha*directional_slope)


def toolkit():
    try:
        module = importlib.import_module("ipctk")
    except ModuleNotFoundError:
        sys.path.insert(0, str(ROOT / ".runtime/ipc-python"))
        module = importlib.import_module("ipctk")
    if module.__version__ != BASELINE["toolkit_version"]:
        raise RuntimeError(
            f"Expected ipctk=={BASELINE['toolkit_version']}, found {module.__version__}"
        )
    return module


def determinant_step(F, D):
    """Keep det(F+tD)>0 throughout the straight P1 trajectory, not just its end."""
    c = np.linalg.det(F)
    a = np.linalg.det(D)
    b = (
        F[:, 0, 0] * D[:, 1, 1]
        + D[:, 0, 0] * F[:, 1, 1]
        - F[:, 0, 1] * D[:, 1, 0]
        - D[:, 0, 1] * F[:, 1, 0]
    )
    if np.any(c <= 0):
        raise ValueError("Initial element inversion")
    roots = np.full((len(c), 2), np.inf)
    linear = abs(a) < 1e-20
    np.divide(-c, b, out=roots[:, 0], where=linear & (b < 0))
    disc = b * b - 4 * a * c
    valid = ~linear & (disc >= 0)
    sd = np.sqrt(np.maximum(disc, 0))
    # Stable quadratic formula; avoid cancellation of the small root.
    q = -0.5 * (b + np.copysign(sd, b))
    np.divide(q, a, out=roots[:, 0], where=valid)
    np.divide(c, q, out=roots[:, 1], where=valid & (q != 0))
    roots[(roots <= 0) | (roots > 1)] = np.inf
    minimum = float(roots.min(initial=np.inf))
    return min(1.0, BASELINE["determinant_step_safety"] * minimum)


class IPCProblem:
    def __init__(
        self,
        case="oblique",
        mu=BASELINE["run"]["friction"],
        dhat=BASELINE["run"]["dhat"],
        stiffness=BASELINE["barrier_stiffness_mpa"],
        max_iterations=BASELINE["run"]["max_iterations"],
        *,
        local_spacing=BASELINE["mesh"]["refinement_mm"],
        surface_spacing=BASELINE["mesh"]["surface_spacing_mm"],
        nose_facets=BASELINE["nose_facets"],
        inclination=BASELINE["model"]["inclination_deg"],
        tip_radius=BASELINE["mesh"]["tip_radius_mm"],
        fine_halfwidth=BASELINE["fine_halfwidth_mm"],
        hessian_mode="projected",
        element="p1",
        linear_solver="direct",
        shear_modulus=FEMConfig().shear_modulus_n_mm2,
        ogden_alpha=2.,
        contact_formulation="improved-max",
        predictor="affine",
        friction_globalization="model-error",
        blade_spacing=None,
        blade_thickness=None,
        blade_angle=None,
        coarse_spacing=None,
    ):
        if not np.isfinite(shear_modulus) or shear_modulus <= 0:
            raise ValueError("Shear modulus must be finite and positive (MPa)")
        self.ipc = toolkit()
        self.case = case
        self.dhat = dhat
        self.max_iterations = max_iterations
        if contact_formulation != 'improved-max':
            raise ValueError('The maintained contact formulation is improved-max')
        self.contact_formulation = contact_formulation
        if hessian_mode not in ("projected", "hybrid", "regularized"):
            raise ValueError("Unknown Hessian mode")
        self.hessian_mode = hessian_mode
        if linear_solver not in ('direct','reused-lu'):
            raise ValueError('Unknown linear solver')
        self.linear_solver = linear_solver
        if predictor not in ('affine', 'secant'):
            raise ValueError('Unknown loading predictor')
        self.predictor = predictor
        if friction_globalization not in ('model-error', 'residual-merit'):
            raise ValueError('Unknown friction globalization')
        self.friction_globalization = friction_globalization
        self._accepted_predictor_history = []
        self.stagnation_window = 0  # Enabled only by the adaptive load controller.
        if not all(np.isfinite(v) and v > 0 for v in (local_spacing, surface_spacing)):
            raise ValueError("Mesh spacings must be finite and positive")
        if not isinstance(nose_facets, int) or nose_facets < 4 or nose_facets % 2:
            raise ValueError("Use an even nose facet count >= 4 to retain the lowest tip vertex")
        self.nose_facets = nose_facets
        self.fine_halfwidth = fine_halfwidth
        if not np.isfinite(fine_halfwidth) or fine_halfwidth <= 0:
            raise ValueError("Fine region halfwidth must be finite and positive")
        self.config = replace(FEMConfig(**BASELINE["model"], friction_coefficient=mu),
                              inclination_deg=inclination,
                              shear_modulus_n_mm2=float(shear_modulus),
                              ogden_alpha=float(ogden_alpha))
        geometry = {name: value for name, value in (
            ('blade_thickness_mm', blade_thickness),
            ('blade_tip_angle_deg', blade_angle),
            ('mesh_size_mm', coarse_spacing)) if value is not None}
        if geometry:
            self.config = replace(self.config, **geometry)
        self.config.validate()
        if element != "p1":
            raise ValueError("The maintained cutting model uses P1 triangles")
        if case == "patch":
            self.config = replace(
                self.config,
                length_mm=1.0,
                height_mm=1.0,
                drop_s_mm=0.5,
                inclination_deg=0.0,
                mesh_size_mm=0.2,
                max_indentation_mm=0.3,
            )
        self.options = replace(
            GeometryOptions(**{k: v for k, v in BASELINE["mesh"].items()
                               if k in GeometryOptions.__dataclass_fields__}),
            refinement_mm=local_spacing,
            surface_spacing_mm=surface_spacing,
            tip_radius_mm=tip_radius,
        )
        self.element = "p1"
        self.mesh, _ = intact_mesh(
            self.config, self.options, fine_halfwidth_mm=self.fine_halfwidth
        )
        tri = self.mesh.triangles
        edges = np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        edges = edges[counts == 1]
        self.enrichment = None  # Retained for old P1 observer compatibility.
        self.nodal_volume = None
        self.start = -2 * dhat / max(np.cos(np.deg2rad(self.config.inclination_deg)), 0.1)
        if case == "patch":
            blade = np.array(
                [
                    [-0.2, 1 - self.start],
                    [1.2, 1 - self.start],
                    [1.2, 1.3 - self.start],
                    [-0.2, 1.3 - self.start],
                ]
            )
        else:
            vertices = blade_boundary_vertices(self.config, self.options, self.start)
            angle = np.deg2rad(self.config.blade_tip_angle_deg / 2)
            _, en = _rotation_vectors(self.config.inclination_deg)
            center = (
                local_to_lab(
                    np.array([[self.config.drop_s_mm, self.config.height_mm]]), self.config
                )[0]
                + self.options.tip_radius_mm * en
                - np.array([0.0, self.start])
            )
            theta = np.linspace(-np.pi + angle, -angle, self.nose_facets + 1)
            arc = center + self.options.tip_radius_mm * np.column_stack(
                (np.cos(theta), np.sin(theta))
            )
            blade = np.vstack((vertices[:2], arc, vertices[-2:]))
        if case=='patch':
            if blade_spacing is not None:raise ValueError('Blade subdivision requires a rounded blade')
            knife_edges=np.column_stack((np.arange(len(blade)),np.roll(np.arange(len(blade)),-1)))
        else:
            blade,knife_edges=subdivide_blade(blade,self.nose_facets,blade_spacing)
        self.blade_spacing=blade_spacing
        self.n = self.mesh.node_count
        self.rest = np.vstack((self.mesh.reference_xy, blade))
        self.total = self.rest.size
        knife_edges=knife_edges+self.n
        edges = np.vstack((edges, knife_edges))
        self.surface = np.unique(edges)
        self.surface_dofs = (2 * self.surface[:, None] + np.arange(2)).ravel()
        self.cmesh = self.ipc.CollisionMesh(
            self.rest[self.surface], np.searchsorted(self.surface, edges)
        )
        self._connected_boundary_pairs = {tuple(sorted(map(int, e))) for e in self.cmesh.edges}
        surface_body = int(np.count_nonzero(self.surface < self.n))
        # Native filter: Python callbacks can deadlock when CCD uses TBB workers.
        self.cmesh.can_collide = self.ipc.make_static_obstacle_filter(surface_body)
        if self.ipc.has_intersections(self.cmesh, self.rest[self.surface]):
            raise ValueError("Intersecting initial collision geometry")
        self.barrier = self.ipc.BarrierPotential(dhat, stiffness * self.config.width_mm, True)
        fixed = (2 * self.mesh.bottom_nodes[:, None] + np.arange(2)).ravel()
        if case == "patch":
            fixed = np.unique(np.r_[fixed, 2 * np.arange(self.n)])
        self.free = np.setdiff1d(np.arange(2 * self.n), fixed)
        self.previous = self.rest.copy()
        self.friction = self.ipc.FrictionPotential(BASELINE["friction_smoothing_mm"])
        self.grad = self.mesh.bulk_data[1]
        self.dofs = self.mesh.bulk_data[2]
        self.reference_contact_check = self.check_reference_contact()

    def check_reference_contact(self):
        """Reject contact-induced prestress in this stress-free intact model.

        A nonintersecting boundary can still have adjacent endpoint barriers
        when its sample spacing is smaller than dhat. Check signed native
        gradients, not collision counts or energy offsets. This is an initial
        consistency check, not a mesh-convergence or loaded-state certificate.
        """
        positions = self.rest[self.surface]
        collisions = self.collisions(self.rest)
        gradient = self.barrier.gradient(collisions, self.cmesh, positions).reshape(-1, 2)
        body_force = float(np.linalg.norm(gradient[self.surface < self.n]))
        energy = float(self.barrier(collisions, self.cmesh, positions))
        if not np.isfinite(body_force) or not np.isfinite(energy) or body_force > BASELINE['force_tolerance_n']:
            raise ValueError(
                f'Reference contact force {body_force:.9g} N exceeds the stress-free '
                f'intact-model limit {BASELINE["force_tolerance_n"]:.9g} N; '
                'revise the contact sampling/activation distance before solving. '
                'Nonintersection alone does not exclude contact-induced prestress.')
        return dict(status='passed', body_contact_force_norm_n=body_force,
                    normal_barrier_energy_n_mm=energy,
                    force_limit_n=BASELINE['force_tolerance_n'])

    def collisions(self, x):
        c = self.ipc.NormalCollisions()
        c.use_area_weighting = True
        c.collision_set_type = self.ipc.NormalCollisions.IMPROVED_MAX_APPROX
        c.build(self.cmesh, x[self.surface], self.dhat)
        return c

    def lag(self, x):
        if self.config.friction_coefficient == 0:
            return None
        c = self.collisions(x)
        f = self.ipc.TangentialCollisions()
        f.build(self.cmesh, x[self.surface], c, self.barrier, self.config.friction_coefficient)
        return f

    def material_bulk(self, u, *, with_tangent=True):
        """P1 constitutive assembly; explicit material candidates override this."""
        return bulk(self.mesh, u, self.config, with_tangent=with_tangent)

    def evaluate(self, x, lag=None, tangent=True, projected=True, *, with_metric=False):
        """Optionally assemble exact and PSD tangents from one material evaluation.

        No cache survives this call: changing x, lag or previous cannot reuse a
        stale tangent. The metric is auxiliary; the physical residual is unchanged.
        """
        if with_metric and not tangent:
            raise ValueError('A paired metric requires the tangent')
        u = (x[: self.n] - self.mesh.reference_xy).ravel()
        phase_start = time.perf_counter()
        r, k, dofs, energy, minj = self.material_bulk(u, with_tangent=tangent)
        self._record_phase('material_bulk', phase_start)
        g = np.zeros(self.total)
        np.add.at(g, dofs.ravel(), r.ravel())
        phase_start = time.perf_counter()
        c = self.collisions(x)
        sx = x[self.surface]
        energy += self.barrier(c, self.cmesh, sx)
        cg = self.barrier.gradient(c, self.cmesh, sx)
        if lag is not None:
            delta = (x - self.previous)[self.surface]
            energy += self.friction(lag, self.cmesh, delta)
            cg += self.friction.gradient(lag, self.cmesh, delta)
        g[self.surface_dofs] += cg
        self._record_phase('contact_force', phase_start)
        phase_start = time.perf_counter()
        h = None
        metric = None
        if tangent:
            if projected or with_metric:
                eigen, v = np.linalg.eigh(k)
                kp = np.einsum("eik,ek,ejk->eij", v, np.maximum(eigen, 0), v)
            rows = np.broadcast_to(dofs[:, :, None], k.shape).ravel()
            cols = np.broadcast_to(dofs[:, None, :], k.shape).ravel()
            def assemble(local, clamp):
                projection = self.ipc.PSDProjectionMethod.CLAMP if clamp else self.ipc.PSDProjectionMethod.NONE
                matrix = coo_matrix((local.ravel(),(rows,cols)),shape=(self.total,self.total)).tocsc()
                ch = self.barrier.hessian(c,self.cmesh,sx,projection)
                if lag is not None:
                    ch += self.friction.hessian(lag,self.cmesh,(x-self.previous)[self.surface],projection)
                ch = ch.tocoo()
                return matrix + coo_matrix((ch.data,(self.surface_dofs[ch.row],self.surface_dofs[ch.col])),shape=matrix.shape).tocsc()
            h = assemble(kp if projected else k,projected)
            if with_metric:
                metric = h if projected else assemble(kp,True)
        self._record_phase('tangent_assembly', phase_start)
        if with_metric:
            return float(energy),g,h,minj,metric
        return float(energy), g, h, minj

    def _record_phase(self, name, started):
        """Optional exclusive phase totals within one loading attempt."""
        if getattr(self, 'solver_diagnostics', False):
            totals = self.__dict__.setdefault('phase_seconds', {})
            totals[name] = totals.get(name, 0.) + time.perf_counter()-started

    def checkpoint_data(self):
        return {}

    def safe_step(self, x, direction):
        phase_start = time.perf_counter()
        F = np.eye(2) + np.einsum(
            "eai,eaJ->eiJ", (x[: self.n] - self.mesh.reference_xy)[self.mesh.triangles], self.grad
        )
        D = np.einsum("eai,eaJ->eiJ", direction[: self.n][self.mesh.triangles], self.grad)
        j_fraction = determinant_step(F, D)
        self._record_phase('positive_j', phase_start)
        phase_start = time.perf_counter()
        ccd_fraction = self.ipc.compute_collision_free_stepsize(
            self.cmesh, x[self.surface], (x + direction)[self.surface])
        self._record_phase('ccd', phase_start)
        if getattr(self, 'solver_diagnostics', False):
            self.step_limits = dict(j_fraction=float(j_fraction), ccd_fraction=float(ccd_fraction))
        return min(j_fraction, ccd_fraction)

    def coupled_friction_tangent(self, x, lag, frozen_hessian):
        """Fresh force Jacobian: frozen Hessian plus contact-geometry coupling.

        The toolkit returns force derivatives; evaluate() uses negative forces.
        This generally nonsymmetric matrix must not receive a PSD projection.
        """
        if lag is None:
            return frozen_hessian
        jac = -self.friction.force_jacobian(
            lag, self.cmesh, self.rest[self.surface],
            (x-self.rest)[self.surface], (x-self.previous)[self.surface],
            self.barrier, self.ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS,
        ).tocoo()
        return frozen_hessian + coo_matrix(
            (jac.data, (self.surface_dofs[jac.row], self.surface_dofs[jac.col])),
            shape=frozen_hessian.shape,
        ).tocsc()

    def _ptc_trial(self, x, direction, maximum):
        """Bound the update by blade displacement, positive J and complete CCD."""
        geometric_fraction = self.safe_step(x, direction)
        alpha = min(1., .5*self.options.tip_radius_mm/maximum, geometric_fraction)
        trial = x+alpha*direction
        return trial, alpha, geometric_fraction

    def linear_direction(self, matrix, rhs):
        """Default arithmetic unchanged; explicit candidates may supply a checked backend."""
        return spsolve(matrix, rhs)


    def solve_friction(self, x, depth, deadline):
        """Preconditioned pseudo-transient continuation of fresh force balance.

        The shift is a solver parameter, not physical viscosity. Intermediate
        residuals may grow; only the original unshifted force test accepts a load.
        previous always remains the last accepted loading state.
        """
        initial_shift = .1
        shift = initial_shift
        best_norm, last_progress = np.inf, 0
        model_backtracks = getattr(self, 'ptc_model_backtracks', 0)
        residual_guard = getattr(self, 'ptc_residual_guard', False)
        shift_policy = getattr(self, 'ptc_shift_policy', 'residual_ratio')
        self.ptc_shift_summary = dict(policy=shift_policy, accepted_directions=0,
            backtrack_increases=0, last_shift_used=None, last_shift_next=None,
            last_residual_ratio_next_shift=None, last_model_halvings=None)
        if model_backtracks:
            self.ptc_model_backtrack_summary = dict(max_backtracks=model_backtracks,
                trial_evaluations=0, shortened_accepted_directions=0,
                maximum_halvings_used=0, exhausted_searches=0)
        linear = None
        if self.linear_solver == 'reused-lu':
            from .linear_solve import ReusedLUSolver
            linear = ReusedLUSolver()
        for iteration in range(self.max_iterations+1):
            if time.monotonic() > deadline:
                raise TimeoutError("IPC friction pseudo-transient wall budget")
            self.trial = x.copy()
            phase_start = time.perf_counter()
            lag = self.lag(x)
            self._record_phase('friction_build', phase_start)
            assembly_start = time.perf_counter()
            assembly_cpu_start = time.process_time()
            # Share the exact tangent and PSD metric at the same state.
            _,g,H,_,paired_metric = self.evaluate(x,lag,projected=False,with_metric=True)
            print(f'PERF_ASSEMBLY seconds={time.perf_counter()-assembly_start:.6f} '
                  f'process_cpu_seconds={time.process_time()-assembly_cpu_start:.6f}',flush=True)
            r = g[self.free]
            norm = float(np.linalg.norm(r))
            print(f"FRICTION_PTC depth_mm={depth:.6f} iteration={iteration} "
                  f"fresh_residual_N={norm:.9g} metric_shift={shift:.9g}", flush=True)
            if norm <= BASELINE["force_tolerance_n"]:
                return x, iteration
            if norm < .9*best_norm:
                best_norm, last_progress = norm, iteration
            if self.stagnation_window and iteration-last_progress >= self.stagnation_window:
                raise LoadStepFailure(f"IPC friction pseudo-transient stagnation: depth={depth:g} mm, "
                                      f"residual={norm:.6g} N, progress_reference={best_norm:.6g} N, "
                                      f"window={self.stagnation_window}")
            if iteration == self.max_iterations:
                break
            phase_start = time.perf_counter()
            A = self.coupled_friction_tangent(x, lag, H)[self.free][:, self.free]
            M = paired_metric[self.free][:,self.free]
            M += diags(np.full(len(self.free), BASELINE["hessian_regularization_n_mm"]))
            self._record_phase('coupled_tangent', phase_start)
            for retry in range(18):
                if time.monotonic() > deadline:
                    raise TimeoutError("IPC friction pseudo-transient wall budget")
                solve_start = time.perf_counter()
                if linear is None:
                    du = self.linear_direction(A+shift*M, -r)
                else:
                    du, linear_info = linear.solve(A+shift*M,-r)
                    print('LINEAR '+json.dumps(linear_info,sort_keys=True),flush=True)
                solve_seconds = time.perf_counter()-solve_start
                self._record_phase('linear_solve', solve_start)
                maximum = float(np.max(np.abs(du)))
                if not np.all(np.isfinite(du)) or maximum == 0:
                    shift = max(initial_shift, 2*shift)
                    continue
                direction = np.zeros(self.total)
                direction[self.free] = du
                direction = direction.reshape(-1, 2)
                trial, alpha, geometric_fraction = self._ptc_trial(x, direction, maximum)
                trial_start = time.perf_counter()
                phase_start = time.perf_counter()
                trial_lag = self.lag(trial)
                self._record_phase('friction_build', phase_start)
                gt = self.evaluate(trial, trial_lag, tangent=False)[1][self.free]
                nt = float(np.linalg.norm(gt))
                # Safeguard the local force model, rather than requiring each
                # auxiliary time step to reduce the physical residual monotonically.
                linear_action = A@du
                error = float(np.linalg.norm(gt-r-alpha*linear_action))
                model_accept = error <= .25*max(norm, nt) or nt <= BASELINE["force_tolerance_n"]
                merit_accept = (not model_accept and self.friction_globalization == 'residual-merit'
                                and residual_merit_decrease(norm, nt, float(r@linear_action), alpha))
                accepted = (np.isfinite(nt) and not np.array_equal(trial, x)
                            and (model_accept or merit_accept))
                if residual_guard:
                    guard_slope = float(r@linear_action)
                    guard_accept = (nt <= BASELINE['force_tolerance_n'] or
                        residual_armijo_decrease(norm, nt, guard_slope, alpha))
                    accepted = accepted and guard_accept
                if model_backtracks:
                    original_alpha = alpha
                    halvings = 0

                    def record_model_trial(index, status):
                        print('PTC_MODEL_BACKTRACK '+json.dumps(dict(
                            depth_mm=float(depth), iteration=iteration, retry=retry,
                            backtrack=index, alpha=float(alpha),
                            ccd_j_safe_alpha=float(geometric_fraction),
                            initial_capped_alpha=float(original_alpha),
                            residual_n=norm, trial_residual_n=nt,
                            model_error_n=error, model_limit_n=.25*max(norm, nt),
                            metric_shift=shift, status=status,
                            criterion=('fresh_residual_merit' if merit_accept else 'model_error')
                                if accepted else 'rejected',
                            **(dict(residual_guard_passed=bool(guard_accept),
                                    residual_merit_slope=guard_slope,
                                    residual_merit_limit_n2=norm**2+2e-4*alpha*guard_slope)
                               if residual_guard else {})), sort_keys=True), flush=True)

                    record_model_trial(0, 'accepted' if accepted else 'rejected')
                    if not accepted:
                        # This searches only a prefix of the exact same CCD/J
                        # safe ray; it changes neither the physical load/time nor
                        # the Jacobian, metric or final force-balance criterion.
                        for halvings in range(1, model_backtracks+1):
                            if time.monotonic() > deadline:
                                raise TimeoutError('IPC friction pseudo-transient wall budget')
                            alpha *= .5
                            trial = x+alpha*direction
                            if np.array_equal(trial, x):
                                gt = r.copy();nt = norm
                                error = float(np.linalg.norm(alpha*linear_action))
                                model_accept = merit_accept = accepted = False
                                if residual_guard:
                                    guard_accept = False
                                record_model_trial(halvings, 'no_representable_update')
                                break
                            phase_start = time.perf_counter()
                            trial_lag = self.lag(trial)
                            self._record_phase('friction_build', phase_start)
                            gt = self.evaluate(trial, trial_lag, tangent=False)[1][self.free]
                            self.ptc_model_backtrack_summary['trial_evaluations'] += 1
                            nt = float(np.linalg.norm(gt))
                            error = float(np.linalg.norm(gt-r-alpha*linear_action))
                            model_accept = error <= .25*max(norm, nt) or nt <= BASELINE['force_tolerance_n']
                            merit_accept = (not model_accept and self.friction_globalization == 'residual-merit'
                                and residual_merit_decrease(norm, nt, float(r@linear_action), alpha))
                            accepted = np.isfinite(nt) and (model_accept or merit_accept)
                            if residual_guard:
                                guard_accept = (nt <= BASELINE['force_tolerance_n'] or
                                    residual_armijo_decrease(norm, nt, guard_slope, alpha))
                                accepted = accepted and guard_accept
                            record_model_trial(halvings, 'accepted' if accepted else 'rejected')
                            if accepted:
                                break
                        self.ptc_model_backtrack_summary['maximum_halvings_used'] = max(
                            self.ptc_model_backtrack_summary['maximum_halvings_used'], halvings)
                        if accepted:
                            self.ptc_model_backtrack_summary['shortened_accepted_directions'] += 1
                        else:
                            self.ptc_model_backtrack_summary['exhausted_searches'] += 1
                next_shift = ((max(initial_shift, shift) if merit_accept else
                    max(1e-10, min(1e6, shift*nt/max(norm, 1e-15)/1.2))) if accepted
                    else max(initial_shift, 2*shift))
                residual_ratio_next_shift = next_shift
                if (shift_policy == 'backtrack_aware' and accepted
                        and model_backtracks and halvings > 0):
                    # Extra force-model backtracking diagnoses an inadequate
                    # local direction. CCD/J clipping alone does not trigger
                    # this heuristic; no physical damping/history is added.
                    next_shift = max(next_shift,
                        min(1e6, max(initial_shift, shift*2.**halvings)))
                if accepted:
                    self.ptc_shift_summary['accepted_directions'] += 1
                    self.ptc_shift_summary['backtrack_increases'] += int(next_shift > residual_ratio_next_shift)
                    self.ptc_shift_summary.update(last_shift_used=float(shift),
                        last_shift_next=float(next_shift),
                        last_residual_ratio_next_shift=float(residual_ratio_next_shift),
                        last_model_halvings=halvings if model_backtracks else 0)
                    if shift_policy == 'backtrack_aware':
                        print('PTC_SHIFT_POLICY '+json.dumps(dict(policy=shift_policy,
                            depth_mm=float(depth), iteration=iteration, retry=retry,
                            model_halvings=halvings, shift_used=float(shift),
                            residual_ratio_next_shift=float(residual_ratio_next_shift),
                            shift_next=float(next_shift)), sort_keys=True), flush=True)
                if getattr(self, 'solver_diagnostics', False):
                    detail = dict(depth_mm=float(depth), iteration=iteration, retry=retry,
                        residual_n=norm, trial_residual_n=nt, residual_ratio=nt/norm,
                        predicted_residual_n=float(np.linalg.norm(r+alpha*linear_action)),
                        linear_relative_residual=float(np.linalg.norm((A+shift*M)@du+r)/norm),
                        residual_squared_slope=float(r@linear_action),
                        model_error_n=error, shift_used=shift, shift_next=next_shift,
                        geometric_fraction=float(geometric_fraction),
                        displacement_cap_alpha=float(.5*self.options.tip_radius_mm/maximum),
                        alpha=float(alpha), displacement_inf_mm=float(alpha*maximum),
                        criterion=('fresh_residual_merit' if merit_accept else 'model_error')
                            if accepted else 'rejected',
                        linear_s=solve_seconds, trial_s=time.perf_counter()-trial_start,
                        **(dict(model_backtracks_taken=halvings,
                                alpha_before_model_backtracking=original_alpha) if model_backtracks else {}),
                        **getattr(self,'step_limits',{}))
                    self.ptc_state = dict(positions=x.copy(), direction=direction.copy(),
                        residual=r.copy(), trial_residual=gt.copy(), **detail)
                    print('PTC_DIAGNOSTIC '+json.dumps(detail, sort_keys=True), flush=True)
                if accepted:
                    x = trial
                    self.trial = x.copy()
                    print(f"FRICTION_PTC_UPDATE alpha={alpha:.9g} residual_N={nt:.9g} "
                          f"model_error_N={error:.9g} retries={retry} "
                          f"criterion={'fresh_residual_merit' if merit_accept else 'model_error'}", flush=True)
                    # A good actual decrease does not make an inaccurate local
                    # model trustworthy: retain stabilization for merit steps.
                    shift = next_shift
                    break
                print(f"FRICTION_PTC_REJECT retry={retry} metric_shift={shift:.9g} "
                      f"alpha={alpha:.9g} residual_N={nt:.9g} "
                      f"model_error_N={error:.9g} "
                      f"model_limit_N={.25*max(norm, nt):.9g}", flush=True)
                # Successful Newton-like steps can make shift arbitrarily small.
                # On rejection recover the initial stabilization scale immediately;
                # otherwise a fixed retry budget may expire before reaching it.
                shift = max(initial_shift, 2*shift)
            else:
                raise LoadStepFailure(f"IPC friction pseudo-transient model mismatch: "
                                   f"depth={depth:g} mm, residual={norm:.6g} N")
        raise LoadStepFailure(f"IPC friction pseudo-transient iteration budget: "
                           f"depth={depth:g} mm, residual={norm:.6g} N, limit={self.max_iterations}")

    def acceptance(self, x, depth, iterations, outer_iterations):
        lag = self.lag(x)
        E, g, _, minj = self.evaluate(x, lag, tangent=False)
        norm = float(np.linalg.norm(g[self.free]))
        if not np.isfinite(norm) or norm > BASELINE["force_tolerance_n"]:
            raise LoadStepFailure(f"Unbalanced IPC acceptance: residual={norm:.6g} N")
        c = self.collisions(x)
        distance = float(np.sqrt(c.compute_minimum_distance(self.cmesh, x[self.surface])))
        if distance <= 0 or minj <= 0 or self.ipc.has_intersections(self.cmesh, x[self.surface]):
            raise RuntimeError("Invalid geometry at acceptance")
        dissipation = 0.
        if lag is not None:
            delta = (x-self.previous)[self.surface]
            dissipation = float(self.friction.gradient(lag, self.cmesh, delta) @ delta.ravel())
            if not np.isfinite(dissipation) or dissipation < -1e-12:
                raise RuntimeError("Negative IPC friction incremental work")
        # Accepted-state diagnostics only. They neither modify contact stencils
        # nor discard the nonadjacent self-contact needed during surface folding.
        normal_gradient = self.barrier.gradient(c, self.cmesh, x[self.surface]).reshape(-1, 2)
        normal_fz = float(-normal_gradient[self.surface >= self.n, 1].sum())
        total_fz = float(-g.reshape(-1, 2)[self.n:, 1].sum())
        adjacent = []
        for stencil in c.vv_collisions:
            pair = tuple(sorted((stencil.vertex0_id, stencil.vertex1_id)))
            if pair in self._connected_boundary_pairs and np.all(self.surface[list(pair)] < self.n):
                adjacent.append(stencil)
        adjacent_energy = sum(float(self.barrier(s, x[self.surface[[s.vertex0_id, s.vertex1_id]]].ravel()))
                              for s in adjacent)
        return dict(depth_mm=depth, reaction_fz_n=float(-g.reshape(-1, 2)[self.n:, 1].sum()),
                    reaction_x_n=float(-g.reshape(-1, 2)[self.n:, 0].sum()),
                    residual_n=norm, min_j=minj,
                    minimum_distance_mm=distance if np.isfinite(distance) else None,
                    iterations=iterations, friction_outer_iterations=outer_iterations,
                    friction_globalization=self.friction_globalization if lag is not None else None,
                    friction_work_n_mm=dissipation, energy_n_mm=E,
                    reaction_normal_fz_n=normal_fz, reaction_friction_fz_n=total_fz-normal_fz,
                    contact_diagnostics=dict(adjacent_body_vv_count=len(adjacent),
                        adjacent_body_vv_energy_n_mm=adjacent_energy,
                        meaning='Connected surface endpoint barrier terms, not proof of physical self-contact'))

    def roundoff_refinement(self, x, trial, E, Et, g, gt, lag=None):
        """Resolve an ambiguous energy subtraction only at an equilibrated tiny step.

        The scale bounds ordinary cancellation, not arbitrary energy increases.
        Independent residual, displacement and directional-work gates all apply.
        """
        tol = BASELINE["force_tolerance_n"]
        norm = float(np.linalg.norm(g[self.free]))
        final_norm = float(np.linalg.norm(gt[self.free]))
        delta = trial-x
        scale = max(1., self.config.length_mm, self.config.height_mm)
        energy_scale = (abs(E)+abs(Et)+self.config.shear_modulus_n_mm2
                        *self.config.length_mm*self.config.height_mm*self.config.width_mm)
        resolution = 64*np.finfo(float).eps*energy_scale
        slope = float(g @ delta.ravel())
        if not (np.isfinite(Et) and 0 < norm <= 10*tol
                and final_norm <= min(tol, .5*norm)
                and 0 < np.max(np.abs(delta)) <= np.sqrt(np.finfo(float).eps)*scale
                and abs(Et-E) <= resolution and -resolution <= slope < 0):
            return False
        # Compare two quadratures of dE/ds. The actual representable displacement
        # is used, including cancellation in x + step, not an idealized direction.
        gm = self.evaluate(x+.5*delta, lag, tangent=False)[1]
        trapezoid = float((g+gt) @ delta.ravel()/2)
        work = float((g+4*gm+gt) @ delta.ravel()/6)
        if not (work < 0 and abs(work-trapezoid) <= .1*abs(work)):
            return False
        print(f"ROUND_OFF energy_change={Et-E:.9g} work={work:.9g} "
              f"energy_resolution={resolution:.9g} residual_N={final_norm:.9g}", flush=True)
        return True

    def blade_boundary_roundoff_tolerance(self, depth):
        """Bound comparison roundoff, not physical blade-position tolerance.

        Repeated prescribed increments in old checkpoints accumulate roughly
        1e-12 mm at 3 mm. Allow 128 ulps of the reference-coordinate scale,
        with an explicit 1e-10 mm ceiling. Fixed material constraints are not
        relaxed by this blade-only comparison tolerance.
        """
        scale = max(1., float(np.max(np.abs(self.rest))), abs(float(depth)))
        return min(1e-10, max(1e-12, 128*np.finfo(float).eps*scale))

    def seed_predictor_history(self, accepted_states):
        """Seed with one/two already validated (depth, positions) accepted states.

        This checks representation and prescribed boundaries, not equilibrium:
        callers must supply genuine accepted checkpoints. No friction history is
        changed. Resume calls this only after its independent history audit.
        """
        states = list(accepted_states)
        if not 1 <= len(states) <= 2:
            raise ValueError('Predictor history needs one or two accepted states')
        checked = []
        fixed = np.setdiff1d(np.arange(2*self.n), self.free)
        for depth, positions in states:
            depth = float(depth)
            positions = np.asarray(positions, dtype=float)
            if (not np.isfinite(depth) or depth < 0 or positions.shape != self.rest.shape
                    or not np.isfinite(positions).all()
                    or (checked and depth <= checked[-1][0])):
                raise ValueError('Invalid accepted predictor history')
            blade = self.prescribed_blade_positions(depth)
            if (not np.allclose(positions.ravel()[fixed], self.rest.ravel()[fixed], rtol=0, atol=1e-12)
                    or not np.allclose(positions[self.n:], blade, rtol=0,
                                       atol=self.blade_boundary_roundoff_tolerance(depth))):
                raise ValueError('Predictor history violates prescribed boundaries')
            checked.append((depth, positions.copy()))
        self._accepted_predictor_history = checked

    def _remember_predictor_acceptance(self, x, depth, old):
        # Invoked only after the unchanged physical acceptance check succeeds.
        history = self._accepted_predictor_history
        if history and (depth <= history[-1][0] or abs(history[-1][0]-old) > 1e-12
                        or not np.array_equal(history[-1][1], self.previous)):
            history = []
        self._accepted_predictor_history = (history+[(float(depth), x.copy())])[-2:]

    def prescribed_blade_positions(self, depth):
        """Prescribed boundary hook; the default retains vertical loading."""
        blade = self.rest[self.n:].copy()
        blade[:, 1] -= depth-self.start
        return blade

    def affine_loading_candidate(self, x, old, depth):
        """Default arithmetic unchanged; explicit trajectory studies override."""
        affine = x.copy()
        delta = depth - old
        affine[: self.n, 1] -= delta * self.mesh.local_sn[:, 1] / self.config.height_mm
        affine[self.n :, 1] -= delta
        return affine

    def _loading_predictor(self, x, old, depth):
        affine = self.affine_loading_candidate(x, old, depth)
        delta = depth - old
        self.trial = affine.copy()
        if self.predictor == 'affine':
            if self.safe_step(x, affine-x) < 1.0-1e-10:
                raise LoadStepFailure('Prescribed-motion predictor is not collision free; reduce loading step')
            return affine
        fixed = np.setdiff1d(np.arange(2*self.n), self.free)
        candidates = [(0., affine)]
        history = self._accepted_predictor_history
        if (len(history) == 2 and abs(history[-1][0]-old) <= 1e-12
                and np.array_equal(history[-1][1], x) and delta > 0):
            before_depth, before = history[0]
            secant = x+(delta/(old-before_depth))*(x-before)
            # Only unconstrained material DOFs are extrapolated. The blade and
            # fixed material DOFs retain the exact prescribed affine values.
            for weight in (1., .5, .25, .125):
                candidate = affine+weight*(secant-affine)
                candidate.ravel()[fixed] = self.rest.ravel()[fixed]
                candidate[self.n:] = affine[self.n:]
                candidates.append((weight, candidate))
        best, best_norm, best_weight = None, np.inf, None
        for weight, candidate in candidates:
            record = dict(depth_mm=float(depth), secant_weight=weight)
            try:
                if (not np.isfinite(candidate).all()
                        or not np.allclose(candidate.ravel()[fixed], self.rest.ravel()[fixed], rtol=0, atol=1e-12)
                        or not np.array_equal(candidate[self.n:], affine[self.n:])):
                    raise ValueError('Nonfinite or invalid prescribed candidate')
                fraction = float(self.safe_step(x, candidate-x))
                record['safe_fraction'] = fraction
                if not np.isfinite(fraction) or fraction < 1.0-1e-10:
                    raise ValueError('Candidate path fails CCD/J protection')
                _, gradient, _, minj = self.evaluate(candidate, self.lag(candidate), tangent=False)
                norm = float(np.linalg.norm(gradient[self.free]))
                if (not np.isfinite(minj) or minj <= 0 or not np.isfinite(gradient).all()
                        or not np.isfinite(norm)
                        or self.ipc.has_intersections(self.cmesh, candidate[self.surface])):
                    raise ValueError('Candidate endpoint has invalid force or geometry')
                record.update(status='safe', fresh_residual_N=norm, min_J=float(minj))
                if norm < best_norm:
                    best, best_norm, best_weight = candidate, norm, weight
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                record.update(status='rejected', reason=f'{type(exc).__name__}: {exc}')
            print('PREDICTOR_CANDIDATE '+json.dumps(record, sort_keys=True), flush=True)
        if best is None:
            raise LoadStepFailure(
                "Prescribed-motion predictor is not collision free; reduce loading step"
            )
        print('PREDICTOR_SELECTED '+json.dumps(dict(depth_mm=float(depth), secant_weight=best_weight,
              fresh_residual_N=best_norm, accepted_history_states=len(history)), sort_keys=True), flush=True)
        return best

    def step(self, x, old, depth, deadline):
        self.phase_seconds = {}
        self.ptc_state = {}
        self.previous = x.copy()
        x = self._loading_predictor(x, old, depth)
        self.trial = x.copy()
        if self.config.friction_coefficient > 0:
            x, iterations = self.solve_friction(x, depth, deadline)
            accepted = self.acceptance(x, depth, iterations+1, 0)
            self._remember_predictor_acceptance(x, depth, old)
            return x, accepted
        iterations = 0
        lag = None
        for inner in range(self.max_iterations):
            self.trial = x.copy()
            if time.monotonic() > deadline:
                raise TimeoutError("IPC reference wall budget")
            E, g, H, minj = self.evaluate(x, lag, projected=self.hessian_mode == "projected")
            norm = float(np.linalg.norm(g[self.free]))
            iterations += 1
            print(
                f"ITER depth_mm={depth:.6f} outer=0 inner={inner} force_residual_N={norm:.6g} energy_Nmm={E:.9g} min_J={minj:.6g}",
                flush=True,
            )
            if norm <= BASELINE["force_tolerance_n"]:
                break
            modes = (True,) if self.hessian_mode == "projected" else (False, True)
            for attempt, projected in enumerate(modes):
                if attempt:
                    H = self.evaluate(x, lag, projected=True)[2]
                A = H[self.free][:, self.free]
                A += diags(np.full(len(self.free), BASELINE["hessian_regularization_n_mm"]))
                if self.hessian_mode == "regularized" and not projected:
                    M = self.evaluate(x, lag, projected=True)[2][self.free][:, self.free]
                    M += diags(np.full(len(self.free), BASELINE["hessian_regularization_n_mm"]))
                    try:
                        A, smallest, shift = regularize_curvature(A, M)
                        print(f"CURVATURE lambda_min={smallest:.6g} metric_shift={shift:.6g}", flush=True)
                    except (RuntimeError, ArpackNoConvergence):
                        print("CURVATURE estimate failed; using projected fallback", flush=True)
                        continue
                try:
                    du = spsolve(A, -g[self.free])
                except RuntimeError:
                    if not projected:
                        continue
                    raise
                direction = np.zeros(self.total)
                direction[self.free] = du
                descent = float(g @ direction)
                if not np.isfinite(descent) or descent >= 0:
                    print(f"DIRECTION rejected projected={projected} descent={descent}", flush=True)
                    continue
                direction = direction.reshape(-1, 2)
                alpha = self.safe_step(x, direction)
                for cut in range(BASELINE["max_line_search_iterations"]):
                    trial = x + alpha * direction
                    if self.hessian_mode == "regularized" and np.array_equal(trial, x):
                        # Smaller alpha cannot provide a representable update either.
                        continue
                    try:
                        Et, gt, _, _ = self.evaluate(trial, lag, tangent=False)
                    except FloatingPointError:
                        Et = np.inf
                    armijo = np.isfinite(Et) and Et <= E + BASELINE["armijo_coefficient"] * alpha * descent
                    refined = (not armijo and self.hessian_mode == "regularized"
                               and np.isfinite(Et)
                               and self.roundoff_refinement(x, trial, E, Et, g, gt, lag))
                    if armijo or refined:
                        x = trial
                        self.trial = x.copy()
                        print(f"UPDATE projected={projected} alpha={alpha:.6g} backtracks={cut}", flush=True)
                        break
                    alpha *= 0.5
                else:
                    continue
                break
            else:
                raise LoadStepFailure("IPC Newton directions/energy line search failed")
        else:
            # The last allowed update may already satisfy the original equations.
            final_g = self.evaluate(x, lag, tangent=False)[1]
            final_norm = float(np.linalg.norm(final_g[self.free]))
            if final_norm > BASELINE["force_tolerance_n"]:
                raise LoadStepFailure(
                    f"IPC inner iteration budget: depth={depth:g} mm, "
                    f"residual={final_norm:.6g} N, limit={self.max_iterations}"
                )
        accepted = self.acceptance(x, depth, iterations, 1)
        self._remember_predictor_acceptance(x, depth, old)
        return x, accepted



def adaptive_load_steps(p, x, old, target, deadline, minimum_increment, on_reject=None,
                        *, nominal_target=None, level=0):
    """Yield every accepted substep; never skip a requested output depth.

    Failed trials cannot become a friction or predictor history. A secant solve
    first retries the same physical increment with the affine guess. Only then
    is the load bisected. Substeps are real history updates, so they must all be
    persisted by the caller and compared for loading-discretization sensitivity.
    Global timeouts and unexpected implementation/geometry errors are not hidden.
    """
    if (not np.isfinite(minimum_increment) or minimum_increment <= 0
            or not np.isfinite(old) or not np.isfinite(target) or target <= old):
        raise ValueError('Adaptive loading requires a positive minimum and increasing finite depths')
    nominal_target = target if nominal_target is None else nominal_target
    predictor = p.predictor
    globalization = getattr(p, 'friction_globalization', 'model-error')
    stagnation_window = getattr(p, 'stagnation_window', 0)
    choices = [(predictor, globalization)]
    if predictor == 'secant' or globalization != 'model-error':
        choices.append(('affine', 'model-error'))
    failure = None
    for choice, friction_globalization in choices:
        p.attempted_depth = float(target)
        p.failed_previous = x.copy()
        p.failed_context = dict(from_depth_mm=float(old), attempted_depth_mm=float(target),
            nominal_target_mm=float(nominal_target), subdivision_level=level, predictor=choice,
            friction_globalization=friction_globalization)
        before_previous = p.previous.copy()
        before_history = [(d, state.copy()) for d, state in p._accepted_predictor_history]
        started = time.monotonic()
        try:
            p.predictor = choice
            p.friction_globalization = friction_globalization
            # A temporary plateau also occurs on known-successful affine paths.
            # Early termination applies only to the optional accelerated guess;
            # retain the full original solve budget for the conservative retry.
            p.stagnation_window = stagnation_window if choice == 'secant' else 0
            value, record = p.step(x.copy(), old, target, deadline)
        except LoadStepFailure as exc:
            failure = exc
            event = dict(from_depth_mm=float(old), attempted_depth_mm=float(target),
                         nominal_target_mm=float(nominal_target), subdivision_level=level,
                         predictor=choice, elapsed_s=time.monotonic()-started,
                         friction_globalization=friction_globalization,
                         error=f'{type(exc).__name__}: {exc}',
                         phase_seconds=getattr(p, 'phase_seconds', {}).copy())
            p.previous = before_previous
            p._accepted_predictor_history = before_history
            print('LOAD_REJECT '+json.dumps(event, sort_keys=True), flush=True)
            if on_reject is not None:
                on_reject(event)
        else:
            record['load_step'] = dict(previous_depth_mm=float(old),
                                      increment_mm=float(target-old),
                                      nominal_target_mm=float(nominal_target),
                                      subdivision_level=level, predictor=choice,
                                      friction_globalization=friction_globalization,
                                      elapsed_s=time.monotonic()-started,
                                      phase_seconds=getattr(p, 'phase_seconds', {}).copy())
            p.predictor = predictor
            p.friction_globalization = globalization
            p.stagnation_window = stagnation_window
            yield value, record
            return
        finally:
            p.predictor = predictor
            p.friction_globalization = globalization
            p.stagnation_window = stagnation_window
    # No negative-depth auxiliary states: initial equilibrium is a separate solve.
    midpoint = .5*(old+target)
    if (old < 0 or .5*(target-old) < minimum_increment*(1-1e-12)
            or not old < midpoint < target):
        raise LoadStepFailure(f'IPC adaptive load exhausted: from={old:g} to={target:g} mm, '
                              f'minimum_increment={minimum_increment:g} mm; {failure}') from failure
    print(f'LOAD_CUTBACK from_mm={old:.9g} target_mm={target:.9g} midpoint_mm={midpoint:.9g}',
          flush=True)
    for endpoint in (midpoint, target):
        for value, record in adaptive_load_steps(p, x, old, endpoint, deadline, minimum_increment,
                                                on_reject, nominal_target=nominal_target,
                                                level=level+1):
            x, old = value, record['depth_mm']
            yield value, record


def regularize_curvature(A, metric, floor=.05):
    """Shift the step model in the PN metric; never alter energy or residual.

    ARPACK estimates the smallest generalized eigenvalue. The floor is a
    dimensionless solver parameter, not material viscosity or stiffness.
    Descent, original energy and geometry still independently gate each step.
    """
    factor = splu(metric.tocsc())
    inverse = LinearOperator(metric.shape, matvec=factor.solve)
    values = eigsh(A, k=1, M=metric, Minv=inverse, which="SA", tol=1e-4,
                   maxiter=100, v0=np.random.default_rng(125).normal(size=A.shape[0]),
                   return_eigenvectors=False)
    smallest = float(values[0])
    if not np.isfinite(smallest):
        raise RuntimeError("Non-finite curvature estimate")
    shift = max(0., floor-smallest)
    return A + shift*metric, smallest, shift


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    toolkit().set_num_threads(args.threads)
    p = IPCProblem(
        args.case, args.friction, args.dhat, max_iterations=args.max_iterations,
        local_spacing=args.local_spacing, surface_spacing=args.surface_spacing,
        nose_facets=args.nose_facets,
        inclination=args.inclination, tip_radius=args.tip_radius,
        fine_halfwidth=args.fine_halfwidth,
        hessian_mode=args.hessian_mode,
        element=args.element,
        linear_solver=args.linear_solver,
        shear_modulus=args.shear_modulus,
        ogden_alpha=args.ogden_alpha,
        contact_formulation=args.contact_formulation,
        predictor=args.predictor,
        friction_globalization=args.friction_globalization,
    )
    if args.adaptive_load:
        p.stagnation_window = 40
    tracker = None
    if args.track_tip_contact:
        from .tip_contact import TipContactTracker
        tracker = TipContactTracker(p.rest, p.mesh.triangles, p.n, p.config, p.dhat, p.nose_facets)
    active_config = {
        k: v
        for k, v in p.config.to_dict().items()
        if k
        in (
            "length_mm",
            "height_mm",
            "width_mm",
            "inclination_deg",
            "drop_s_mm",
            "blade_thickness_mm",
            "blade_tip_angle_deg",
            "shear_modulus_n_mm2",
            "ogden_alpha",
            "poisson_ratio",
            "friction_coefficient",
        )
    }
    report = dict(
        status="running",
        completed=False,
        case=args.case,
        target_mm=args.depth,
        config=active_config,
        material_parameter_status=dict(
            experimentally_calibrated=False,
            shear_modulus_role=("current_catalogue_user_adopted"
                                if args.shear_modulus == FEMConfig().shear_modulus_n_mm2 else
                                "baseline_catalogue_initial_value"
                                if args.shear_modulus == BASELINE["model"]["shear_modulus_n_mm2"] else
                                "diagnostic_or_calibration_candidate"),
            baseline_shear_modulus_mpa=BASELINE["model"]["shear_modulus_n_mm2"],
            ogden_alpha_role=("neo_hookean_reference" if args.ogden_alpha == 2. else
                              "uncalibrated_shape_candidate"),
            ogden_coefficient_mpa=2*args.shear_modulus/args.ogden_alpha,
            bulk_modulus_mpa=p.config.bulk_modulus_n_mm2,
            poisson_ratio_fixed=True,
            barrier_stiffness_scales_with_shear_modulus=False,
        ),
        simplifications=dict(
            displacement_element="P1, fixed mesh",
            tip_radius_mm=p.options.tip_radius_mm,
            nose_facets=p.nose_facets,
            fracture=False,
            material="existing finite-strain plane-strain elastic law",
            host=("local IPC finite-element harness, NOT PolyFEM" if args.friction else
                  "local projected-Newton harness, NOT PolyFEM"),
            friction_smoothing_mm=BASELINE["friction_smoothing_mm"],
        ),
        mesh_settings=dict(
            bulk_spacing_mm=p.config.mesh_size_mm,
            local_spacing_mm=p.options.refinement_mm,
            surface_spacing_mm=p.options.surface_spacing_mm,
            fine_halfwidth_mm=p.fine_halfwidth,
        ),
        baseline_id=BASELINE["id"] if args.baseline else None,
        contact_formulation=args.contact_formulation,
        contact_formulation_status="intact_reference_discretization",
        input_preset="current_20261011" if args.current else None,
        blade_parameter_status=dict(model="99755", thickness_and_full_angle="user_specification_table",
            radius="user_specified_not_measured", bevel_profile="single_straight_bevel_assumption",
            friction="unmeasured_interface_assumption"),
        baseline_config_sha256=hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest(),
        ipc_version=p.ipc.__version__,
        dhat_mm=args.dhat,
        reference_contact_check=p.reference_contact_check,
        barrier_stiffness_mpa=BASELINE["barrier_stiffness_mpa"],
        ipc_binary_sha256=hashlib.sha256(Path(p.ipc.__file__).read_bytes()).hexdigest(),
        loading_increment_mm=args.increment,
        wall_budget_s=args.seconds,
        wall_time_limit_enabled=args.seconds > 0,
        force_tolerance_n=BASELINE["force_tolerance_n"],
        max_inner_iterations=args.max_iterations,
        friction_solver="coupled_pseudo_transient" if args.friction else None,
        friction_globalization=args.friction_globalization,
        friction_merit_settings=(dict(armijo_coefficient=1e-4, residual_ratio_limit=.9,
                                      requires_negative_true_directional_slope=True,
                                      accepted_shift='retain_at_least_initial',
                                      adaptive_fallback='affine/model-error' if args.adaptive_load else None)
                                 if args.friction_globalization == 'residual-merit' else None),
        friction_solver_settings=(dict(initial_metric_shift=.1, model_error_fraction=.25,
                                       pseudo_time_growth=1.2, max_tip_radius_fraction=.5,
                                       max_model_retries=18,
                                       rejection_metric_shift_floor=.1) if args.friction else None),
        hessian_mode=args.hessian_mode,
        curvature_settings=(dict(metric="projected Hessian", eigenvalue_floor=.05,
                                 eigenvalue_tolerance=1e-4, eigensolver_max_iterations=100)
                            if args.hessian_mode == "regularized" and args.friction == 0 else None),
        line_search=(("pseudo_transient_model_error_or_residual_merit"
                      if args.friction_globalization == 'residual-merit' else
                      "pseudo_transient_model_error") if args.friction else
                     "armijo_with_roundoff_refinement" if args.hessian_mode == "regularized"
                     else "armijo"),
        source_sha256=source_hashes(),
        nodes=p.n,
        elements=len(p.mesh.triangles),
        threads=args.threads,
        process_policy=args.process_policy,
        pid=os.getpid(),
        steps=[],
        linear_solver=args.linear_solver,
        predictor=args.predictor,
        predictor_settings=(dict(secant_weights=[1., .5, .25, .125],
                                 selection='minimum fresh residual including affine',
                                 safety='full old-to-candidate CCD and determinant path; finite endpoint and fixed boundaries',
                                 history='last two accepted states; no trial history')
                            if args.predictor == 'secant' else None),
    )
    if args.adaptive_load:
        report['load_control'] = dict(mode='adaptive_bisection',
            minimum_increment_mm=args.minimum_increment,
            maximum_increment_mm=args.increment, output_grid_increment_mm=args.increment,
            retry_predictor='affine', stagnation_window=40, stagnation_improvement_fraction=.1)
        report['load_control']['stagnation_scope'] = 'secant_only; affine retains original iteration budget'
        report['rejected_load_attempts'] = []
    if tracker is not None:
        report["tip_contact_tracking"] = dict(tracker.metadata, status="running", errors=[])
    shutil.copyfile(BASELINE_PATH, out / "baseline_config.json")
    visual_source = ROOT / "tools/visualize_ipc.py"
    report["source_sha256"]["tools/visualize_ipc.py"] = hashlib.sha256(visual_source.read_bytes()).hexdigest()
    x = p.rest.copy()
    old = p.start
    start = time.monotonic()
    (out / "states").mkdir()
    targets = np.r_[0.0, np.arange(args.increment, args.depth - 1e-12, args.increment), args.depth]
    if args.resume is not None:
        x, old = restore_accepted_prefix(args.resume, out, report, p, targets)
        targets = targets[targets > old + 1e-12]
        if tracker is not None:
            for index, step in enumerate(report["steps"]):
                with np.load(out / "states" / f"{index:04d}.npz", allow_pickle=False) as state:
                    record_tip_contact(tracker, state["positions"], step, report)
            export_contact_table(out, report)
    write_json(out / "result.json", report)
    with (
        (out / "run.log").open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        threadpool_limits(limits=args.threads),
        compute_process_policy(args.process_policy) as runtime_policy,
    ):
        from threadpoolctl import threadpool_info
        report['runtime_threadpools'] = threadpool_info()
        report['runtime_process_policy'] = runtime_policy
        print('RUNTIME_POLICY '+json.dumps(runtime_policy, sort_keys=True), flush=True)
        write_json(out / 'result.json', report)
        try:
            for depth in targets:
                deadline = start+args.seconds if args.seconds > 0 else float("inf")
                def rejected(event):
                    report['rejected_load_attempts'].append(event)
                    write_json(out / 'result.json', report)
                p.attempted_depth = float(depth)
                accepted = (adaptive_load_steps(p, x, old, float(depth), deadline,
                                               args.minimum_increment, rejected)
                            if args.adaptive_load else [p.step(x, old, float(depth), deadline)])
                for x, d in accepted:
                    accepted_depth = float(d['depth_mm'])
                    checkpoint = out / "states" / f'{len(report["steps"]):04d}.npz'
                    save_npz(
                        checkpoint,
                        reference_xy=p.rest,
                        triangles=p.mesh.triangles,
                        positions=x,
                        previous_positions=p.previous,
                        depth_mm=accepted_depth,
                        body_nodes=p.n,
                        **p.checkpoint_data(),
                    )
                    report["steps"].append(d)
                    old = accepted_depth
                    report["last_checkpoint"] = checkpoint.relative_to(out).as_posix()
                    report["last_depth_mm"] = old
                    if tracker is not None:
                        record_tip_contact(tracker, x, d, report)
                        export_contact_table(out, report)
                    write_json(out / "result.json", report)
                    print(
                        f'ACCEPT depth_mm={old:.6f} Fz_N={d["reaction_fz_n"]:.6g} residual_N={d["residual_n"]:.3g} gap_mm={d["minimum_distance_mm"]}',
                        flush=True,
                    )
            report.update(status="completed", completed=True)
        except Exception as e:
            save_npz(
                out / "failed_trial.npz",
                reference_xy=p.rest,
                triangles=p.mesh.triangles,
                positions=getattr(p, "trial", x),
                previous_positions=getattr(p, 'failed_previous', p.previous),
                depth_mm=p.attempted_depth,
                body_nodes=p.n,
                **p.checkpoint_data(),
            )
            report.update(
                status="failed", attempted_depth_mm=float(p.attempted_depth), error=f"{type(e).__name__}: {e}"
            )
            print(report["error"], flush=True)
    report["elapsed_s"] = time.monotonic() - start
    if tracker is not None:
        tracking = report["tip_contact_tracking"]
        tracking["status"] = "partial" if tracking["errors"] else "accepted_states_tracked"
    write_json(out / "result.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "last_depth_mm", "elapsed_s", "error")}))
    if report["completed"] and args.case == "oblique" and not args.no_visualize:
        export_visualization(out, report)
    return report["completed"]


def record_tip_contact(tracker, positions, step, report):
    """Read-only observation errors must not relabel a valid mechanical state."""
    try:
        step["tip_contact"] = tracker.observe(positions, step["depth_mm"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        step["tip_contact"] = dict(status="error", depth_mm=step["depth_mm"], error=error)
        report["tip_contact_tracking"]["errors"].append(dict(depth_mm=step["depth_mm"], error=error))
    try:
        step['blade_contact_footprint'] = tracker.observe_footprint(positions, step['depth_mm'])
    except Exception as exc:
        error = f'Whole-blade footprint {type(exc).__name__}: {exc}'
        step['blade_contact_footprint'] = dict(status='error', error=error)
        report['tip_contact_tracking']['errors'].append(dict(depth_mm=step['depth_mm'], error=error))
    contact = step["tip_contact"]
    print(f'TIP_CONTACT depth_mm={step["depth_mm"]:.6f} status={contact["status"]} '
          f'offset_mm={contact.get("reference_offset_mm")} gap_mm={contact.get("gap_mm")} '
          f'scope=lowest_blade_point nose_status={contact.get("nose_status", "not_recorded")} '
          f'offset_kind=reference_material_coordinate_not_accumulated_slip', flush=True)


def export_contact_table(out, report):
    from .contact_trace import write_trace_csv
    try:
        write_trace_csv(out / "contact_trace.csv", report["steps"])
    except Exception as exc:
        error = f"CSV export {type(exc).__name__}: {exc}"
        report["tip_contact_tracking"]["errors"].append(dict(error=error))
        print(error, flush=True)


def restore_accepted_prefix(source, out, report, p, targets):
    """Resume only accepted physical history, retaining its distinct provenance."""
    previous = json.loads((source / "result.json").read_text(encoding="utf-8"))
    if previous.get('contact_formulation', 'improved-max') != report['contact_formulation']:
        raise ValueError('Resume mismatch: contact_formulation')
    for key in ("case", "config", "mesh_settings", "simplifications", "dhat_mm",
                "barrier_stiffness_mpa", "ipc_binary_sha256", "loading_increment_mm",
                "force_tolerance_n", "baseline_config_sha256"):
        if previous[key] != report[key]:
            raise ValueError(f"Resume mismatch: {key}")
    if previous.get('discretization') != report.get('discretization'):
        raise ValueError('Resume mismatch: displacement discretization')
    if p.config.friction_coefficient > 0:
        if previous.get("friction_solver") != report.get("friction_solver"):
            raise ValueError("Resume mismatch: friction_solver")
        before = dict(previous.get("friction_solver_settings") or {})
        after = dict(report.get("friction_solver_settings") or {})
        # The rejection recovery changes only auxiliary iterations, not the
        # physical equation, slip history or acceptance of saved states.
        old_floor = before.pop("rejection_metric_shift_floor", None)
        new_floor = after.pop("rejection_metric_shift_floor", None)
        if before != after or (old_floor != new_floor and (old_floor, new_floor) != (None, .1)):
            raise ValueError("Resume mismatch: friction_solver_settings")
    if set(previous["source_sha256"]) != set(report["source_sha256"]):
        raise ValueError("Resume source inventory differs")
    for name, fingerprint in previous["source_sha256"].items():
        # Runtime scheduling/output helpers do not change the physical kernel;
        # the accepted prefix is independently rechecked below before reuse.
        if name not in ("tools/visualize_ipc.py", "prototype/contact_trace.py",
                        "run_support.py"):
            if report["source_sha256"].get(name) != fingerprint:
                raise ValueError(f"Resume physics source changed: {name}")
    steps = previous["steps"]
    if not steps or steps[-1]['depth_mm'] >= targets[-1]-1e-12:
        raise ValueError("Resume requires a nonempty accepted prefix shorter than target")
    from .contact_trace import validate_loading_schedule
    validate_loading_schedule(previous)
    old_mode = (previous.get('load_control') or {}).get('mode', 'fixed')
    new_mode = (report.get('load_control') or {}).get('mode', 'fixed')
    if old_mode == 'adaptive_bisection' and new_mode != 'adaptive_bisection':
        raise ValueError('An adaptive history requires the adaptive loading controller on resume')
    if new_mode != 'adaptive_bisection':
        np.testing.assert_allclose([s['depth_mm'] for s in steps], targets[:len(steps)],
                                   rtol=0, atol=1e-12)
    prior_x = None
    fingerprints = {}
    fixed_body = np.setdiff1d(np.arange(2*p.n), p.free)
    for i, step in enumerate(steps):
        required = ('depth_mm', 'residual_n', 'min_j', 'minimum_distance_mm',
                    'reaction_fz_n', 'friction_work_n_mm')
        if (any(step.get(k) is None or not np.isfinite(step[k]) for k in required)
                or not 0 <= step['residual_n'] <= report['force_tolerance_n']
                or step['min_j'] <= 0 or step['minimum_distance_mm'] <= 0
                or step['friction_work_n_mm'] < -1e-12):
            raise ValueError("Resume prefix includes an unaccepted state")
        relative = Path("states") / f"{i:04d}.npz"
        with np.load(source / relative) as state:
            np.testing.assert_array_equal(state["reference_xy"], p.rest)
            np.testing.assert_array_equal(state["triangles"], p.mesh.triangles)
            np.testing.assert_array_equal(state['body_nodes'], p.n)
            for key,value in p.checkpoint_data().items():
                np.testing.assert_array_equal(state[key],value)
            np.testing.assert_allclose(float(state["depth_mm"]), step["depth_mm"], rtol=0, atol=1e-12)
            if prior_x is not None:
                np.testing.assert_array_equal(state["previous_positions"], prior_x)
            else:
                np.testing.assert_array_equal(state["previous_positions"], p.rest)
            last_previous = state["previous_positions"].copy()
            x = state["positions"].copy()
            if x.shape != p.rest.shape or not np.isfinite(x).all():
                raise ValueError("Invalid resume positions")
            np.testing.assert_allclose(x.ravel()[fixed_body], p.rest.ravel()[fixed_body],
                                       rtol=0, atol=1e-12)
            blade = p.rest[p.n:].copy()
            blade[:, 1] -= step['depth_mm']-p.start
            np.testing.assert_allclose(x[p.n:], blade, rtol=0,
                                       atol=p.blade_boundary_roundoff_tolerance(step['depth_mm']))
            prior_x = x
        fingerprints[relative.as_posix()] = hashlib.sha256((source / relative).read_bytes()).hexdigest()
    # Friction must be reassembled using the slip of the last accepted increment.
    # Setting previous=x would erase this slip and test a different equilibrium.
    p.previous = last_previous
    _, g, _, minj = p.evaluate(x, p.lag(x), tangent=False)
    norm = float(np.linalg.norm(g[p.free]))
    if (not np.isfinite(norm) or norm > report["force_tolerance_n"]
            or not np.isfinite(minj) or minj <= 0
            or p.ipc.has_intersections(p.cmesh, x[p.surface])):
        raise ValueError("Last accepted state failed independent equilibrium/geometry check")
    for relative in fingerprints:
        shutil.copyfile(source / relative, out / relative)
    shutil.copyfile(source / "result.json", out / "resume_source_result.json")
    report.update(steps=steps, last_depth_mm=steps[-1]["depth_mm"], last_checkpoint=relative)
    report["resume"] = dict(source=str(source.resolve()), accepted_states=len(steps),
                          source_result_sha256=hashlib.sha256((source / "result.json").read_bytes()).hexdigest(),
                          checkpoints_sha256=fingerprints,
                          previous_friction_solver_settings=previous.get("friction_solver_settings"),
                          previous_hessian_mode=previous.get("hessian_mode", "projected"))
    report['resume']['previous_linear_solver'] = previous.get('linear_solver','direct')
    report['resume']['previous_predictor'] = previous.get('predictor', 'affine')
    report['resume']['previous_predictor_settings'] = previous.get('predictor_settings')
    report['resume']['previous_load_control'] = previous.get('load_control')
    # Globalization changes auxiliary iterations, not the physical equation.
    # Preserve its provenance for the copied prefix; the last accepted state
    # has just been independently checked with its actual friction history.
    report['resume']['previous_friction_globalization'] = previous.get('friction_globalization', 'model-error')
    report['resume']['previous_friction_merit_settings'] = previous.get('friction_merit_settings')
    report['resume']['previous_process_policy'] = previous.get('process_policy', 'system')
    # last_previous was verified against the penultimate accepted checkpoint.
    # A single accepted state has an unloaded predecessor, not a second accepted state.
    history = ([(steps[-2]['depth_mm'], last_previous)] if len(steps) >= 2 else [])
    p.seed_predictor_history(history+[(steps[-1]['depth_mm'], x)])
    return x, steps[-1]["depth_mm"]


def export_visualization(out, report):
    """Postprocessing has its own status and cannot change mechanical acceptance."""
    report["visualization"] = dict(status="running", directory="visualization")
    write_json(out / "result.json", report)
    try:
        # Record current postprocessor provenance; no duplicate source tree is retained.
        with (out / "visualization.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-u", str(ROOT / "tools/visualize_ipc.py"),
                 "--source", str(out), "--output", str(out / "visualization")],
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        report["visualization"]["status"] = "completed"
    except Exception as exc:
        report["visualization"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_json(out / "result.json", report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the elastic P1 reference preset to 2 mm; reject parameter changes",
    )
    parser.add_argument("--current", action="store_true",
                        help="Current fine-mesh intact 60-degree preset, G=0.47 MPa and 0.02 mm tip; explicit options override it")
    parser.add_argument("--case", choices=("patch", "oblique"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=float)
    parser.add_argument("--increment", type=float)
    parser.add_argument("--friction", type=float)
    parser.add_argument("--shear-modulus", type=float,
                        help="Small-strain shear modulus in MPa; current catalogue default 0.47, fixed Poisson ratio and barrier stiffness")
    parser.add_argument("--ogden-alpha", type=float,
                        help="One-term Ogden shape at fixed small-strain G; default 2 (neo-Hookean), other values uncalibrated")
    parser.add_argument('--contact-formulation', choices=('improved-max',),
                        help='Complete native improved-max contact and self-contact')
    parser.add_argument('--friction-globalization', choices=('model-error', 'residual-merit'),
                        help='Optional fresh-residual sufficient-decrease fallback for rejected PTC trials; unchanged final force tolerance')
    parser.add_argument('--process-policy', choices=('system', 'compute'),
                        help='compute requests Windows process HighQoS during solving and restores it afterwards; no-op on other OSes')
    parser.add_argument("--dhat", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seconds", type=float, help="Wall-clock budget; 0 disables the time limit")
    parser.add_argument("--max-iterations", type=int,
                        help="Newton limit without friction; pseudo-transient update limit with friction")
    parser.add_argument("--local-spacing", type=float, help="Local body mesh size in mm")
    parser.add_argument("--surface-spacing", type=float, help="Contact surface mesh size in mm")
    parser.add_argument("--nose-facets", type=int, help="Even circular nose segment count")
    parser.add_argument("--inclination", type=float, help="Workpiece angle to horizontal in degrees")
    parser.add_argument("--tip-radius", type=float, help="Blade nose radius in mm; not inferred from thickness")
    parser.add_argument("--fine-halfwidth", type=float, help="Fine surface region halfwidth in reference s, mm")
    parser.add_argument("--no-visualize", action="store_true", help="Skip automatic postprocessing")
    parser.add_argument("--track-tip-contact", action=argparse.BooleanOptionalAction, default=None,
                        help="Record rounded-tip contact in undeformed material surface coordinates")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--element", choices=("p1",), default="p1")
    parser.add_argument('--linear-solver',choices=('direct','reused-lu'),default='direct',
                        help='Reused LU preconditions new linear systems; exact residual check and direct fallback')
    parser.add_argument('--predictor', choices=('affine', 'secant'),
                        help='Optional accepted-history secant initial guess, screened by full safety and fresh residual')
    parser.add_argument('--adaptive-load', action=argparse.BooleanOptionalAction, default=None,
                        help='Retry rejected steps with affine prediction, then bisect; preserve every accepted substep')
    parser.add_argument('--minimum-increment', type=float, default=.0003125,
                        help='Smallest failure-cutback increment in mm; used only with --adaptive-load')
    parser.add_argument("--hessian-mode", choices=("projected", "hybrid", "regularized"),
                        help="Frictionless Hessian mode; friction uses the coupled pseudo-transient solver")
    parser.add_argument("--resume", type=Path, help="Copy and verify accepted prefix from an earlier run")
    args = parser.parse_args(argv)
    if args.current and args.baseline:
        parser.error('--current and --baseline are distinct presets')
    if args.baseline and args.linear_solver!='direct':
        parser.error('Frozen baseline uses the direct linear solver')
    if args.baseline and args.predictor not in (None, 'affine'):
        parser.error('Frozen baseline uses the affine loading predictor')
    if args.baseline and args.adaptive_load:
        parser.error('Frozen baseline uses fixed loading increments')
    defaults = BASELINE["run"] if args.baseline else dict(BASELINE["run"], depth=0.5, seconds=180.0)
    defaults = dict(defaults, local_spacing=BASELINE["mesh"]["refinement_mm"],
                    surface_spacing=BASELINE["mesh"]["surface_spacing_mm"],
                    nose_facets=BASELINE["nose_facets"],
                    inclination=BASELINE["model"]["inclination_deg"],
                    shear_modulus=(BASELINE["model"]["shear_modulus_n_mm2"] if args.baseline
                                   else FEMConfig().shear_modulus_n_mm2), ogden_alpha=2.,
                    tip_radius=BASELINE["mesh"]["tip_radius_mm"],
                    fine_halfwidth=BASELINE["fine_halfwidth_mm"], hessian_mode="projected",
                    predictor='affine', adaptive_load=False, track_tip_contact=False)
    defaults['contact_formulation'] = 'improved-max'
    defaults['friction_globalization'] = 'model-error'
    defaults['process_policy'] = 'system' if args.baseline else 'compute'
    if args.current:
        defaults.update(CURRENT_PRESET)
    for name, value in defaults.items():
        supplied = getattr(args, name)
        if args.baseline and supplied is not None and supplied != value:
            parser.error(
                f"--baseline fixes {name}={value}; omit --baseline for a separate diagnostic"
            )
        if supplied is None:
            setattr(args, name, value)
    if args.adaptive_load and (not np.isfinite(args.minimum_increment)
                              or not 0 < args.minimum_increment <= args.increment):
        parser.error('Adaptive minimum increment must be finite, positive and no larger than the nominal increment')
    if (
        not all(
            np.isfinite(v) and v > 0 for v in (args.depth, args.increment, args.dhat)
        )
        or not np.isfinite(args.seconds) or args.seconds < 0
        or args.depth > 2
        or not np.isfinite(args.friction)
        or args.friction < 0
        or not np.isfinite(args.shear_modulus) or args.shear_modulus <= 0
        or not np.isfinite(args.ogden_alpha) or args.ogden_alpha <= 0
        or args.threads < 1
        or args.max_iterations < 1
        or not all(np.isfinite(v) and v > 0 for v in (args.local_spacing, args.surface_spacing))
        or args.nose_facets < 4
        or args.nose_facets % 2
        or not np.isfinite(args.inclination) or not 0 <= args.inclination < 90
        or not np.isfinite(args.tip_radius) or not 0 < args.tip_radius < BASELINE["model"]["blade_thickness_mm"] / 2
        or not np.isfinite(args.fine_halfwidth) or args.fine_halfwidth <= 0
    ):
        parser.error("Invalid experiment or budget")
    if args.case == "patch" and args.depth > 0.3:
        parser.error("Patch depth must not exceed 0.3 mm")
    if args.case != "oblique" and args.track_tip_contact:
        parser.error("Tip contact tracking requires the rounded oblique blade")
    if args.resume is not None and args.baseline:
        parser.error("Resume requires a diagnostic variant, not the frozen baseline")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty")
    return args


def main():
    args = parse_args()
    if args.detach:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            "prototype.ipc_reference",
            *[x for x in sys.argv[1:] if x != "--detach"],
        ]
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[key] = str(args.threads)
        startup = None
        flags = 0
        if os.name == "nt":
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
            flags = subprocess.CREATE_NO_WINDOW
        with (
            args.output.with_suffix(".stdout.log").open("wb") as log,
            args.output.with_suffix(".stderr.log").open("wb") as error,
        ):
            p = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=error,
                startupinfo=startup,
                creationflags=flags,
                start_new_session=os.name != "nt",
            )
        write_json(
            args.output.with_suffix(".launch.json"),
            dict(pid=p.pid, command=command, threads=args.threads),
        )
        print(p.pid)
    else:
        sys.exit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
