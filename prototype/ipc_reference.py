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
from run_support import write_json, save_npz, source_hashes
from .model_config import FEMConfig, local_to_lab, _rotation_vectors
from .initiation import InitiationOptions, intact_mesh, _blade_boundary_vertices
from .material import bulk

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).with_name("ipc_baseline.json")
BASELINE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


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
    ):
        self.ipc = toolkit()
        self.case = case
        self.dhat = dhat
        self.max_iterations = max_iterations
        if hessian_mode not in ("projected", "hybrid", "regularized"):
            raise ValueError("Unknown Hessian mode")
        self.hessian_mode = hessian_mode
        if not all(np.isfinite(v) and v > 0 for v in (local_spacing, surface_spacing)):
            raise ValueError("Mesh spacings must be finite and positive")
        if not isinstance(nose_facets, int) or nose_facets < 4 or nose_facets % 2:
            raise ValueError("Use an even nose facet count >= 4 to retain the lowest tip vertex")
        self.nose_facets = nose_facets
        self.fine_halfwidth = fine_halfwidth
        if not np.isfinite(fine_halfwidth) or fine_halfwidth <= 0:
            raise ValueError("Fine region halfwidth must be finite and positive")
        self.config = replace(FEMConfig(**BASELINE["model"], friction_coefficient=mu),
                              inclination_deg=inclination)
        self.config.validate()
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
            InitiationOptions(**BASELINE["mesh"]),
            refinement_mm=local_spacing,
            surface_spacing_mm=surface_spacing,
            tip_radius_mm=tip_radius,
        )
        self.mesh, _ = intact_mesh(
            self.config, self.options, fine_halfwidth_mm=self.fine_halfwidth
        )
        tri = self.mesh.triangles
        edges = np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        edges = edges[counts == 1]
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
            vertices = _blade_boundary_vertices(self.config, self.options, self.start)
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
        self.n = self.mesh.node_count
        self.rest = np.vstack((self.mesh.reference_xy, blade))
        self.total = self.rest.size
        knife_edges = (
            np.column_stack((np.arange(len(blade)), np.roll(np.arange(len(blade)), -1))) + self.n
        )
        edges = np.vstack((edges, knife_edges))
        self.surface = np.unique(edges)
        self.surface_dofs = (2 * self.surface[:, None] + np.arange(2)).ravel()
        self.cmesh = self.ipc.CollisionMesh(
            self.rest[self.surface], np.searchsorted(self.surface, edges)
        )
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

    def evaluate(self, x, lag=None, tangent=True, projected=True):
        u = (x[: self.n] - self.mesh.reference_xy).ravel()
        r, k, dofs, energy, minj = bulk(self.mesh, u, self.config, with_tangent=tangent)
        g = np.zeros(self.total)
        np.add.at(g, dofs.ravel(), r.ravel())
        c = self.collisions(x)
        sx = x[self.surface]
        energy += self.barrier(c, self.cmesh, sx)
        cg = self.barrier.gradient(c, self.cmesh, sx)
        if lag is not None:
            delta = (x - self.previous)[self.surface]
            energy += self.friction(lag, self.cmesh, delta)
            cg += self.friction.gradient(lag, self.cmesh, delta)
        g[self.surface_dofs] += cg
        h = None
        if tangent:
            if projected:
                eigen, v = np.linalg.eigh(k)
                k = np.einsum("eik,ek,ejk->eij", v, np.maximum(eigen, 0), v)
            projection = self.ipc.PSDProjectionMethod.CLAMP if projected else self.ipc.PSDProjectionMethod.NONE
            rows = np.broadcast_to(dofs[:, :, None], k.shape).ravel()
            cols = np.broadcast_to(dofs[:, None, :], k.shape).ravel()
            h = coo_matrix((k.ravel(), (rows, cols)), shape=(self.total, self.total)).tocsc()
            ch = self.barrier.hessian(c, self.cmesh, sx, projection)
            if lag is not None:
                ch += self.friction.hessian(
                    lag,
                    self.cmesh,
                    (x - self.previous)[self.surface],
                    projection,
                )
            ch = ch.tocoo()
            h += coo_matrix(
                (ch.data, (self.surface_dofs[ch.row], self.surface_dofs[ch.col])), shape=h.shape
            ).tocsc()
        return float(energy), g, h, minj

    def safe_step(self, x, direction):
        F = np.eye(2) + np.einsum(
            "eai,eaJ->eiJ", (x[: self.n] - self.mesh.reference_xy)[self.mesh.triangles], self.grad
        )
        D = np.einsum("eai,eaJ->eiJ", direction[: self.n][self.mesh.triangles], self.grad)
        return min(
            determinant_step(F, D),
            self.ipc.compute_collision_free_stepsize(
                self.cmesh, x[self.surface], (x + direction)[self.surface]
            ),
        )

    def coupled_friction_tangent(self, x, lag, frozen_hessian):
        """Fresh force Jacobian: frozen Hessian plus contact-geometry coupling.

        The toolkit returns force derivatives; evaluate() uses negative forces.
        This generally nonsymmetric matrix must not receive a PSD projection.
        """
        jac = -self.friction.force_jacobian(
            lag, self.cmesh, self.rest[self.surface],
            (x-self.rest)[self.surface], (x-self.previous)[self.surface],
            self.barrier, self.ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS,
        ).tocoo()
        return frozen_hessian + coo_matrix(
            (jac.data, (self.surface_dofs[jac.row], self.surface_dofs[jac.col])),
            shape=frozen_hessian.shape,
        ).tocsc()

    def solve_friction(self, x, depth, deadline):
        """Preconditioned pseudo-transient continuation of fresh force balance.

        The shift is a solver parameter, not physical viscosity. Intermediate
        residuals may grow; only the original unshifted force test accepts a load.
        previous always remains the last accepted loading state.
        """
        initial_shift = .1
        shift = initial_shift
        for iteration in range(self.max_iterations+1):
            if time.monotonic() > deadline:
                raise TimeoutError("IPC friction pseudo-transient wall budget")
            self.trial = x.copy()
            lag = self.lag(x)
            _, g, H, _ = self.evaluate(x, lag, projected=False)
            r = g[self.free]
            norm = float(np.linalg.norm(r))
            print(f"FRICTION_PTC depth_mm={depth:.6f} iteration={iteration} "
                  f"fresh_residual_N={norm:.9g} metric_shift={shift:.9g}", flush=True)
            if norm <= BASELINE["force_tolerance_n"]:
                return x, iteration
            if iteration == self.max_iterations:
                break
            A = self.coupled_friction_tangent(x, lag, H)[self.free][:, self.free]
            M = self.evaluate(x, lag, projected=True)[2][self.free][:, self.free]
            M += diags(np.full(len(self.free), BASELINE["hessian_regularization_n_mm"]))
            for retry in range(18):
                if time.monotonic() > deadline:
                    raise TimeoutError("IPC friction pseudo-transient wall budget")
                du = spsolve(A+shift*M, -r)
                maximum = float(np.max(np.abs(du)))
                if not np.all(np.isfinite(du)) or maximum == 0:
                    shift = max(initial_shift, 2*shift)
                    continue
                direction = np.zeros(self.total)
                direction[self.free] = du
                direction = direction.reshape(-1, 2)
                alpha = min(1., .5*self.options.tip_radius_mm/maximum,
                            self.safe_step(x, direction))
                trial = x+alpha*direction
                gt = self.evaluate(trial, self.lag(trial), tangent=False)[1][self.free]
                nt = float(np.linalg.norm(gt))
                # Safeguard the local force model, rather than requiring each
                # auxiliary time step to reduce the physical residual monotonically.
                error = float(np.linalg.norm(gt-r-alpha*(A@du)))
                if (np.isfinite(nt) and not np.array_equal(trial, x)
                        and (error <= .25*max(norm, nt) or nt <= BASELINE["force_tolerance_n"])):
                    x = trial
                    self.trial = x.copy()
                    print(f"FRICTION_PTC_UPDATE alpha={alpha:.9g} residual_N={nt:.9g} "
                          f"model_error_N={error:.9g} retries={retry}", flush=True)
                    shift = max(1e-10, min(1e6, shift*nt/max(norm, 1e-15)/1.2))
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
                raise RuntimeError(f"IPC friction pseudo-transient model mismatch: "
                                   f"depth={depth:g} mm, residual={norm:.6g} N")
        raise RuntimeError(f"IPC friction pseudo-transient iteration budget: "
                           f"depth={depth:g} mm, residual={norm:.6g} N, limit={self.max_iterations}")

    def acceptance(self, x, depth, iterations, outer_iterations):
        lag = self.lag(x)
        E, g, _, minj = self.evaluate(x, lag, tangent=False)
        norm = float(np.linalg.norm(g[self.free]))
        if not np.isfinite(norm) or norm > BASELINE["force_tolerance_n"]:
            raise RuntimeError(f"Unbalanced IPC acceptance: residual={norm:.6g} N")
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
        return dict(depth_mm=depth, reaction_fz_n=float(-g.reshape(-1, 2)[self.n:, 1].sum()),
                    residual_n=norm, min_j=minj,
                    minimum_distance_mm=distance if np.isfinite(distance) else None,
                    iterations=iterations, friction_outer_iterations=outer_iterations,
                    friction_work_n_mm=dissipation, energy_n_mm=E)

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

    def step(self, x, old, depth, deadline):
        self.previous = x.copy()
        candidate = x.copy()
        delta = depth - old
        candidate[: self.n, 1] -= delta * self.mesh.local_sn[:, 1] / self.config.height_mm
        candidate[self.n :, 1] -= delta
        self.trial = candidate.copy()
        if self.safe_step(x, candidate - x) < 1.0 - 1e-10:
            raise RuntimeError(
                "Prescribed-motion predictor is not collision free; reduce loading step"
            )
        x = candidate
        if self.config.friction_coefficient > 0:
            x, iterations = self.solve_friction(x, depth, deadline)
            return x, self.acceptance(x, depth, iterations+1, 0)
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
                raise RuntimeError("IPC Newton directions/energy line search failed")
        else:
            # The last allowed update may already satisfy the original equations.
            final_g = self.evaluate(x, lag, tangent=False)[1]
            final_norm = float(np.linalg.norm(final_g[self.free]))
            if final_norm > BASELINE["force_tolerance_n"]:
                raise RuntimeError(
                    f"IPC inner iteration budget: depth={depth:g} mm, "
                    f"residual={final_norm:.6g} N, limit={self.max_iterations}"
                )
        return x, self.acceptance(x, depth, iterations, 1)



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
    )
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
        baseline_config_sha256=hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest(),
        ipc_version=p.ipc.__version__,
        dhat_mm=args.dhat,
        barrier_stiffness_mpa=BASELINE["barrier_stiffness_mpa"],
        ipc_binary_sha256=hashlib.sha256(Path(p.ipc.__file__).read_bytes()).hexdigest(),
        loading_increment_mm=args.increment,
        wall_budget_s=args.seconds,
        wall_time_limit_enabled=args.seconds > 0,
        force_tolerance_n=BASELINE["force_tolerance_n"],
        max_inner_iterations=args.max_iterations,
        friction_solver="coupled_pseudo_transient" if args.friction else None,
        friction_solver_settings=(dict(initial_metric_shift=.1, model_error_fraction=.25,
                                       pseudo_time_growth=1.2, max_tip_radius_fraction=.5,
                                       max_model_retries=18,
                                       rejection_metric_shift_floor=.1) if args.friction else None),
        hessian_mode=args.hessian_mode,
        curvature_settings=(dict(metric="projected Hessian", eigenvalue_floor=.05,
                                 eigenvalue_tolerance=1e-4, eigensolver_max_iterations=100)
                            if args.hessian_mode == "regularized" and args.friction == 0 else None),
        line_search=("pseudo_transient_model_error" if args.friction else
                     "armijo_with_roundoff_refinement" if args.hessian_mode == "regularized"
                     else "armijo"),
        source_sha256=source_hashes(snapshot=out / "source"),
        nodes=p.n,
        elements=len(p.mesh.triangles),
        threads=args.threads,
        pid=os.getpid(),
        steps=[],
    )
    shutil.copyfile(BASELINE_PATH, out / "baseline_config.json")
    visual_source = ROOT / "tools/visualize_ipc.py"
    visual_snapshot = out / "source/tools/visualize_ipc.py"
    visual_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(visual_source, visual_snapshot)
    report["source_sha256"]["tools/visualize_ipc.py"] = hashlib.sha256(visual_source.read_bytes()).hexdigest()
    x = p.rest.copy()
    old = p.start
    start = time.monotonic()
    (out / "states").mkdir()
    targets = np.r_[0.0, np.arange(args.increment, args.depth - 1e-12, args.increment), args.depth]
    if args.resume is not None:
        x, old = restore_accepted_prefix(args.resume, out, report, p, targets)
        targets = targets[targets > old + 1e-12]
    write_json(out / "result.json", report)
    with (
        (out / "run.log").open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        threadpool_limits(limits=args.threads),
    ):
        try:
            for depth in targets:
                previous = x.copy()
                deadline = start+args.seconds if args.seconds > 0 else float("inf")
                x, d = p.step(x, old, float(depth), deadline)
                report["steps"].append(d)
                old = float(depth)
                checkpoint = out / "states" / f'{len(report["steps"])-1:04d}.npz'
                save_npz(
                    checkpoint,
                    reference_xy=p.rest,
                    triangles=p.mesh.triangles,
                    positions=x,
                    previous_positions=previous,
                    depth_mm=depth,
                    body_nodes=p.n,
                )
                report["last_checkpoint"] = checkpoint.relative_to(out).as_posix()
                report["last_depth_mm"] = old
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
                previous_positions=p.previous,
                depth_mm=depth,
                body_nodes=p.n,
            )
            report.update(
                status="failed", attempted_depth_mm=float(depth), error=f"{type(e).__name__}: {e}"
            )
            print(report["error"], flush=True)
    report["elapsed_s"] = time.monotonic() - start
    write_json(out / "result.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "last_depth_mm", "elapsed_s", "error")}))
    if report["completed"] and args.case == "oblique" and not args.no_visualize:
        export_visualization(out, report)
    return report["completed"]


def restore_accepted_prefix(source, out, report, p, targets):
    """Resume only accepted physical history, retaining its distinct provenance."""
    previous = json.loads((source / "result.json").read_text(encoding="utf-8"))
    for key in ("case", "config", "mesh_settings", "simplifications", "dhat_mm",
                "barrier_stiffness_mpa", "ipc_binary_sha256", "loading_increment_mm",
                "force_tolerance_n", "baseline_config_sha256"):
        if previous[key] != report[key]:
            raise ValueError(f"Resume mismatch: {key}")
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
    for name, fingerprint in previous["source_sha256"].items():
        if name not in ("prototype/ipc_reference.py", "tools/visualize_ipc.py"):
            if report["source_sha256"].get(name) != fingerprint:
                raise ValueError(f"Resume physics source changed: {name}")
    steps = previous["steps"]
    if not steps or len(steps) >= len(targets):
        raise ValueError("Resume requires a nonempty accepted prefix shorter than target")
    np.testing.assert_allclose([s["depth_mm"] for s in steps], targets[:len(steps)], rtol=0, atol=1e-12)
    prior_x = None
    fingerprints = {}
    for i, step in enumerate(steps):
        if step["residual_n"] > report["force_tolerance_n"] or step["min_j"] <= 0:
            raise ValueError("Resume prefix includes an unaccepted state")
        relative = Path("states") / f"{i:04d}.npz"
        with np.load(source / relative) as state:
            np.testing.assert_array_equal(state["reference_xy"], p.rest)
            np.testing.assert_array_equal(state["triangles"], p.mesh.triangles)
            np.testing.assert_allclose(float(state["depth_mm"]), step["depth_mm"], rtol=0, atol=1e-12)
            if prior_x is not None:
                np.testing.assert_array_equal(state["previous_positions"], prior_x)
            else:
                np.testing.assert_array_equal(state["previous_positions"], p.rest)
            last_previous = state["previous_positions"].copy()
            x = state["positions"].copy()
            if x.shape != p.rest.shape or not np.isfinite(x).all():
                raise ValueError("Invalid resume positions")
            prior_x = x
        fingerprints[relative.as_posix()] = hashlib.sha256((source / relative).read_bytes()).hexdigest()
    # Friction must be reassembled using the slip of the last accepted increment.
    # Setting previous=x would erase this slip and test a different equilibrium.
    p.previous = last_previous
    _, g, _, minj = p.evaluate(x, p.lag(x), tangent=False)
    if np.linalg.norm(g[p.free]) > report["force_tolerance_n"] or minj <= 0 or p.ipc.has_intersections(p.cmesh, x[p.surface]):
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
    return x, steps[-1]["depth_mm"]


def export_visualization(out, report):
    """Postprocessing has its own status and cannot change mechanical acceptance."""
    report["visualization"] = dict(status="running", directory="visualization")
    write_json(out / "result.json", report)
    try:
        # Run against the saved source so later workspace edits cannot change this export.
        with (out / "visualization.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-u", str(out / "source/tools/visualize_ipc.py"),
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
        help="Run the frozen ipc-frictionless-v1 preset; reject parameter changes",
    )
    parser.add_argument("--case", choices=("patch", "oblique"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=float)
    parser.add_argument("--increment", type=float)
    parser.add_argument("--friction", type=float)
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
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--hessian-mode", choices=("projected", "hybrid", "regularized"),
                        help="Frictionless Hessian mode; friction uses the coupled pseudo-transient solver")
    parser.add_argument("--resume", type=Path, help="Copy and verify accepted prefix from an earlier run")
    args = parser.parse_args(argv)
    defaults = BASELINE["run"] if args.baseline else dict(BASELINE["run"], depth=0.5, seconds=180.0)
    defaults = dict(defaults, local_spacing=BASELINE["mesh"]["refinement_mm"],
                    surface_spacing=BASELINE["mesh"]["surface_spacing_mm"],
                    nose_facets=BASELINE["nose_facets"],
                    inclination=BASELINE["model"]["inclination_deg"],
                    tip_radius=BASELINE["mesh"]["tip_radius_mm"],
                    fine_halfwidth=BASELINE["fine_halfwidth_mm"], hessian_mode="projected")
    for name, value in defaults.items():
        supplied = getattr(args, name)
        if args.baseline and supplied is not None and supplied != value:
            parser.error(
                f"--baseline fixes {name}={value}; omit --baseline for a separate diagnostic"
            )
        if supplied is None:
            setattr(args, name, value)
    if (
        not all(
            np.isfinite(v) and v > 0 for v in (args.depth, args.increment, args.dhat)
        )
        or not np.isfinite(args.seconds) or args.seconds < 0
        or args.depth > 3
        or not np.isfinite(args.friction)
        or args.friction < 0
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
