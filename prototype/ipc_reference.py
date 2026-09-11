"""Small quasi-static IPC experiment; official contact kernels, local P1 host.

This is NOT PolyFEM and does not restore the retired multiplier/IPM solver.
The host implements barrier-aware projected Newton, with exact original energy
gradients, CCD, and a determinant step bound. No fracture or mesh adaptation.
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
from scipy.sparse.linalg import spsolve
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
    ):
        self.ipc = toolkit()
        self.case = case
        self.dhat = dhat
        self.max_iterations = max_iterations
        self.config = FEMConfig(**BASELINE["model"], friction_coefficient=mu)
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
        self.options = InitiationOptions(**BASELINE["mesh"])
        self.mesh, _ = intact_mesh(
            self.config, self.options, fine_halfwidth_mm=BASELINE["fine_halfwidth_mm"]
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
            theta = np.linspace(-np.pi + angle, -angle, BASELINE["nose_facets"] + 1)
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

    def evaluate(self, x, lag=None, tangent=True):
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
            eigen, v = np.linalg.eigh(k)
            k = np.einsum("eik,ek,ejk->eij", v, np.maximum(eigen, 0), v)
            rows = np.broadcast_to(dofs[:, :, None], k.shape).ravel()
            cols = np.broadcast_to(dofs[:, None, :], k.shape).ravel()
            h = coo_matrix((k.ravel(), (rows, cols)), shape=(self.total, self.total)).tocsc()
            ch = self.barrier.hessian(c, self.cmesh, sx, self.ipc.PSDProjectionMethod.CLAMP)
            if lag is not None:
                ch += self.friction.hessian(
                    lag,
                    self.cmesh,
                    (x - self.previous)[self.surface],
                    self.ipc.PSDProjectionMethod.CLAMP,
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
        iterations = 0
        for outer in range(BASELINE["max_friction_iterations"]):
            lag = self.lag(x)
            for inner in range(self.max_iterations):
                self.trial = x.copy()
                if time.monotonic() > deadline:
                    raise TimeoutError("IPC reference wall budget")
                E, g, H, minj = self.evaluate(x, lag)
                norm = float(np.linalg.norm(g[self.free]))
                iterations += 1
                print(
                    f"ITER depth_mm={depth:.6f} outer={outer} inner={inner} force_residual_N={norm:.6g} energy_Nmm={E:.9g} min_J={minj:.6g}",
                    flush=True,
                )
                if norm <= BASELINE["force_tolerance_n"]:
                    break
                A = H[self.free][:, self.free]
                A += diags(np.full(len(self.free), BASELINE["hessian_regularization_n_mm"]))
                du = spsolve(A, -g[self.free])
                direction = np.zeros(self.total)
                direction[self.free] = du
                descent = float(g @ direction)
                if not np.isfinite(descent) or descent >= 0:
                    raise RuntimeError("Non-descent IPC Newton direction")
                direction = direction.reshape(-1, 2)
                alpha = self.safe_step(x, direction)
                for cut in range(BASELINE["max_line_search_iterations"]):
                    trial = x + alpha * direction
                    try:
                        Et = self.evaluate(trial, lag, tangent=False)[0]
                    except FloatingPointError:
                        Et = np.inf
                    if (
                        np.isfinite(Et)
                        and Et <= E + BASELINE["armijo_coefficient"] * alpha * descent
                    ):
                        x = trial
                        break
                    alpha *= 0.5
                else:
                    raise RuntimeError("IPC energy line search failed")
            else:
                raise RuntimeError("IPC inner iteration budget")
            # Rebuild friction at the actual final geometry before acceptance.
            fresh = self.lag(x)
            E, g, _, minj = self.evaluate(x, fresh, tangent=False)
            norm = float(np.linalg.norm(g[self.free]))
            if norm <= BASELINE["force_tolerance_n"]:
                c = self.collisions(x)
                # Toolkit stores squared stencil distances.
                distance = float(np.sqrt(c.compute_minimum_distance(self.cmesh, x[self.surface])))
                if (
                    distance <= 0
                    or minj <= 0
                    or self.ipc.has_intersections(self.cmesh, x[self.surface])
                ):
                    raise RuntimeError("Invalid geometry at acceptance")
                reaction = float(-g.reshape(-1, 2)[self.n :, 1].sum())
                return x, dict(
                    depth_mm=depth,
                    reaction_fz_n=reaction,
                    residual_n=norm,
                    min_j=minj,
                    minimum_distance_mm=distance if np.isfinite(distance) else None,
                    iterations=iterations,
                    friction_outer_iterations=outer + 1,
                    energy_n_mm=E,
                )
        raise RuntimeError("IPC friction outer iteration budget")


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    toolkit().set_num_threads(args.threads)
    p = IPCProblem(args.case, args.friction, args.dhat, max_iterations=args.max_iterations)
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
            displacement_element="P1, coarse fixed mesh",
            tip_radius_mm=p.options.tip_radius_mm,
            nose_facets=BASELINE["nose_facets"],
            fracture=False,
            material="existing finite-strain plane-strain elastic law",
            host="local projected-Newton harness, NOT PolyFEM",
            friction_smoothing_mm=BASELINE["friction_smoothing_mm"],
        ),
        mesh_settings=dict(
            bulk_spacing_mm=p.config.mesh_size_mm,
            local_spacing_mm=p.options.refinement_mm,
            surface_spacing_mm=p.options.surface_spacing_mm,
            fine_halfwidth_mm=BASELINE["fine_halfwidth_mm"],
        ),
        baseline_id=BASELINE["id"] if args.baseline else None,
        baseline_config_sha256=hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest(),
        ipc_version=p.ipc.__version__,
        dhat_mm=args.dhat,
        barrier_stiffness_mpa=BASELINE["barrier_stiffness_mpa"],
        ipc_binary_sha256=hashlib.sha256(Path(p.ipc.__file__).read_bytes()).hexdigest(),
        loading_increment_mm=args.increment,
        wall_budget_s=args.seconds,
        force_tolerance_n=BASELINE["force_tolerance_n"],
        max_inner_iterations=args.max_iterations,
        source_sha256=source_hashes(snapshot=out / "source"),
        nodes=p.n,
        elements=len(p.mesh.triangles),
        threads=args.threads,
        pid=os.getpid(),
        steps=[],
    )
    shutil.copyfile(BASELINE_PATH, out / "baseline_config.json")
    x = p.rest.copy()
    old = p.start
    start = time.monotonic()
    write_json(out / "result.json", report)
    (out / "states").mkdir()
    targets = np.r_[0.0, np.arange(args.increment, args.depth - 1e-12, args.increment), args.depth]
    with (
        (out / "run.log").open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        threadpool_limits(limits=args.threads),
    ):
        try:
            for depth in targets:
                previous = x.copy()
                x, d = p.step(x, old, float(depth), start + args.seconds)
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
    return report["completed"]


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
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    defaults = BASELINE["run"] if args.baseline else dict(BASELINE["run"], depth=0.5, seconds=180.0)
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
            np.isfinite(v) and v > 0 for v in (args.depth, args.increment, args.dhat, args.seconds)
        )
        or args.depth > 3
        or not np.isfinite(args.friction)
        or args.friction < 0
        or args.threads < 1
        or args.max_iterations < 1
    ):
        parser.error("Invalid experiment or budget")
    if args.case == "patch" and args.depth > 0.3:
        parser.error("Patch depth must not exceed 0.3 mm")
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
