"""Bounded equation audit: native contact kinematics, smoothing, Maxwell state.

No FE advancement, fitting, source edits, or modified physical acceptance.
Each invocation uses independent output and verifies its immutable input hashes.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".runtime/ipc-python"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def table(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stencils(c):
    for kind, items in (("vv", c.vv_collisions), ("ev", c.ev_collisions)):
        for s in items:
            yield kind, s


def ids_of(p, s):
    import numpy as np
    return np.array([i for i in s.vertex_ids(p.cmesh.edges, p.cmesh.faces) if i >= 0], int)


def rotation(a):
    import numpy as np
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def fixture_checks(out, evidence, deadline):
    import numpy as np
    path = ROOT / "tests/test_ipc_contact_formulation.py"
    evidence[str(path)] = sha(path)
    spec = importlib.util.spec_from_file_location("contact_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    t = module.ContactFormulationTests()
    t.setUpClass()
    p = t.rigid_contact("improved-max")
    for name in ("prototype/ipc_reference.py", "mechanics/finite_strain.py"):
        evidence[str(ROOT/name)] = sha(ROOT/name)

    def work(p, x, old):
        lag = p.lag(x)
        delta = x-old
        g = p.friction.gradient(lag, p.cmesh, delta)
        smooth = []
        for _, s in stencils(lag):
            rel = s.relative_velocity(s.dof(delta, p.cmesh.edges, p.cmesh.faces))
            smooth.append(float(np.linalg.norm(s.tangent_basis.T@rel)))
        return dict(work_n_mm=float(g@delta.ravel()), gradient_norm_n=float(np.linalg.norm(g)),
                    max_slip_mm=max(smooth, default=0.), count=len(smooth)), g

    # Coordinate covariance is distinct from physical rigid motion over time.
    old = p.rest.copy()
    x = old.copy()
    x[:4, 0] += 2e-5
    base, g = work(p, x, old)
    Q = rotation(.47)
    rotated, gr = work(p, x@Q.T+[.02, -.03], old@Q.T+[.02, -.03])
    np.testing.assert_allclose(gr.reshape(-1, 2), g.reshape(-1, 2)@Q.T, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(rotated["work_n_mm"], base["work_n_mm"], rtol=1e-9, atol=1e-14)
    translation, _ = work(p, old+[.03, -.02], old)
    assert translation["work_n_mm"] < 1e-20
    rows = []
    for gap in (.0015, .00075, .00015):
        q = t.rigid_contact("improved-max")
        q.rest[4:, 1] -= .0015-gap
        initial = q.rest.copy()
        for subdivisions in (1, 2, 4, 8, 16, 32):
            if time.monotonic() > deadline:
                raise TimeoutError("rigid-motion fixture budget")
            for eps_mode in ("fixed_displacement", "fixed_velocity"):
                q.friction = q.ipc.FrictionPotential(1e-5 if eps_mode == "fixed_displacement" else 1e-5/subdivisions)
                total = 0.
                maximum = 0.
                for i in range(subdivisions):
                    a = initial@rotation(.1*i/subdivisions).T
                    b = initial@rotation(.1*(i+1)/subdivisions).T
                    result, _ = work(q, b, a)
                    total += result["work_n_mm"]
                    maximum = max(maximum, result["max_slip_mm"])
                c = q.collisions(initial)
                ng = q.barrier.gradient(c, q.cmesh, initial).reshape(-1, 2)
                normal = float(np.linalg.norm(ng[:4].sum(axis=0)))
                rows.append(dict(gap_mm=gap, angle_rad=.1, subdivisions=subdivisions,
                    epsilon_mode=eps_mode, work_n_mm=total, normal_load_n=normal,
                    work_over_mu_normal_mm=total/(.35*normal), max_increment_slip_mm=maximum))
    table(out/"rigid_motion.csv", rows)
    # One stable point-edge, fixed normal force/basis; exact polynomial control.
    p = t.rigid_contact("improved-max")
    s = p.lag(p.rest).ev_collisions[0]
    ids = ids_of(p, s)
    local = np.zeros((len(ids), 2))
    local[0, 0] = 1.
    mu_n = s.weight*s.normal_force_magnitude*s.mu_s_effective_lagged
    smooth = []
    for velocity in (2e-5, 1e-4, 2e-4, 1e-3):
        for dt in (.05, .025, .0125):
            for mode in ("fixed_displacement", "fixed_velocity"):
                eps = 1e-5 if mode == "fixed_displacement" else 2e-4*dt
                pot = p.ipc.FrictionPotential(eps)
                delta = local*velocity*dt
                value = pot.gradient(s, delta.ravel()).reshape(-1, 2)[0, 0]
                ratio = velocity*dt/eps
                predicted = 1. if ratio >= 1 else 2*ratio-ratio**2
                np.testing.assert_allclose(value/mu_n, predicted, atol=2e-14)
                smooth.append(dict(velocity_mm_s=velocity, dt_s=dt, epsilon_mode=mode,
                    epsilon_u_mm=eps, epsilon_v_mm_s=eps/dt,
                    slip_over_epsilon=ratio, force_over_mu_normal=float(value/mu_n)))
    table(out/"smoothing_control.csv", smooth)
    # Current fixture already demonstrates reference-neighbor repulsion.
    t.test_straight_connected_edge_compression_has_no_ogc_contact_force()
    result = dict(covariance=base, transformed_covariance=rotated, translation=translation,
        physical_rotation=rows, smoothing=smooth, adjacent_fixture="passed",
        interpretation="Physical common rotation is zero-slip. Finite barrier gap gives different "
        "velocities at two distinct sampled positions. Subdivision and gap controls separate effects; "
        "OGC is a fixture comparison only, not a qualified production alternative.")
    write(out/"fixtures.json", result)


def elastic_state(F, metric, material):
    """Actual committed Maxwell elastic state; includes F33 and Cv33."""
    import numpy as np
    from mechanics.finite_strain import principal
    full = np.broadcast_to(np.eye(3), (len(F), 3, 3)).copy()
    full[:, :2, :2] = F
    be = full@metric@full.transpose(0, 2, 1)
    values, vectors = np.linalg.eigh(be)
    assert np.all(values > 0)
    e = .5*np.log(values)
    energy, stress, _ = principal(e, material, True)
    tau = (vectors*stress[:, None, :])@vectors.transpose(0, 2, 1)
    norm = np.linalg.norm(stress, axis=1)
    J = np.linalg.det(F)
    P = tau[:, :2, :2]@np.linalg.inv(F).transpose(0, 2, 1)
    return dict(P=P, J=J, energy=energy, norm=norm, tau=tau,
                elastic_stretches=np.sqrt(values),
                elastic_iso_stretches=np.sqrt(values)/J[:, None]**(1/3))


def holding_probe(F, metric, material, volumes, deadline):
    import numpy as np
    from mechanics.finite_strain import response
    base = elastic_state(F, metric, material)
    initial = float(volumes@base["norm"])
    if initial <= 1e-20:
        return dict(status="zero_branch_stress"), np.full(len(F), np.inf)
    rates = []
    taus = []
    for h in (1e-4, 5e-5):
        q = response(F, metric, h, material, tangent=False, kinematics="plane_strain")
        after = elastic_state(F, q["metric"], material)
        derivative = (base["tau"]-after["tau"])/h
        den = np.einsum("eij,eij->e", base["tau"], derivative)
        tau = np.divide(base["norm"]**2, den, out=np.full(len(F), np.inf), where=den > 1e-20)
        taus.append(tau)
        rates.append(float((volumes@after["norm"])/initial))
    rows = []
    for dt in (.005, .0025):
        A = metric.copy()
        for i in range(round(.1/dt)):
            if time.monotonic() > deadline:
                raise TimeoutError("bounded material hold budget")
            q = response(F, A, dt, material, tangent=False, kinematics="plane_strain")
            A = q["metric"]
            if (i+1) in (round(.01/dt), round(.05/dt), round(.1/dt)):
                after = elastic_state(F, A, material)
                rows.append(dict(dt_s=dt, time_s=(i+1)*dt,
                    weighted_stress_ratio=float(volumes@after["norm"])/initial,
                    energy_ratio=float(volumes@after["energy"])/max(float(volumes@base["energy"]), 1e-30)))
    weights = volumes*base["norm"]
    tau = taus[-1]
    finite = np.isfinite(tau)
    # Stress-weighted fractions; no arbitrary max-stretch point selection.
    result = dict(status="completed", points=len(F), nominal_tau_s=material.relaxation_s,
        initial_norm_volume_n_mm=initial, short_probe_stress_ratios=rates,
        fraction_weight_tau_below_half=float(weights[tau < .5*material.relaxation_s].sum()/weights.sum()),
        fraction_weight_tau_below_tenth=float(weights[tau < .1*material.relaxation_s].sum()/weights.sum()),
        finite_rate_weight_fraction=float(weights[finite].sum()/weights.sum()),
        short_probe_rate_relative_change=float(np.max(abs(taus[0][finite]-tau[finite])/tau[finite])) if finite.any() else None,
        hold=rows, note="Fixed F and own committed metric; stress norm, not energy relaxation time; not a new FE path.")
    return result, tau


def field_contact(p, x, old, out, label):
    import numpy as np
    sx = x[p.surface]
    delta = (x-old)[p.surface]
    c = p.collisions(x)
    lag = p.lag(x)
    groups = {k: np.zeros_like(sx) for k in ("blade", "adjacent_body_vv", "other_body")}
    fg = {k: np.zeros_like(sx) for k in groups}
    smoothg = {k: np.zeros_like(sx) for k in groups}
    rows = []
    blade_bearing = np.zeros_like(x[:p.n])

    def category(kind, s, ids):
        global_ids = p.surface[ids]
        if np.any(global_ids >= p.n):
            return "blade"
        if kind == "vv" and tuple(sorted(ids)) in p._connected_boundary_pairs:
            return "adjacent_body_vv"
        return "other_body"

    for kind, s in stencils(c):
        ids = ids_of(p, s)
        key = category(kind, s, ids)
        g = p.barrier.gradient(s, sx[ids].ravel()).reshape(-1, 2)
        np.add.at(groups[key], ids, g)
        if key == "blade":
            body = p.surface[ids] < p.n
            np.add.at(blade_bearing, p.surface[ids][body], g[body])
    dt = p.material_dt
    eps = 1e-5
    alternate = p.ipc.FrictionPotential(eps/2)
    half_fixed = np.zeros_like(sx)
    half_scaled = np.zeros_like(sx)
    for kind, s in stencils(lag):
        ids = ids_of(p, s)
        key = category(kind, s, ids)
        d = delta[ids].ravel()
        rel = s.relative_velocity(d)
        slip = float(np.linalg.norm(s.tangent_basis.T@rel))
        g = p.friction.gradient(s, d).reshape(-1, 2)
        np.add.at(fg[key], ids, g)
        if slip < eps:
            np.add.at(smoothg[key], ids, g)
        np.add.at(half_fixed, ids, p.friction.gradient(s, d/2).reshape(-1, 2))
        np.add.at(half_scaled, ids, alternate.gradient(s, d/2).reshape(-1, 2))
        rows.append(dict(kind=kind, category=key, surface_nodes=" ".join(map(str, p.surface[ids])),
            signed_weight=float(s.weight), normal_force_n=float(s.weight*s.normal_force_magnitude),
            slip_mm=slip, slip_over_epsilon=slip/eps, epsilon_v_mm_s=eps/dt,
            smooth=slip < eps, force_norm_n=float(np.linalg.norm(g)),
            work_n_mm=float(g.ravel()@d)))
    normal = p.barrier.gradient(c, p.cmesh, sx).reshape(-1, 2)
    friction = p.friction.gradient(lag, p.cmesh, delta).reshape(-1, 2)
    np.testing.assert_allclose(sum(groups.values()), normal, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(sum(fg.values()), friction, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(half_scaled, friction, rtol=1e-10, atol=1e-11)
    result = dict(dt_s=dt, epsilon_u_mm=eps, epsilon_v_mm_s=eps/dt, groups={},
        half_step_fixed_geometry=dict(
            fixed_epsilon_force_change_norm_n=float(np.linalg.norm(half_fixed-friction)),
            scaled_epsilon_force_change_norm_n=float(np.linalg.norm(half_scaled-friction)),
            note="Same stencil, normal force and velocity; half displacement, no FE solve/history change."))
    for key in groups:
        rr = [r for r in rows if r["category"] == key]
        total_abs = sum(abs(r["normal_force_n"]) for r in rr)
        smooth_abs = sum(abs(r["normal_force_n"]) for r in rr if r["smooth"])
        global_g = np.zeros_like(x)
        global_g[p.surface] = groups[key]+fg[key]
        result["groups"][key] = dict(count=len(rr),
            normal_body_norm_n=float(np.linalg.norm(groups[key][p.surface < p.n])),
            friction_body_norm_n=float(np.linalg.norm(fg[key][p.surface < p.n])),
            normal_fz_n=float(-groups[key][p.surface >= p.n, 1].sum()),
            friction_fz_n=float(-fg[key][p.surface >= p.n, 1].sum()),
            smooth_friction_fz_n=float(-smoothg[key][p.surface >= p.n, 1].sum()),
            normal_work_n_mm=float(np.sum(groups[key]*delta)),
            friction_work_n_mm=float(np.sum(fg[key]*delta)),
            smooth_count=sum(r["smooth"] for r in rr),
            absolute_stencil_load_n=total_abs,
            smooth_absolute_load_fraction=smooth_abs/total_abs if total_abs else None,
            signed_stencil_load_n=sum(r["normal_force_n"] for r in rr),
            free_force_norm_n=float(np.linalg.norm(global_g.ravel()[p.free])))
    table(out/f"{label}_contacts.csv", rows)
    return result, groups, fg, blade_bearing


def field_material(p, x, old, committed_metric, bearing, out, label, deadline):
    import numpy as np
    from mechanics.finite_strain import elastic_response
    u = x[:p.n]-p.mesh.reference_xy
    F = np.eye(2)+np.einsum("eai,eaJ->eiJ", u[p.mesh.triangles], p.grad)
    du = x[:p.n]-old[:p.n]
    dF = np.einsum("eai,eaJ->eiJ", du[p.mesh.triangles], p.grad)
    q = p.material_response(x)
    np.testing.assert_allclose(q["metric"], committed_metric, rtol=1e-9, atol=1e-10)
    state = elastic_state(F, committed_metric, p.material)
    eq = elastic_response(F, p.material, tangent=False, diagnostics=False)
    J = state["J"]
    K = p.material.moduli()[1]
    inv = np.linalg.inv(F).transpose(0, 2, 1)
    volP = (K*J*(J-1))[:, None, None]*inv
    isoP = eq["P"]-volP
    np.testing.assert_allclose(eq["P"]+state["P"], q["P"], rtol=1e-8, atol=1e-9)
    # All fields in this audit are constant-strain P1, one point per triangle.
    # Old frozen classes predate the convenience method with this same formula.
    assert p.element == "p1" and p.enrichment is None
    weights = p.mesh.bulk_data[0]*p.config.width_mm
    work = {key: float(np.einsum("e,eij,eij->", weights, P, dF))
            for key, P in (("equilibrium_iso", isoP), ("volume", volP), ("maxwell", state["P"]))}
    nodal = np.linalg.norm(bearing, axis=1)
    order = np.argsort(nodal)[::-1]
    cumulative = np.cumsum(nodal[order])
    count = int(np.searchsorted(cumulative, .9*cumulative[-1])+1)
    nodes = order[:count]
    mask = np.isin(p.mesh.triangles, nodes).any(axis=1)
    assert mask.any()
    hold, tau = holding_probe(F[mask], committed_metric[mask], p.material, weights[mask], deadline)
    centres = x[p.mesh.triangles].mean(axis=1)
    ids = np.flatnonzero(mask)
    rows = []
    for j, i in enumerate(ids):
        rows.append(dict(element=int(i), x_mm=float(centres[i, 0]), z_mm=float(centres[i, 1]),
            reference_volume_mm3=float(weights[i]), J=float(J[i]),
            elastic_iso_stretch_max=float(state["elastic_iso_stretches"][i].max()),
            maxwell_kirchhoff_norm_mpa=float(state["norm"][i]),
            total_inplane_P_norm_mpa=float(np.linalg.norm(q["P"][i])),
            maxwell_energy_mpa=float(state["energy"][i]),
            local_stress_decay_s=float(tau[j]) if np.isfinite(tau[j]) else None))
    table(out/f"{label}_bearing_elements.csv", rows)
    def stats(m):
        w = weights[m]
        return dict(points=int(m.sum()), J_min=float(J[m].min()),
            J_volume_average=float(w@J[m]/w.sum()),
            volume_fraction_J_below_08=float(w[J[m] < .8].sum()/w.sum()),
            volume_fraction_J_below_05=float(w[J[m] < .5].sum()/w.sum()),
            elastic_iso_stretch_max=float(state["elastic_iso_stretches"][m].max()),
            maxwell_energy_n_mm=float(w@state["energy"][m]),
            bulk_energy_n_mm=float(w@(K/2*(J[m]-1)**2)))
    return dict(work_n_mm=work, all_body=stats(np.ones(len(F), bool)), bearing=stats(mask),
        bearing_definition="Triangles incident to nodes accounting for 90% sum of assembled blade-normal nodal force magnitudes; "
        "not 90% of external Fz and not a causal force partition.",
        bearing_nodes=len(nodes), hold=hold,
        nominal_tau_s=p.material.relaxation_s, bulk_modulus_mpa=K,
        instantaneous_shear_mpa=p.material.moduli()[0]+p.material.moduli(True)[0])


def baseline_environment(angle):
    """Construct from that completed case's OWN frozen physical modules."""
    paths = {
        "0": "outputs/relaxation_partition_20261015/angles30_0_2mm/02_angle0",
        "33": "outputs/native_angles_2mm_20261016/01_angle33",
        "57": "outputs/native_angles_2mm_20261016/02_angle57"}
    folder = ROOT/paths[angle]
    result = read(folder/"result.json")
    source = folder/"source"
    evidence = {}
    for name, digest in result["source_sha256"].items():
        path = source/name
        assert sha(path) == digest, path
        evidence[str(path)] = digest
    sys.path.insert(0, str(source))
    from prototype.ipc_viscoelastic import ViscoelasticProblem
    import inspect
    assert Path(inspect.getfile(ViscoelasticProblem)).resolve().is_relative_to(source.resolve())
    c = result["signature"]["config"]
    m = result["signature"]["material"]
    K = 2*c["shear_modulus_n_mm2"]*(1+c["poisson_ratio"])/(3*(1-2*c["poisson_ratio"]))
    p = ViscoelasticProblem(maxwell_shear=m["mu_ov"][0], relaxation_s=m["relaxation_s"],
        loading_rate=result["signature"]["loading_rate"], bulk_modulus=K,
        case="oblique", mu=c["friction_coefficient"], shear_modulus=c["shear_modulus_n_mm2"],
        dhat=.0025, local_spacing=.05, surface_spacing=.005, nose_facets=64,
        inclination=float(angle), tip_radius=.02, fine_halfwidth=3.,
        max_iterations=500, predictor="secant", hessian_mode="regularized", element="p1")
    assert json.loads(json.dumps(p.material.to_dict())) == m
    assert hashlib.sha256(p.rest.tobytes()+p.mesh.triangles.tobytes()).hexdigest() == result["signature"]["mesh_sha256"]
    return p, result, evidence, folder


def fields(out, study, cases, evidence, deadline, baseline=False):
    import numpy as np
    from tools.diagnose_ptc_directions import environment, load
    results = []
    for case in cases:
        if baseline:
            p, result, sources, folder = baseline_environment(case)
            case = "baseline_"+case
        else:
            p, result, sources, _, _ = environment(study, case)
            folder = study/case
        evidence.update(sources)
        evidence[str(folder/"result.json")] = sha(folder/"result.json")
        p.ipc.set_num_threads(4)
        try:
            steps = result["steps"]
            depths = (.1, .25, .5, .75, 1.) if baseline else (.1, .25, .5)
            indices = (sorted(set([int(np.argmin([abs(s["depth_mm"]-d) for s in steps]))
                                  for d in depths])) if baseline or not case.endswith("_57") else [len(steps)-1])
            jobs = [(f"a{i:04d}", folder/"states"/f"{i:04d}.npz", steps[i]) for i in indices]
            if (folder/"failed_trial.npz").exists():
                jobs.append(("failed", folder/"failed_trial.npz", None))
            for tag, path, record in jobs:
                if time.monotonic() > deadline:
                    raise TimeoutError("field audit budget")
                evidence[str(path)] = sha(path)
                z = load(path)
                x = z["positions"]
                old = z["previous_positions"]
                p.previous = old.copy()
                p.previous_material_metric = z["previous_material_metric"].copy()
                p.material_dt = float(z["material_dt_s"])
                label = case+"_"+tag
                if record is not None:
                    audit = p.audit_state(z, record)
                    metric = z["material_metric"]
                    depth = float(z["depth_mm"])
                else:
                    audit = None
                    metric = p.material_response(x)["metric"]
                    depth = float(result["rejected_load_attempts"][-1]["depth_mm"])
                lag = p.lag(x)
                _, g, _, minj = p.evaluate(x, lag, tangent=False)
                contact, ng, fg, bearing = field_contact(p, x, old, out, label)
                material = field_material(p, x, old, metric, bearing, out, label, deadline)
                adj = np.zeros_like(x)
                adj[p.surface] = ng["adjacent_body_vv"]+fg["adjacent_body_vv"]
                normal_work = sum(v["normal_work_n_mm"] for v in contact["groups"].values())
                friction_work = sum(v["friction_work_n_mm"] for v in contact["groups"].values())
                total_work = sum(material["work_n_mm"].values())+normal_work+friction_work
                np.testing.assert_allclose(total_work, g@(x-old).ravel(), atol=1e-10, rtol=1e-8)
                dd = p.material_dt*p.loading_rate
                row = dict(label=label, case=case, depth_mm=depth,
                    status="accepted" if record is not None else "failed_trial",
                    fresh_audit=audit, residual_n=float(np.linalg.norm(g[p.free])),
                    residual_without_adjacent_n=float(np.linalg.norm((g-adj.ravel())[p.free])),
                    min_j=float(minj), fz_n=float(-g.reshape(-1, 2)[p.n:, 1].sum()),
                    contacts=contact, material=material,
                    endpoint_work_equivalent_force_n=total_work/dd,
                    note="Deleted-force residual is algebraic diagnostic, not reequilibrium. "
                    "Endpoint-work components are bookkeeping on this path, not integrated energy or causal Fz fractions.")
                results.append(row)
                write(out/"fields.json", results)
                print(f"FIELD {label} depth={depth:.6g} residual={row['residual_n']:.6g} "
                      f"adj={contact['groups']['adjacent_body_vv']['free_force_norm_n']:.6g} "
                      f"hold_ratio={material['hold']['hold'][-1]['weighted_stress_ratio']:.6g}", flush=True)
        finally:
            if getattr(p, "parallel_linear", None) is not None:
                p.parallel_linear.close()
    return results


def material_controls(out, deadline):
    import numpy as np
    from mechanics.finite_strain import Material
    rows = []
    m = Material(mu_eq=(.14,), alpha_eq=(2.,), mu_ov=(1.24,), alpha_ov=(2.,),
                 relaxation_s=1.7, viscous_flow="isochoric_maxwell", volumetric="quadratic_j")
    for stretch in (1.001, 1.5, 3., 6.):
        F = np.array([[[stretch, 0.], [0., 1/stretch]]])
        A = np.eye(3)[None]
        result, tau = holding_probe(F, A, m, np.ones(1), deadline)
        rows.append(dict(stretch=stretch, instantaneous_stress_decay_s=float(tau[0]), result=result))
    assert abs(rows[0]["instantaneous_stress_decay_s"]/1.7-1) < .002
    write(out/"material_controls.json", rows)


def report(out):
    """Present completed audits; never runs mechanics or changes a saved field."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from html import escape
    folders = ["fixtures", "fields", "baseline_0_v2", "baseline_33_v2", "baseline_57_v2"]
    provenance = {}
    elapsed = {}
    for folder in folders:
        status = read(out/folder/"status.json")
        assert status["status"] == "completed" and status["input_hashes_unchanged"]
        elapsed[folder] = status["elapsed_s"]
        for name, digest in status["input_sha256"].items():
            path = Path(name)
            if path.name == Path(__file__).name:
                path = out/folder/"diagnostic_source.py"
            assert sha(path) == digest, path
            provenance[str(path)] = digest
        for path in (out/folder).glob("*"):
            if path.is_file():
                provenance[str(path)] = sha(path)
    fixture = read(out/"fixtures/fixtures.json")
    controls = read(out/"fixtures/material_controls.json")
    cases = read(out/"fields/fields.json")
    baselines = {a: read(out/f"baseline_{a}_v2/fields.json") for a in (0, 33, 57)}
    fields_all = cases+sum(baselines.values(), [])
    assert len(fields_all) == 39
    raw_summary = []
    for s in fields_all:
        b = s["contacts"]["groups"]["blade"]
        adj = s["contacts"]["groups"]["adjacent_body_vv"]
        m = s["material"]
        h = m["hold"]
        dd = s["contacts"]["dt_s"]*.2
        raw_summary.append(dict(case=s["case"], label=s["label"], status=s["status"],
            depth_mm=s["depth_mm"], fz_n=s["fz_n"], residual_n=s["residual_n"],
            adjacent_force_norm_n=adj["free_force_norm_n"],
            smooth_blade_absolute_stencil_load_fraction=b["smooth_absolute_load_fraction"],
            smooth_blade_friction_fz_n=b["smooth_friction_fz_n"],
            blade_friction_fz_n=b["friction_fz_n"],
            maxwell_virtual_power_n=m["work_n_mm"]["maxwell"]/dd,
            volume_virtual_power_n=m["work_n_mm"]["volume"]/dd,
            equilibrium_iso_virtual_power_n=m["work_n_mm"]["equilibrium_iso"]/dd,
            bearing_J_min=m["bearing"]["J_min"],
            bearing_J_mean=m["bearing"]["J_volume_average"],
            bearing_elastic_iso_max=m["bearing"]["elastic_iso_stretch_max"],
            fast_half_stress_weight_fraction=h["fraction_weight_tau_below_half"],
            fast_tenth_stress_weight_fraction=h["fraction_weight_tau_below_tenth"],
            hold_01_stress_ratio=h["hold"][-1]["weighted_stress_ratio"]))
    table(out/"state_summary.csv", raw_summary)
    experimental = {}
    slope_rows = []
    experiments = {0: "1008n0a2__Fz__curve_1.csv", 33: "1008n33a1__Fz__curve_2.csv",
                   57: "1014n57a1__Fz__curve_0.csv"}
    source_folders = {
        0: "outputs/relaxation_partition_20261015/angles30_0_2mm/02_angle0",
        33: "outputs/native_angles_2mm_20261016/01_angle33",
        57: "outputs/native_angles_2mm_20261016/02_angle57"}
    for a, name in experiments.items():
        path = ROOT/"docs"/name
        provenance[str(path)] = sha(path)
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header[1] == "Fz (N)"
            data = np.array([[float(row[0])/1000, float(row[1])] for row in reader])
        experimental[a] = data
        path = ROOT/source_folders[a]/"result.json"
        provenance[str(path)] = sha(path)
        fe = np.array([[r["depth_mm"], r["reaction_fz_n"]] for r in read(path)["steps"]])
        for lo, hi in ((.1, .25), (.25, .5)):
            rr = {}
            for key, arr in (("experiment", data), ("fe", fe)):
                sub = arr[(arr[:, 0] >= lo-1e-10) & (arr[:, 0] <= hi+1e-10)]
                rr[key+"_slope_n_mm"] = float(np.polyfit(sub[:, 0], sub[:, 1], 1)[0])
                rr[key+"_samples"] = len(sub)
            ends = [next(s for s in raw_summary if s["case"] == f"baseline_{a}" and
                         abs(s["depth_mm"]-d) < 1e-9) for d in (lo, hi)]
            for key in ("maxwell", "volume", "equilibrium_iso"):
                rr[key+"_power_secant_n_mm"] = (
                    ends[1][key+"_virtual_power_n"]-ends[0][key+"_virtual_power_n"])/(hi-lo)
            slope_rows.append(dict(angle=a, start_mm=lo, end_mm=hi, **rr))
    table(out/"slope_diagnostics.csv", slope_rows)
    regressions = read(out/"regressions.json")
    assert regressions == dict(tests=29, failures=0, errors=0, skipped=0, passed=True)
    hold_difference = max(abs(s["material"]["hold"]["hold"][2]["weighted_stress_ratio"]-
                              s["material"]["hold"]["hold"][5]["weighted_stress_ratio"]) for s in fields_all)
    analysis = dict(scope="read-only equations and saved fields; 36 accepted + 3 failed, no new FE loads",
        states=len(fields_all), accepted=sum(s["status"] == "accepted" for s in fields_all),
        failure_fields=sum(s["status"] != "accepted" for s in fields_all),
        input_sha256=provenance, elapsed_s=elapsed, regressions=regressions,
        max_hold_time_refinement_ratio_change=hold_difference,
        max_short_rate_probe_relative_change=max(s["material"]["hold"]["short_probe_rate_relative_change"]
                                                for s in fields_all),
        plots=dict(smoothing="none", curves="straight segments between measured states",
            uncertainty="deterministic probes, no statistical interval",
            force_comparison="original experimental zeros/N retained; um->mm; only .1--.5 slopes",
            load_coverage="absolute signed-stencil load weights, not a fraction of macroscopic force",
            bearing_region="triangles touching nodes with 90% cumulative assembled blade-normal nodal magnitude"),
        renderer_sha256=sha(__file__))
    write(out/"analysis.json", analysis)
    colors = {0: "#0072B2", 33: "#D55E00", 57: "#6B4C9A"}
    markers = {0: "o", 33: "s", 57: "^"}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "path", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
        ax = axes[0, 0]
        for gap, color in zip((.0015, .00075, .00015), colors.values()):
            for mode, style in (("fixed_displacement", "--"), ("fixed_velocity", "-")):
                rows = [r for r in fixture["physical_rotation"] if r["gap_mm"] == gap and r["epsilon_mode"] == mode]
                ax.plot([r["subdivisions"] for r in rows], [1000*r["work_over_mu_normal_mm"] for r in rows],
                        style, marker="o" if mode == "fixed_velocity" else "x", color=color,
                        label=f"gap {1000*gap:g} um, "+("fixed eps_v" if style == "-" else "fixed eps_u"))
        ax.axhline(0, color="black", lw=.8)
        ax.set_xscale("log", base=2)
        ax.set(title="A. Common rotation: nonzero apparent slip", xlabel="Subdivisions of 0.1 rad rotation",
               ylabel="Work / (mu * normal load) (um)")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2)
        for a, rows in baselines.items():
            depths = [r["depth_mm"] for r in rows]
            style = markers[a]+"-"
            axes[0, 1].plot(depths, [100*r["contacts"]["groups"]["blade"]["smooth_absolute_load_fraction"]
                                    for r in rows], style, color=colors[a], label=f"{a} deg")
            axes[1, 0].plot(depths, [r["material"]["hold"]["hold"][-1]["weighted_stress_ratio"]
                                    for r in rows], style, color=colors[a], label=f"{a} deg")
            axes[1, 1].plot(depths, [r["material"]["bearing"]["J_min"] for r in rows],
                            style, color=colors[a], label=f"{a} deg minimum")
            axes[1, 1].plot(depths, [r["material"]["bearing"]["J_volume_average"] for r in rows],
                            "--", color=colors[a], label=f"{a} deg volume mean")
        axes[0, 1].set(title="B. Baseline contact in smoothing region", xlabel="Depth (mm)",
                      ylabel="Absolute stencil-load fraction (%)", ylim=(-3, 103))
        axes[0, 1].legend()
        axes[1, 0].axhline(np.exp(-.1/1.7), color="black", ls=":", label="Small-strain exp(-0.1 / 1.7)")
        axes[1, 0].set(title="C. Own elastic state held at fixed F for 0.1 s", xlabel="Depth (mm)",
                      ylabel="Stress-norm integral / initial value", ylim=(.85, 1.005))
        axes[1, 0].legend(fontsize=9)
        axes[1, 1].set(title="D. Bearing-region volume change", xlabel="Depth (mm)", ylabel="J (dimensionless)",
                      ylim=(0, 1.05))
        axes[1, 1].legend(fontsize=8, ncol=2)
        for ax in axes.ravel():
            ax.grid(alpha=.2)
        fig.suptitle("Contact kinematics, regularization and actual Maxwell state")
        fig.savefig(out/"audit.png", dpi=170)
        fig.savefig(out/"audit.svg")
        plt.close(fig)
    def html_table(headers, rows):
        return "<table><tr>"+"".join(f"<th>{escape(h)}</th>" for h in headers)+"</tr>"+"".join(
            "<tr>"+"".join(f"<td>{escape(str(v))}</td>" for v in row)+"</tr>" for row in rows)+"</table>"
    failures_table = html_table(["失败场", "原残差 N", "相邻作用自由力范数 N", "代数移除后残差 N"],
        [[s["case"], f"{s['residual_n']:.6f}",
          f"{s['contacts']['groups']['adjacent_body_vv']['free_force_norm_n']:.6f}",
          f"{s['residual_without_adjacent_n']:.6f}"] for s in cases if s["status"] != "accepted"])
    state_table = html_table(["角度/.5 mm", "平滑承载覆盖", "平滑项Fz N", "承载区平均/最小J",
                             "支路弹性最大等容伸长", "局部τ<0.85 s应力权重", "停留.1 s应力余量"],
        [[a, f"{s['smooth_blade_absolute_stencil_load_fraction']:.1%}", f"{s['smooth_blade_friction_fz_n']:.3f}",
          f"{s['bearing_J_mean']:.3f} / {s['bearing_J_min']:.3f}", f"{s['bearing_elastic_iso_max']:.3f}",
          f"{s['fast_half_stress_weight_fraction']:.1%}", f"{s['hold_01_stress_ratio']:.1%}"]
         for a in (0, 33, 57) for s in raw_summary if s["case"] == f"baseline_{a}" and abs(s["depth_mm"]-.5) < 1e-9])
    slopes_table = html_table(["角度", "实验斜率：前→后 N/mm", "FE斜率：前→后 N/mm",
                               "Maxwell功率分量割线：前→后 N/mm"],
        [[a, f"{ss[0]['experiment_slope_n_mm']:.2f} → {ss[1]['experiment_slope_n_mm']:.2f}",
          f"{ss[0]['fe_slope_n_mm']:.2f} → {ss[1]['fe_slope_n_mm']:.2f}",
          f"{ss[0]['maxwell_power_secant_n_mm']:.2f} → {ss[1]['maxwell_power_secant_n_mm']:.2f}"]
         for a in (0, 33, 57) for ss in [[r for r in slope_rows if r["angle"] == a]]])
    logs_html = "".join("<details><summary>"+escape(folder)+"：已结束，"
                       +f"{elapsed[folder]:.2f} s</summary><pre>"
                       +escape((out/folder/"run.log").read_text(encoding="utf-8"))+"</pre></details>"
                       for folder in folders)
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>接触运动学、时间正则与实际Maxwell状态审查</title>
<style>body{{max-width:1200px;margin:30px auto;padding:0 22px;font:16px/1.7 system-ui,sans-serif;color:#1b2d38}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ccc;text-align:left}}
img{{max-width:100%}}a{{color:#075a9c}}.box{{padding:18px;background:#eef4f8;border-radius:10px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f6;padding:12px}}</style>
<h1>接触运动学、摩擦正则与实际Maxwell状态</h1>
<p>2026-10-20；全部有界检查结束。36个既有接受态重新通过原验收，另审查3个拒绝场。
没有新增FE加载、材料拟合或主线源码改动。原共同材料三角度.1/.25/.5/.75/1 mm共15态，
近期三组候选共24态。超过已知起裂或完整性不明区间只作为数值场，不作完整体实验标定。</p>
<div class="box"><b>结论：</b>已实测共同刚体转动下的摩擦功、固定滑移正则引入的时间步依赖，
以及部分实际场中的相邻边界排斥。实际承载区存在加速松弛，但没有出现材料点极端示例的十倍加速。
这支持先审查接触方程，再重新判断材料；尚不能给几类误差分配因果百分比。</div>
<h2>1. 共同转动与相邻边界</h2>
<p>同一坐标旋转的协变检查通过，纯共同平移摩擦功为0；但保持间隙的共同刚体转动产生非零摩擦功。
夹具间隙1.5 µm、共同转角0.1 rad时，摩擦功2.03028×10⁻⁴ N·mm。
固定速度阈值后，32个转动子步仍给出等效滑程约0.150 µm，未趋零。
三种间隙下均符合局部投影的g·sin(Δθ)，细分后累计趋近g·θ。</p>
<p>原因是屏障使两个作用位置相隔有限距离g；共同转动时这两个不同位置的速度差被当前切向投影计为滑移。
这暴露有限间隙摩擦运动学的非客观作用，不是单纯坐标变换错误，也不能由线性求解容差消除。
固定εu细分后功变小混入了摩擦阈值改变，不可据此宣布客观性恢复。
本刀具实际只平移，因此夹具的功不能直接当作实际切割中的虚假功占比。</p>
{failures_table}
<p>fast/57的两对相邻材料节点为10101–10104、10104–10110：法向力范数.350586 N，
摩擦力范数.122705 N，合成自由力范数.371439 N。它们位于同一连续材料边界，库仑项也参与平衡；
这不是需要真实接触的独立裂面。代数移除后残差反而增大，是因为原状态已经平衡了部分这些作用，
不能据此认定该作用合理，也不能把移除后的场当新解。其它两失败场该项为零，排除它作为三次失败共同原因。</p>
<p>39态逐项装配法向/摩擦力并核对总和；直接相邻VV与其它自接触分开。
不删除全部自接触，不把improved-max负组合权重当负压力。原生刀鼻分支切换的既有证据仍然保留。</p>
<h2>2. 固定滑移阈值会改变缩步后的摩擦近似</h2>
<p>主线固定εu=10⁻⁵ mm，用实际位移增量求摩擦；Δt=Δd/v。
当前Δd=.01 mm、v=.2 mm/s给εv=.0002 mm/s；半步则εv=.0004 mm/s。
固定几何/法向力测试中，v滑=.0001 mm/s时，固定εu使摩擦/μN从.75变为.4375，
再半步为.234375；采用εu=εvΔt则三者均为.75。24个单接触控制与原核多项式吻合。</p>
<p>这一关系也见<a href="https://ipc-sim.github.io/file/IPC-paper-350ppi.pdf">IPC原文第5.3节</a>。
阈值在本项目是数值正则，尚无实验依据把它认定为真实界面速度尺度。
后续应明确选择固定速度正则，或做趋于库仑极限的控制；两者不能与材料时间细分混用。</p>
{state_table}
<p>“覆盖”按接触项法向载荷绝对值加权，保留正负组合；不是宏观力占比。
平滑项Fz使用真实带符号装配。平滑区含近似黏着，进入平滑区本身不证明摩擦错误。
全体39场也核实：固定几何/法向力下，位移和εu同时减半恢复同一摩擦力；
只减半位移则改变摩擦。这些是条件代数控制，没有推进加载。</p>
<h2>3. 实际Maxwell弹性状态与可压缩性</h2>
<p>从每个积分点自己的提交逆黏性度量恢复三维平面应变bₑ=F Cᵥ⁻¹ Fᵀ，
不是用总伸长冒充支路弹性伸长。平衡等容、体积、Maxwell应力之和与原核重算一致。
停留曲线是三维支路Kirchhoff应力Frobenius范数的参考体积积分，按自身初值归一化；
局部时间用应力范数的瞬时对数衰减率倒数估计，不假设整段为单指数。
载荷区按装配后刀具法向节点力的90%累计幅值选节点，再取关联三角形；这是明确的局部选区，不是90% Fz。</p>
<p>材料点等容伸长1.001/1.5/3/6时，局部应力衰减时间分别为
{controls[0]['instantaneous_stress_decay_s']:.3f}/{controls[1]['instantaneous_stress_decay_s']:.3f}/
{controls[2]['instantaneous_stress_decay_s']:.3f}/{controls[3]['instantaneous_stress_decay_s']:.3f} s。
1.7 s是小应变应力松弛时间，大变形可快很多。但本次39个实际场的选定承载区，
应力权重中τ局部&lt;0.1τ名义的比例全部为0；因此不能把极端材料点示例直接搬到整个切割模型。</p>
<p>实际状态在固定总F下停留.1 s，原主线.5 mm的支路应力积分下降约9.6%/11.2%/8.8%；
小应变名义律约下降5.7%。加速确实存在，但不是所有深度单调增强。
停留步长.005/.0025 s对最终应力比最大差{100*hold_difference:.4f}个百分点；
瞬时衰减估计1e−4/5e−5 s也进行了复核。没有以能量衰减时间冒充应力松弛时间。</p>
<p>体积能仍为K(J−1)²/2，体积响应不松弛。原主线1 mm的承载区最小J约.266/.140/.456，
可压缩性不可忽略。K固定、剪切松弛意味着小应变等效泊松比由瞬时约.471变到长期约.497；
这不自动构成错误，但不是全程固定同一泊松比。低J在所有3个本轮失败场中不是同样严重，
不能将体积项直接认定为共同失效根因。</p>
<h2>4. 与实验斜率差的联系</h2>
{slopes_table}
<p>前/后区间是.1–.25/.25–.5 mm。实验与FE斜率用各自区间全部样本直线回归；
Maxwell列为两端实际增量功率记账的割线，不是独立分量FE求解或相同回归估计。
实验保留各自牛顿值与零点，仅µm转mm。实际功率分解含接触项并逐态闭合到总节点功。</p>
<p>0°/33°的仿真斜率下降伴随Maxwell功率分量增速下降，而实验在同窗增斜率；
它定位了与差异同向的承载分量。普通历史松弛、局部非线性加速、接触重分配和体积约束仍耦合，
不能把该记账当作材料是唯一原因，更不能按各角度分别改G或τ。</p>
<img src="audit.png" alt="共同转动摩擦功、摩擦平滑承载覆盖、实际材料固定变形松弛以及承载区体积变化四面板。">
<p>图无平滑或外推；连线仅连接已检查状态；没有统计置信区间。C以各自初始支路应力积分归一化。
D同时显示最小J和参考体积加权平均，避免平均值掩盖压缩。原数据见下方CSV/JSON。</p>
<h2>5. 下一步与边界</h2>
<p>先建立同时满足共同转动零滑移、固定速度下时间细分一致、连续相邻边界不虚假排斥的接触候选，
并保持独立边界碰撞及真正自接触。需要一致残差/切线，不能只改日志或从力里随意扣项。
先过小夹具和相同失败场的有界原验收，再核查邻近加载；本轮没有擅自采用候选或开长队列。</p>
<p>然后在统一的接触语义下重新比较共同材料，才决定是否改变松弛谱或体积律。
当前无法给“松弛、可压缩性、接触误差”分配因果百分比；需要重新平衡的隔离对照。
29项已有接触/材料回归通过，0跳过。既有回归通过不等于新增共同转动物理检验通过。</p>
<p>旧冻结类缺少便利积分权重方法，三个基线审查首次在数据适配阶段停止；
适配器改用原P1面积×厚度后完成，物理源未修改。这不是FE求解失败。</p>
<h2>已结束的运行日志</h2>{logs_html}
<h2>可复核数据与日志</h2><ul>
<li><a href="analysis.json">出处哈希、统计方法与预算</a> · <a href="state_summary.csv">39态汇总</a> ·
<a href="slope_diagnostics.csv">斜率与功率分量</a> · <a href="regressions.log">29项回归</a></li>
<li><a href="fixtures/fixtures.json">全部小夹具</a> · <a href="fixtures/rigid_motion.csv">刚体运动表</a> ·
<a href="fixtures/smoothing_control.csv">固定法向力摩擦表</a> · <a href="fixtures/material_controls.json">材料点控制</a></li>
<li><a href="fields/fields.json">近期24态全部分量</a> · <a href="fields/run.log">24态审查日志</a> ·
<a href="fields/fast_57_failed_contacts.csv">fast57失败场逐接触项</a></li>
<li><a href="baseline_0_v2/fields.json">原主线0°</a> · <a href="baseline_33_v2/fields.json">原主线33°</a> ·
<a href="baseline_57_v2/fields.json">原主线57°</a></li></ul></html>"""
    (out/"index.html").write_text(html, encoding="utf-8")
    print(json.dumps(dict(states=len(fields_all), accepted=analysis["accepted"],
        verified_inputs=len(provenance), hold_refinement_max=hold_difference, report=str(out/"index.html"))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fixtures", "fields", "baseline", "report"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--study", type=Path, default=ROOT/"outputs/common_material_fe_20261020")
    p.add_argument("--cases", nargs="+", default=["soft_0", "soft_33", "soft_57",
                   "fast_0", "fast_33", "fast_57", "partition_0", "partition_33", "partition_57"])
    p.add_argument("--seconds", type=float, default=600.)
    args = p.parse_args()
    out = args.output.resolve()
    if args.mode == "report":
        report(out)
        return
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/"diagnostic_source.py")
    evidence = {str(Path(__file__).resolve()): sha(__file__)}
    started = time.monotonic()
    status = dict(status="running", mode=args.mode, budget_s=args.seconds)
    write(out/"status.json", status)
    with (out/"run.log").open("w", encoding="utf-8", buffering=1) as f, contextlib.redirect_stdout(f):
        try:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(1):
                if args.mode == "fixtures":
                    fixture_checks(out, evidence, started+args.seconds)
                    material_controls(out, started+args.seconds)
                else:
                    if args.mode == "baseline" and (len(args.cases) != 1 or args.cases[0] not in ("0", "33", "57")):
                        raise ValueError("One baseline angle per process preserves frozen-module isolation")
                    fields(out, args.study.resolve(), args.cases, evidence, started+args.seconds,
                           baseline=args.mode == "baseline")
            status["status"] = "completed"
        except Exception as exc:
            status.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            print(traceback.format_exc(), flush=True)
        status.update(elapsed_s=time.monotonic()-started, input_sha256=evidence,
            input_hashes_unchanged=all(sha(path) == digest for path, digest in evidence.items()))
        write(out/"status.json", status)
    print(json.dumps({k:v for k,v in status.items() if k not in ("input_sha256", "traceback")}))
    if status["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
