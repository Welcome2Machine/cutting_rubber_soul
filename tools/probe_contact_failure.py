"""Exact-history contact failure probes; no failed iterate is an equilibrium.

Import the failed case's frozen physical implementation, then load an explicit
current solver candidate separately. Never weaken production resume checks.
"""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("probe", "candidate", "trace", "components", "branches"), default="probe")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--seconds", type=float, default=600)
    args = parser.parse_args()
    folder, out = args.case.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT/".runtime/ipc-python"))
    sys.path.insert(0, str(folder/"source"))
    import numpy as np
    from scipy.sparse import diags
    from scipy.sparse.linalg import lsmr
    from threadpoolctl import threadpool_limits
    from tools.run_contact_controls import make
    from tools.run_common_material_study import verify_case
    from prototype.material_case import signature
    from prototype.ipc_reference import BASELINE, IPCProblem
    from prototype.pardiso import PardisoDirection
    from run_support import write_json, save_npz
    # The probe helpers are analysis-only; imported explicitly from this tree.
    spec = importlib.util.spec_from_file_location("failure_helpers", ROOT/"tools/diagnose_ptc_directions.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    r = helper.read(folder/"result.json")
    evidence = verify_case(folder)
    p = make(r["job"]["angle"], r["job"]["parameters"], r["job"]["group"])
    assert json.loads(json.dumps(signature(p))) == r["signature"]
    p.parallel_linear = PardisoDirection(8, runtime=str(ROOT/".runtime/pardiso"))
    summary = dict(mode=args.mode, source_case=str(folder), evidence=evidence, seconds=args.seconds,
                   interpretation="Conditional solver diagnostic, not a new material model")
    began = time.monotonic()
    try:
        with (out/"run.log").open("w", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), threadpool_limits(1):
            last, target, failed = helper.restore_prefix(p, folder, r, evidence)
            summary.update(target_mm=target, accepted_start_mm=float(last["depth_mm"]))
            if args.mode == "trace":
                summary["trace"] = helper.trace(p, r, folder, out, args.seconds, evidence, IPCProblem)
            elif args.mode == "candidate":
                spec = importlib.util.spec_from_file_location("prototype.residual_trust", ROOT/"prototype/residual_trust.py")
                candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
                evidence[str(ROOT/"prototype/residual_trust.py")] = helper.sha(ROOT/"prototype/residual_trust.py")
                p.__class__ = type("ResidualTrustControl", (candidate.ResidualTrustMixin, type(p)), {})
                p.probe_directory = out
                x, old = last["positions"].copy(), float(last["depth_mm"])
                summary["steps"] = []
                deadline = began+args.seconds
                # Same original interval, then three neighbors, with own histories.
                for i in range(4):
                    depth = round(target+i*r["job"]["increment"], 12)
                    x, row = p.step(x, old, depth, deadline)
                    data = dict(positions=x, previous_positions=p.previous,
                        reference_xy=p.rest, triangles=p.mesh.triangles, depth_mm=depth,
                        body_nodes=p.n, **p.checkpoint_data())
                    save_npz(out/f"state_{i:02d}.npz", **data)
                    row["fresh_audit"] = p.audit_state(data, row)
                    summary["steps"].append(row)
                    write_json(out/"summary.json", summary)
                    old = depth
                summary["status"] = "completed"
            else:
                if args.snapshot:
                    failed = helper.load(args.snapshot)
                    evidence[str(args.snapshot.resolve())] = helper.sha(args.snapshot)
                x = failed["positions"]
                p.previous = failed["previous_positions"].copy()
                p.previous_material_metric = failed["previous_material_metric"].copy()
                p.material_dt = float(failed["material_dt_s"])
                p._contact_initial_preparation = bool(failed.get("contact_initial_preparation", False))
                lag = p.lag(x)
                _, g, H, j, M = p.evaluate(x, lag, projected=False, with_metric=True)
                residual = g[p.free]; norm = float(np.linalg.norm(residual))
                A = p.coupled_friction_tangent(x, lag, H)[p.free][:, p.free].tocsc()
                M = M[p.free][:, p.free]+diags(np.full(len(p.free), BASELINE["hessian_regularization_n_mm"]))
                import re
                lines = (folder/"run.log").read_text(encoding="utf-8")
                shift = float(re.findall(r"FRICTION_PTC depth_mm=\S+ iteration=\d+ fresh_residual_N=\S+ metric_shift=(\S+)", lines)[-1])
                summary.update(norm_n=norm, min_j=float(j), dofs=len(residual), nnz=A.nnz, actual_shift=shift, directions=[])
                if args.mode == "branches":
                    du=p.linear_direction(A,-residual)
                    direction=np.zeros(p.total);direction[p.free]=du;direction=direction.reshape(x.shape)
                    unit=direction/np.max(abs(direction))
                    branch_matrices=[]
                    for sign in (-1,1):
                        y=x+sign*1e-7*unit
                        if p.safe_step(x,sign*1e-7*unit)<1.-1e-10:raise ValueError("Branch sampling geometry limited")
                        lag_y=p.lag(y);_,_,hy,_=p.evaluate(y,lag_y,projected=False)
                        branch_matrices.append(p.coupled_friction_tangent(y,lag_y,hy)[p.free][:,p.free])
                    minus,plus=branch_matrices
                    for weight in (0.,.25,.5,.75,1.):
                        matrix=(1-weight)*minus+weight*plus
                        du=p.linear_direction(matrix,-residual)
                        d=np.zeros(p.total);d[p.free]=du;d=d.reshape(x.shape)
                        maximum=float(abs(du).max());trial,alpha,geo=p._ptc_trial(x,d,maximum)
                        row=dict(plus_weight=weight,geometric_fraction=float(geo),trials=[])
                        for fraction in (1.,.5,.125,.03125):
                            a=alpha*fraction;y=x+a*d
                            rt=p.evaluate(y,p.lag(y),tangent=False)[1][p.free]
                            row["trials"].append(dict(alpha=a,norm_n=float(np.linalg.norm(rt)),
                                displacement_infinity_mm=a*maximum))
                        summary["directions"].append(row)
                    summary["status"]="completed"
                    write_json(out/"summary.json",summary)
                    return
                if args.mode == "components":
                    du=p.linear_direction(A,-residual)
                    direction=np.zeros(p.total);direction[p.free]=du;direction=direction.reshape(x.shape)
                    save_npz(out/"newton_field.npz",positions=x,direction=direction,r=residual,
                        previous_positions=p.previous,previous_material_metric=p.previous_material_metric,
                        material_dt_s=p.material_dt,shift=0.,alpha=1.)
                    summary["components"]=helper.component_probe(p,out/"newton_field.npz",out,args.seconds)
                    base=helper.force_parts(p,x)
                    normal_h=p.barrier.hessian(p.collisions(x),p.cmesh,x[p.surface],
                                               p.ipc.PSDProjectionMethod.NONE)
                    unit=direction/np.max(abs(direction))
                    an=np.zeros(p.total);an[p.surface_dofs]=normal_h@unit[p.surface].ravel()
                    summary["one_sided_normal"]=[]
                    for h in (1e-6,1e-7,1e-8,1e-9,1e-10):
                        plus=helper.force_parts(p,x+h*unit)["normal"]
                        minus=helper.force_parts(p,x-h*unit)["normal"]
                        dp=(plus-base["normal"])/h;dm=(base["normal"]-minus)/h
                        err=(dp-dm).reshape(-1,2)[:p.n]
                        ids=np.argsort(np.linalg.norm(err,axis=1))[-3:][::-1]
                        summary["one_sided_normal"].append(dict(h_mm=h,
                            plus_error=float(np.linalg.norm((dp-an)[p.free])),
                            minus_error=float(np.linalg.norm((dm-an)[p.free])),
                            plus_norm=float(np.linalg.norm(dp[p.free])),
                            minus_norm=float(np.linalg.norm(dm[p.free])),
                            cross_gap_force_n=float(np.linalg.norm((plus-minus)[p.free])),
                            nodes=[helper.local_contact_geometry(p,x,int(i)) for i in ids]))
                    summary["status"]="completed"
                    write_json(out/"summary.json",summary)
                    return
                keys = helper.feature_keys(p.collisions(x))
                scale = 1/np.maximum(np.sqrt(np.asarray(A.power(2).sum(axis=0)).ravel()), 1e-12)
                B = A@diags(scale)
                for name, amount in (("actual_ptc", shift), ("newton", 0.), ("lm", .001), ("lm", .01), ("lm", .1)):
                    if time.monotonic()-began > args.seconds: raise TimeoutError("Fixed-field probe")
                    tick = time.monotonic(); detail = {}
                    if name == "lm":
                        solution = lsmr(B, -residual, damp=amount, atol=1e-9, btol=1e-9, maxiter=1500)
                        du = scale*solution[0]
                        detail.update(lsmr_stop=int(solution[1]), lsmr_iterations=int(solution[2]))
                    else:
                        du = p.linear_direction(A+amount*M, -residual)
                    action = A@du; direction = np.zeros(p.total); direction[p.free] = du; direction = direction.reshape(x.shape)
                    maximum = float(abs(du).max())
                    geo = float(p.safe_step(x, direction)); cap = .5*p.options.tip_radius_mm/maximum
                    alpha = min(1., geo, cap)
                    row = dict(name=name, shift_or_damp=amount, solve_s=time.monotonic()-tick,
                        slope_over_norm2=float(residual@action/norm**2), geometric_fraction=geo,
                        tip_cap_alpha=cap, maximum_mm=maximum, trials=[], **detail)
                    for fraction in (1., .5, .125, .03125, .0078125, .001953125):
                        a = alpha*fraction; trial = x+a*direction
                        rt = p.evaluate(trial, p.lag(trial), tangent=False)[1][p.free]
                        row["trials"].append(dict(alpha=a, displacement_infinity_mm=a*maximum,
                            predicted_norm_n=float(np.linalg.norm(residual+a*action)),
                            norm_n=float(np.linalg.norm(rt)),
                            model_error_n=float(np.linalg.norm(rt-residual-a*action)),
                            features_changed=len(keys ^ helper.feature_keys(p.collisions(trial)))))
                    summary["directions"].append(row)
                    write_json(out/"summary.json", summary)
                summary["status"] = "completed"
    except Exception as exc:
        summary.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    finally:
        summary["elapsed_s"] = time.monotonic()-began
        summary["evidence"] = evidence
        write_json(out/"summary.json", summary)
        p.parallel_linear.close()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("evidence", "directions", "steps")}, indent=2))


if __name__ == "__main__":
    main()
