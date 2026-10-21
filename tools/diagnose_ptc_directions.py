"""Bounded instrumentation of a frozen, exact-history failed PTC increment.

Trace calls the original solve_friction without replacing arithmetic or decisions.
Probes use saved actual directions and shifts; none is an accepted FE state.
"""
import argparse
import ast
import contextlib
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def environment(study, case):
    result = read(study / case / "result.json")
    evidence = {}
    for name, digest in result["source_sha256"].items():
        path = study / "source" / name
        assert sha(path) == digest, name
        evidence[str(path)] = digest
    sys.path.insert(0, str(ROOT / ".runtime/ipc-python"))
    sys.path.insert(0, str(study / "source"))
    from tools.run_common_material_study import make, signature, PardisoDirection
    from prototype.ipc_reference import IPCProblem, BASELINE
    p = make(result["job"]["angle"], result["job"]["parameters"])
    assert json.loads(json.dumps(signature(p))) == result["signature"]
    assert p.enrichment is None and p.nodal_volume is None
    assert BASELINE["hessian_regularization_n_mm"] == 1e-9
    p.parallel_linear = PardisoDirection(8, runtime=str(ROOT / ".runtime/pardiso"))
    return p, result, evidence, IPCProblem, BASELINE


def restore_prefix(p, folder, result, evidence):
    import numpy as np
    count = len(result["steps"])
    states = []
    for i in (count - 2, count - 1):
        path = folder / "states" / f"{i:04d}.npz"
        evidence[str(path)] = sha(path)
        z = load(path)
        p.audit_state(z, result["steps"][i])
        states.append(z)
    older, last = states
    p.previous = last["positions"].copy()
    p.material_metric = last["material_metric"].copy()
    p.material_dissipation = float(last["cumulative_viscous_dissipation_n_mm"])
    p.seed_predictor_history([(float(s["depth_mm"]), s["positions"]) for s in states])
    target = result["rejected_load_attempts"][-1]["depth_mm"]
    failed = load(folder / "failed_trial.npz")
    evidence[str(folder / "failed_trial.npz")] = sha(folder / "failed_trial.npz")
    np.testing.assert_array_equal(failed["previous_positions"], last["positions"])
    np.testing.assert_array_equal(failed["previous_material_metric"], last["material_metric"])
    return last, target, failed


def trace(p, result, folder, out, seconds, evidence, IPCProblem):
    import numpy as np
    last, target, original_failed = restore_prefix(p, folder, result, evidence)
    old = float(last["depth_mm"])
    source_lines, first = inspect.getsourcelines(IPCProblem.solve_friction)
    accept_line = first + next(i for i, line in enumerate(source_lines) if line.strip() == "x = trial")
    reject_line = first + next(i for i, line in enumerate(source_lines)
                              if 'print(f"FRICTION_PTC_REJECT' in line)
    rows = []
    snapshots = []
    observed = set()
    # Include the full final stagnation window and its turning points. Save only
    # the first two predicted-growth events there, plus two preceding updates.
    from_log = (folder / "run.log").read_text(encoding="utf-8")
    import re
    entries = [(int(i), float(r)) for d, i, r in re.findall(
        r"FRICTION_PTC depth_mm=([\d.eE+-]+) iteration=(\d+) fresh_residual_N=([\d.eE+-]+)", from_log)
        if abs(float(d) - target) < 1e-9]
    best, progress = float("inf"), 0
    for i, norm in entries:
        if norm < .9 * best:
            best, progress = norm, i
    selected_growth = 0

    def local(frame, event, arg):
        nonlocal selected_growth
        if event != "line" or frame.f_lineno not in (accept_line, reject_line):
            return local
        v = frame.f_locals
        event_key = (int(v["iteration"]), int(v["retry"]), frame.f_lineno == accept_line)
        # Python can emit the starting line repeatedly while evaluating a
        # multi-line f-string. That is one rejected trial, not multiple retries.
        if event_key in observed:
            return local
        observed.add(event_key)
        r, du, A, M = (v[k] for k in ("r", "du", "A", "M"))
        action = v["linear_action"]
        md = M @ du
        hd = v["H"][p.free][:, p.free] @ du
        alpha, norm, shift = (float(v[k]) for k in ("alpha", "norm", "shift"))
        pred = r + alpha * action
        accepted = frame.f_lineno == accept_line
        slope = float(r @ action)
        row = dict(iteration=int(v["iteration"]), retry=int(v["retry"]), accepted=accepted,
            norm_n=norm, nt_n=float(v["nt"]), shift_used=shift, alpha=alpha,
            geometric_fraction=float(v["geometric_fraction"]),
            tip_cap_alpha=.5*p.options.tip_radius_mm/float(v["maximum"]),
            displacement_infinity_mm=alpha*float(v["maximum"]),
            predicted_norm_n=float(np.linalg.norm(pred)), residual_ratio=float(v["nt"])/norm,
            model_error_n=float(v["error"]), slope=slope, slope_over_norm2=slope/norm**2,
            predicted_best_alpha=float(np.clip(-slope/max(float(action@action),1e-300),0,alpha)),
            linear_relative_residual=float(np.linalg.norm(action+shift*md+r)/norm),
            action_norm_n=float(np.linalg.norm(action)), metric_action_norm_n=float(np.linalg.norm(md)),
            frozen_hessian_action_norm_n=float(np.linalg.norm(hd)),
            friction_coupling_action_norm_n=float(np.linalg.norm(action-hd)),
            psd_metric_change_action_norm_n=float(np.linalg.norm(md-hd)),
            directional_metric=float(du@md), directional_jacobian=float(du@action))
        rows.append(row)
        with (out/"trace.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
        save = accepted and row["iteration"] in (progress-1, progress)
        if accepted and row["iteration"] > progress and row["predicted_norm_n"] > norm:
            if selected_growth < 2:
                save = True
            selected_growth += 1
        if save:
            name = f'iteration_{row["iteration"]:03d}.npz'
            np.savez_compressed(out/name, positions=v["x"], direction=v["direction"], r=r,
                previous_positions=p.previous, previous_material_metric=p.previous_material_metric,
                material_dt_s=p.material_dt, alpha=alpha, shift=shift)
            snapshots.append(dict(file=name, row=row, sha256=sha(out/name)))
        return local

    def dispatch(frame, event, arg):
        return local if frame.f_code is IPCProblem.solve_friction.__code__ else None

    report = dict(scope="original frozen PTC; original histories, initial guess, backend and stop rules",
                  old_depth_mm=old, target_depth_mm=target, prior_progress_iteration=progress,
                  solve_budget_s=seconds)
    began = time.monotonic()
    try:
        sys.settrace(dispatch)
        x, record = p.step(last["positions"].copy(), old, target, began+seconds)
        report.update(status="completed", record=record)
    except Exception as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    finally:
        sys.settrace(None)
    report.update(solve_s=time.monotonic()-began, snapshots=snapshots,
        updates=sum(r["accepted"] for r in rows), rejected=sum(not r["accepted"] for r in rows),
        terminal_position_max_difference_mm=float(np.max(abs(p.trial-original_failed["positions"]))),
        linear_calls=len(p.parallel_linear.rows),
        linear_fallbacks=sum(r["fallback_reason"] is not None for r in p.parallel_linear.rows))
    write(out/"trace_summary.json",report)
    return report


def feature_keys(c):
    keys = {("vv",int(s.vertex0_id),int(s.vertex1_id)) for s in c.vv_collisions}
    keys |= {("ev",int(s.edge_id),int(s.vertex_id)) for s in c.ev_collisions}
    return keys


def force_parts(p,x,lag=None,normal=None):
    import numpy as np
    c=p.collisions(x) if normal is None else normal
    lag=p.lag(x) if lag is None else lag
    r,_,dofs,_,_=p.material_bulk((x[:p.n]-p.mesh.reference_xy).ravel(),with_tangent=False)
    bulk=np.zeros(p.total);np.add.at(bulk,dofs.ravel(),r.ravel())
    normal_g=np.zeros(p.total);friction_g=np.zeros(p.total)
    normal_g[p.surface_dofs]=p.barrier.gradient(c,p.cmesh,x[p.surface])
    friction_g[p.surface_dofs]=p.friction.gradient(lag,p.cmesh,(x-p.previous)[p.surface])
    return dict(bulk=bulk,normal=normal_g,friction=friction_g)


def local_contact_geometry(p,x,node):
    """Explicit point/segment geometry and native stencils, without filtering them."""
    import numpy as np
    c=p.collisions(x);pairs=[]
    for kind,stencils in (("vv",c.vv_collisions),("ev",c.ev_collisions)):
        for s in stencils:
            local=([int(s.vertex0_id),int(s.vertex1_id)] if kind=="vv" else
                   [int(s.vertex_id),*map(int,p.cmesh.edges[s.edge_id])])
            ids=p.surface[local]
            if node not in ids:continue
            weight=getattr(s,"weight",None)
            pairs.append(dict(kind=kind,global_nodes=list(map(int,ids)),
                blade_local_nodes=[int(i-p.n) for i in ids if i>=p.n],
                weight=None if weight is None else float(weight)))
    edges=[]
    for i,edge in enumerate(p.cmesh.edges):
        ids=p.surface[edge]
        if not np.all(ids>=p.n):continue
        a,b=x[ids];v=b-a; t=float((x[node]-a)@v/(v@v))
        distance=float(np.linalg.norm(x[node]-(a+np.clip(t,0,1)*v)))
        if distance<=2*p.dhat:
            edges.append(dict(collision_edge=i,blade_local_nodes=(ids-p.n).tolist(),
                projection_parameter=t,distance_mm=distance))
    return dict(node=node,xy=x[node].tolist(),stencils=pairs,nearby_blade_edges=edges)


def component_probe(p,snapshot,out,seconds):
    import numpy as np
    from scipy.sparse import coo_matrix
    z=load(snapshot);x=z["positions"];direction=z["direction"]
    p.previous=z["previous_positions"].copy()
    p.previous_material_metric=z["previous_material_metric"].copy()
    p.material_dt=float(z["material_dt_s"])
    deadline=time.monotonic()+seconds
    lag=p.lag(x);c=p.collisions(x)
    _,g,H,j=p.evaluate(x,lag,projected=False)
    A=p.coupled_friction_tangent(x,lag,H)
    def embed(m):
        m=m.tocoo()
        return coo_matrix((m.data,(p.surface_dofs[m.row],p.surface_dofs[m.col])),
                          shape=(p.total,p.total)).tocsc()
    mode=p.ipc.PSDProjectionMethod.NONE
    hn=embed(p.barrier.hessian(c,p.cmesh,x[p.surface],mode))
    hf=embed(p.friction.hessian(lag,p.cmesh,(x-p.previous)[p.surface],mode))
    matrices=dict(bulk=H-hn-hf,normal=hn,friction=hf+A-H)
    base=force_parts(p,x,lag,c)
    np.testing.assert_allclose(sum(base.values()),g,rtol=1e-10,atol=1e-12)
    actions={name:mat@direction.ravel() for name,mat in matrices.items()}
    report=dict(scope="fixed saved iterate; component residual/Jacobian audit",min_j=float(j),
        norm_n=float(np.linalg.norm(g[p.free])),fractions=[],finite_differences=[])
    hotspot=None
    fractions=((1/512,1/2048,1/4096) if snapshot.name=="newton_fallback.npz" else (1.,.5,.25))
    for fraction in fractions:
        if time.monotonic()>deadline:raise TimeoutError("Component probe budget")
        point=x+fraction*direction
        actual=force_parts(p,point)
        frozen=force_parts(p,point,lag,c)
        detail={}
        err=np.zeros(p.total)
        for name in base:
            error=actual[name]-base[name]-fraction*actions[name]
            err+=error
            detail[name]=dict(change_norm_n=float(np.linalg.norm((actual[name]-base[name])[p.free])),
                predicted_change_norm_n=float(np.linalg.norm(fraction*actions[name][p.free])),
                model_error_n=float(np.linalg.norm(error[p.free])),
                actual_minus_frozen_norm_n=float(np.linalg.norm((actual[name]-frozen[name])[p.free])))
        norms=np.linalg.norm(err.reshape(-1,2)[:p.n],axis=1)
        ids=np.argsort(norms)[-8:][::-1]
        if hotspot is None:
            hotspot=int(ids[0])
            report["base_contact_geometry"]=local_contact_geometry(p,x,hotspot)
        report["fractions"].append(dict(alpha=fraction,
            maximum_displacement_mm=fraction*float(np.max(abs(direction))),
            norm_n=float(np.linalg.norm(sum(actual.values())[p.free])),components=detail,
            error_nodes=[dict(node=int(i),error_n=float(norms[i]),xy=x[i].tolist()) for i in ids],
            contact_geometry=local_contact_geometry(p,point,hotspot)))
    maximum=float(np.max(abs(direction)));unit=direction/maximum
    for h in (1e-6,1e-7):
        if time.monotonic()>deadline:raise TimeoutError("Component probe budget")
        assert min(p.safe_step(x,h*unit),p.safe_step(x,-h*unit))>=1.-1e-10
        plus=force_parts(p,x+h*unit);minus=force_parts(p,x-h*unit)
        values={}
        for name in base:
            fd=(plus[name]-minus[name])/(2*h)
            pred=actions[name]/maximum
            values[name]=dict(fd_norm=float(np.linalg.norm(fd[p.free])),
                error_norm=float(np.linalg.norm((fd-pred)[p.free])),
                relative_error=float(np.linalg.norm((fd-pred)[p.free])/max(np.linalg.norm(fd[p.free]),1e-30)))
        report["finite_differences"].append(dict(h_mm=h,components=values))
    write(out/"components.json",report)
    return report


def probe(p, snapshot, out, seconds, baseline):
    import numpy as np
    from scipy.sparse import diags
    z = load(snapshot)
    x = z["positions"]
    p.previous=z["previous_positions"].copy()
    p.previous_material_metric=z["previous_material_metric"].copy()
    p.material_dt=float(z["material_dt_s"])
    deadline=time.monotonic()+seconds
    lag=p.lag(x)
    _,g,H,j,M=p.evaluate(x,lag,projected=False,with_metric=True)
    r=g[p.free]
    A=p.coupled_friction_tangent(x,lag,H)[p.free][:,p.free]
    M=M[p.free][:,p.free]+diags(np.full(len(p.free),baseline["hessian_regularization_n_mm"]))
    saved=z["direction"].ravel()[p.free]
    shift=float(z["shift"])
    np.testing.assert_allclose(r,z["r"],rtol=1e-7,atol=1e-10)
    initial_keys=feature_keys(p.collisions(x))
    report=dict(scope="fixed-iterate probes; not equilibria or load advancement",min_j=float(j),
                saved_shift=shift,norm_n=float(np.linalg.norm(r)),directions=[],derivatives=[])
    mr=p.linear_direction(M,r)
    mr_norm2=float(r@mr)
    choices=(("actual",shift),) if snapshot.name=="newton_fallback.npz" else (
        ("actual",shift),("newton",0.),("stronger_ptc",4*shift))
    for label,amount in choices:
        if time.monotonic()>deadline:raise TimeoutError("Direction probe wall budget")
        du=saved if label=="actual" else p.linear_direction(A+amount*M,-r)
        action=A@du
        full=np.zeros(p.total);full[p.free]=du;full=full.reshape(x.shape)
        maximum=float(np.max(abs(du)));safe=float(p.safe_step(x,full))
        cap=.5*p.options.tip_radius_mm/maximum
        alpha=min(1.,cap,safe)
        pred=r+alpha*action
        mp=p.linear_direction(M,pred)
        row=dict(name=label,shift=amount,alpha=alpha,geometric_fraction=safe,tip_cap_alpha=cap,
            displacement_infinity_mm=alpha*maximum,slope=float(r@action),
            slope_over_norm2=float(r@action)/float(r@r),predicted_norm_n=float(np.linalg.norm(pred)),
            linear_relative_residual=float(np.linalg.norm(action+amount*(M@du)+r)/np.linalg.norm(r)),
            predicted_metric_norm_ratio=float(np.sqrt(max(0.,pred@mp)/mr_norm2)),
            metric_initial_squared=mr_norm2,trials=[])
        fractions=tuple(2.**-i for i in range(14)) if snapshot.name=="newton_fallback.npz" else (
            1.,.5,.25,.125,.0625)
        for fraction in fractions:
            if time.monotonic()>deadline:raise TimeoutError("Direction probe wall budget")
            a=alpha*fraction;trial=x+a*full
            gt=p.evaluate(trial,p.lag(trial),tangent=False)[1][p.free]
            row["trials"].append(dict(alpha=a,norm_n=float(np.linalg.norm(gt)),
                model_error_n=float(np.linalg.norm(gt-r-a*action)),
                features_changed=len(initial_keys^feature_keys(p.collisions(trial)))))
        report["directions"].append(row)
        if label=="actual":
            unit=full/maximum;analytic=action/maximum
            for h in (1e-6,1e-7,1e-8):
                if min(p.safe_step(x,h*unit),p.safe_step(x,-h*unit))<1.-1e-10:
                    report["derivatives"].append(dict(h_mm=h,status="geometry_limited"));continue
                plus=x+h*unit;minus=x-h*unit
                gp=p.evaluate(plus,p.lag(plus),tangent=False)[1][p.free]
                gm=p.evaluate(minus,p.lag(minus),tangent=False)[1][p.free]
                fd=(gp-gm)/(2*h)
                report["derivatives"].append(dict(h_mm=h,status="evaluated",
                    relative_error=float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(fd),1e-30)),
                    one_sided_disagreement=float(np.linalg.norm((gp+gm-2*r)/h)/max(np.linalg.norm(fd),1e-30)),
                    features_changed_plus=len(initial_keys^feature_keys(p.collisions(plus))),
                    features_changed_minus=len(initial_keys^feature_keys(p.collisions(minus)))))
    write(out/"probe.json",report)
    return report


def candidate(p, result, folder, out, seconds, evidence, capture_fallback=False):
    """Existing 24-step residual Newton phase; same checked PARDISO backend."""
    import numpy as np
    from prototype import residual_newton
    last,target,_=restore_prefix(p,folder,result,evidence)
    original_type=type(p)
    original_linear=residual_newton.spsolve
    # Explicit single-process backend injection, restored in finally. This is
    # the ONLY change to the existing opt-in Newton implementation. Keep its
    # exact Armijo/backtracking and combined 500-direction fallback budget.
    residual_newton.spsolve=p.linear_direction
    p.__class__=type("DiagnosticPardisoNewton",(residual_newton.ResidualNewtonMixin,original_type),{})
    report=dict(scope="existing residual Newton phase plus original PTC fallback",
        linear_backend="same guarded PARDISO8 as control; injected into Newton spsolve call",
        original_depth_mm=float(last["depth_mm"]),target_mm=target,
        budget_s=seconds,maximum_directions=p.max_iterations,newton_phase_limit=p.newton_phase_limit,
        physical_acceptance="unchanged",steps=[])
    x=last["positions"].copy();old=float(last["depth_mm"])
    began=time.monotonic();deadline=began+seconds
    class CapturedFallback(Exception):
        pass
    function=residual_newton.ResidualNewtonMixin.solve_friction
    lines,first=inspect.getsourcelines(function)
    capture_line=first+next(i for i,line in enumerate(lines) if "print(f'NEWTON_FALLBACK original_PTC" in line)
    def capture(frame,event,arg):
        if event=="line" and frame.f_lineno==capture_line:
            v=frame.f_locals
            np.savez_compressed(out/"newton_fallback.npz",positions=v["x"],direction=v["direction"],
                r=v["r"],previous_positions=p.previous,previous_material_metric=p.previous_material_metric,
                material_dt_s=p.material_dt,shift=0.,alpha=v["alpha"])
            report["captured"]=dict(iteration=int(v["iteration"]),used=int(v["used"]),
                norm_n=float(v["norm"]),last_alpha=float(v["alpha"]),last_nt_n=float(v["nt"]),
                linear_relative=float(v["relative"]),slope=float(v["slope"]))
            raise CapturedFallback("Captured original Newton fallback; no PTC or physical acceptance")
        return capture
    def dispatch(frame,event,arg):
        return capture if frame.f_code is function.__code__ else None
    try:
        if capture_fallback:sys.settrace(dispatch)
        # Neighbors run only if the original failed increment independently
        # passes. They use this candidate's committed histories, no cutbacks.
        for index in range(4):
            depth=target+.01*index
            x,row=p.step(x,old,depth,deadline)
            state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,
                previous_positions=p.previous,depth_mm=depth,body_nodes=p.n,**p.checkpoint_data())
            path=out/f"accepted_{index:02d}.npz"
            np.savez_compressed(path,**state)
            row["fresh_audit"]=p.audit_state(state,row)
            row["checkpoint_sha256"]=sha(path)
            report["steps"].append(row);old=depth
            write(out/"candidate.json",report)
            print(f"CANDIDATE_ACCEPT depth_mm={depth:.9g} residual_n={row['residual_n']:.9g}",flush=True)
        report["status"]="completed_four_increments"
    except CapturedFallback:
        report["status"]="captured_before_ptc"
    except Exception as exc:
        report.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
        if hasattr(p,"trial"):
            np.savez_compressed(out/"failed_trial.npz",positions=p.trial,
                previous_positions=p.previous,**p.checkpoint_data())
    finally:
        sys.settrace(None)
        residual_newton.spsolve=original_linear
        p.__class__=original_type
    report.update(elapsed_s=time.monotonic()-began,linear_calls=len(p.parallel_linear.rows),
                  linear_fallbacks=sum(r["fallback_reason"] is not None for r in p.parallel_linear.rows))
    write(out/"candidate.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("trace","probe","components","candidate"))
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--case",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--snapshot",type=Path)
    parser.add_argument("--seconds",type=float,default=180.)
    parser.add_argument("--capture-newton-fallback",action="store_true",
                        help="Diagnostic stop at the original Newton-to-PTC handoff")
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise ValueError("Use independent output")
    out.mkdir(parents=True)
    shutil.copy2(__file__,out/"diagnostic_source.py")
    study=args.study.resolve()
    evidence={str(Path(__file__).resolve()):sha(__file__)}
    status=dict(status="starting",mode=args.mode,case=args.case,budget_s=args.seconds)
    write(out/"status.json",status)
    with (out/"run.log").open("w",encoding="utf-8",buffering=1) as log,contextlib.redirect_stdout(log):
        p=None
        try:
            p,result,source,IPCProblem,baseline=environment(study,args.case)
            evidence.update(source)
            from threadpoolctl import threadpool_limits
            from run_support import compute_process_policy
            for name in ("result.json","run.log"):
                path=study/args.case/name;evidence[str(path)]=sha(path)
            with threadpool_limits(1),compute_process_policy():
                if args.mode=="trace":
                    detail=trace(p,result,study/args.case,out,args.seconds,evidence,IPCProblem)
                elif args.mode=="candidate":
                    detail=candidate(p,result,study/args.case,out,args.seconds,evidence,args.capture_newton_fallback)
                else:
                    assert args.snapshot is not None
                    path=args.snapshot.resolve();evidence[str(path)]=sha(path)
                    detail=(component_probe(p,path,out,args.seconds) if args.mode=="components" else
                            probe(p,path,out,args.seconds,baseline))
            status.update(status="finished",result=detail)
        except Exception as exc:
            status.update(status="diagnostic_failed",error=repr(exc),traceback=traceback.format_exc())
            print(traceback.format_exc(),flush=True)
        finally:
            if p is not None and p.parallel_linear is not None:p.parallel_linear.close()
            status["input_sha256"]=evidence
            status["input_hashes_unchanged"]=all(sha(Path(path))==digest for path,digest in evidence.items())
            write(out/"status.json",status)
    print(json.dumps({k:v for k,v in status.items() if k not in ("result","input_sha256","traceback")}))


if __name__=="__main__":
    main()
