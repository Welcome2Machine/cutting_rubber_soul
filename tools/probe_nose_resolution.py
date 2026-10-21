"""Conditional polygon-resolution check with unchanged rubber histories.

This is NOT a production resume: only rigid prescribed blade arrays change
size. The body coordinates, previous material metric, physical dt and material
parameters are exact. A successful candidate still needs an independent path.
"""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--facets",type=int,required=True)
    parser.add_argument("--seconds",type=float,default=600)
    args=parser.parse_args()
    folder=args.case.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/".runtime/ipc-python"));sys.path.insert(0,str(folder/"source"))
    import numpy as np
    from threadpoolctl import threadpool_limits
    from prototype.ipc_viscoelastic import ViscoelasticProblem
    from prototype.contact_controls import ContactControlMixin
    from tools.run_common_material_study import ComputeMaterial, verify_case, K
    from tools.run_contact_controls import make, GROUPS
    from prototype.pardiso import PardisoDirection
    from run_support import save_npz, write_json
    spec=importlib.util.spec_from_file_location("failure_helpers",ROOT/"tools/diagnose_ptc_directions.py")
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    r=helper.read(folder/"result.json");evidence=verify_case(folder)
    original=make(r["job"]["angle"],r["job"]["parameters"],r["job"]["group"])
    last,target,_=helper.restore_prefix(original,folder,r,evidence)
    ge,gv,tau=r["job"]["parameters"]
    p=ViscoelasticProblem(maxwell_shear=gv,relaxation_s=tau,loading_rate=.2,
        bulk_modulus=K,case="oblique",mu=.35,shear_modulus=ge,ogden_alpha=2.,
        dhat=.0025,local_spacing=.05,surface_spacing=.005,nose_facets=args.facets,
        inclination=r["job"]["angle"],tip_radius=.02,fine_halfwidth=3.,
        max_iterations=500,predictor="secant",hessian_mode="regularized",element="p1")
    p.__class__=type("ConditionalNose",(ContactControlMixin,ComputeMaterial),{})
    p.enable_contact_controls(velocity_epsilon=GROUPS[r["job"]["group"]]["velocity_epsilon"])
    p.stagnation_window=40;p.ipc.set_num_threads(4)
    p.parallel_linear=PardisoDirection(8,runtime=str(ROOT/".runtime/pardiso"))
    np.testing.assert_array_equal(p.mesh.reference_xy,original.mesh.reference_xy)
    np.testing.assert_array_equal(p.mesh.triangles,original.mesh.triangles)
    assert p.material==original.material
    def transfer(positions,depth):
        return np.vstack((positions[:p.n],p.prescribed_blade_positions(depth)))
    old=float(last["depth_mm"])
    x=transfer(last["positions"],old)
    p.previous=x.copy();p.material_metric=last["material_metric"].copy()
    p.material_dissipation=float(last["cumulative_viscous_dissipation_n_mm"])
    older=helper.load(folder/"states"/f'{len(r["steps"])-2:04d}.npz')
    p.seed_predictor_history([(float(s["depth_mm"]),transfer(s["positions"],float(s["depth_mm"])))
                            for s in (older,last)])
    report=dict(status="running",facets=args.facets,source_case=str(folder),steps=[],
        interpretation="Conditional geometry control; old body/material histories exact, blade arrays reconstructed rigidly; not full-path accuracy",
        evidence=evidence,original_depth_mm=old,target_mm=target,budget_s=args.seconds)
    began=time.monotonic()
    with (out/"run.log").open("w",encoding="utf-8",buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1):
        try:
            for i in range(4):
                depth=round(target+i*r["job"]["increment"],12)
                x,row=p.step(x,old,depth,began+args.seconds)
                state=dict(positions=x,previous_positions=p.previous,reference_xy=p.rest,
                    triangles=p.mesh.triangles,depth_mm=depth,body_nodes=p.n,**p.checkpoint_data())
                save_npz(out/f"state_{i:02d}.npz",**state)
                row["fresh_audit"]=p.audit_state(state,row)
                report["steps"].append(row);write_json(out/"summary.json",report)
                old=depth
            report["status"]="completed"
        except Exception as exc:
            report.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
            save_npz(out/"failed_trial.npz",positions=getattr(p,"trial",x),
                previous_positions=p.previous,**p.checkpoint_data())
        finally:
            p.parallel_linear.close()
            report["elapsed_s"]=time.monotonic()-began
            write_json(out/"summary.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("evidence","steps")},indent=2))


if __name__=="__main__":
    main()
