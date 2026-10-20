"""Opt-in bounded replay of the actually-affine predictor stagnation rule.

The production solver is unchanged. A secant chooser may return exactly the
affine trial (weight zero). In this diagnostic only, that trial receives the
existing affine policy (no accelerated-guess stagnation cutoff), while every
force, geometry, 500-iteration and wall-budget check remains unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np


class ActualAffinePolicy:
    """Mixin; exact equality, never a tolerance-based guess at predictor type."""
    def _loading_predictor(self, x, old, depth):
        self._actual_affine_selected = False
        affine = x.copy()
        delta = depth-old
        affine[:self.n, 1] -= delta*self.mesh.local_sn[:, 1]/self.config.height_mm
        affine[self.n:, 1] -= delta
        value = super()._loading_predictor(x, old, depth)
        self._actual_affine_selected = self.predictor == "secant" and np.array_equal(value, affine)
        return value

    def solve_friction(self, x, depth, deadline):
        prior = self.stagnation_window
        use_affine = self.predictor == "secant" and getattr(self, "_actual_affine_selected", False)
        if use_affine:
            self.stagnation_window = 0
            print("ACTUAL_AFFINE_POLICY "+json.dumps(dict(depth_mm=depth,
                prior_stagnation_window=prior, active_stagnation_window=0,
                physical_acceptance="unchanged", max_iterations=self.max_iterations)), flush=True)
        try:
            return super().solve_friction(x, depth, deadline)
        finally:
            self.stagnation_window = prior


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def worker(args):
    from threadpoolctl import threadpool_limits
    from run_support import compute_process_policy, save_npz, write_json
    from prototype.ipc_viscoelastic import ViscoelasticProblem
    from prototype.tip_contact import TipContactTracker
    from tools.run_material_candidate import make_problem, signature
    class Candidate(ActualAffinePolicy, ViscoelasticProblem):
        pass
    report = dict(status="starting", variant=args.variant, replay_depth_mm=args.depth,
                  scope="one conditional increment, exact original two histories; not a new force curve")
    started=time.monotonic()
    solve_start=None
    shutil.copy2(Path(__file__),args.output/"replay_source.py")
    with (args.output/"run.log").open("w",encoding="utf-8") as log, contextlib.redirect_stdout(log),\
         threadpool_limits(2), compute_process_policy() as policy:
        try:
            original=read(args.source/"result.json")
            audits=read(args.source/"audit.json")
            records=original["steps"]
            index=next(i for i,r in enumerate(records) if abs(r["depth_mm"]-args.depth)<1e-9)
            if index<2:raise ValueError("Two accepted prior states are required")
            # Reuse the stored target's exact floating-point representation,
            # not the visually equivalent CLI decimal (e.g. .21 vs .210...02).
            target_depth=float(records[index]["depth_mm"])
            report.update(requested_depth_mm=args.depth,replay_depth_mm=target_depth,
                          original_dt_s=records[index]["material"]["dt_s"])
            paths=[args.source/"states"/f"{i:04d}.npz" for i in (index-2,index-1,index)]
            report["input_sha256"]={str(p):sha(p) for p in paths}
            for i,path in zip((index-2,index-1,index),paths):
                if sha(path)!=audits["rows"][i]["checkpoint_sha256"]:
                    raise ValueError("Original audit checkpoint hash mismatch")
            source_names=[n for n in original["source_sha256"] if n.startswith(("prototype/","mechanics/"))]
            report["physical_source_sha256"]={n:sha(Path(n)) for n in source_names}
            for n,digest in report["physical_source_sha256"].items():
                if digest!=original["source_sha256"][n]:
                    raise ValueError("Current physical source differs from original: "+n)
            settings=original["settings"]
            p=make_problem(**{k:settings[k] for k in ("angle","equilibrium_shear","maxwell_shear", "relaxation_s","loading_rate","bulk_modulus","mu")})
            if json.loads(json.dumps(signature(p)))!=original["signature"]:
                raise ValueError("Physical signature differs")
            states=[]
            for i,path in zip((index-2,index-1,index),paths):
                with np.load(path,allow_pickle=False) as source:
                    state={k:source[k].copy() for k in source.files}
                p.audit_state(state,records[i]);states.append(state)
            if args.variant=="actual-affine":p.__class__=Candidate
            older,previous,target=states
            x=previous["positions"].copy();old=float(previous["depth_mm"])
            p.previous=x.copy();p.material_metric=previous["material_metric"].copy()
            p.material_dissipation=float(previous["cumulative_viscous_dissipation_n_mm"])
            p.seed_predictor_history([(float(older["depth_mm"]),older["positions"]),(old,x)])
            if (target_depth-old)/p.loading_rate!=float(target['material_dt_s']):
                raise ValueError('Stored-target material dt is not bitwise reproducible')
            solve_start=time.monotonic()
            value,record=p.step(x,old,target_depth,solve_start+args.seconds)
            report["solve_elapsed_s"]=time.monotonic()-solve_start
            output=args.output/"accepted.npz"
            save_npz(output,reference_xy=p.rest,triangles=p.mesh.triangles,positions=value,
                previous_positions=p.previous,depth_mm=target_depth,body_nodes=p.n,**p.checkpoint_data())
            with np.load(output,allow_pickle=False) as state:
                fresh=p.audit_state(state,record)
            baseline=records[index]
            tracker=TipContactTracker(p.rest,p.mesh.triangles,p.n,p.config,p.dhat,p.nose_facets)
            original_tip=tracker.observe(target['positions'],target_depth)
            candidate_tip=tracker.observe(value,target_depth)
            same_intervals_shape=np.shape(original_tip['nose_reference_intervals_mm'])==np.shape(candidate_tip['nose_reference_intervals_mm'])
            interval_difference=(float(np.max(np.abs(np.asarray(original_tip['nose_reference_intervals_mm'])-
                np.asarray(candidate_tip['nose_reference_intervals_mm'])),initial=0.)) if same_intervals_shape else None)
            differences=dict(force_n=record['reaction_fz_n']-baseline['reaction_fz_n'],
                min_j=record['min_j']-baseline['min_j'],
                positions_max_mm=float(np.max(np.abs(value-target['positions']))),
                metric_max=float(np.max(np.abs(p.material_metric-target['material_metric']))),
                tip_gap_mm=candidate_tip['gap_mm']-original_tip['gap_mm'],
                nose_interval_endpoints_max_mm=interval_difference,
                blade_positions_max_mm=float(np.max(np.abs(value[p.n:]-target['positions'][p.n:]))))
            tolerances=dict(force_abs_n=1e-3,min_j_abs=1e-3,positions_max_mm=1e-4,
                metric_max=1e-3,tip_gap_abs_mm=1e-5,nose_interval_endpoints_max_mm=1e-4,
                meaning='predeclared conditional engineering agreement; not mesh/time convergence')
            gates=dict(force=abs(differences['force_n'])<=tolerances['force_abs_n'],
                min_j=abs(differences['min_j'])<=tolerances['min_j_abs'],
                positions=differences['positions_max_mm']<=tolerances['positions_max_mm'],
                metric=differences['metric_max']<=tolerances['metric_max'],
                tip_gap=abs(differences['tip_gap_mm'])<=tolerances['tip_gap_abs_mm'],
                nose_intervals=interval_difference is not None and interval_difference<=tolerances['nose_interval_endpoints_max_mm'],
                tip_status=original_tip['status']==candidate_tip['status'],
                nose_status=original_tip['nose_status']==candidate_tip['nose_status'])
            report.update(status="accepted",record=record,fresh_audit=fresh,
                original_record=baseline,
                original_tip=original_tip,candidate_tip=candidate_tip,
                differences=differences,comparison_tolerances=tolerances,comparison_gates=gates,
                conditional_agreement_passed=all(gates.values()),
                exact_dt_reproduced=p.material_dt==float(target['material_dt_s']),
                actual_affine_selected=getattr(p,"_actual_affine_selected",False),
                unchanged_stagnation_window_after=p.stagnation_window,
                force_difference_n=record["reaction_fz_n"]-baseline["reaction_fz_n"],
                positions_max_difference_mm=float(np.max(np.abs(value-target["positions"]))),
                metric_max_difference=float(np.max(np.abs(p.material_metric-target["material_metric"]))),
                accepted_sha256=sha(output))
        except Exception as exc:
            report.update(status="budget_exhausted" if isinstance(exc,TimeoutError) else "failed",
                          error=f"{type(exc).__name__}: {exc}")
            if solve_start is not None:report["solve_elapsed_s"]=time.monotonic()-solve_start
            traceback.print_exc(file=log)
        report["total_elapsed_s"]=time.monotonic()-started
        report["runtime_policy"]=policy
        report["tool_sha256"]=sha(Path(__file__))
        write_json(args.output/"outcome.json",report)
    print(json.dumps({k:report.get(k) for k in ("status","solve_elapsed_s","force_difference_n","error")}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("outputs/native_angles_2mm_20261016/02_angle57"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--depth",type=float,default=.21)
    parser.add_argument("--seconds",type=float,default=180.)
    parser.add_argument("--variant",choices=("actual-affine","original"),default="actual-affine")
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    if args.worker:return worker(args)
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,"-u","-m","tools.replay_affine_policy","--worker", "--source",str(args.source.resolve()),
             "--output",str(args.output.resolve()),"--depth",str(args.depth),"--seconds",str(args.seconds),"--variant",args.variant]
    started=time.monotonic()
    with (args.output/"stdout.log").open("w") as out, (args.output/"stderr.log").open("w") as err:
        child=subprocess.Popen(command,stdout=out,stderr=err,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            code=child.wait(timeout=args.seconds+60.)
            external=dict(status="exited",returncode=code)
        except subprocess.TimeoutExpired:
            child.kill();child.wait();external=dict(status="external_timeout",returncode=child.returncode)
    external.update(wall_elapsed_s=time.monotonic()-started,timeout_s=args.seconds+60.,command=command)
    (args.output/"external.json").write_text(json.dumps(external,indent=2),encoding="utf-8")
    print(json.dumps(external))


if __name__=="__main__":main()
