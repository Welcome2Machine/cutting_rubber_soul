"""Recover an IO-interrupted control with frozen physics and explicit IO overlay.

The old source hashes and strict resume checks remain intact. The separately
hashed overlay replaces ONLY write_json/save_npz/HTML output functions. A
bounded guardian reconciles previously orphaned solvers before the new case.
"""
import argparse
from datetime import datetime
import hashlib
from html import escape
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def overlay(out):
    spec=importlib.util.spec_from_file_location("recovery_io",out/"harness/run_support.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def check(out,m):
    for rel,h in m["source_sha256"].items():
        if sha(out/"source"/rel)!=h:raise ValueError("Frozen physical source changed: "+rel)
    for rel,h in m["overlay_sha256"].items():
        if sha(out/rel)!=h:raise ValueError("IO overlay changed: "+rel)
    for path,h in m["inputs"].items():
        if sha(path)!=h:raise ValueError("Resume input changed: "+path)


def render(out,state,io):
    body=f"<h1>输出修复后的连续验证与旧进程接管</h1><p>{escape(state['status'])}</p>"
    body+=("<p>0°固定阈值半步组从已验收0.845 mm补齐至1 mm。保持旧冻结物理源码、准确材料及摩擦历史、"
           "原500次PTC/40次停滞和1e−4 N验收。唯一执行覆盖是有哈希记录的输出IO有限重试。"
           "57°求解候选尚未通过门槛，不在这里重开。</p><pre>"+escape(json.dumps(state,ensure_ascii=False,indent=2))+"</pre>")
    result=out/"recovered_fixed_half_0/result.json"
    if result.exists():
        r=read(result);last=(r["steps"] or [{}])[-1]
        body+=f"<p>算例：{escape(r['status'])}；已接受 {last.get('depth_mm')} mm；Fz={last.get('reaction_fz_n')} N。</p>"
        log=out/"recovered_fixed_half_0/run.log"
        if log.exists():
            with log.open("rb") as f:
                f.seek(0,2);f.seek(max(0,f.tell()-24000));text=f.read().decode("utf-8",errors="replace")
            body+="<pre>"+escape(text)+"</pre>"
    text=('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
          '<title>连续验证接管</title><style>body{max-width:1100px;margin:2em auto;font-family:system-ui}'
          'pre{white-space:pre-wrap;word-break:break-all;background:#eee;padding:1em}</style>'+body)
    temp=out/"index.html.tmp";temp.write_text(text,encoding="utf-8");io.replace_with_retry(temp,out/"index.html")
    io.write_json(out/"status.json",state)


def prepare(out,source_case):
    import psutil
    from tools.run_common_material_study import verify_case
    out.mkdir(parents=True,exist_ok=False);(out/"harness").mkdir()
    r=read(source_case/"result.json")
    if "PermissionError" not in r.get("error",""):raise ValueError("Not an IO-interrupted case")
    inputs=verify_case(source_case)
    for name in ("run_support.py","tools/recover_contact_validation.py"):
        dest=out/"harness"/Path(name).name;shutil.copy2(ROOT/name,dest)
    for rel,h in r["source_sha256"].items():
        src=source_case/"source"/rel
        if sha(src)!=h:raise ValueError("Old source mismatch")
        dest=out/"source"/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    study=source_case.parent;state=read(study/"queue_status.json");dependencies=[]
    for job in state["jobs"]:
        if not job.get("pid"):continue
        try:p=psutil.Process(job["pid"]);command=p.cmdline()
        except psutil.NoSuchProcess:continue
        if str(study) not in " ".join(command) or "tools.run_contact_controls" not in command:continue
        if job["name"] not in " ".join(command):continue
        dependencies.append(dict(pid=p.pid,created=p.create_time(),command=command,name=job["name"],
            folder=str(study/job["name"]),deadline_epoch=p.create_time()+job["seconds"]+900))
    m=dict(source_sha256=r["source_sha256"],inputs=inputs,
        overlay_sha256={str(f.relative_to(out)):sha(f) for f in (out/"harness").iterdir()},
        overlay_symbols=["run_support.write_json","run_support.save_npz",
            "tools.run_common_material_study.write_json/save_npz",
            "prototype.material_case.write_json","tools.run_contact_controls.write_json/save_npz",
            "tools.render_solver_dashboard.atomic_write"],
        runtime_path=str(ROOT/".runtime/ipc-python"),pardiso_runtime=str(ROOT/".runtime/pardiso"),
        python=sys.executable,dependencies=dependencies,
        job=dict(r["job"],name="recovered_fixed_half_0",status="pending",
                 resume=str(source_case),target_mm=1.,seconds=5400),
        physical_source_note="Identical old sources and strict resume; separate IO overlay is part of execution provenance")
    io=overlay(out);io.write_json(out/"manifest.json",m)
    render(out,dict(status="prepared",dependencies=dependencies),io)
    print(json.dumps(dict(prepared=True,dependency_pids=[d["pid"] for d in dependencies])))


def case(out):
    m=read(out/"manifest.json");check(out,m)
    sys.path.insert(0,m["runtime_path"]);sys.path.insert(0,str(out/"source"))
    from tools import run_contact_controls as control,run_common_material_study as host,render_solver_dashboard as dashboard
    from prototype import material_case
    import run_support
    io=overlay(out)
    # Function bodies are unmodified except the bounded Path.replace retry.
    for module in (host,control,run_support):
        module.write_json=io.write_json;module.save_npz=io.save_npz
    material_case.write_json=io.write_json
    def html_write(path,text):
        temp=path.with_suffix(path.suffix+".tmp");temp.write_text(text,encoding="utf-8")
        io.replace_with_retry(temp,path)
    dashboard.atomic_write=html_write
    from prototype.ipc_reference import IPCProblem
    import numpy as np
    lines,first=inspect.getsourcelines(IPCProblem.solve_friction)
    accept_line=first+next(i for i,line in enumerate(lines) if line.strip()=="x = trial")
    reject_line=first+next(i for i,line in enumerate(lines) if 'print(f"FRICTION_PTC_REJECT' in line)
    latest={};seen=set()
    def local(frame,event,arg):
        if event!="line" or frame.f_lineno not in (accept_line,reject_line):return local
        v=frame.f_locals;accepted=frame.f_lineno==accept_line
        key=(v["depth"],v["iteration"],v["retry"],accepted)
        if key in seen:return local
        seen.add(key)
        alpha=float(v["alpha"]);direction=v["direction"];p=v["self"]
        row=dict(depth_mm=float(v["depth"]),iteration=v["iteration"],retry=v["retry"],
            accepted=accepted,shift_used=float(v["shift"]),norm_n=float(v["norm"]),
            trial_norm_n=float(v["nt"]),alpha=alpha,geometric_fraction=float(v["geometric_fraction"]),
            tip_cap_alpha=.5*p.options.tip_radius_mm/float(v["maximum"]),
            displacement_infinity_mm=alpha*float(v["maximum"]),
            predicted_norm_n=float(np.linalg.norm(v["r"]+alpha*v["linear_action"])),
            model_error_n=float(v["error"]))
        with (out/"actual_ptc_directions.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
        latest.update(positions=v["x"],direction=direction,previous_positions=p.previous,
            previous_material_metric=p.previous_material_metric,material_dt_s=p.material_dt,
            r=v["r"],shift=v["shift"],alpha=alpha,depth_mm=v["depth"])
        return local
    def dispatch(frame,event,arg):
        return local if frame.f_code is IPCProblem.solve_friction.__code__ else None
    try:
        sys.settrace(dispatch)
        result=control.case(out,m["job"])
    finally:
        sys.settrace(None)
        if latest:io.save_npz(out/"last_actual_ptc_direction.npz",**latest)
    result["execution_io_overlay"]=m["overlay_sha256"]
    io.write_json(out/m["job"]["name"]/"result.json",result)


def worker(out):
    import psutil
    m=read(out/"manifest.json");check(out,m);io=overlay(out)
    started=out/"worker_started.json"
    with started.open("x",encoding="utf-8") as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
    state=dict(status="waiting_for_existing_validations",dependencies=m["dependencies"],reconciled=[])
    render(out,state,io)
    for dep in m["dependencies"]:
        try:
            p=psutil.Process(dep["pid"])
            if abs(p.create_time()-dep["created"])>1e-6:continue
            while p.is_running():
                if time.time()>dep["deadline_epoch"]:
                    # Only this observed, exact command/process identity is owned.
                    if p.cmdline()!=dep["command"]:raise RuntimeError("Process identity changed")
                    p.terminate();p.wait(timeout=30);dep["guardian_timeout"]=True;break
                try:render(out,state,io)
                except OSError as exc:print("REPORT_IO_ERROR",repr(exc),flush=True)
                try:p.wait(timeout=30)
                except psutil.TimeoutExpired:pass
        except psutil.NoSuchProcess:pass
        folder=Path(dep["folder"])
        result=read(folder/"result.json")
        audit=read(folder/"audit.json") if (folder/"audit.json").exists() else None
        state["reconciled"].append(dict(name=dep["name"],case_status=result["status"],
            accepted_states=len(result["steps"]),last_depth_mm=result.get("last_depth_mm"),
            audit_passed=bool(audit and audit["passed"] and audit["states"]==len(result["steps"])),
            reporting_error_is_not_mechanical_failure=True))
    state["status"]="running_recovered_case";render(out,state,io)
    env=os.environ.copy();env.update(PYTHONPATH=m["runtime_path"],OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",SOFTMODELING_PARDISO_RUNTIME=m["pardiso_runtime"],PYTHONDONTWRITEBYTECODE="1")
    command=[m["python"],"-B","-X","utf8","-u",str(out/"harness/recover_contact_validation.py"),
             "case","--output",str(out)]
    with (out/"case.stdout.log").open("wb") as stdout,(out/"case.stderr.log").open("wb") as stderr:
        p=subprocess.Popen(command,cwd=out/"source",env=env,stdout=stdout,stderr=stderr,
            stdin=subprocess.DEVNULL,**io.background_child_options())
        state["case_pid"]=p.pid;tick=time.monotonic()
        try:
            while p.poll() is None:
                if time.monotonic()-tick>6300:p.terminate();p.wait(timeout=30);state["timeout"]=True;break
                try:render(out,state,io)
                except OSError as exc:print("REPORT_IO_ERROR",repr(exc),flush=True)
                time.sleep(10)
        finally:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
    state.update(status="finished",returncode=p.returncode)
    folder=out/m["job"]["name"]
    if (folder/"result.json").exists():
        r=read(folder/"result.json");state["result_status"]=r["status"]
        state["accepted_states"]=len(r["steps"]);state["last_depth_mm"]=r.get("last_depth_mm")
        audit=read(folder/"audit.json") if (folder/"audit.json").exists() else {}
        state["audit_passed"]=bool(audit.get("passed") and audit.get("states")==len(r["steps"]))
        if state["audit_passed"]:
            state["checkpoint_hashes_verified"]=all(
                sha(folder/"states"/f"{i:04d}.npz")==a["checkpoint_sha256"] for i,a in enumerate(audit["rows"]))
    check(out,m);render(out,state,io)


def launch(out):
    m=read(out/"manifest.json");check(out,m);io=overlay(out)
    if (out/"launch.json").exists():raise ValueError("Already launched")
    with (out/"worker.stdout.log").open("wb") as stdout,(out/"worker.stderr.log").open("wb") as stderr:
        p=subprocess.Popen([m["python"],"-B","-X","utf8","-u",str(out/"harness/recover_contact_validation.py"),
            "worker","--output",str(out)],cwd=out,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            **io.background_child_options())
    io.write_json(out/"launch.json",dict(pid=p.pid,time=time.time()));print(p.pid)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("phase",choices=("prepare","case","worker","launch"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-case",type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.phase=="prepare":prepare(out,args.source_case.resolve())
    elif args.phase=="case":case(out)
    elif args.phase=="worker":worker(out)
    else:launch(out)
