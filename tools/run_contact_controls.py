"""Frozen, bounded longer diagnostics of time smoothing and adjacent contact.

Six independent paths to 1 mm; no material selection, automatic cutback or
budget extension. This studies identified numerical effects, not a qualified
objective-friction replacement or a claim that the old failures are repaired.
"""
import argparse
import contextlib
from functools import partial
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from prototype.contact_controls import ContactControlMixin
from prototype.material_case import signature, load_force_csv
from prototype.pardiso import PardisoDirection
from run_support import write_json, save_npz, background_child_options, compute_process_policy
from tools import run_common_material_study as host
from tools.render_solver_dashboard import shell, atomic_write, stamp

ROOT = Path(__file__).resolve().parents[1]
MODULE = "tools.run_contact_controls"
GROUPS = {
    "fixed_half": dict(velocity_epsilon=None, exclude_adjacent=False, increment=.005),
    "velocity_half": dict(velocity_epsilon=.0002, exclude_adjacent=False, increment=.005),
}


def make(angle, parameters, group):
    p = host.make(angle, parameters)
    p.__class__ = type("ContactControlProblem", (ContactControlMixin, type(p)), {})
    policy = GROUPS[group]
    p.enable_contact_controls(velocity_epsilon=policy["velocity_epsilon"],
                              exclude_adjacent=policy["exclude_adjacent"])
    return p


def schedule(end, increment):
    if not 0 < end <= 1. or increment not in (.005, .01):
        raise ValueError("Control scope: 0--1 mm, .005/.01 mm increments only")
    return np.r_[0., np.arange(increment, end-1e-12, increment), end]


def case(out, job):
    return host.case(out, job, factory=partial(make, group=job["group"]),
                     targets=schedule(job["target_mm"], job["increment"]))


def render(out, state):
    body = f"<h1>接触数值作用：连续加载验证</h1><p>{stamp()} · {escape(state['status'])}</p>"
    body += ("<p>共同材料不变；每组独立从静止到1 mm。A：固定位移阈值半步；"
             "B：固定速度阈值半步。"
             "A/B对比隔离正则时间尺度，B/旧原步长对比时间离散。"
             "起始静止到零深度准备相同，属于零深度之后的时间细化。</p>"
             "<p>直接删相邻VV的候选已被排除：特征切换出现有限力跳变，且原失败增量未通过；不进入长算例。</p>"
             "<p>本轮不标定材料、不改原PTC/物理验收。有限间隙转动客观性尚未全面修复；"
             "新的局部投影滑移核没有混入以下计算。长路径是机制对照，不是已验证的新模型。</p>")
    body += '<p><a href="analysis.html">自动比较</a> · <a href="qualification.html">资格与限制</a> · <a href="manifest.json">冻结方案</a></p>'
    body += "<table><tr><th>算例</th><th>状态</th><th>深度mm</th><th>Fz N</th><th>步耗时s</th><th>日志</th></tr>"
    for job in state["jobs"]:
        folder = out/job["name"]
        r = host.read(folder/"result.json") if (folder/"result.json").exists() else {}
        z = (r.get("steps") or [{}])[-1]
        body += (f"<tr><td>{escape(job['name'])}</td><td>{escape(job['status'])}</td>"
                 f"<td>{z.get('depth_mm','—')}</td><td>{z.get('reaction_fz_n','—')}</td>"
                 f"<td>{z.get('wall_step_s','—')}</td><td>")
        if r:
            page = (f"<h1>{escape(job['name'])}</h1><p>{stamp()} · {escape(r['status'])}</p>"
                    f"<p>已接受 {len(r['steps'])} 态；深度 {z.get('depth_mm')} mm；"
                    f"Fz {z.get('reaction_fz_n')} N；minJ {z.get('min_j')}。</p>"
                    "<p>最大500次PTC、原40次停滞判据；无自动缩步。"
                    "新鲜残差、几何、双历史和耗散验收均保留。</p>"
                    f"<pre>{escape(json.dumps(r.get('signature',{}).get('contact_control',{}),ensure_ascii=False,indent=2))}</pre>"
                    '<p><a href="../index.html">队列</a> · <a href="result.json">状态</a> · '
                    '<a href="fz_depth.csv">终态力CSV</a> · <a href="contact_trace.csv">接触材料身份CSV</a></p>')
            page += "<p>"+escape(r.get("error","") or r.get("audit_error",""))+"</p>"
            if (folder/"run.log").exists():
                with (folder/"run.log").open("rb") as f:
                    f.seek(0,2); f.seek(max(0,f.tell()-60000))
                    page += "<details open><summary>运行日志末尾</summary><pre>"+escape(f.read().decode("utf-8",errors="replace"))+"</pre></details>"
            atomic_write(folder/"index.html",shell("接触对照日志",page,refresh=r["status"] in ("running","auditing","initializing")))
            body += f'<a href="{job["name"]}/index.html">状态/日志</a>'
        body += "</td></tr>"
    body += ("</table><p>每个正式算例求解预算5400 s，另有900 s准备/审查上限；"
             "这不是完成时间保证。失败或超时自动进入下一组，不重复尝试。"
             "PARDISO8 / IPC4 / BLAS1，后台串行，避免资源争抢。</p>")
    atomic_write(out/"index.html",shell("连续接触对照",body,refresh=state["status"]=="running"))
    write_json(out/"queue_status.json",state)


def supervise(out, m, job, state):
    # Reuse the bounded supervisor, but use THIS CLI and truthful control pages.
    prior_module, prior_render = host.MODULE, host.render
    host.MODULE, host.render = MODULE, render
    try:
        return host.supervise(out,m,job,state)
    finally:
        host.MODULE, host.render = prior_module, prior_render


def metrics(d, f, ed, ef):
    values = {}
    for lo, hi in ((.1,.25),(.25,.5),(.5,1.)):
        if d[-1] < hi-1e-10: continue
        grid = np.linspace(lo,hi,101)
        err = np.interp(grid,d,f)-np.interp(grid,ed,ef)
        values[f"{lo:g}--{hi:g}"] = dict(rmse_n=float(np.sqrt(np.mean(err**2))),
            fe_secant_n_mm=float(np.diff(np.interp([lo,hi],d,f))[0]/(hi-lo)),
            experiment_secant_n_mm=float(np.diff(np.interp([lo,hi],ed,ef))[0]/(hi-lo)))
    return values


def analysis(out, m, state):
    rows = []; controls = []
    for angle in host.ANGLES:
        baseline = host.read(Path(m["baselines"][str(angle)])/"result.json")
        bd = np.array([z["depth_mm"] for z in baseline["steps"]])
        bf = np.array([z["reaction_fz_n"] for z in baseline["steps"]])
        ed, ef, _ = load_force_csv(out/f"experiment{angle}.csv")
        curves = {"reference": (bd,bf)}
        histories = {"reference": baseline["steps"]}
        rows.append(dict(name=f"reference_{angle}",angle=angle,status="previous_complete_baseline",
                         windows=metrics(bd,bf,ed,ef)))
        for job in state["jobs"]:
            if job["angle"] != angle or job.get("stage") == "preflight": continue
            folder = out/job["name"]
            if not (folder/"result.json").exists(): continue
            r = host.read(folder/"result.json"); a = host.read(folder/"audit.json") if (folder/"audit.json").exists() else {}
            if not a.get("passed") or a.get("states") != len(r["steps"]) or not r["steps"]: continue
            d = np.array([z["depth_mm"] for z in r["steps"]]); f = np.array([z["reaction_fz_n"] for z in r["steps"]])
            curves[job["group"]] = (d,f)
            histories[job["group"]] = r["steps"]
            tip = [z.get("tip_contact") for z in r["steps"]]
            rows.append(dict(name=job["name"],angle=angle,status=r["status"],
                accepted_depth_mm=float(d[-1]),states=len(d),windows=metrics(d,f,ed,ef),
                path_min_j=min(z["min_j"] for z in r["steps"]),
                solve_s=r.get("solve_s"),
                unique_tip_observation_count=sum((t or {}).get("reference_offset_mm") is not None for t in tip),
                missing_unique_tip_count=sum((t or {}).get("reference_offset_mm") is None for t in tip),
                contact_note="Unique contact may be absent; see full CSV, do not replace with proximity"))
        for left, right in (("fixed_half","velocity_half"),("reference","velocity_half")):
            if left not in curves or right not in curves: continue
            a,b = curves[left],curves[right]; end=min(1.,a[0][-1],b[0][-1])
            if end < .1: continue
            grid = np.unique(np.r_[a[0][a[0]<=end],b[0][b[0]<=end]])
            change = np.interp(grid,*b)-np.interp(grid,*a)
            # Contact identity is compared only at exact shared accepted depths.
            ha={round(z["depth_mm"],10):z for z in histories[left] if z["depth_mm"]<=end+1e-10}
            hb={round(z["depth_mm"],10):z for z in histories[right] if z["depth_mm"]<=end+1e-10}
            shared=sorted(ha.keys()&hb.keys());offset_changes=[];availability_mismatch=0
            for d in shared:
                u=(ha[d].get("tip_contact") or {}).get("reference_offset_mm")
                v=(hb[d].get("tip_contact") or {}).get("reference_offset_mm")
                if u is not None and v is not None: offset_changes.append(v-u)
                availability_mismatch+=int((u is None)!=(v is None))
            relative=np.abs(change)/np.maximum(np.abs(np.interp(grid,*a)),1e-12)
            interior=grid>=.1-1e-10
            controls.append(dict(angle=angle,left=left,right=right,common_depth_mm=float(end),
                max_force_difference_n=float(abs(change).max()),
                end_force_difference_n=float(change[-1]),
                force_sensitive=bool(np.any((abs(change)>.1)|(interior&(relative>.02)))),
                exact_shared_tip_depths=len(shared),both_unique_tip_count=len(offset_changes),
                unique_availability_mismatch_count=availability_mismatch,
                max_unique_reference_offset_difference_mm=max(map(abs,offset_changes),default=None),
                interpretation="Audited common accepted prefix only; no extrapolation"))
    write_json(out/"analysis.json",dict(rows=rows,controls=controls,
        note="No material fitted; numerical diagnostics. Experimental intactness not certified."))
    body = ("<h1>连续验证自动分析</h1><p>只使用已经完成终态新鲜审查的接受前缀；"
            "未完成深度不外推。0°当前a2起裂未知，33°约1.2–1.3 mm为粗估，57°未知。"
            "1 mm计算主要验证路径和时间敏感性，不能认证实验全段未裂。57°曾参与选参。</p>"
            '<p><a href="index.html">队列</a> · <a href="analysis.json">全部指标</a></p>'
            "<table><tr><th>组别</th><th>状态</th><th>区间mm</th><th>RMSE N</th><th>FE割线 N/mm</th><th>实验割线</th></tr>")
    for row in rows:
        for window, z in row["windows"].items():
            body += (f"<tr><td>{row['name']}</td><td>{row['status']}</td><td>{window}</td>"
                     f"<td>{z['rmse_n']:.5f}</td><td>{z['fe_secant_n_mm']:.5f}</td><td>{z['experiment_secant_n_mm']:.5f}</td></tr>")
    body += "</table><h2>隔离对照（相同已接受深度）</h2><pre>"+escape(json.dumps(controls,ensure_ascii=False,indent=2))+"</pre>"
    atomic_write(out/"analysis.html",shell("接触连续验证分析",body))


def replay(out):
    """Conditional old-history factor test, not a new independent material path."""
    folder = ROOT/"outputs/common_material_fe_20261020/fast_57"
    result = host.read(folder/"result.json")
    inputs = host.verify_case(folder)
    count = len(result["steps"])
    z = []
    for i in (count-2,count-1):
        with np.load(folder/"states"/f"{i:04d}.npz",allow_pickle=False) as archive:
            z.append({k:archive[k].copy() for k in archive.files})
    report = dict(scope="Conditional adjacent-VV removal only; original fast/57 .19 history; not a full-path repair",
                  inputs=inputs,neighbors_requested=3,solve_budget_s=360,status="running",steps=[])
    dest = out/"conditional_fast57"; dest.mkdir(exist_ok=False)
    t = time.monotonic()
    with (dest/"run.log").open("w",encoding="utf-8",buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy():
        p = host.make(57,result["job"]["parameters"])
        assert json.loads(json.dumps(signature(p))) == result["signature"]
        for state, row in zip(z,result["steps"][-2:]): p.audit_state(state,row)
        p.__class__ = type("ConditionalAdjacent", (ContactControlMixin,type(p)), {})
        p.enable_contact_controls(velocity_epsilon=None,exclude_adjacent=True)
        p.previous = z[-1]["positions"].copy()
        p.material_metric = z[-1]["material_metric"].copy()
        p.material_dissipation = float(z[-1]["cumulative_viscous_dissipation_n_mm"])
        p.seed_predictor_history([(float(v["depth_mm"]),v["positions"]) for v in z])
        p.parallel_linear = PardisoDirection(8)
        x, old = z[-1]["positions"].copy(), float(z[-1]["depth_mm"])
        try:
            for depth in (.20,.21,.22):
                x,row = p.step(x,old,depth,t+360)
                state = dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,
                             previous_positions=p.previous,depth_mm=depth,**p.checkpoint_data())
                report["steps"].append(dict(record=row,audit=p.audit_state(state,row)))
                save_npz(dest/f"{depth:.2f}.npz",**state); old=depth
            report["status"] = "completed_three_conditional_increments"
        except Exception as exc:
            report.update(status="failed_or_budget_exhausted",error=repr(exc))
            traceback.print_exc()
            save_npz(dest/"failed_trial.npz",positions=getattr(p,"trial",x),
                     previous_positions=p.previous,**p.checkpoint_data())
        finally:
            p.parallel_linear.close()
    report["elapsed_s"] = time.monotonic()-t
    write_json(dest/"result.json",report)
    body = "<h1>已知fast/57失败增量：仅排除相邻VV</h1><pre>"+escape(json.dumps({k:v for k,v in report.items() if k!="inputs"},ensure_ascii=False,indent=2))+"</pre>"
    body += "<details><summary>日志</summary><pre>"+escape((dest/"run.log").read_text(encoding="utf-8"))+"</pre></details>"
    atomic_write(dest/"index.html",shell("隔离增量重放",body))
    return report


def switch_check():
    from prototype.ipc_reference import toolkit
    ipc=toolkit();ipc.set_num_threads(1)
    rest=np.array([[0.,0.],[.005,0.],[.01,0.],[.015,0.]])
    mesh=ipc.CollisionMesh(rest,np.array([[0,1],[1,2],[2,3]]))
    adjacent={(0,1),(1,2),(2,3)};barrier=ipc.BarrierPotential(.0025,132.,True)
    rows=[]
    for eps in (1e-6,1e-8,1e-10,1e-12):
        native=[];filtered=[]
        for sign in (-1,1):
            x=np.array([[0.,0.],[.005,0.],[.005+sign*eps,.001],[.01,.001]])
            c=ipc.NormalCollisions();c.use_area_weighting=True;c.collision_set_type=ipc.NormalCollisions.IMPROVED_MAX_APPROX
            c.build(mesh,x,.0025);native.append(barrier.gradient(c,mesh,x))
            c.vv_collisions=[s for s in c.vv_collisions if tuple(sorted((s.vertex0_id,s.vertex1_id))) not in adjacent]
            filtered.append(barrier.gradient(c,mesh,x))
        rows.append(dict(perturbation_mm=eps,native_force_jump_n=float(np.linalg.norm(native[1]-native[0])),
                         deleted_vv_force_jump_n=float(np.linalg.norm(filtered[1]-filtered[0]))))
    if rows[-1]["native_force_jump_n"]>=1e-8 or rows[-1]["deleted_vv_force_jump_n"]<=5:
        raise ValueError("Feature-switch counterexample no longer reproduces; reassess before launch")
    return dict(adjacent_deletion_qualified=False,rows=rows,
                conclusion="Do not delete adjacent VV terms in isolation: native EV/VV cancellation is broken")


def prepare(out, qualification_from):
    out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists(): raise ValueError("Already prepared")
    tests = host.read(out/"tests.json")
    if not tests["passed"] or tests["skipped"]: raise ValueError("Regressions must pass without skips")
    qualification_from=qualification_from.resolve()
    replay_result=host.read(qualification_from/"conditional_fast57/result.json")
    prior=host.read(qualification_from/"manifest.json");host.check(prior,qualification_from)
    crossing=switch_check();write_json(out/"switch_check.json",crossing)
    # Preserve the failed conditional probe and harness errors; never transfer
    # its material or displacement history to the independent production controls.
    inputs = {str(out/"tests.json"):host.sha(out/"tests.json")}
    for name in ("conditional_fast57/result.json","conditional_fast57/failed_trial.npz",
                 "conditional_fast57/run.log","manifest.json"):
        path=qualification_from/name;inputs[str(path)]=host.sha(path)
    for name,h in prior["source_sha256"].items():
        inputs[str(qualification_from/"source"/name)]=h
    inputs[str(out/"switch_check.json")]=host.sha(out/"switch_check.json")
    anchors = []
    with threadpool_limits(1),compute_process_policy():
        for angle in host.ANGLES:
            folder = ROOT/host.REFERENCES[angle]; r=host.read(folder/"result.json")
            au = host.read(folder/"audit.json")
            if not r["completed"] or not au["passed"]: raise ValueError("Reference incomplete")
            p = host.make(angle,host.BASE)
            if json.loads(json.dumps(signature(p))) != r["signature"]: raise ValueError("Reference signature differs")
            for file in ("result.json","audit.json"): inputs[str(folder/file)] = host.sha(folder/file)
            for name, digest in r["source_sha256"].items():
                path=folder/"source"/name
                if host.sha(path)!=digest: raise ValueError("Reference source changed")
                inputs[str(path)]=digest
            for i,row in enumerate(r["steps"]):
                if row["depth_mm"]>1.+1e-10: break
                path=folder/"states"/f"{i:04d}.npz"; digest=au["rows"][i]["checkpoint_sha256"]
                if host.sha(path)!=digest: raise ValueError("Reference state changed")
                inputs[str(path)] = digest
                if any(abs(row["depth_mm"]-v)<1e-10 for v in (0.,.1,.5,1.)):
                    with np.load(path,allow_pickle=False) as archive:
                        z={k:archive[k] for k in archive.files}
                    anchors.append(dict(angle=angle,**p.audit_state(z,row)))
            dest=out/f"experiment{angle}.csv";shutil.copy2(ROOT/"docs"/host.EXPERIMENTS[angle],dest)
            if load_force_csv(dest)[2] != .2: raise ValueError("Speed differs")
            inputs[str(dest)] = host.sha(dest)
    source = host.snapshot(out)
    for name in ("tools/run_contact_controls.py",):
        dest=out/"source"/name;shutil.copy2(ROOT/name,dest);source[name]=host.sha(dest)
    jobs = [dict(name=f"{group}_{a}",group=group,angle=a,parameters=host.BASE.tolist(),
                 increment=policy["increment"],target_mm=1.,seconds=5400,status="pending")
            for group,policy in GROUPS.items() for a in (57,33,0)]
    m = dict(created=stamp(),policy="bounded-contact-controls-20261021",source_sha256=source,inputs=inputs,
             baselines={str(a):str(ROOT/f) for a,f in host.REFERENCES.items()},jobs=jobs,
             python=sys.executable,runtime_path=str(ROOT/".runtime/ipc-python"),
             pardiso_runtime=str(ROOT/".runtime/pardiso"),groups=GROUPS,
             budgets=dict(case_solve_s=5400,case_setup_audit_s=900,preflight_solve_s=240),
             controls="old complete .01 baseline vs A fixed .005 vs B velocity .005",
             limitations="No full objective friction; no material fit; no automatic retries; identical initial preparation",
             qualification=dict(tests=tests,baseline_anchors=anchors,feature_switch=crossing,
                                prior_diagnostics=str(qualification_from),
                                conditional_replay_status=replay_result["status"]))
    write_json(out/"manifest.json",m)
    state=dict(status="preparing",jobs=jobs)
    # Real fine-P1 preflights for every distinct formulation. Other angles have
    # their own preflight later, and may only continue their identical prefix.
    for group in GROUPS:
        job=next(j for j in jobs if j["group"]==group and j["angle"]==57)
        pilot=dict(job,name="pilot_"+job["name"],stage="preflight",target_mm=.01,seconds=240)
        state["jobs"].append(pilot)
        if supervise(out,m,pilot,state):
            job["resume"]=str(out/pilot["name"]); inputs.update(host.verify_case(out/pilot["name"]))
        else:
            for j in jobs:
                if j["group"]==group and j.get("stage")!="preflight":
                    j.update(status="not_qualified",reason="57-degree formulation preflight failed")
        write_json(out/"manifest.json",m)
    m["jobs"]=[j for j in jobs if j.get("stage")!="preflight"]
    m["inputs"]=inputs;write_json(out/"manifest.json",m)
    body=("<h1>本轮资格、范围和预定解释</h1>"
          "<p>新增物理测试：速度正则；共同初始准备；相邻压缩零排斥且保留CCD/折叠；"
          "保留刀具作用和折叠导数；刷新摩擦切线；局部投影滑移客观性及梯度。</p>"
          "<p>局部投影滑移 s=L(β−β₀) 仅对持久主边已合格。跨边/VV历史传递、全局残差和切线尚未完成，"
          "因此不接入长验证。A/B仍保留原生有限间隙摩擦的客观性限制。</p>"
          "<p>直接删相邻VV在特征切换时力跳变5.59 N，原失败增量也失败；"
          "此候选已经否决，不进入连续队列。不能由此认定原相邻排斥物理合理，"
          "而应更换完整势/候选构造后再验证。</p>"
          f'<p><a href="../{qualification_from.name}/conditional_fast57/index.html">原失败增量隔离对照</a> · <a href="index.html">连续队列</a></p>'
          "<p>每组原500次PTC、原40次停滞、原真实力/J/CCD/耗散验收；不增加迭代/缩步求通过。"
          "真实时间阈值εv=.0002 mm/s使原.01 mm步的εu仍为1e−5 mm。后零深度半步εu=5e−6 mm。"
          "所有0/33/57使用G∞=.14、Gv=1.24、τ=1.7、K=23.34333333333331、μ=.35、v=.2。</p>"
          "<p>目标：记录至1 mm的力幅值/曲率、参考接触身份、minJ、真实耗时；共同深度差"
          "超过2%或0.1 N时标记时间/接触敏感，不称收敛。缺失唯一接触点保留缺失，"
          "不把鼻部邻近区间叫滑移真值。未通过者保留前缀及失败场，不补算/外推。</p>"
          "<pre>"+escape(json.dumps(m["qualification"],ensure_ascii=False,indent=2))+"</pre>")
    atomic_write(out/"qualification.html",shell("接触对照资格",body))
    state=dict(status="prepared",jobs=m["jobs"]);analysis(out,m,state);render(out,state)
    print(json.dumps(dict(prepared=True,replay=replay_result["status"],
        qualified_groups=[g for g in GROUPS if any(j["group"]==g and j.get("resume") for j in m["jobs"])],
        source_count=len(source),input_count=len(inputs))))


def worker(out):
    m=host.read(out/"manifest.json");host.check(m,out)
    with (out/"worker_started.json").open("x",encoding="utf-8") as f:
        json.dump(dict(pid=os.getpid(),started=stamp()),f)
    state=dict(status="running",jobs=m["jobs"]);render(out,state)
    for job in list(state["jobs"]):
        if job["status"]=="not_qualified": continue
        try:
            if not job.get("resume"):
                pilot=dict(job,name="pilot_"+job["name"],stage="preflight",target_mm=.01,seconds=240)
                state["jobs"].append(pilot)
                if not supervise(out,m,pilot,state):
                    job.update(status="not_qualified",reason="Own prefix preflight failed");continue
                job["resume"]=str(out/pilot["name"])
            supervise(out,m,job,state)
        except Exception as exc:
            job.update(status="worker_case_failed",error=repr(exc));traceback.print_exc()
        analysis(out,m,state);render(out,state)
    state.update(status="finished",finished=stamp());analysis(out,m,state);render(out,state)


def launch(out):
    m=host.read(out/"manifest.json");host.check(m,out)
    if (out/"launch.json").exists() or (out/"worker_started.json").exists():
        raise ValueError("Already launched")
    if not any(j.get("resume") for j in m["jobs"]): raise ValueError("No qualified formulation")
    with (out/"worker.stdout.log").open("wb") as stdout,(out/"worker.stderr.log").open("wb") as stderr:
        p=subprocess.Popen([m["python"],"-B","-X","utf8","-u","-m",MODULE,"worker","--output",str(out)],
            cwd=out/"source",env=host.child_env(m),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            **background_child_options())
    value=dict(pid=p.pid,time=stamp());write_json(out/"launch.json",value);print(json.dumps(value))


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("phase",choices=("prepare","case","worker","launch","report"))
    parser.add_argument("--output",required=True,type=Path);parser.add_argument("--job-json")
    parser.add_argument("--qualification-from",type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.phase=="case":case(out,json.loads(args.job_json))
    elif args.phase=="prepare":
        if args.qualification_from is None:parser.error("prepare requires --qualification-from")
        prepare(out,args.qualification_from)
    elif args.phase=="worker":worker(out)
    elif args.phase=="launch":launch(out)
    else:
        m=host.read(out/"manifest.json");s=host.read(out/"queue_status.json");analysis(out,m,s);render(out,s)
