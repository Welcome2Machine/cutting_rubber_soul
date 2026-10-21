"""Timestamped, hash-checked review of completed and incomplete time controls."""
import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prototype.material_case import load_force_csv
from tools import run_common_material_study as host
from tools.render_solver_dashboard import atomic_write, shell, stamp
from run_support import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=Path("outputs/contact_time_paths_20261021"))
    parser.add_argument("--output", type=Path, default=Path("outputs/contact_time_review_20261021"))
    args = parser.parse_args()
    out, study = args.output.resolve(), args.study.resolve()
    out.mkdir(parents=True, exist_ok=True)
    q = host.read(study/"queue_status.json")
    report = dict(snapshot=stamp(), study=str(study), cases=[], inputs={},
        limitations=["No extrapolation beyond accepted prefixes", "57 zero .51 N retained",
          "0 experiment is a2; onset estimate for a1 cannot be transferred",
          "Early experimental windows are not certified intact",
          "57 used in exploratory calibration; not independent validation",
          "Equilibrium audits are not spatial/time convergence or contact-migration validation"])
    with plt.rc_context({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "savefig.facecolor":"white"}):
        fig, axes = plt.subplots(2, 3, figsize=(13,7.4), sharex=True, sharey="row")
        for col, angle in enumerate((0,33,57)):
            ep = host.ROOT/"docs"/host.EXPERIMENTS[angle]
            ed, ef, _ = load_force_csv(ep)
            report["inputs"][str(ep)] = host.sha(ep)
            ax, diff = axes[:,col]
            mask = ed <= 1.
            ax.plot(ed[mask],ef[mask],color="black",lw=1.5,label="Experiment (raw)")
            jobs = [dict(name="reference", angle=angle, folder=host.ROOT/host.REFERENCES[angle])]
            jobs += [dict(j, folder=study/j["name"]) for j in q["jobs"]
                     if j["angle"]==angle and not j["name"].startswith("pilot_")]
            for job in jobs:
                folder = job["folder"]
                if not (folder/"result.json").exists():
                    report["cases"].append(dict(name=job["name"],angle=angle,status=job["status"]));continue
                r = host.read(folder/"result.json"); steps = r["steps"]
                if not steps: continue
                terminal = r["status"] not in ("initializing","running","auditing")
                if terminal:
                    evidence = host.verify_case(folder); report["inputs"].update(evidence)
                else:
                    # Explicit snapshot, not a terminal fresh-audit assertion.
                    evidence = {}
                d = np.array([s["depth_mm"] for s in steps])
                f = np.array([s["reaction_fz_n"] for s in steps])
                name = job["name"]
                if name=="reference": label, color, style = "Original 0.01 mm", "#999999", ":"
                elif job["group"]=="fixed_half": label,color,style = "Half-step, fixed slip epsilon", "#0072B2","-"
                else: label,color,style = "Half-step, fixed speed epsilon","#D55E00","--"
                keep=d<=1.+1e-12
                ax.plot(d[keep],f[keep],color=color,ls=style,lw=1.8,label=label)
                delta=f-np.interp(d,ed,ef)
                diff.plot(d[keep],delta[keep],color=color,ls=style,lw=1.8)
                if d[-1]<1.-1e-10:
                    ax.plot(d[-1],f[-1],"x",color=color,ms=8)
                record=dict(name=name,angle=angle,status=r["status"], error=r.get("error"),
                    supervisor_status=job.get("status"),
                    accepted_depth_mm=float(d[-1]), states=len(d),
                    fresh_terminal_audit=terminal, verified_files=len(evidence),
                    path_min_j=min(s["min_j"] for s in steps),
                    failure_category=("output_io" if "PermissionError" in r.get("error","") else
                                      "mechanical" if terminal and not r["completed"] else None),
                    windows=[])
                for a,b in ((.1,.25),(.25,.5),(.5,1.)):
                    if d[-1]<b-1e-10: continue
                    grid=np.linspace(a,b,101); errs=np.interp(grid,d,f)-np.interp(grid,ed,ef)
                    fs=float(np.diff(np.interp([a,b],d,f))[0]/(b-a))
                    es=float(np.diff(np.interp([a,b],ed,ef))[0]/(b-a))
                    record["windows"].append(dict(start_mm=a,end_mm=b,fe_secant_n_mm=fs,
                        experimental_secant_n_mm=es,rmse_n=float(np.sqrt(np.mean(errs**2)))))
                record["anchors"]=[]
                for depth in (.25,.5,.845,1.):
                    if d[-1]<depth-1e-10:continue
                    fe=float(np.interp(depth,d,f)); exp=float(np.interp(depth,ed,ef))
                    record["anchors"].append(dict(depth_mm=depth,fe_n=fe,experiment_n=exp,
                        difference_n=fe-exp,relative_percent=100*(fe/exp-1) if exp else None))
                report["cases"].append(record)
            ax.set_title(f"{angle} degrees");ax.set_ylim(0,25);ax.set_xlim(0,1.)
            diff.axhline(0,color="black",lw=.8);diff.set_ylim(-3,6)
            for a in (ax,diff):a.grid(alpha=.2)
            diff.set_xlabel("Depth (mm)")
        axes[0,0].set_ylabel("Fz (N)");axes[1,0].set_ylabel("FE - experiment (N)")
        axes[0,0].legend(fontsize=8,loc="upper left")
        fig.suptitle("Common Maxwell model: accepted paths vs same-angle experiment")
        fig.text(.5,.01,"Cross = incomplete prefix. A completed solve is not a validated physical model. No zero shift or extrapolation.",
                 ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.96))
        fig.savefig(out/"force_review.png",dpi=180);plt.close(fig)
    write_json(out/"review.json",report)
    body=f"<h1>本轮接触时间对照与实际失败诊断</h1><p>快照：{report['snapshot']}</p>"
    body+=('<p><strong>主要判断：33°步长减半后的力仍偏高，时间步本身没有解释该偏差；'
           '57°两种正则都在0.135 mm停滞；0°固定阈值组的退出来自结果文件替换失败，不能算力学失败。</strong></p>'
           '<p>共同材料不变，保留实验原始零点。灰线为已有完整0.01 mm路径；新结果只画到已接受位置。'
           '运行中曲线不是终态验收结果。早期区间仅暂作完整体比较。</p>'
           '<img src="force_review.png" style="width:100%;max-width:1300px" alt="三个角度的Fz和实验偏差">'
           '<p>特别说明：velocity_half_33的监控器曾因queue_status.json替换失败提前退出本组；'
           '求解子进程继续运行，不能按监控器的worker_case_failed判定力学失败。'
           '独立有界接管器负责等待、核查终态，见下方连续验证入口。</p>'
           '<table><tr><th>组别</th><th>求解器状态</th><th>监控器状态</th><th>已接受mm</th><th>minJ</th><th>失败类别</th><th>终态审查</th></tr>')
    for r in report["cases"]:
        body+=f'<tr><td>{r["angle"]}° {r["name"]}</td><td>{r["status"]}</td><td>{r.get("supervisor_status") or "—"}</td><td>{r.get("accepted_depth_mm","—")}</td><td>{r.get("path_min_j","—")}</td><td>{r.get("failure_category") or "—"}</td><td>{r.get("fresh_terminal_audit","—")}</td></tr>'
    body+='</table><h2>幅值与形状必须分别检验</h2><table><tr><th>组别</th><th>区间mm</th><th>FE斜率N/mm</th><th>实验斜率</th><th>RMSE N</th></tr>'
    for r in report["cases"]:
        for w in r.get("windows",[]):
            body+=f'<tr><td>{r["angle"]}° {r["name"]}</td><td>{w["start_mm"]}–{w["end_mm"]}</td><td>{w["fe_secant_n_mm"]:.3f}</td><td>{w["experimental_secant_n_mm"]:.3f}</td><td>{w["rmse_n"]:.3f}</td></tr>'
    body+=('</table><h2>机制与下一轮门槛</h2><p>0/33°当前仿真响应的前段斜率递减，实验前段则递增。'
           '这不是统一乘一个力系数就能修复的形状差异。接触时间离散对照用于排查数值来源，不证明本构是唯一原因。'
           '保持原始正J、CCD、1e−4 N自由残差及准确双历史验收。</p>'
           '<p>已修复报告文件原子替换的短暂占用处理：有限重试，持续拒绝仍报错。'
           '57°设置原失败场的PTC/Newton/阻尼最小二乘方向探针，记录预测/真实残差、几何截断、步长和特征变化。'
           '候选需通过原增量及三个邻近增量的新鲜审查，才能进入连续路径。无效候选不追加预算。</p>'
           '<p><a href="diagnostics.html">实际机制、候选否决与下一步</a> · '
           '<a href="../contact_io_recovery_20261021/index.html">修复后的连续验证/独立接管</a></p>'
           '<p><a href="review.json">全部指标、输入哈希和限制</a> · '
           '<a href="directions/summary.json">实际失败场方向探针</a> · '
           '<a href="trust_conditional/summary.json">条件修复检验</a> · '
           '<a href="../contact_time_paths_20261021/index.html">原队列实时HTML</a></p>')
    atomic_write(out/"index.html",shell("接触时间验证审查",body))
    print(str(out/"index.html"))


if __name__=="__main__":
    main()
