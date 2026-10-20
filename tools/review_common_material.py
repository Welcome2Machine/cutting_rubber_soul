"""Snapshot accepted common-material results, including incomplete parameter sets.

Analysis only: no load steps, fitting, queue edits, or restarts. Reassemble
terminal endpoints and rejected trials from the study's frozen physical source.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from datetime import datetime
from html import escape


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(study, out):
    if out.exists():
        raise ValueError("Use a new review directory; preserve prior snapshots")
    out.mkdir(parents=True)
    begun = time.monotonic()
    # All mechanics imports resolve to the frozen study, not the working tree.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / ".runtime/ipc-python"))
    sys.path.insert(0, str(study / "source"))
    import numpy as np
    from threadpoolctl import threadpool_limits
    from tools.run_common_material_study import make, signature, check, verify_case
    from tools.run_material_candidate import load_force_csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = read(study / "manifest.json")
    check(manifest, study)
    queue = read(study / "queue_status.json")
    (out / "queue_snapshot.json").write_text(json.dumps(queue, indent=2), encoding="utf-8")
    evidence = {str(study / "source" / p): h for p, h in manifest["source_sha256"].items()}
    evidence.update(manifest["inputs"])
    experiments = {}
    for angle in (0, 33, 57):
        path = study / f"experiment{angle}.csv"
        d, f, _ = load_force_csv(path)
        experiments[angle] = (d, f)
        evidence[str(path)] = sha(path)

    records, curves, failures, rows = [], {}, [], []
    jobs = [dict(name=f"baseline_{a}", angle=a, group="baseline",
                 folder=Path(manifest["baselines"][str(a)])) for a in (0, 33, 57)]
    jobs += [dict(j, folder=study / j["name"]) for j in queue["jobs"] if j["stage"] == "train"]
    for job in jobs:
        folder = job["folder"]
        if not (folder / "result.json").exists():
            records.append(dict(name=job["name"], angle=job["angle"], group=job["group"],
                                status="not_started", states=0))
            continue
        # Preserve the exact result bytes used even if the live case advances.
        content = (folder / "result.json").read_bytes()
        r = json.loads(content)
        (out / (job["name"] + ".snapshot.json")).write_bytes(content)
        ss = [s for s in r["steps"] if s["depth_mm"] <= .5 + 1e-10]
        if not ss:
            records.append(dict(name=job["name"], angle=job["angle"], group=job["group"],
                                status=r["status"], states=0))
            continue
        terminal = r["status"] in ("completed", "failed", "budget_exhausted")
        audited = terminal and (folder / "audit.json").exists() and read(folder / "audit.json")["passed"]
        if audited:
            evidence.update(verify_case(folder))
        else:
            for i in range(len(ss)):
                path = folder / "states" / f"{i:04d}.npz"
                evidence[str(path)] = sha(path)
        d = np.array([s["depth_mm"] for s in ss])
        f = np.array([s["reaction_fz_n"] for s in ss])
        assert np.isfinite(f).all() and np.all(np.diff(d) > 0)
        angle = job["angle"]
        ed, ef = experiments[angle]
        curves[job["name"]] = (d, f)
        end = float(d[-1])
        anchors = []
        for dep in (.1, .25, .5):
            if dep <= end + 1e-10:
                v = float(np.interp(dep, d, f))
                e = float(np.interp(dep, ed, ef))
                anchors.append(dict(depth_mm=dep, fe_n=v, experiment_n=e, error_n=v-e))
        secants = []
        for a, b in ((.1, .25), (.25, .5)):
            if b <= end + 1e-10:
                v = float(np.diff(np.interp([a, b], d, f))[0] / (b-a))
                e = float(np.diff(np.interp([a, b], ed, ef))[0] / (b-a))
                secants.append(dict(window_mm=[a, b], fe_n_mm=v, experiment_n_mm=e,
                                    relative_error=v/e-1))
        metric = None
        if end >= .5 - 1e-10:
            grid = np.arange(10, 51) / 100
            err = np.interp(grid, d, f) - np.interp(grid, ed, ef)
            metric = dict(window_mm=[.1, .5], rmse_n=float(np.sqrt(np.mean(err**2))),
                          max_abs_n=float(abs(err).max()), bias_n=float(err.mean()))
        log = (folder / "run.log").read_text(encoding="utf-8") if (folder / "run.log").exists() else ""
        ls = r.get("linear_statistics", {})
        rec = dict(name=job["name"], angle=angle, group=job["group"], status=r["status"],
                   snapshot_time=datetime.now().astimezone().isoformat(),
                   source_result_sha256=hashlib.sha256(content).hexdigest(),
                   states=len(ss), depth_mm=end, fz_n=float(f[-1]),
                   path_min_j=min(s["min_j"] for s in ss), terminal_audited=bool(audited),
                   error=r.get("error"), anchors=anchors, secants=secants, metric=metric,
                   parameters=r.get("job", {}).get("parameters"),
                   solve_s=r.get("solve_s"), prior_elapsed_s=r.get("prior_elapsed_s"),
                   linear_statistics=ls, accepted_step_seconds=sum(s.get("wall_step_s", 0) for s in ss),
                   unique_tip_states=sum(s.get("tip_contact", {}).get("status") == "unique" for s in ss))
        if ls and r.get("solve_s"):
            rec["linear_fraction_of_solve"] = ls["total_s"] / r["solve_s"]
        records.append(rec)
        for s in ss:
            rows.append(dict(case=job["name"], angle=angle, depth_mm=s["depth_mm"],
                             fz_n=s["reaction_fz_n"],
                             experiment_n=float(np.interp(s["depth_mm"], ed, ef)),
                             min_j=s["min_j"], terminal_audited=bool(audited)))

        if job["group"] == "baseline" or not audited:
            continue
        with threadpool_limits(1):
            p = make(angle, r["job"]["parameters"])
            assert json.loads(json.dumps(signature(p))) == r["signature"]
            with np.load(folder / "states" / f"{len(r['steps'])-1:04d}.npz") as z:
                rec["fresh_endpoint"] = p.audit_state(z, r["steps"][-1])
            if not (folder / "failed_trial.npz").exists():
                continue
            failed_path = folder / "failed_trial.npz"
            evidence[str(failed_path)] = sha(failed_path)
            depth = r["rejected_load_attempts"][-1]["depth_mm"]
            with np.load(failed_path) as z, np.load(folder / "states" / f"{len(r['steps'])-1:04d}.npz") as prev:
                x = z["positions"].copy()
                p.previous = z["previous_positions"].copy()
                p.previous_material_metric = z["previous_material_metric"].copy()
                p.material_dt = float(z["material_dt_s"])
                assert np.array_equal(p.previous, prev["positions"])
                assert np.array_equal(p.previous_material_metric, prev["material_metric"])
                assert np.array_equal(z["reference_xy"], p.rest)
                assert np.array_equal(z["triangles"], p.mesh.triangles)
                assert abs(p.material_dt - (depth-end)/p.loading_rate) < 1e-12
            energy, g, _, minj = p.evaluate(x, p.lag(x), tangent=False)
            residual = float(np.linalg.norm(g[p.free]))
            c = p.collisions(x)
            sx = x[p.surface]
            nodal_g = np.zeros_like(g)
            nodal_g[p.free] = g[p.free]
            norm = np.linalg.norm(nodal_g.reshape(-1, 2)[:p.n], axis=1)
            ids = np.argsort(norm)[-10:][::-1]
            blade = p.prescribed_blade_positions(depth)
            tip = blade[np.argmin(blade[:, 1])]
            blade_error = float(np.max(abs(x[p.n:]-blade)))
            assert blade_error <= p.blade_boundary_roundoff_tolerance(depth)
            assert np.array_equal(x[p.mesh.bottom_nodes], p.rest[p.mesh.bottom_nodes])
            entries = [dict(iteration=int(i), residual_n=float(v), shift=float(s))
                       for dep, i, v, s in re.findall(
                           r"FRICTION_PTC depth_mm=([\deE.+-]+) iteration=(\d+) fresh_residual_N=([\deE.+-]+) metric_shift=([\deE.+-]+)", log)
                       if abs(float(dep)-depth) < 1e-10]
            assert entries and abs(entries[-1]["residual_n"]-residual) < 1e-8
            ref, progress = float("inf"), 0
            for ent in entries:
                if ent["residual_n"] < .9*ref:
                    ref, progress = ent["residual_n"], ent["iteration"]
            failures.append(dict(case=job["name"], attempted_depth_mm=depth,
                residual_n=residual, ratio_to_tolerance=residual/1e-4, min_j=float(minj),
                distance_mm=float(np.sqrt(c.compute_minimum_distance(p.cmesh, sx))),
                intersections=bool(p.ipc.has_intersections(p.cmesh, sx)),
                blade_error_mm=blade_error, previous_histories_match=True,
                bottom_matches=True, material_dt_s=p.material_dt,
                top10_squared_residual_fraction=float(np.sum(norm[ids]**2)/residual**2),
                max_top10_distance_to_tip_mm=float(np.max(np.linalg.norm(x[ids]-tip, axis=1))),
                top10_nodes=[dict(node=int(i), residual_n=float(norm[i]), xy=x[i].tolist()) for i in ids],
                log_iterations=entries, last_progress_iteration=progress,
                iterations_without_10percent_improvement=entries[-1]["iteration"]-progress,
                minimum_logged_residual_n=min(v["residual_n"] for v in entries),
                linear_statistics=ls, interpretation="Rejected field; not an equilibrium or proof of a specific friction/contact bug"))
        print(job["name"], "verified", flush=True)

    for path, h in evidence.items():
        if sha(path) != h:
            raise ValueError("Evidence changed: " + path)
    with (out / "accepted_force_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey="row", layout="constrained")
        colors = {"baseline": "#666666", "soft": "#0072B2", "fast": "#D55E00", "partition": "#009E73"}
        styles = {"baseline": "--", "soft": "-", "fast": "-.", "partition": ":"}
        for col, angle in enumerate((0, 33, 57)):
            ed, ef = experiments[angle]
            mask = ed <= .5
            axes[0, col].plot(ed[mask], ef[mask], "k-", lw=1.6, label="Experiment (raw)")
            for group in colors:
                name = f"{group}_{angle}"
                if name not in curves:
                    continue
                d, f = curves[name]
                rec = next(r for r in records if r["name"] == name)
                suffix = " [partial]" if d[-1] < .5-1e-10 else ""
                if not rec["terminal_audited"]:
                    suffix += " [live]"
                label = group + suffix
                for ax, v in ((axes[0, col], f), (axes[1, col], f-np.interp(d, ed, ef))):
                    ax.plot(d, v, styles[group], color=colors[group], lw=1.6, label=label)
                    ax.plot(d[-1], v[-1], "s" if suffix else "o", color=colors[group], ms=5)
            axes[0, col].set_title(f"{angle} degrees")
            axes[0, col].legend(fontsize=8)
            axes[1, col].axhline(0, color="black", lw=.7)
            axes[1, col].set_xlabel("Lab vertical depth (mm)")
            for ax in axes[:, col]:
                ax.set_xlim(0, .51)
                ax.grid(alpha=.2)
        axes[0, 0].set_ylabel("Fz (N)")
        axes[0, 0].set_ylim(bottom=0)
        axes[1, 0].set_ylabel("FE minus experiment (N)")
        fig.suptitle("Common material probes: accepted prefixes only; no extrapolation")
        fig.savefig(out / "prefix_comparison.png", dpi=170)
        fig.savefig(out / "prefix_comparison.svg")
        plt.close(fig)
        if failures:
            fig, axes = plt.subplots(1, len(failures), figsize=(6*len(failures), 4), squeeze=False, layout="constrained")
            for ax, f in zip(axes[0], failures):
                ax.semilogy([v["iteration"] for v in f["log_iterations"]],
                            [v["residual_n"] for v in f["log_iterations"]], color="#0072B2")
                ax.axhline(1e-4, color="black", ls="--", label="Original acceptance")
                ax.set(title=f'{f["case"]}: rejected {f["attempted_depth_mm"]:.2f} mm',
                       xlabel="PTC iteration", ylabel="Fresh free residual (N)")
                ax.legend()
                ax.grid(alpha=.2)
            fig.savefig(out / "failure_residuals.png", dpi=160)
            plt.close(fig)
    report = dict(time=datetime.now().astimezone().isoformat(), queue_snapshot_status=queue["status"],
                  scope="Read-only snapshot; terminal endpoint/failure assembly, no loading/fitting/queue change",
                  records=records, failures=failures, evidence=evidence,
                  verified_files=len(evidence), accepted_rows=len(rows),
                  code_sha256=sha(Path(__file__)), elapsed_s=time.monotonic()-begun)
    (out / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("time", "verified_files", "accepted_rows", "elapsed_s")}))
    render(out)


def render(out):
    """Render the immutable review snapshots, without revisiting a running case."""
    import numpy as np
    report = read(out / "review.json")
    records = report["records"]
    by_name = {r["name"]: r for r in records}
    sensitivity = []
    for angle in (0, 33, 57):
        with (out.parent / f"experiment{angle}.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        dk = next(k for k in rows[0] if k.startswith("Depth"))
        fk = next(k for k in rows[0] if k.startswith("Fz"))
        curves = {"experiment": (np.array([float(r[dk])/1000 for r in rows]),
                                 np.array([float(r[fk]) for r in rows]))}
        for group in ("baseline", "soft", "fast", "partition"):
            path = out / f"{group}_{angle}.snapshot.json"
            if not path.exists():
                continue
            ss = read(path)["steps"]
            if not ss or ss[-1]["depth_mm"] < .5 - 1e-10:
                continue
            curves[group] = (np.array([r["depth_mm"] for r in ss]),
                             np.array([r["reaction_fz_n"] for r in ss]))
        for label, (d, f) in curves.items():
            values = []
            for start in (.08, .1, .12):
                for split in (.23, .25, .27):
                    for end in (.48, .5):
                        slopes = []
                        for lo, hi in ((start, split), (split, end)):
                            keep = (d >= lo-1e-10) & (d <= hi+1e-10)
                            slopes.append(float(np.polyfit(d[keep], f[keep], 1)[0]))
                        values.append(dict(window_mm=[start, split, end],
                                           slopes_n_mm=slopes, ratio=slopes[1]/slopes[0]))
            sensitivity.append(dict(angle=angle, case=label, windows=values,
                min_ratio=min(v["ratio"] for v in values),
                max_ratio=max(v["ratio"] for v in values),
                count_above_one=sum(v["ratio"] > 1 for v in values)))
    (out / "slope_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
    style = """body{font:16px/1.65 system-ui,sans-serif;color:#17212d;max-width:1200px;margin:32px auto;padding:0 24px;background:#f8fafc}
    h1{font-size:27px}h2{margin-top:32px;font-size:22px}table{border-collapse:collapse;width:100%;background:white}
    td,th{padding:9px;border:1px solid #cbd5e1;text-align:right}th:first-child,td:first-child{text-align:left}
    th{background:#e7eef5}img{width:100%;height:auto;background:white}.box{border-left:5px solid #0072b2;background:#eaf4fa;padding:15px}
    .caution{border-left-color:#a64600;background:#fff2e3}a{color:#005b91}small{color:#435469}code{word-break:break-all}"""
    body = f"""<h1>共同材料检验：当前全部接受结果与失败分析</h1>
    <p>审查快照：{escape(report['time'])}。本页是固定快照，正在运行的 partition 前缀可能已推进。
    <a href="../index.html">后台最新状态</a> · <a href="review.json">核查数值</a> ·
    <a href="accepted_force_comparison.csv">全部已接受前缀对比CSV</a></p>
    <div class="box"><b>结论：</b>已有共同参数能大幅改善0°/33°的早期力幅值，所以不能把原偏差直接判定为“材料律必然错误”。
    但降低刚度、加快松弛均未修正前后段斜率关系；两组57°又因局部平衡停滞而缺少完整响应。
    当前没有一组通过跨角度验收，不能选取“能算完且像实验”的57°参数当成材料真值。</div>
    <h2>1. 各组实际状态</h2><p>三个方向都使用共同材料参数，K、摩擦、刀具、网格、边界与原始验收不变。
    soft：两剪切模量同乘0.7；fast：τ从1.7降至0.85 s；partition：固定G₀=1.38 MPa，将长期份额从10.1%升至30%。</p>
    <table><tr><th>算例</th><th>状态</th><th>接受深度 mm</th><th>Fz N</th><th>路径最小J</th><th>终态重新审查</th></tr>"""
    for r in records:
        if r["group"] == "baseline":
            continue
        vals = [r["name"], r["status"], f'{r["depth_mm"]:.2f}' if r["states"] else "—",
                f'{r["fz_n"]:.4f}' if r["states"] else "—",
                f'{r["path_min_j"]:.4f}' if r["states"] else "—",
                "通过" if r.get("terminal_audited") else "待终态审查/未开始"]
        body += "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in vals) + "</tr>"
    body += """</table><p>预检不重复计为独立材料算例。未完成曲线只绘其接受前缀；失败尝试场不进入Fz对比。
    两个失败57°均非超时，partition/57在此快照尚未开始。</p>
    <img src="prefix_comparison.png" alt="三角度实验与基线、soft、fast、partition的已接受前缀，以及逐深度力误差。未完成曲线在实际终点停止。">
    <p><a href="prefix_comparison.svg">矢量图</a>。黑线为原始实验；没有平移、再归零、倍乘或平滑。
    0°为1008n0a2，33°为1008n33a1，57°为1014n57a1。57保留0.51 N初值，与另两条的上游归零惯例不同。
    这会影响早期绝对偏差，常数力偏移不能消除分段斜率差。0.1–0.5 mm是暂定早期窗口，未宣称已逐帧证实完整。</p>
    <h2>2. 幅值显著改善，但终点接近掩盖了形状问题</h2>
    <table><tr><th>组别/角度</th><th>F(.25) N</th><th>F(.5) N</th><th>.1–.5 RMSE N</th><th>.1–.25斜率</th><th>.25–.5斜率 N/mm</th></tr>"""
    for a in (0, 33):
        e = by_name[f"baseline_{a}"]
        av = e["anchors"]
        body += f'<tr><td>实验 {a}°</td><td>{av[1]["experiment_n"]:.3f}</td><td>{av[2]["experiment_n"]:.3f}</td><td>—</td><td>{e["secants"][0]["experiment_n_mm"]:.2f}</td><td>{e["secants"][1]["experiment_n_mm"]:.2f}</td></tr>'
        for group in ("baseline", "soft", "fast", "partition"):
            r = by_name.get(f"{group}_{a}")
            if not r or not r.get("metric"):
                continue
            av, ss = r["anchors"], r["secants"]
            body += f'<tr><td>{group} {a}°</td><td>{av[1]["fe_n"]:.3f}</td><td>{av[2]["fe_n"]:.3f}</td><td>{r["metric"]["rmse_n"]:.3f}</td><td>{ss[0]["fe_n_mm"]:.2f}</td><td>{ss[1]["fe_n_mm"]:.2f}</td></tr>'
    body += """</table><p>soft将0°/33° RMSE降低75.2%/71.9%；fast降低60.4%/58.6%。这说明共同材料参数确有幅值调整能力。
    但是soft/0后段斜率低21.8%，soft/33前段斜率高30.1%；fast/0后段低31.1%，fast/33前段高33.2%。
    <b>即使补齐57°，soft和fast这两个原样参数集也已不能通过预设的所有角度斜率误差≤20%门槛。</b>
    对此不能靠放宽门槛或只报0.5 mm终点来“通过”。</p>
    <p>partition/0的F(.5)=13.317 N，比实验8.270 N高61.0%，也比原基线11.485 N高15.95%；幅值方向不合适。
    但它的后段/前段割线比从基线0.838提高到0.954，说明降低松弛份额有助于减轻斜率下滑。
    它是有用的机制方向，并非可以单独采用的候选；不能据此把未计算的组合参数称作有效FE模型。</p>
    <p>18组窗口敏感性：起点取.08/.10/.12，分段点.23/.25/.27，终点.48/.50 mm；各段用该段全部原始点线性回归，比较后/前斜率。
    这是描述性趋势核查，不是置信区间或材料切线模量。</p>
    <table><tr><th>角度/曲线</th><th>后/前回归斜率比范围</th><th>18个窗口中大于1的数量</th></tr>"""
    for s in sensitivity:
        body += f'<tr><td>{s["angle"]}° {s["case"]}</td><td>{s["min_ratio"]:.3f}–{s["max_ratio"]:.3f}</td><td>{s["count_above_one"]}/18</td></tr>'
    body += """</table><p><a href="slope_sensitivity.json">完整窗口与斜率</a>。
    0°/33°实验轻微增斜率、soft/fast减斜率的相反关系在全部窗口均成立。
    57基线并非完美：早期力有负偏，后期转正偏，.25–.5割线高33.7%；其小RMSE存在正负偏差跨深度变化，不能证明机制正确。</p>
    <h2>3. 57°失败已定位到很小的刀鼻区域，但尚未唯一确定接触算法缺陷</h2>
    <table><tr><th>算例</th><th>拒绝深度 mm</th><th>真实残差 N</th><th>残差/容限</th><th>失败场minJ</th><th>前10节点残差平方占比</th><th>这些节点距刀尖最大距离 mm</th></tr>"""
    for f in report["failures"]:
        body += f'<tr><td>{f["case"]}</td><td>{f["attempted_depth_mm"]:.2f}</td><td>{f["residual_n"]:.6f}</td><td>{f["ratio_to_tolerance"]:.1f}</td><td>{f["min_j"]:.4f}</td><td>{100*f["top10_squared_residual_fraction"]:.2f}%</td><td>{f["max_top10_distance_to_tip_mm"]:.5f}</td></tr>'
    body += """</table><img src="failure_residuals.png" alt="两组57°的失败步残差先下降后形成平台，均高于原始1e-4 N容限。">
    <p>soft在迭代102、fast在迭代63触发原40次停滞判据；判据是没有再次降低至少10%，不是40次完全没有变化。
    两个失败场重新装配与日志残差相符；上一接受位置与材料历史完全一致，dt=.05 s，刀具/底面边界正确；
    无边界相交，最小间距分别.002060/.001644 mm，未见J≤0。不能把失败认作橡胶破裂、物理解不存在或内存不足。</p>
    <p>两失败路径分别执行699/869次线性求解，均无后端回退记录；直接线性求解耗时54.45/67.74 s，占各路径计时约8.8%/7.9%。
    证据支持优先检查刀鼻接触邻接、特征切换、真实残差Jacobian与PTC方向。
    当前代码已经有接触刷新耦合导数，不能未经核查就宣称“漏掉了摩擦导数”。局部材料—接触耦合也尚未排除。</p>
    <p>soft/57最后接受.09 mm的力比同深度基线低28.16%；fast/57在.19 mm低3.65%，已开始改变原57吻合。
    这些仅是已有浅层前缀，不外推.5 mm，也不拿数值失败筛选所谓“正确材料”。唯一刀尖材料点观测仍有缺失：soft/57前缀10态均无唯一点，fast/57仅9/20态有唯一点，不能将缺失画成零滑移。</p>
    <h2>4. 下一步重点与停止条件</h2>
    <ol><li><b>先使57°的诊断可用。</b>仅针对这两个已保存失败增量，保持准确历史，检查刀具/材料具体接触对、
    鼻部面片或端点分支、正常力/摩擦项及实际Jacobian的双尺度方向差分，并用已成功基线邻近场作对照。
    根据证据选择求解全局化或接触几何修复。验收为这两个原物理增量达到原残差/几何门槛，并控制力和接触材料身份变化；
    不是追加迭代数、缩步或重开大队列。</li>
    <li><b>材料方向改为“降低起步刚度，同时减少随后松弛导致的斜率下降”。</b>
    先检验较低G₀与较小松弛份额的共同组合；单独继续缩短τ已显示不利形状方向。
    partition幅值过高不意味着这个方向对组合没有价值，但当前结果不足以指定成功的新参数。
    尤其57基线已有偏大的后段斜率，减少松弛可能与57的改善方向冲突；因此这个组合只是待检验假设，
    不是根据0/33结果直接决定的新材料参数。
    用三角度幅值、分段斜率及已能可靠提取的表面运动联合筛选，最多推进一个有依据的共同候选。
    缺失57响应时不进行三角度组合拟合，不能用固定场代理力冒充新FE。</li>
    <li><b>若共同参数只在角度间转移误差，转向模型结构。</b>优先用不同角度实际压缩/剪切路径和表面迁移约束
    检查共同材料的跨模式响应、刀鼻接触及二维/体积约束；只沿证据最强的一项干预。
    不分别给0/33/57设材料，也不把已知裂后的数据用于完整体标定。</li></ol>
    <p>当前自动队列按原有限预算继续；本次没有停止、续跑或修改其冻结源。由于soft/fast的57缺失，
    原流程不会生成三方向组合提案；partition/0也已不满足联合幅值门槛，故本轮原样候选没有依据延伸至.8 mm。
    剩余partition角度仅补充响应方向和求解行为，不能使其0°已失败的幅值门槛变成通过。</p>
    <h2>5. 理论依据与证据边界</h2>
    <p>在线性小应变单支路示意中，G(t)=G∞+Gv exp(−t/τ)：固定G₀提高G∞减少松弛，减小τ使松弛更早。
    此式仅解释参数方向；当前是有限应变局部历史模型，切削力还由变形、接触、摩擦和约束共同决定，不能用整条Fz的卷积替代FE。
    <a href="https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-timevisco.htm">Abaqus官方时间域黏弹性说明</a>
    支持剪切松弛与大应变历史处理的区别，不等于其模型和本项目实现完全相同。57曲线曾用于选参，因此不属于独立验证；
    目前能确认选择偏向与形状/转移不足，尚不能区分统计过拟合、参数未识别与模型补偿误差的唯一主因。</p>"""
    body += f"""<p>核查：{report['verified_files']}份冻结源/输入/结果/检查点文件哈希一致，
    图表含{report['accepted_rows']}个接受记录；7个已终态新算例末端按冻结物理代码重新装配通过，另重新装配2个拒绝场。
    运行中前缀没有被冒称为终态审查完成。原603个基线状态及各终态新路径使用已有逐态审查与检查点哈希，
    本次没有再次重算每个加载步，没有新平衡求解。
    <a href="queue_snapshot.json">队列快照</a> · <a href="review.json">完整证据</a></p>"""
    if (out / "latest_status.json").exists():
        latest = read(out / "latest_status.json")
        body += f'<div class="box"><b>交付前一次状态快照：</b>{escape(latest["time"])}。'
        body += "；".join(escape(f'{r["name"]}：{r["status"]}，已接受{r.get("depth_mm", "—")} mm')
                         for r in latest["rows"] if r["name"].startswith("partition"))
        body += '。这一补充不改变上面固定图表快照，不构成新的终态审查。</div>'
    (out / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>共同材料当前结果核查</title><style>'+style+'</style><body>'+body+'</body></html>', encoding="utf-8")
    files = ["review.json", "prefix_comparison.png", "failure_residuals.png",
             "accepted_force_comparison.csv", "slope_sensitivity.json", "index.html"]
    (out / "render_manifest.json").write_text(json.dumps(dict(
        time=datetime.now().astimezone().isoformat(), code_sha256=sha(Path(__file__)),
        inputs_and_outputs={p: sha(out / p) for p in files},
        transform="Raw experiment; accepted FE prefixes only; 18 paired-window OLS sensitivity; no fit or extrapolation"),
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render(args.output.resolve())
    else:
        main(args.study.resolve(), args.output.resolve())
