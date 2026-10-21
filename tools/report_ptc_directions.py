"""Render the bounded PTC direction investigation; does not run mechanics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histories(path):
    rows=[]
    phase="original";offset=0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("NEWTON_RESIDUAL "):
            phase="newton"
            m=re.search(r"iteration=(\d+) residual_N=([\deE.+-]+)",line)
            rows.append((int(m[1]),float(m[2]),phase))
        elif line.startswith("NEWTON_FALLBACK original_PTC"):
            offset=int(re.search(r"used_directions=(\d+)",line)[1])
            phase="fallback"
        elif line.startswith("FRICTION_PTC "):
            m=re.search(r"iteration=(\d+) fresh_residual_N=([\deE.+-]+)",line)
            rows.append((offset+int(m[1]),float(m[2]),phase))
    return rows


def main(out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trace=read(out/"fast_trace/trace_summary.json")
    candidate=read(out/"fast_newton/candidate.json")
    capture=read(out/"newton_handoff/candidate.json")
    probes=[read(out/f"fast_probe{i}/probe.json") for i in (23,24)]
    handoff=read(out/"newton_handoff_probe/probe.json")
    components=read(out/"handoff_geometry/components.json")
    ptc_components=read(out/"ptc23_components/components.json")
    # The first collector's multiline print tracing emitted repeated rejected
    # callbacks. Preserve it, deduplicate by actual (iteration,retry,accepted),
    # and require duplicate records to be numerically identical.
    raw=[json.loads(s) for s in (out/"fast_trace/trace.jsonl").read_text().splitlines()]
    unique={}
    for row in raw:
        key=(row["iteration"],row["retry"],row["accepted"])
        if key in unique:assert row==unique[key]
        unique[key]=row
    rows=list(unique.values());accepted=[r for r in rows if r["accepted"]]
    assert len(accepted)==63 and len(rows)==88
    assert trace["linear_calls"]==len(rows) and trace["linear_fallbacks"]==0
    assert len(candidate["steps"])==0
    with (out/"actual_directions.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    inputs={}
    for folder in out.iterdir():
        if not folder.is_dir() or not (folder/"status.json").exists():continue
        status=read(folder/"status.json")
        assert status["status"]=="finished" and status["input_hashes_unchanged"]
        for name,digest in status["input_sha256"].items():
            path=Path(name)
            if path.name=="diagnose_ptc_directions.py":
                path=folder/"diagnostic_source.py"
            assert sha(path)==digest,path
            inputs[str(path)]=digest
        for name in ("status.json","run.log"):
            inputs[str(folder/name)]=sha(folder/name)
    base_edge=next(e for e in components["base_contact_geometry"]["nearby_blade_edges"]
                   if e["blade_local_nodes"]==[25,26])
    f=components["fractions"][0]
    trial_edge=next(e for e in f["contact_geometry"]["nearby_blade_edges"]
                    if e["blade_local_nodes"]==[25,26])
    t0=base_edge["projection_parameter"]
    derivative=(trial_edge["projection_parameter"]-t0)/f["alpha"]
    crossing=-t0/derivative
    max_direction=f["maximum_displacement_mm"]/f["alpha"]
    data=dict(scope="one failed fast/57 increment plus fixed-field probes; no accepted new loads",
        raw_trace_callback_rows=len(raw),unique_linear_trials=len(rows),accepted_ptc_updates=len(accepted),
        duplicate_callback_note="Repeated multiline print trace events were deduplicated; collector now guards event identity.",
        original_replay=trace,candidate=candidate,newton_handoff=capture,
        endpoint_event=dict(node=10031,blade_edge=[25,26],alpha=crossing,
                            maximum_direction_displacement_nm=crossing*max_direction*1e6),
        input_sha256=inputs,figures=dict(medium="local research HTML; not journal submission",
            smoothing="none; straight segments only guide between measured probes",
            uncertainty="deterministic solver diagnostics, no statistical intervals",
            normalization="panel B residual ratio; panel C 100*(trial/current-1)",
            exclusions="panel C explicitly restricted to alpha<=1/32; complete data in probe.json",
            matplotlib_version=matplotlib.__version__))
    (out/"analysis.json").write_text(json.dumps(data,indent=2),encoding="utf-8")

    with plt.rc_context({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "savefig.facecolor":"white","svg.fonttype":"path"}):
        fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
        ax=axes[0,0]
        for folder,label,color,style in [
            ("fast_trace","Original PTC","#0072B2","-"),
            ("fast_newton","Newton phase + PTC","#D55E00","--")]:
            h=histories(out/folder/"run.log")
            ax.plot([r[0] for r in h],[r[1] for r in h],style,color=color,label=label)
        ax.axhline(1e-4,color="black",ls=":",label="Physical tolerance")
        ax.set(yscale="log",xlabel="Outer iteration index (not equal cost)",ylabel="Residual norm (N)",
               title="A. Both bounded paths failed")
        ax.legend(fontsize=9)
        ax=axes[0,1]
        for i,p,color,marker in zip((23,24),probes,("#0072B2","#D55E00"),("o","s")):
            d=p["directions"][0];tt=sorted(d["trials"],key=lambda t:t["alpha"])
            a=np.array([t["alpha"] for t in tt]);n=p["norm_n"]
            row=next(r for r in accepted if r["iteration"]==i)
            predicted=np.sqrt(n*n+2*a*row["slope"]+a*a*row["action_norm_n"]**2)
            ax.plot(a,[t["norm_n"]/n for t in tt],marker+"-",color=color,label=f"Iter {i}: actual")
            ax.plot(a,predicted/n,"--",color=color,label=f"Iter {i}: linear model")
        ax.axhline(1,color="black",ls=":")
        ax.set(xlabel="Fraction of saved PTC direction",ylabel="Trial / current residual norm",
               title="B. Finite-step error vs ascent direction")
        ax.legend(fontsize=9)
        ax=axes[1,0]
        d=handoff["directions"][0];tt=sorted(
            [t for t in d["trials"] if t["alpha"]<=1/32],key=lambda t:t["alpha"])
        a=np.array([t["alpha"] for t in tt]);n=handoff["norm_n"]
        ax.plot(a,[100*(t["norm_n"]/n-1) for t in tt],"o-",color="#0072B2",label="Fresh force")
        ax.plot(a,-100*a,"--",color="#D55E00",label="Newton linear prediction")
        ax.axhline(0,color="black",lw=.8)
        ax.axvline(1/512,color="#555555",ls=":",label="Original last backtrack")
        ax.axvline(crossing,color="#6B4C9A",ls="-.",label="Edge endpoint crossing")
        ax.set_xscale("log",base=2)
        ax.set(xlabel="Newton fraction alpha",ylabel="Residual change (%)",
               title="C. Handoff detail: alpha <= 1/32")
        ax.legend(fontsize=8)
        ax=axes[1,1]
        ff=sorted(components["fractions"],key=lambda t:t["alpha"])
        for name,color,marker in (("bulk","#0072B2","o"),("normal","#D55E00","s"),
                                  ("friction","#6B4C9A","^")):
            ax.plot([t["alpha"] for t in ff],[t["components"][name]["model_error_n"] for t in ff],
                    marker+"-",color=color,label=name)
        ax.axvline(crossing,color="#555555",ls=":")
        ax.set_xscale("log",base=2);ax.set_yscale("log")
        ax.set(xlabel="Newton fraction alpha",ylabel="Component model error (N)",
               title="D. Normal-contact reconstruction dominates")
        ax.legend()
        for ax in axes.ravel():ax.grid(alpha=.2)
        fig.suptitle("fast / 57 degrees, 0.19 -> 0.20 mm: rejected increment diagnostics")
        fig.savefig(out/"directions.png",dpi=170)
        fig.savefig(out/"directions.svg")
        plt.close(fig)
    html=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>实际PTC方向、度量与全局化核查</title>
<style>body{{max-width:1180px;margin:32px auto;padding:0 22px;font:16px/1.7 system-ui,sans-serif;color:#192b38}}
img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:9px;border-bottom:1px solid #ccc;text-align:left}}a{{color:#075a9c}}
.box{{background:#eef4f8;padding:18px;border-radius:10px}}code{{overflow-wrap:anywhere}}</style>
<h1>实际PTC方向、度量与全局化核查</h1>
<p>2026-10-20。针对fast/57°的0.19→0.20 mm，使用原冻结源、双历史、初猜、物理方程与PARDISO8。
原PTC与Newton对照均限180秒，不增加原500次主迭代预算，不缩加载步，不放宽验收。</p>
<div class="box"><b>结论：已找到具体数值机制，但尚未修复原失败增量。</b>
原PTC存在接受预测上升方向的阶段，也存在边内—顶点接触切换造成有限步模型失准的阶段。
现有Newton回溯没有解决它；没有新的物理接受态，邻近加载未启动。</div>
<h2>重放可靠性</h2>
<p>原失败在{trace["solve_s"]:.2f}秒内复现：63次接受更新、25次拒绝、88次线性求解；
末态与原保存场最大位置差{trace["terminal_position_max_difference_mm"]:.3g} mm，PARDISO无回退。
记录的实际方向、shift与1e−9度量正则来自原循环。</p>
<p>首版Python多行print跟踪重复发出拒绝回调，报告已按迭代/重试去重并逐项验证重复值完全相同；
真实线性调用数仍为88。原始跟踪保留，采集器已修正，未改变物理求解。</p>
<h2>两种反弹，不能混为一谈</h2>
<table><tr><th>原PTC</th><th>当前残差N</th><th>预测完整步N</th><th>实际完整步N</th><th>现场含义</th></tr>
<tr><td>第23次</td><td>0.0208490</td><td>0.0194399</td><td>0.0226876</td>
<td>方向局部下降；完整步跨接触分支。半步实际降到0.0201295 N。</td></tr>
<tr><td>第24次</td><td>0.0226876</td><td>0.0267002</td><td>0.0267873</td>
<td>方向本身上升，rᵀAd/||r||²=+0.1662。shift增至4倍仍上升。</td></tr></table>
<p>第24次M方向二次型为正，A方向二次型却为负。M是辅助PSD度量，并未保证
新鲜残差的下降方向；其M⁻¹加权预测范数也增长约13.1%，所以不能只归咎于欧氏范数选择。
线性相对残差约4×10⁻¹⁴，排除本次线性求解精度不足。</p>
<h2>接触切换已定位到节点与刀鼻面片</h2>
<p>原PTC第23次：材料节点10022在刀鼻局部节点9–10的边上，投影参数由+0.03257变为−0.02766，
在alpha≈0.54078切到顶点接触。完整步法向模型误差0.00410083 N，材料项仅0.00008253 N；
半步不跨该分支，法向重建差接近零。</p>
<p>Newton回溯停滞点：材料节点10031对刀鼻局部节点25–26的投影参数为+0.000349416，
沿实际方向在alpha={crossing:.8g}越过端点；最大位移仅{crossing*max_direction*1e6:.3f} nm。
边—点项消失，相关顶点项权重由−0.000428118变为+0.002071882。
improved-max含带符号组合权重，这不是负压力或内核错误的证明。</p>
<p>原最后回溯alpha=1/512的模型误差：材料5.07×10⁻⁸、法向1.046×10⁻⁴、
摩擦3.662×10⁻⁵ N。误差几乎集中于节点10031，法向误差几乎等于重建接触与固定接触的力差。
不跨分支的1/4096步，法向误差降至5.63×10⁻⁹ N。这里主要是法向接触分支，摩擦随其变化；
不能把它笼统称为黏滑切换。</p>
<h2>同起点Newton对照：失败记录保留</h2>
<p>复用仓库已有24方向Newton+原PTC回退，仅把Newton线性调用接到相同PARDISO。
22次Newton方向后，残差降到0.0157557 N；10次回溯失败，回到PTC后再次40次停滞，
最终0.0331714 N，用时{candidate["elapsed_s"]:.2f}秒。零个新增接受态。</p>
<p>对Newton回退点另作固定场探针：所测回溯序列到alpha=1/2048才开始下降，最大位移0.420 nm，
残差只下降约0.0248%。这解释了为何简单加回溯次数可能仅产生极慢推进，并非已经解决。
在1e−7 mm扰动下总切线相对差约6.27×10⁻⁴；跨分支的1e−6 mm探针则误差很大。
更小扰动的舍入/活动集噪声增加，不能称全局导数收敛验证。</p>
<img src="directions.png" alt="四面板：两条失败残差路径；PTC第23和24次的实际与预测步；Newton回溯端点切换；材料、法向与摩擦模型误差。">
<p>图为确定性求解诊断，不是实验力曲线。A横轴不是等成本时间；B以各自起点残差归一化；
C明确只显示alpha≤1/32，完整探针保留JSON；D为分量模型误差。没有平滑、外推或统计置信区间。</p>
<h2>修复重点与限制</h2>
<p>下一步应在同一方程上做刀鼻端点分支感知的方向/全局化对照，检查能否跨过已定位的具体事件；
若该离散本身妨碍稳健计算，再评估连续刀鼻表示，不能直接推广此前深部漏积分的旧解析方案。
优先级已从增预算、统一增shift或多加回溯，转向实际接触分支。尚未证明这是全部工况的唯一根因。</p>
<p>主线源码未改，未启用新默认、未开长队列。9项已有Newton/回退回归通过。
本次不是材料失配或实验跨角度误差已解决的证明。</p>
<h2>可复核记录</h2><ul>
<li><a href="analysis.json">汇总、出处哈希及图形变换</a> · <a href="actual_directions.csv">88个真实线性试算</a></li>
<li><a href="fast_trace/run.log">原PTC重放日志</a> · <a href="fast_newton/run.log">Newton完整失败日志</a></li>
<li><a href="fast_probe23/probe.json">第23次方向探针</a> · <a href="fast_probe24/probe.json">第24次方向探针</a></li>
<li><a href="ptc23_components/components.json">原PTC接触事件</a> · <a href="handoff_geometry/components.json">Newton端点几何与分量</a></li>
<li><a href="newton_handoff_probe/probe.json">Newton回退点完整回溯探针</a> · <a href="newton_tests.log">9项回归</a></li>
</ul></html>"""
    (out/"index.html").write_text(html,encoding="utf-8")
    print("Report and two figure formats written; original input hashes verified.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args().output.resolve())
