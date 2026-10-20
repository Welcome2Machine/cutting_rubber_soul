"""Plot the three explicitly supplied Fz CSVs against saved common-Maxwell FE.

No mechanics imports, new solves, smoothing, fitted scaling or zero/depth shifts.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from html import escape
import argparse
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    0: ("1008n0a2__Fz__curve_1.csv",
        "outputs/relaxation_partition_20261015/angles30_0_2mm/02_angle0"),
    33: ("1008n33a1__Fz__curve_2.csv",
         "outputs/native_angles_2mm_20261016/01_angle33"),
    57: ("1014n57a1__Fz__curve_0.csv",
         "outputs/native_angles_2mm_20261016/02_angle57"),
}
COLORS = {0: "#0072B2", 33: "#D55E00", 57: "#008568"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment(path, angle):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, strict=True))
    if not 2 <= len(rows) <= 100000:
        raise ValueError("Unexpected experiment length")
    metadata = {}
    for key in ("label", "original_filename", "depth_unit", "force_unit",
                "zero_applied", "speed_um_s", "component", "curve_type", "kz"):
        values = {r[key] for r in rows}
        if len(values) != 1:
            raise ValueError("Nonconstant metadata: " + key)
        metadata[key] = values.pop()
    if (metadata["depth_unit"] != "μm" or metadata["force_unit"] != "N"
            or metadata["component"] != "Fz" or metadata["curve_type"] != "experiment"
            or float(metadata["speed_um_s"]) != 200.):
        raise ValueError("Unexpected units/channel/rate")
    if f"n{angle}a" not in metadata["label"]:
        raise ValueError("Angle and trial label disagree")
    d = np.array([float(r["Depth (μm)"]) / 1000 for r in rows])
    f = np.array([float(r["Fz (N)"]) for r in rows])
    if not (np.isfinite(d).all() and np.isfinite(f).all()
            and abs(d[0]) < 1e-12 and np.all(np.diff(d) > 0)):
        raise ValueError("Nonfinite or nonmonotone initial-contact series")
    metadata.update(rows=len(rows), depth_end_mm=float(d[-1]),
                    first_force_n=float(f[0]), max_gap_mm=float(np.diff(d).max()))
    return d, f, metadata


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    curves = {}; inputs = {}; signatures = {}; anchors = []; matched = []
    for angle, (csv_name, folder_name) in CASES.items():
        ep = ROOT / "docs" / csv_name
        ed, ef, meta = experiment(ep, angle)
        folder = ROOT / folder_name
        r = read(folder / "result.json"); audit = read(folder / "audit.json")
        if not (r["completed"] and audit["passed"]
                and len(r["steps"]) == audit["states"] == len(audit["rows"]) == 201):
            raise ValueError("Incomplete or unaudited saved FE path")
        sig = json.loads(json.dumps(r["signature"]))
        sig.pop("mesh_sha256")
        if sig["config"].pop("inclination_deg") != angle:
            raise ValueError("Wrong FE angle")
        signatures[angle] = sig
        for name, digest in r["source_sha256"].items():
            inputs[str((folder / "source" / name).resolve())] = digest
        for i, (row, ar) in enumerate(zip(r["steps"], audit["rows"])):
            if not ar["passed"] or abs(ar["depth_mm"] - row["depth_mm"]) > 1e-10:
                raise ValueError("Saved audit ordering mismatch")
            if abs(ar["reaction_fz_n"] - row["reaction_fz_n"]) > 1e-7:
                raise ValueError("Saved audited force mismatch")
            inputs[str((folder / "states" / f"{i:04d}.npz").resolve())] = ar["checkpoint_sha256"]
        for path in (ep, folder / "result.json", folder / "audit.json"):
            inputs[str(path.resolve())] = sha(path)
        sd = np.array([row["depth_mm"] for row in r["steps"]])
        sf = np.array([row["reaction_fz_n"] for row in r["steps"]])
        if not (np.isfinite(sf).all() and np.all(np.diff(sd) > 0)
                and abs(sd[-1] - 2.) < 1e-12 and sd[-1] <= ed[-1]):
            raise ValueError("Invalid FE plotting domain")
        curves[angle] = (ed, ef, sd, sf, meta)
        for depth, fe, exp in zip(sd, sf, np.interp(sd, ed, ef)):
            matched.append(dict(angle_deg=angle, trial=meta["label"],
                                depth_mm=float(depth), experiment_interpolated_n=float(exp),
                                simulation_n=float(fe), simulation_minus_experiment_n=float(fe-exp)))
        for depth in (.5, 1., 1.5, 2.):
            exp = float(np.interp(depth, ed, ef)); fe = float(np.interp(depth, sd, sf))
            anchors.append(dict(angle_deg=angle, trial=meta["label"], depth_mm=depth,
                                experiment_n=exp, simulation_n=fe,
                                difference_n=fe-exp, relative_difference_percent=100*(fe-exp)/exp))
    if not signatures[0] == signatures[33] == signatures[57]:
        raise ValueError("FE common-model signatures differ beyond angle/mesh hash")
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError("Changed source data: " + path)

    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"}):
        for name, end, ymax in (("comparison_15mm", 1.5, 33), ("comparison_full", 3., 46)):
            fig, axs = plt.subplots(1, 3, figsize=(14, 5.2), sharex=True, sharey=True,
                                    layout="constrained")
            for ax, (angle, data) in zip(axs, curves.items()):
                ed, ef, sd, sf, meta = data
                ax.plot(ed, ef, "-", color=COLORS[angle], lw=1.4,
                        label=f"Experiment: {meta['label']}")
                ax.plot(sd, sf, "--", color="#1b1b1b", lw=1.8, label="Intact FE: common Maxwell")
                if angle == 33:
                    ax.axvspan(1.2, 1.3, color="#999999", alpha=.18, zorder=0)
                    ax.text(.04, .94, "Onset estimate: 1.2-1.3 mm", transform=ax.transAxes,
                            va="top", fontsize=9, color="#555555")
                elif angle == 0:
                    ax.text(.04, .94, "a2 trial; prior a1 onset not transferred",
                            transform=ax.transAxes, va="top", fontsize=9, color="#555555")
                else:
                    ax.text(.04, .94, "Selected trial; onset unknown", transform=ax.transAxes,
                            va="top", fontsize=9, color="#555555")
                ax.set(title=f"{angle}°", xlabel="Depth (mm)", xlim=(0, end), ylim=(0, ymax))
                ax.grid(alpha=.18)
                ax.legend(loc="lower right", fontsize=9)
                if end > 2:
                    ax.plot(sd[-1], sf[-1], "s", color="#1b1b1b", ms=4)
            axs[0].set_ylabel("Fz (N)")
            fig.suptitle("Measured CSV force vs saved intact simulation", fontsize=15)
            fig.supxlabel("Original exported force values; no added scaling, zero/depth shifts or smoothing.\n"
                          "Late experiment includes fracture; the intact FE is not a post-fracture prediction.",
                          fontsize=10)
            fig.savefig(out / f"{name}.png", dpi=170)
            if end == 1.5:
                fig.savefig(out / f"{name}.svg")
            plt.close(fig)
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matched[0]))
        writer.writeheader(); writer.writerows(matched)
    summary = dict(
        created=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        scope="Plot only; saved complete 2 mm common-Maxwell paths. No new solve/assembly/calibration.",
        experiment={str(a): c[4] for a, c in curves.items()}, anchors=anchors,
        transformations=["Depth μm / 1000 -> mm", "No further force scaling or zero/depth shift",
                         "Raw rows plotted unchanged; linear interpolation ONLY for tabulated comparisons"],
        onset={"0": "Prior ~1 mm estimate concerned 1008n0a1; not transferred to 1008n0a2",
               "33": "User rough estimate 1.2-1.3 mm for 1008n33a1; not independently detected",
               "57": "Unknown; exploratory selection trial, not independent validation"},
        input_sha256=inputs, code_sha256=sha(Path(__file__)),
        preserved_initial_force_n={str(a): float(c[1][0]) for a, c in curves.items()},
        full_curve_errors_are_not_intact_validation=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    table = "<table><thead><tr><th>工况/试次</th><th>深度 mm</th><th>实验 N</th><th>仿真 N</th><th>仿真−实验 N</th></tr></thead><tbody>"
    for row in anchors:
        table += (f'<tr><td>{row["angle_deg"]}° / {escape(row["trial"])}</td>'
                  f'<td>{row["depth_mm"]:g}</td><td>{row["experiment_n"]:.3f}</td>'
                  f'<td>{row["simulation_n"]:.3f}</td><td>{row["difference_n"]:+.3f}</td></tr>')
    table += "</tbody></table>"
    description = "".join(
        f'<li>{a}°：{escape(c[4]["label"])}，{c[4]["rows"]}点，0–{c[4]["depth_end_mm"]:.4f} mm；'
        f'导出zero_applied={c[4]["zero_applied"]}，初始Fz={c[4]["first_force_n"]:.2f} N。</li>'
        for a, c in curves.items())
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三条实验CSV与共同Maxwell仿真对比</title><style>
body{{font:16px/1.65 'Segoe UI','Microsoft YaHei',sans-serif;background:#f4f7fa;color:#1c2d40;margin:0}}
main{{max-width:1320px;padding:28px;margin:auto}}section{{background:white;padding:22px;border:1px solid #d9e2e9;margin:18px 0}}
img{{width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}
a{{color:#006785}}.note{{border-left:4px solid #b46b16;background:#fff5e5;padding:14px}}code{{overflow-wrap:anywhere}}
</style><main><h1>三条完整实验CSV与仿真对比</h1>
<p>2026-10-20 · 仅读取/绘图，未重算、调参或启动队列。</p>
<p class="note">直接使用docs中的三个数值CSV，替代本轮比较中的人工截图读数。0°为1008n0a2，不是前图1008n0a1；
此前0°约1 mm的起裂估计不直接移植为a2已知起裂点。33°标出用户估计1.2–1.3 mm；57°起裂未知。
实验后段已包含断裂，图上保留全曲线，但不据裂后差异标定完整体模型。</p>
<section><h2>0–1.5 mm：当前重点范围</h2><img src="comparison_15mm.png" alt="0、33、57度分别比较实测CSV实线与完整体仿真虚线。0和33度仿真偏高，57度接近所选试次。">
<p>各面板使用相同坐标范围。实线为实验，黑色虚线为对应角度的完整体仿真；没有平滑、额外归零、深度平移或倍率拟合。</p></section>
<section><h2>完整实验范围与已有2 mm仿真</h2><img src="comparison_full.png" alt="三组实验保留至约3毫米，仿真在已完成的2毫米终止，末端用方块标识，不进行外推。">
<p>实验保留到各自文件终点，仿真只到已有2 mm。不能把实验末端与尚不存在的3 mm仿真比较。</p></section>
<section><h2>相同深度数值</h2>{table}<p>实验仅在表中以相邻原始点线性插值到指定深度；图使用原始全部采样点。
这些差值是描述性比较，不是把完整范围认定为未裂后的验证误差。57°已参与探索选参，吻合不构成独立验证。</p></section>
<section><h2>数据核对</h2><ul>{description}</ul><p>三者均为Fz/N、深度μm、200 μm/s。CSV中的kz=100是导出元数据，
不再对已标为N的Fz乘100。0/33已由上游导出归零，57保留.51 N初值；本轮不额外改变任何零点。
沿用导出深度起点，未独立核验新增0/33 CSV的机械首触时刻。旧V01原通道未使用。</p>
<p>共同模型G∞=.14、Gv=1.24、K=23.343333 MPa，τ=1.7 s，μ=.35，竖直速度.2 mm/s。
本轮核对原终态审查、603个保存状态及冻结源哈希；没有新的有限元装配。局部压缩/空间误差限制仍见上轮报告。</p>
<p><a href="comparison.csv">对齐深度CSV</a> · <a href="comparison_15mm.svg">1.5 mm矢量图</a> ·
<a href="summary.json">来源、锚点及哈希</a> · <a href="../index.html">此前保存场审查</a></p></section>
</main></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps(dict(output=str(out),checked_files=len(inputs),anchors=anchors), ensure_ascii=False))


def calibration_review(out):
    """Trace selection and compute descriptive transfer errors, without fitting."""
    screen_path = ROOT / "outputs/relaxation_partition_20261015/screen/screen.json"
    screen = read(screen_path); inputs = {str(screen_path): sha(screen_path)}
    for value in screen["sources"].values():
        if sha(Path(value["file"])) != value["sha256"]:
            raise ValueError("Historical selection source changed")
        inputs[value["file"]] = value["sha256"]
    previous = read(out / "summary.json")
    for name, digest in previous["input_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Prior plot evidence changed")
    inputs.update(previous["input_sha256"])
    inputs[str(out / "summary.json")] = sha(out / "summary.json")
    records = []; table_rows = []; slope_rows = []
    for angle, (csv_name, folder_name) in CASES.items():
        ed, ef, meta = experiment(ROOT / "docs" / csv_name, angle)
        result = read(ROOT / folder_name / "result.json")
        sd = np.array([r["depth_mm"] for r in result["steps"]])
        sf = np.array([r["reaction_fz_n"] for r in result["steps"]])
        for depth in (.25, .5, .75, 1.):
            exp = float(np.interp(depth, ed, ef)); fe = float(np.interp(depth, sd, sf))
            record = dict(angle_deg=angle, depth_mm=depth, experiment_n=exp, simulation_n=fe,
                          difference_n=fe-exp, output_amplitude_ratio_only=exp/fe)
            table_rows.append(record)
        for low, high in ((.1, .5), (.1, .8), (.25, .5), (.5, .75)):
            mask = (sd >= low-1e-10) & (sd <= high+1e-10)
            predicted = sf[mask]; observed = np.interp(sd[mask], ed, ef)
            records.append(dict(angle_deg=angle, window_mm=[low, high], points=int(mask.sum()),
                                bias_n=float(np.mean(predicted-observed)),
                                rmse_n=float(np.sqrt(np.mean((predicted-observed)**2))),
                                note="Descriptive on FE depth grid; no statistical uncertainty or certified intactness"))
            slope_rows.append(dict(angle_deg=angle, window_mm=[low, high],
                                   experiment_n_mm=float((observed[-1]-observed[0])/(high-low)),
                                   simulation_n_mm=float((predicted[-1]-predicted[0])/(high-low))))
    sources = [
        "docs/history/topics/material_candidate.md",
        "references/native_material_audit_20261018.md",
        "outputs/angle_modes_20261015/README.md",
        "outputs/native_material_audit_20261018/parameter_screen_summary.json",
    ]
    for name in sources:
        inputs[str(ROOT / name)] = sha(ROOT / name)
    analysis = dict(
        created=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        scope="Selection provenance plus existing-force diagnostics; no FE solve, fit or new parameter choice.",
        confirmed_selection=screen["selected"], selected_data_window_mm=screen["window_mm"],
        selection_experiment=screen["sources"]["experiment"],
        surrogate_formula=screen["formula"], surrogate_transform=screen["transformation"],
        selection_limitations=screen["limitations"], anchors=table_rows,
        windows=records, secants=slope_rows, input_sha256=inputs, code_sha256=sha(Path(__file__)),
        conclusion="57 participated in selection. Cross-angle predictive validity not established. "
                   "Calibration bias/parameter compensation is evidenced; noise overfitting, unique "
                   "constitutive or discretization cause, and impossibility of a common fit are not proven.",
        target="One common model for all three angles; not maximization of the 0/33 gap.",
        official_sources=[
            "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-hyperelastic.htm",
            "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-timevisco.htm"])
    (out / "calibration_review.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    numbers = "<table><tr><th>深度</th><th>角度</th><th>实验N</th><th>FE N</th><th>FE−实验N</th></tr>"
    for depth in (.25, .5, .75):
        for r in table_rows:
            if r["depth_mm"] != depth:
                continue
            numbers += (f'<tr><td>{depth:g}</td><td>{r["angle_deg"]}°</td><td>{r["experiment_n"]:.3f}</td>'
                        f'<td>{r["simulation_n"]:.3f}</td><td>{r["difference_n"]:+.3f}</td></tr>')
    numbers += "</table>"
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>57°吻合与跨角度失配：选参及模型证据</title><style>
body{{font:16px/1.7 'Segoe UI','Microsoft YaHei',sans-serif;color:#182c40;background:#f2f6f9;margin:0}}
main{{max-width:1050px;margin:auto;padding:30px}}section{{background:white;padding:24px;margin:18px 0;border:1px solid #d8e2ea}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}
a{{color:#006b86}}.notice{{padding:16px;background:#fff4dc;border-left:5px solid #af6610}}code{{overflow-wrap:anywhere}}
</style><main><h1>不是放大0/33°差距，而是检验共同模型</h1><p>2026-10-20 · 追溯选参记录与三条CSV · 没有新求解、拟合或参数修改</p>
<p class="notice"><b>目前最有证据的判断：57°吻合含有针对这条曲线的参数选择，不能当独立验证；
当前共同模型对其它角度的预测还未成立。</b>这可能同时包含参数补偿与模型近似误差。
不能在“57°过拟合／模型错／0和33漏项”之间强行只选一个，也不能把未知原因说成已经定位。</p>
<section><h2>1. 选参偏向不是猜测，有原始记录</h2>
<p><a href="../../relaxation_partition_20261015/screen/screen.json">历史screen.json</a>记录：
用1014n57a1（当时按名义60°FE）的0.05–2 mm数据，扫描τ及平衡/松弛剪切分配，
选取G∞=.14、Gv=1.24 MPa、τ=1.7 s，K固定23.343333 MPa。
本轮重新核对其三个原输入文件的SHA256，与历史记录一致。</p>
<p>该筛选使用整体弹性力的历史卷积代理，并将每条曲线各自减去F(0)；
没有在筛选阶段重新求局部接触/变形史，整体力缩放还混入了体积及接触效应。
随后真实FE对这条同一试次较接近，说明此组参数具有拟合能力，不能证明它等于材料真值。
严格说，这是<b>单工况选参偏向和可辨识性不足</b>的证据，还不是“拟合了随机噪声”的统计过拟合证明。</p>
<p>该材料假设长期剪切模量仅为瞬时值的10.1%。实际橡胶是否有如此大的松弛幅度，没有独立材料数据认证。
筛选到2 mm时，57°的未裂区间没有确认；因此存在把未识别断裂或其它效应吸收到松弛参数的风险，
但不能据此宣布57°在某深度一定已经断裂。</p></section>
<section><h2>2. 当前偏差不只是后段断裂或一个共同倍率</h2>{numbers}
<p>这些较早区间按现有起裂粗估用于暂定诊断，不认证全部未裂；0°CSV为a2而此前起裂估计来自a1。
在.5 mm若仅把FE输出乘一个数来对齐，三个角度分别需要约.720/.654/.934。
这是输出层面的代数检查，<b>不是材料G的识别，也不是重新求解后的力</b>；
它仅排除“一个共同力倍率就足够”的解释，不能排除改变共同参数后产生不同的角度响应。</p>
<p>57°也不是所有特征均正确：.25–.5 mm实验割线刚度15.40 N/mm，FE20.59 N/mm（高约34%）；
对应力误差从−.70 N变成+.60 N。整段绝对力较接近，仍可能掩盖局部斜率不一致。
因此不应把57°当作必须完全保持不动的正确基准。</p></section>
<section><h2>3. 三种说法的证据等级</h2>
<table><tr><th>说法</th><th>当前判断</th></tr>
<tr><td>57°针对该曲线选参</td><td>已确认；同一曲线参与选取G∞、Gv、τ，吻合不是独立预测。</td></tr>
<tr><td>57°只是盲预测的巧合</td><td>不准确，它没有被留作独立检验；不同误差在57°互相抵消仍有可能，但尚未分解证明。</td></tr>
<tr><td>0/33°欠考虑某些物理</td><td>可能。共同本构、面外/支承近似、接触和离散可在不同变形史下产生不同误差，尚未识别唯一项。</td></tr>
<tr><td>材料公式或程序已证明错误</td><td>未证明。已做的导数、转动客观性、材料历史及平衡核查未发现能解释本偏差的实现错误；空间/物理验证仍不完整。</td></tr>
<tr><td>必须给三个角度不同材料</td><td>不成立，违反同一材料的已知实验前提。</td></tr></table>
<p>“欠考虑0/33”应指共同模型对它们激发的变形模式/约束响应不充分，而不是为角度增设专用软化参数。
同一材料受到不同局部压缩、剪切和接触史时，单曲线选出的参数可能只在其中一条路径补偿得好。</p></section>
<section><h2>4. 为什么现在不能直接点名某个物理项</h2>
<p>已有材料转动/导数核查与13个保存场支持实现一致性，但不是本构真实性证明。
曾做的共同G∞分配、τ小扰动主要让两角度同向变化，且只是固定运动史筛选，不是新平衡路径；
不能据此宣布所有共同参数都无解。底滑和平面应力条件对照也同时降低大倾角力，
不能仅凭0/33变好就采用与已知约束不符的边界。无松弛对照把.5 mm的0/33力升至约19.3 N，也没有解决幅值偏差。</p>
<p>官方有限元资料支持多种变形模式联合核验，并指出受限变形对压缩性敏感；
它们支持检查共同材料与约束耦合，不能指定本项目唯一根因。
<a href="https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-hyperelastic.htm">Abaqus超弹性/压缩性</a>；
<a href="https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-timevisco.htm">时间域黏弹性</a>。
本项目乘法Maxwell与其Prony实现不同，不以文档等同认证。</p></section>
<section><h2>5. 下一步应做什么，怎样判定</h2>
<ol><li><b>改变判定目标。</b>一套共同模型同时解释三角度的暂定裂前力幅值、斜率和表面运动；
不单独最大化0/33差距，也不把57°原吻合设为不能改变的硬约束。</li>
<li><b>做有界的共同参数可行性检验。</b>先在较早共同窗口（如.1–.5 mm）内，用各角度等权指标，
检查可信参数范围内能否同时降低三组误差。真正候选必须从零传播自己的材料/摩擦历史并重新平衡，
不能拿固定场虚功代理冒充预测。后续.5–.8 mm和不参与该次选参的角度检查迁移。
这些数据已被看过，应称探索性交叉检验，不宣称新的盲验证。</li>
<li><b>根据结果决定是否改模型。</b>若共同参数能够同时解释各角度，支持原参数识别偏向；
若只会在角度间搬运误差，则支持当前模型组合的结构不足，需用表面运动/反力分量和受控离散/约束对照定位。
有限范围搜索失败不等于数学上证明任何参数都不可行；数值失败也不作物理反证。
改变物理项必须能改变已测偏差模式，而不只是降低总力。</li></ol>
<p>上述是下一阶段判别方案，本轮没有启动队列或重新标定，也没有选择新的材料参数。</p></section>
<p><a href="index.html">三条CSV原始对比</a> · <a href="calibration_review.json">本轮追溯与数值证据</a></p>
</main></html>"""
    (out / "calibration_review.html").write_text(html, encoding="utf-8")
    print(json.dumps(dict(status="selection_and_transfer_review_completed",
                         source_files=len(inputs), selection_window=screen["window_mm"],
                         new_loading_solves=0), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/angle_emergence_20261020/experiment_csv")
    parser.add_argument("--diagnose-selection", action="store_true")
    args = parser.parse_args()
    if args.diagnose_selection:
        calibration_review(args.output.resolve())
    else:
        main(args.output.resolve())
