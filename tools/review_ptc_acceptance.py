"""Read-only audit of accepted PTC steps in the three saved 57-degree failures.

Uses original logs and frozen source. Does not reconstruct a direction with
invented stabilization, run FE, change acceptance, or restart a queue.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from html import escape

ROOT = Path(__file__).resolve().parents[1]
CASES = ("soft_57", "fast_57", "partition_57")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def close(a, b):
    # Original logs print nine significant digits, not full-precision states.
    return math.isclose(a, b, rel_tol=2e-8, abs_tol=2e-12)


def fields(line):
    return {k: float(v) for k, v in re.findall(r"(\w+)=([\d.eE+-]+)", line)}


def parse_failure(folder):
    result = read(folder / "result.json")
    assert result["status"] == "failed"
    assert "pseudo-transient stagnation" in result["error"]
    assert result["steps"][-1]["friction_globalization"] == "model-error"
    depth = result["rejected_load_attempts"][-1]["depth_mm"]
    states, updates, rejections = [], [], []
    current = None
    for number, line in enumerate((folder / "run.log").read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("FRICTION_PTC "):
            current = fields(line)
            if abs(current["depth_mm"] - depth) < 1e-9:
                current.update(line=number, rejects=[])
                states.append(current)
            else:
                current = None
        elif current is not None and line.startswith("FRICTION_PTC_REJECT "):
            row = fields(line)
            row.update(line=number, iteration=int(current["iteration"]))
            current["rejects"].append(row)
            rejections.append(row)
        elif current is not None and line.startswith("FRICTION_PTC_UPDATE "):
            row = fields(line)
            retry = int(row["retries"])
            assert len(current["rejects"]) == retry, "Unlogged retry; do not invent shift_used"
            used = current["metric_shift"]
            for i, rejected in enumerate(current["rejects"]):
                assert int(rejected["retry"]) == i and close(used, rejected["metric_shift"])
                used = max(.1, 2 * used)
            norm = current["fresh_residual_N"]
            nt = row["residual_N"]
            criterion = line.split("criterion=", 1)[1].strip()
            assert criterion == "model_error"
            # All accepted steps in these failures are full steps. Therefore
            # neither geometry threshold (.001 or .25) is approached, even
            # allowing for the nine-digit printing precision.
            assert row["alpha"] == 1., "Hypothesis cannot be excluded by this full-step audit"
            next_shift = max(1e-10, min(1e6, used * nt / max(norm, 1e-15) / 1.2))
            error = row["model_error_N"]
            assert error <= .25 * max(norm, nt) * (1 + 2e-8)
            updates.append(dict(
                case=folder.name, depth_mm=depth, iteration=int(current["iteration"]),
                source_line=number, norm_n=norm, nt_n=nt, residual_ratio=nt / norm,
                shift_at_iteration=current["metric_shift"], retries=retry,
                shift_used_reconstructed=used, shift_next_reconstructed=next_shift,
                alpha=row["alpha"], criterion=criterion, model_error_n=error,
                pred_norm_lower_bound_n=max(0., nt - error),
                pred_norm_upper_bound_n=nt + error,
                predicted_growth_proved=nt - error > norm * (1 + 1e-7),
                # Nulls are deliberate: scalar logs do not recover vectors.
                pred_norm_n=None, geometric_fraction=None, tip_cap_alpha=None,
                displacement_infinity_mm=None,
                geometry_and_tip_cap_lower_bound=1 - 1e-8,
                displacement_infinity_upper_bound_mm=.01 * (1 + 1e-8),
            ))
    assert len(states) == len(updates) + 1
    assert [int(s["iteration"]) for s in states] == list(range(len(states)))
    for row, following in zip(updates, states[1:]):
        assert close(row["nt_n"], following["fresh_residual_N"])
        assert close(row["shift_next_reconstructed"], following["metric_shift"])
    reference, last_progress = math.inf, 0
    for state in states:
        if state["fresh_residual_N"] < .9 * reference:
            reference, last_progress = state["fresh_residual_N"], int(state["iteration"])
    assert int(states[-1]["iteration"]) - last_progress == 40
    tail = updates[last_progress:]
    assert len(tail) == 40
    minimum = min(s["fresh_residual_N"] for s in states[last_progress:])
    summary = dict(
        case=folder.name, attempted_depth_mm=depth, accepted_depth_mm=result["steps"][-1]["depth_mm"],
        terminal_iteration=int(states[-1]["iteration"]), last_progress_iteration=last_progress,
        accepted_ptc_updates=len(updates), terminal_residual_n=states[-1]["fresh_residual_N"],
        terminal_shift=states[-1]["metric_shift"], progress_reference_n=reference,
        tail_minimum_residual_n=minimum, tail_best_relative_improvement=1 - minimum / reference,
        tail_shift_used_range=[min(r["shift_used_reconstructed"] for r in tail),
                               max(r["shift_used_reconstructed"] for r in tail)],
        tail_shift_next_range=[min(r["shift_next_reconstructed"] for r in tail),
                               max(r["shift_next_reconstructed"] for r in tail)],
        tail_rejected_trials=sum(r["retries"] for r in tail),
        tail_nondecreasing_updates=sum(r["residual_ratio"] >= 1 for r in tail),
        tail_proved_predicted_growth=sum(r["predicted_growth_proved"] for r in tail),
        all_alpha_min=min(r["alpha"] for r in updates),
        all_alpha_max=max(r["alpha"] for r in updates),
        hypothesis="excluded_for_these_accepted_updates",
    )
    return summary, updates, rejections


def render(out, report, rows):
    table = ""
    for s in report["cases"]:
        vals = [s["case"], s["attempted_depth_mm"], s["accepted_ptc_updates"],
                f'{s["last_progress_iteration"]}–{s["terminal_iteration"]-1}',
                "1 / 1", f'{s["tail_shift_used_range"][0]:.6g}–{s["tail_shift_used_range"][1]:.6g}',
                f'{s["terminal_shift"]:.6g}', f'{s["terminal_residual_n"]:.8g}',
                s["tail_nondecreasing_updates"], s["tail_proved_predicted_growth"]]
        table += "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in vals) + "</tr>"
    examples = [r for r in rows if (r["case"], r["iteration"]) in
                {("soft_57", 69), ("fast_57", 51), ("partition_57", 50)}]
    examples_html = "".join(
        f'<li>{r["case"]}，迭代{r["iteration"]}：残差 {r["norm_n"]:.9g} → {r["nt_n"]:.9g} N；'
        f'模型误差 {r["model_error_n"]:.9g} N，所以预测残差至少 '
        f'{r["pred_norm_lower_bound_n"]:.9g} N，已经高于起点。</li>' for r in examples)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PTC几何截短—稳定化反馈核查</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1250px;margin:30px auto;padding:0 22px;color:#192b38}}
table{{border-collapse:collapse;font-size:14px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ccc;text-align:left}}
.box{{background:#eef4f8;padding:18px;border-radius:8px}}code{{overflow-wrap:anywhere}}a{{color:#075a9b}}</style>
<h1>PTC几何截短—稳定化反馈核查</h1>
<p>基准提交7185d5c。仅读取真实失败日志和冻结源；没有新FE加载、没有修改求解器、没有重开队列。</p>
<div class="box"><b>结论：这条具体机制被最近三次57°失败的现场记录排除。</b>
253次已接受PTC更新全部alpha=1；各自末40次也全部是完整步。
因此几何比例与刀尖位移上限都没有截短这些已接受步，新增的两条P1几何保护不会作用于它们。
中间试算确有被拒绝的小步，但不能当成“连续接受小步”的证据。</div>
<h2>实际记录</h2>
<div style="overflow:auto"><table><tr><th>组别</th><th>失败深度mm</th><th>PTC更新数</th>
<th>末40次迭代</th><th>alpha最小/最大</th><th>实际使用shift范围</th><th>退出shift</th>
<th>退出残差N</th><th>末40次不降残差</th><th>末40次可证明预测增长</th></tr>{table}</table></div>
<p>shift_used从迭代入口与每次拒绝后的max(.1,2*shift)重建；shift_next按冻结公式重建，
逐项核对下一行原日志。精度限于原日志9位有效数字。不是使用固定shift=.1的旧探针。</p>
<h2>更符合日志的解释</h2>
<p>真实过程是完整PTC步的残差反弹与稳定化往复：若干下降步之后残差上升，
局部模型误差超限时拒绝并加倍shift，然后再次接受并回落。
40次停止条件要求残差相对最近进展基准再改善10%；不是要求每次单调下降。</p>
<p>partition末40次的最低残差比进展基准低约9.84%，未达到10%门槛；
这说明停止计数与既有规则一致，不证明增大预算就能收敛。</p>
<h2>不重建方向也能证明的一点</h2>
<p>由error=||r_trial−r_pred||与三角不等式，预测残差位于
[max(0,nt−error),nt+error]。以下记录的下界已高于原残差：</p><ul>{examples_html}</ul>
<p>因此至少这些反弹不能全部归咎于“本来预测下降、实际却因分支跳变上升”；
该PTC方向的线性预测本身就允许残差范数增大。这不证明Jacobian错误，也不证明非单调接受本身应被删除。</p>
<h2>控制流与证据边界</h2>
<ul><li>当前与失败冻结源的solve_friction、_ptc_trial、safe_step主体AST一致。</li>
<li>residual-merit确实只是额外通道；既有测试有意允许精确模型下残差增长。
本轮没有把“开关仍失败”当作排除依据。</li>
<li>没有记录精确pred_norm、CCD/J各自比例或位移向量；CSV对应字段保持空。
alpha=1已经足以排除小几何比例，不能据此宣称位移本身很大。</li>
<li>旧diagnose_path_coupling使用shift=.1和度量正则1e−8，与当前动态shift和1e−9不同；
其方向不被当作本次失败方向。</li>
<li>未做两保护补丁对照或连续加载：触发前提未命中。未证明这条机制在所有其它算例都不存在。</li></ul>
<h2>下一步重点</h2>
<p>若继续做物理求解诊断，应保存“下降转反弹”前后的实际方向、准确shift与接触特征，
比较预测残差、方向导数及有限步误差；在同一起点、双历史和预算下检验一种非单调PTC全局化。
先查方向与度量为何不产生足够净进展，不先扩大几何保护、切换同一个merit开关或延长预算。
该方向是待检验方案，不是已确定根因，也不在本轮自动开算。</p>
<p><a href="report.json">完整审查与输入哈希</a> · <a href="accepted_updates.csv">253次已接受更新</a> ·
<a href="rejected_trials.csv">拒绝试算</a> ·
<a href="../../docs/research_status.md">研究状态</a></p></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study, out = args.study.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError("Use a new review directory")
    report = dict(scope="read-only original logs; not FE or reconstructed directions",
                  cases=[], evidence_sha256={}, method_ast_equal={},
                  scalar_log_relative_tolerance=2e-8)
    frozen = study / "source/prototype/ipc_reference.py"
    for name in ("solve_friction", "_ptc_trial", "safe_step"):
        same = method(frozen, name) == method(ROOT / "prototype/ipc_reference.py", name)
        assert same, "Current control flow differs; reassess before using this review"
        report["method_ast_equal"][name] = same
    all_rows, all_rejected = [], []
    for case in CASES:
        folder = study / case
        result = read(folder / "result.json")
        for relative, digest in result["source_sha256"].items():
            path = folder / "source" / relative
            assert sha(path) == digest, path
            report["evidence_sha256"][str(path)] = digest
        assert sha(frozen) == result["source_sha256"]["prototype/ipc_reference.py"]
        for name in ("result.json", "run.log", "failed_trial.npz", "audit.json"):
            path = folder / name
            report["evidence_sha256"][str(path)] = sha(path)
        baseline = read(folder / "source/prototype/ipc_baseline.json")
        assert baseline["hessian_regularization_n_mm"] == 1e-9
        summary, rows, rejected = parse_failure(folder)
        report["cases"].append(summary)
        all_rows.extend(rows)
        all_rejected.extend(dict(case=case, **r) for r in rejected)
    for name in ("prototype/ipc_reference.py", "tools/review_ptc_acceptance.py",
                 "tools/diagnose_path_coupling.py", "tests/test_ipc_friction_merit.py"):
        report["evidence_sha256"][str(ROOT / name)] = sha(ROOT / name)
    for name, digest in report["evidence_sha256"].items():
        assert sha(Path(name)) == digest, "Input changed during review"
    out.mkdir(parents=True)
    for filename, rows in (("accepted_updates.csv", all_rows), ("rejected_trials.csv", all_rejected)):
        with (out / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    render(out, report, all_rows)
    print(json.dumps(dict(accepted_updates=len(all_rows), rejections=len(all_rejected),
                         hypotheses=[s["hypothesis"] for s in report["cases"]])))


if __name__ == "__main__":
    main()
