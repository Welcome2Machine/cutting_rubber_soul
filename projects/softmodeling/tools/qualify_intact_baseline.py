"""Verify/freshly audit the completed common-material 0/33/57 two-mm baseline.

No FE loading is launched by this tool. Original results and source trees remain
read-only. Fresh audits use today's native equations on every saved exact history;
they are not a new from-rest loading run. Original source trees are not retained. Recorded fingerprints identify provenance;
they do not provide an archived executable or authorize cross-version resumption.
"""
import argparse
import contextlib
import csv
from html import escape
import json
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.material_case import audit, make_problem, signature
from run_support import source_hashes, write_json
from tools.render_solver_dashboard import atomic_write, shell, stamp

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'prototype/intact_2mm_baseline.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_case(folder):
    """Verify stored report/checkpoint fingerprints without requiring source copies."""
    folder = Path(folder)
    report = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    audit_data = json.loads((folder/'audit.json').read_text(encoding='utf-8'))
    steps = report['steps']
    if (not audit_data['passed'] or audit_data['states'] != len(steps)
            or len(audit_data['rows']) != len(steps)):
        raise ValueError('Incomplete audit')
    files = {}  # Source fingerprints remain metadata; source snapshots were retired.
    for i, (step, row) in enumerate(zip(steps, audit_data['rows'])):
        if not row['passed'] or abs(row['depth_mm']-step['depth_mm']) > 1e-10:
            raise ValueError('Rejected or mismatched audit state')
        files[str(folder/'states'/f'{i:04d}.npz')] = row['checkpoint_sha256']
    for name in ('result.json', 'audit.json'):
        files[str(folder/name)] = sha(folder/name)
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('Changed case input: '+path)
    return files


def validate_report(report, stored_audit, spec, angle):
    """Reject incomplete paths, parameter mixing and acceptance-policy drift."""
    steps = report['steps']
    depths = np.array([s['depth_mm'] for s in steps])
    wanted = np.arange(201) / 100
    if (not report['completed'] or report['status'] != 'completed'
            or report['target_mm'] != spec['depth_mm'] or len(steps) != len(wanted)
            or not np.allclose(depths, wanted, rtol=0, atol=1e-12)):
        raise ValueError('Baseline requires the complete accepted 0--2 mm path')
    if not stored_audit['passed'] or stored_audit['states'] != len(steps):
        raise ValueError('Incomplete stored audit')
    sig = report['signature']
    mat = sig['material']
    expected = dict(mu_eq=[spec['equilibrium_shear_mpa']], alpha_eq=[2.],
                    mu_ov=[spec['maxwell_shear_mpa']], alpha_ov=[2.],
                    relaxation_s=spec['relaxation_s'], volumetric='quadratic_j',
                    viscous_flow='isochoric_maxwell')
    if any(mat[k] != value for k, value in expected.items()):
        raise ValueError('Different material in baseline case')
    if (sig['config']['inclination_deg'] != angle
            or sig['config']['friction_coefficient'] != spec['mu']
            or sig['loading_rate'] != spec['loading_rate_mm_s']
            or sig['physical_force_tolerance_n'] != spec['solver']['force_tolerance_n']
            or sig['contact_formulation'] != 'improved-max' or sig['dhat'] != .0025):
        raise ValueError('Different contact/loading/acceptance signature')
    for step in steps:
        if (not np.isfinite([step['residual_n'], step['min_j'], step['reaction_fz_n']]).all()
                or step['residual_n'] > spec['solver']['force_tolerance_n']
                or step['min_j'] <= 0 or step['load_step']['subdivision_level'] != 0):
            raise ValueError('Baseline state violates original acceptance/schedule')
    return dict(angle_deg=angle, states=len(steps), endpoint_mm=float(depths[-1]),
        endpoint_fz_n=steps[-1]['reaction_fz_n'], minimum_j=min(s['min_j'] for s in steps),
        maximum_residual_n=max(s['residual_n'] for s in steps),
        affine_accepts=sum(s['load_step']['predictor'] == 'affine' for s in steps),
        rejected_attempts=len(report['rejected_load_attempts']), cutbacks=0,
        recorded_elapsed_s=report['elapsed_s'], prior_elapsed_s=report.get('prior_elapsed_s', 0.),
        stored_audit_passed=True, fresh_states=0, fresh_audit_passed=False)


def render(out, state):
    rows = ''.join('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in
        (c['angle_deg'], c['states'], f"{c['endpoint_fz_n']:.6f}",
         f"{c['minimum_j']:.6g}", f"{c['maximum_residual_n']:.6g}",
         c['affine_accepts'], c['fresh_states'], c['fresh_audit_passed']))+'</tr>'
        for c in state['cases'].values())
    content = f'''<h1>共同完整体 2 mm 计算基线</h1>
<p>状态：{escape(state['status'])}；更新：{escape(state['updated'])}。</p>
<p>G∞=.14、Gv=1.24、G0=1.38 MPa，q=.1014492754，τ=1.7 s，K=23.34333333 MPa，μ=.35。
0°/33°/57°均为独立材料及摩擦历史、0–2 mm、名义步长.01 mm、速度.2 mm/s。</p>
<table><thead><tr><th>角度</th><th>历史状态数</th><th>2 mm Fz/N</th><th>全程最小J</th>
<th>全程最大残差/N</th><th>仿射接受次数</th><th>本轮新鲜审查数</th><th>新鲜审查通过</th></tr></thead>
<tbody>{rows}</tbody></table>
<p>割线初猜失败后从同一已接受状态重试仿射初猜；前者停滞窗口40，后者无40步早停、仍受500次及墙钟预算约束。
每次尝试各有500次上限，并非合计500。三条保存路径实际未发生物理缩步；失败尝试不提交历史。</p>
<p>原冻结源码树已清理；仅保留来源指纹，不能再运行原冻结引擎。本轮只核查已有结果：报告与检查点哈希/原审查逐项验证；若启用 fresh，对每个保存状态用当前原生物理重组。
这不等于本轮重新从静止算到2 mm，也不证明未验证参数可解或当前源码逐步轨迹完全相同。</p>
<p>计算基线并非三角度实验拟合成功。旧0°/33°偏硬，材料参数未独立标定；完整体后段也不能描述实验断裂。
本轮不放宽1e-4 N残差、正J、接触不穿透、边界或历史验收，不启用断裂。</p>
<p>已校验文件：{state.get('verified_files', 0)}。
<a href="analysis.json">机器可读审查</a> · <a href="fz_depth.csv">三角度Fz CSV</a> ·
<a href="audit.log">实际审查日志</a></p>
<pre>{escape(state.get('error', ''))}</pre>'''
    atomic_write(out/'index.html', shell('完整体 2 mm 基线', content))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh', action='store_true', help='Reassemble all 603 exact-history saved states')
    parser.add_argument('--audit-seconds', type=float, default=3600., help='Fresh audit wall budget per angle')
    args = parser.parse_args()
    if not np.isfinite(args.audit_seconds) or args.audit_seconds <= 0:
        parser.error('Positive finite audit budget required')
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'outputs'):
        parser.error('Derived baseline evidence must be in repository outputs')
    out.mkdir(parents=True, exist_ok=False)
    spec = json.loads(SPEC.read_text(encoding='utf-8'))
    state = dict(baseline_id=spec['id'], status='verifying', updated=stamp(), cases={},
        scope='Existing complete paths, not a new loading run', fresh_requested=args.fresh,
        baseline_spec_sha256=sha(SPEC), inputs={}, source_sha256=source_hashes())
    for src in (SPEC, Path(__file__).resolve()):
        state['source_sha256'][src.relative_to(ROOT).as_posix()] = sha(src)
    state['original_source_trees_available'] = False

    def save():
        state['updated'] = stamp(); state['elapsed_s'] = time.monotonic()-started
        write_json(out/'analysis.json', state); render(out, state)

    started = time.monotonic(); save()
    reports = {}
    try:
        with (out/'audit.log').open('w', encoding='utf-8', buffering=1) as log, contextlib.redirect_stdout(log), threadpool_limits(2):
            common_config = None
            for angle in spec['angles_deg']:
                entry = spec['cases'][str(angle)]; folder = ROOT/entry['path']
                for name in ('result', 'audit'):
                    if sha(folder/f'{name}.json') != entry[f'{name}_sha256']:
                        raise ValueError('Pinned baseline report changed: '+str(folder/name))
                state['inputs'].update(verify_case(folder))
                r = json.loads((folder/'result.json').read_text(encoding='utf-8'))
                a = json.loads((folder/'audit.json').read_text(encoding='utf-8'))
                state['cases'][str(angle)] = validate_report(r, a, spec, angle)
                cfg = dict(r['signature']['config']); cfg.pop('inclination_deg')
                if common_config is not None and cfg != common_config:
                    raise ValueError('Cases differ in more than blade inclination')
                common_config = cfg; reports[angle] = r
            state['verified_files'] = len(state['inputs']); save()
            with (out/'fz_depth.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(['depth_mm', 'fz_0_n', 'fz_33_n', 'fz_57_n'])
                for i in range(201):
                    writer.writerow([reports[0]['steps'][i]['depth_mm']]+[
                        reports[a]['steps'][i]['reaction_fz_n'] for a in spec['angles_deg']])
            if args.fresh:
                for angle in spec['angles_deg']:
                    key = str(angle); state['status'] = f'fresh_audit_angle{angle}'; save()
                    r = reports[angle]; folder = ROOT/spec['cases'][key]['path']
                    p = make_problem(angle=angle, equilibrium_shear=spec['equilibrium_shear_mpa'],
                        maxwell_shear=spec['maxwell_shear_mpa'], relaxation_s=spec['relaxation_s'],
                        loading_rate=spec['loading_rate_mm_s'], bulk_modulus=spec['bulk_modulus_mpa'], mu=spec['mu'])
                    if json.loads(json.dumps(signature(p))) != r['signature']:
                        raise ValueError('Current problem differs from archived physical signature')
                    p.ipc.set_num_threads(2)
                    def progress(row, number, total):
                        state['cases'][key]['fresh_states'] = number
                        state['cases'][key]['fresh_last_depth_mm'] = row['depth_mm']
                        save()
                    t0 = time.monotonic()
                    result = audit(p, folder, r, destination=out/f'angle{angle}_fresh_audit.json',
                        progress=progress, deadline=t0+args.audit_seconds)
                    state['cases'][key].update(fresh_audit_passed=result['passed'], fresh_elapsed_s=time.monotonic()-t0)
                    save(); del p
            for path, digest in state['inputs'].items():
                if sha(path) != digest:
                    raise ValueError('Baseline evidence changed during qualification: '+path)
            for name, digest in state['source_sha256'].items():
                if sha(ROOT/name) != digest:
                    raise ValueError('Audit source changed during qualification: '+name)
            state['status'] = 'qualified_fresh' if args.fresh else 'qualified_stored_evidence'
            state['passed'] = True
    except Exception as exc:
        state.update(status='qualification_failed', passed=False, error=f'{type(exc).__name__}: {exc}')
        with (out/'audit.log').open('a', encoding='utf-8') as stream:traceback.print_exc(file=stream)
    finally:
        save()
    print(json.dumps({k:state.get(k) for k in ('status','passed','elapsed_s','error')}))
    return 0 if state.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
