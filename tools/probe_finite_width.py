"""Bounded same-history finite-width compatibility probes, never loading runs."""
import argparse
import gc
from html import escape
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from mechanics.finite_strain_3d import response_3d
from prototype.finite_width_probe import deformation, embed, transverse_blocks, release_direction, integrated_bulk
from prototype.material_case import signature
from prototype.pardiso import PardisoDirection
from tools import run_common_material_study as host
from tools.render_solver_dashboard import atomic_write, shell, stamp
from run_support import write_json

ROOT = Path(__file__).resolve().parents[1]


def render(out, result):
    write_json(out/'result.json', result)
    body = '<h1>有限宽相容约束：条件方向诊断</h1><p>'+stamp()+' · '+escape(result['status'])+'</p>'
    body += '<p>不是新Fz曲线，也没有新增加载态。在既有平面应变受载场和准确上一接受历史上，释放维持面外零位移的虚拟约束反力。采用20 mm有限宽、跨宽连续面外位移；面内位移仍跨宽一致。原面内IPC切线保留，未建模面外摩擦及跨宽面内翘曲，所以不能把方向外推为真实三维切割反力。</p>'
    body += '<p>列出的dFz/dε是ε=0处条件导数；ε表示约束反力释放幅度，ε=1不是已求得的平衡。每场1/4/8个半宽分层，4层方向经两尺度新鲜三维残差检验；底面面外固定，另在.5 mm作底面面外自由控制。不得按角度选边界。</p>'
    body += '<table><tr><th>角度/深度</th><th>半宽层数</th><th>底面面外</th><th>dFz/dε N</th><th>线性残差</th><th>方向检验</th></tr>'
    for f in result['fields']:
        for v in f.get('directions', []):
            body += f'<tr><td>{f["angle"]}° / {f["depth_mm"]} mm</td><td>{v["layers"]}</td><td>{v["clamp_bottom"]}</td><td>{v["derivative_fz_n"]:.6g}</td><td>{v["linear_relative"]:.3g}</td><td>{v.get("qualified", "未逐场非线性检验")}</td></tr>'
        if f.get('error'): body += '<tr><td colspan="6">'+escape(f['error'])+'</td></tr>'
    body += '</table><p>每场900 s检查预算；没有新切割加载、参数拟合、旧失败增量重试或物理验收放宽。三维核已检验二维一致性、凝聚切线、旋转、增量势导数与Maxwell极限；这些不是实验材料验证。</p>'
    body += '<p><a href="result.json">全部数值/资格</a> · <a href="plan.md">冻结方案</a> · <a href="run.log">文本运行日志</a></p>'
    log = (out/'run.log').read_text(encoding='utf-8', errors='replace') if (out/'run.log').exists() else ''
    body += '<details><summary>运行日志</summary><pre>'+escape(log[-20000:])+'</pre></details>'
    atomic_write(out/'index.html', shell('有限宽条件诊断', body, refresh=result['status']=='running'))


def probe(out):
    out.mkdir(parents=True, exist_ok=False)
    source = host.snapshot(out)
    extras = ('tools/probe_finite_width.py', 'tests/test_finite_strain_3d.py', 'tests/test_finite_width_probe.py')
    for name in extras:
        dest = out/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest); source[name] = host.sha(dest)
    shutil.copy2(ROOT/'docs/finite_width_diagnostic.md', out/'plan.md')
    result = dict(created=stamp(), status='running', new_loading_states=0, fields=[], source_sha256=source,
                  plan_sha256=host.sha(out/'plan.md'), inputs={}, common_material=host.BASE.tolist())
    render(out, result)
    log = (out/'run.log').open('w', encoding='utf-8')
    def note(message):
        text = stamp()+' '+message
        print(text, flush=True); print(text, file=log, flush=True); render(out, result)
    try:
        with threadpool_limits(1):
            for angle in (0, 33, 57):
                folder = ROOT/host.REFERENCES[angle]
                result['inputs'].update(host.verify_case(folder))
                report = host.read(folder/'result.json')
                p = host.make(angle, host.BASE)
                if json.loads(json.dumps(signature(p))) != report['signature']:
                    raise ValueError('Reference signature differs')
                for depth in (.16, .5):
                    f = dict(angle=angle, depth_mm=depth, status='running', directions=[])
                    result['fields'].append(f); tick = time.monotonic()
                    note(f'FIELD angle={angle} depth={depth}')
                    try:
                        idx = next(i for i, s in enumerate(report['steps']) if abs(s['depth_mm']-depth)<1e-10)
                        with np.load(folder/'states'/f'{idx:04d}.npz', allow_pickle=False) as z:
                            state = dict(z)
                        f['fresh_audit'] = p.audit_state(state, report['steps'][idx])
                        x = state['positions']; p.previous = state['previous_positions']
                        p.previous_material_metric = state['previous_material_metric']; p.material_dt = float(state['material_dt_s'])
                        f['baseline_fz_n'] = report['steps'][idx]['reaction_fz_n']
                        F = deformation(p, x)
                        q = response_3d(embed(F), p.previous_material_metric, p.material_dt, p.material)
                        q2 = p.constitutive_response(F, tangent=True)
                        planeidx = [0, 1, 3, 4]
                        f['3d_2d_stress_max_error'] = float(np.max(np.abs(q['P'][:, :2, :2]-q2['P'])))
                        f['3d_2d_tangent_relative'] = float(np.linalg.norm(q['tangent'][:, planeidx][:, :, planeidx]-q2['tangent'])/np.linalg.norm(q2['tangent']))
                        if f['3d_2d_stress_max_error'] > 1e-7 or f['3d_2d_tangent_relative'] > 1e-8:
                            raise RuntimeError('Real-field 3-D/2-D equivalence failed')
                        lag = p.lag(x); _, g, H, _ = p.evaluate(x, lag, projected=False)
                        A = p.coupled_friction_tangent(x, lag, H)
                        choices = [(1, True), (4, True), (8, True)] + ([(4, False)] if depth==.5 else [])
                        for layers, clamp in choices:
                            if time.monotonic()-tick > 900: raise TimeoutError('Field diagnostic budget 900 s')
                            note(f'DIRECTION angle={angle} depth={depth} layers={layers} clamp={clamp}')
                            b = transverse_blocks(p, q, layers, clamp_bottom=clamp)
                            solver = PardisoDirection(8)
                            try:
                                du, dw, v = release_direction(p, A, b, solver)
                                v['linear_calls'] = solver.rows
                            finally: solver.close()
                            v.update(layers=layers, clamp_bottom=clamp, z_mm=b['z'].tolist())
                            f['directions'].append(v)
                            np.savez_compressed(out/f'direction_{angle}_{depth:g}_{layers}_{int(clamp)}.npz',
                                                du=du, dw=dw, z=b['z'])
                            note(f'LINEAR dFz={v["derivative_fz_n"]:.8g} relative={v["linear_relative"]:.3g}')
                            if layers==4 and clamp:
                                v['checks'] = []
                                predicted_u = A@du.ravel()+b['B']@dw.ravel()
                                predicted_w = b['B'].T@du.ravel()+b['K']@dw.ravel()
                                for eps in (1e-4, 3e-5):
                                    if time.monotonic()-tick > 900: raise TimeoutError('Field diagnostic budget 900 s')
                                    safe = float(p.safe_step(x, eps*du))
                                    if safe < 1-1e-10:
                                        v['checks'].append(dict(eps=eps, safe=safe, passed=False)); continue
                                    y = x+eps*du
                                    # Reassemble original, fully refreshed in-plane contact, then
                                    # substitute only the integrated 3-D bulk term.
                                    gy = p.evaluate(y, p.lag(y), tangent=False)[1]
                                    local, _, dofs, _, _ = p.material_bulk((y[:p.n]-p.mesh.reference_xy).ravel(), with_tangent=False)
                                    original_bulk = np.zeros(p.total); np.add.at(original_bulk, dofs.ravel(), local.ravel())
                                    bulk, rw, diagnostics = integrated_bulk(p, y, eps*dw, b['z'])
                                    gy += bulk-original_bulk
                                    uerr = gy-g-eps*predicted_u
                                    werr = rw-b['r']-eps*predicted_w
                                    err = float(np.linalg.norm(np.r_[uerr[p.free], werr[b['free']]])/
                                                max(eps*np.linalg.norm(b['r'][b['free']]), 1e-12))
                                    actual = float(-(gy-g).reshape(-1, 2)[p.n:, 1].sum()/eps)
                                    ferr = abs(actual-v['derivative_fz_n'])/max(abs(v['derivative_fz_n']), 1e-5)
                                    check = dict(eps=eps, safe=safe, residual_relative=err, force_relative=ferr,
                                        actual_fz_derivative_n=actual, **diagnostics,
                                        passed=bool(err<.02 and ferr<.02 and diagnostics['min_j']>0 and diagnostics['max_history_det_error']<1e-8))
                                    v['checks'].append(check)
                                    note(f'CHECK eps={eps} residual_error={err:.5g} force_error={ferr:.5g} passed={check["passed"]}')
                                v['qualified'] = bool(len(v['checks'])==2 and all(c['passed'] for c in v['checks']))
                            del b, du, dw; gc.collect()
                        v4, v8 = f['directions'][1:3]
                        f['layer_4_8_force_relative_change'] = abs(v8['derivative_fz_n']-v4['derivative_fz_n'])/max(abs(v8['derivative_fz_n']), 1e-5)
                        f['status'] = 'completed'
                    except Exception as exc:
                        f.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
                        note('FIELD_FAILED '+repr(exc))
                    finally:
                        f['elapsed_s'] = time.monotonic()-tick; render(out, result); gc.collect()
            result['status'] = 'ended'
    except Exception as exc:
        result.update(status='failed', error=repr(exc)); raise
    finally:
        result['ended'] = stamp(); log.close(); render(out, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True, type=Path)
    probe(parser.parse_args().output.resolve())
