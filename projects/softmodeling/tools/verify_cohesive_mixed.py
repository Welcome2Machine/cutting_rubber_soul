"""Independent, bounded local verification of the mixed-mode cohesive model.

The example strengths and toughness are numerical test inputs. No experimental
data, cutting model, fitted onset threshold or physical onset depth is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import brentq

from prototype.cohesive_mixed import MixedModeLaw, interface_response


EXAMPLE_PARAMETERS = dict(normal_strength_mpa=.2, shear_strength_mpa=.3,
                          normal_stiffness_mpa_per_mm=50., toughness_n_per_mm=.006)


def _tick(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Mixed-mode verification execution budget exhausted')


def _gate(gates, name, passed, **metrics):
    gates.append(dict(name=name, passed=bool(passed), **metrics))


def _polyline(vertices, subdivisions):
    points = [np.asarray(vertices[0], float)[None, :]]
    for start, end in zip(vertices[:-1], vertices[1:]):
        points.append(np.linspace(start, end, subdivisions+1)[1:])
    return np.concatenate(points)


def integrate_path(law, jumps, deadline=float('inf')):
    """Commit local history sequentially, independently integrate traction work."""
    jumps = np.asarray(jumps, float)
    if jumps.ndim != 2 or jumps.shape[1] != 2 or len(jumps) < 2:
        raise ValueError('At least two (normal opening, tangential slip) samples required')
    old = 0.
    values = {key: [] for key in ('traction', 'stored', 'dissipated', 'history', 'damage')}
    for index, (normal, shear) in enumerate(jumps):
        if index % 64 == 0:
            _tick(deadline)
        response = law.response(normal, shear, old)
        old = float(response['history'])
        for key in values:
            values[key].append(np.asarray(response[key]).copy())
    values = {key: np.asarray(value) for key, value in values.items()}
    increments = np.sum(.5*(values['traction'][1:]+values['traction'][:-1])
                        * np.diff(jumps, axis=0), axis=1)
    work = np.r_[0., np.cumsum(increments)]
    accounted = values['stored']-values['stored'][0] + values['dissipated']-values['dissipated'][0]
    residual = work-accounted
    return dict(**values, jumps=jumps, work=work, energy_residual=residual,
                max_absolute_energy_residual=float(np.max(np.abs(residual))),
                final_energy_residual=float(residual[-1]))


def _path_summary(path, subdivisions):
    return dict(subdivisions_per_leg=int(subdivisions), samples=len(path['jumps']),
                external_work_n_per_mm=float(path['work'][-1]),
                final_stored_n_per_mm=float(path['stored'][-1]),
                final_dissipation_n_per_mm=float(path['dissipated'][-1]),
                final_damage=float(path['damage'][-1]),
                minimum_dissipation_increment_n_per_mm=float(np.min(np.diff(path['dissipated']))),
                minimum_history_increment_mm=float(np.min(np.diff(path['history']))),
                max_absolute_energy_residual_n_per_mm=path['max_absolute_energy_residual'],
                final_energy_residual_n_per_mm=path['final_energy_residual'])


def _finite_difference_checks(law, gates, deadline):
    d0, df = law.peak_opening_mm, law.final_opening_mm
    beta = np.sqrt(law.shear_stiffness_mpa_per_mm/law.normal_stiffness_mpa_per_mm)
    trials = [('elastic', [.4*d0, .3*d0/beta], 0.),
              ('growing_damage', [.35*df, .25*df/beta], 0.),
              ('unloading', [.1*df, .2*df/beta], .65*df),
              ('closed_shear', [-d0, .5*d0/beta], 0.),
              ('failed', [.1*df, .1*df/beta], 1.1*df)]
    rows = []
    h = 1e-7
    for name, jump, old in trials:
        _tick(deadline)
        jump = np.asarray(jump)
        response = law.response(*jump, old)
        gradient = np.zeros(2)
        tangent = np.zeros((2, 2))
        for column in range(2):
            step = np.eye(2)[column]*h
            plus, minus = law.response(*(jump+step), old), law.response(*(jump-step), old)
            gradient[column] = (plus['incremental_potential']-minus['incremental_potential'])/(2*h)
            tangent[:, column] = (plus['traction']-minus['traction'])/(2*h)
        gradient_error = float(np.max(np.abs(gradient-response['traction'])))
        tangent_error = float(np.max(np.abs(tangent-response['tangent'])))
        _gate(gates, 'constitutive_derivatives_'+name,
              gradient_error < 1e-8 and tangent_error < 1e-5,
              gradient_absolute_error_mpa=gradient_error,
              tangent_absolute_error_mpa_per_mm=tangent_error)
        rows.append(dict(branch=name, gradient_absolute_error_mpa=gradient_error,
                         tangent_absolute_error_mpa_per_mm=tangent_error))

    # This distorted four-node element exercises the moving director derivatives.
    x = np.array([[-.003, 0.], [.026, .007], [-.008, .8], [.031, .807]])
    segments, gaps, areas, old = np.array([[0, 1, 2, 3]]), np.array([.006]), np.array([1.6]), np.zeros((1, 2))

    def assembled(positions):
        return interface_response(positions, segments, gaps, areas, old, law)

    response = assembled(x)
    gradient, tangent = np.zeros(8), np.zeros((8, 8))
    for column in range(8):
        _tick(deadline)
        step = np.eye(8)[column].reshape(4, 2)*h
        plus, minus = assembled(x+step), assembled(x-step)
        gradient[column] = (plus['incremental_potential_n_mm']-minus['incremental_potential_n_mm'])/(2*h)
        tangent[:, column] = (plus['gradient']-minus['gradient'])/(2*h)
    gradient_error = float(np.max(np.abs(gradient-response['gradient'])))
    tangent_error = float(np.max(np.abs(tangent-response['hessian'].toarray())))
    _gate(gates, 'moving_director_gradient_and_tangent',
          gradient_error < 1e-8 and tangent_error < 1e-5,
          gradient_absolute_error_n=gradient_error, tangent_absolute_error_n_per_mm=tangent_error)

    angle = .73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = assembled(x@rotation.T+np.array([3.7, -2.1]))
    transform = np.kron(np.eye(4), rotation)
    force_error = float(np.max(np.abs(transformed['gradient']-transform@response['gradient'])))
    hessian_error = float(np.max(np.abs(transformed['hessian'].toarray()
                                       -transform@response['hessian'].toarray()@transform.T)))
    energy_error = float(abs(response['incremental_potential_n_mm']-transformed['incremental_potential_n_mm']))
    _gate(gates, 'rigid_rotation_objectivity', force_error < 1e-10 and hessian_error < 1e-8 and energy_error < 1e-12,
          force_error_n=force_error, tangent_error_n_per_mm=hessian_error, potential_error_n_mm=energy_error)
    _gate(gates, 'assembled_force_and_moment_balance',
          np.linalg.norm(response['gradient'].reshape(4, 2).sum(axis=0)) < 1e-12
          and abs(np.sum(x[:, 0]*response['gradient'].reshape(4, 2)[:, 1]
                         -x[:, 1]*response['gradient'].reshape(4, 2)[:, 0])) < 1e-12)
    return dict(constitutive=rows, assembled_gradient_error_n=gradient_error,
                assembled_tangent_error_n_per_mm=tangent_error,
                finite_difference_step_mm=h, nonsmooth_switches_excluded=True)


def _series_path(law, subdivisions, deadline):
    """One stable spring-interface series system; area=1 mm2, pure opening."""
    kn, strength = law.normal_stiffness_mpa_per_mm, law.normal_strength_mpa
    d0, df = law.peak_opening_mm, law.final_opening_mm
    softening_magnitude = strength/(df-d0)
    external_stiffness = 3.*softening_magnitude
    displacement = np.linspace(0., 1.07*df, subdivisions+1)
    old = 0.
    rows = []
    peak_total = d0+strength/external_stiffness
    for index, prescribed in enumerate(displacement):
        if index % 32 == 0:
            _tick(deadline)
        if prescribed == 0.:
            opening = 0.
        else:
            opening = brentq(lambda candidate: external_stiffness*(prescribed-candidate)
                             -float(law.response(candidate, 0., old)['traction'][0]),
                             0., prescribed, xtol=1e-14, rtol=1e-14)
        response = law.response(opening, 0., old)
        old = float(response['history'])
        force = external_stiffness*(prescribed-opening)
        if prescribed <= peak_total:
            analytic = external_stiffness*prescribed/(external_stiffness+kn)
        elif prescribed < df:
            analytic = (external_stiffness*prescribed-softening_magnitude*df)/(external_stiffness-softening_magnitude)
        else:
            analytic = prescribed
        rows.append((opening, force, .5*external_stiffness*(prescribed-opening)**2,
                     float(response['stored']), float(response['dissipated']),
                     force-float(response['traction'][0]), opening-analytic))
    values = np.asarray(rows)
    work = np.r_[0., np.cumsum(.5*(values[1:, 1]+values[:-1, 1])*np.diff(displacement))]
    residual = work-values[:, 2]-values[:, 3]-values[:, 4]
    return dict(subdivisions=subdivisions, external_stiffness_n_per_mm=external_stiffness,
                softening_tangent_magnitude_n_per_mm=softening_magnitude,
                minimum_total_energy_second_derivative_n_per_mm=external_stiffness-softening_magnitude,
                maximum_force_residual_n=float(np.max(np.abs(values[:, 5]))),
                maximum_analytic_opening_error_mm=float(np.max(np.abs(values[:, 6]))),
                max_absolute_energy_residual_n_mm=float(np.max(np.abs(residual))),
                final_work_n_mm=float(work[-1]), final_dissipation_n_mm=float(values[-1, 4]),
                final_spring_stored_n_mm=float(values[-1, 2]),
                final_cohesive_stored_n_mm=float(values[-1, 3]),
                displacement=displacement, opening=values[:, 0], force=values[:, 1], energy_residual=residual)


def verify(law=None, *, deadline=float('inf'), subdivisions=(128, 256, 512)):
    """Return evidence and explicit pass/fail gates; no source or input writes."""
    law = MixedModeLaw(**EXAMPLE_PARAMETERS) if law is None else law
    if len(subdivisions) < 3 or any(int(n) != n or n < 8 for n in subdivisions) or any(
            b <= a for a, b in zip(subdivisions[:-1], subdivisions[1:])):
        raise ValueError('At least three strictly increasing subdivision counts >=8 required')
    _tick(deadline)
    gates, cases, traces = [], {}, {}
    df, gc = law.final_opening_mm, law.toughness_n_per_mm
    beta = np.sqrt(law.shear_stiffness_mpa_per_mm/law.normal_stiffness_mpa_per_mm)
    for name, angle in [('mode_I', 0.), ('mode_II', np.pi/2), ('mixed_30', np.pi/6), ('mixed_65', 65*np.pi/180)]:
        vertices = [[0., 0.], [1.07*df*np.cos(angle), 1.07*df*np.sin(angle)/beta]]
        paths = [integrate_path(law, _polyline(vertices, n), deadline) for n in subdivisions]
        cases[name] = [_path_summary(path, n) for path, n in zip(paths, subdivisions)]
        fine = paths[-1]
        errors = [p['max_absolute_energy_residual'] for p in paths]
        _gate(gates, name+'_work_to_common_toughness',
              abs(fine['work'][-1]-gc)/gc < 5e-4 and abs(fine['dissipated'][-1]-gc)/gc < 1e-12
              and fine['damage'][-1] == 1. and abs(fine['stored'][-1]) < 1e-14,
              work_relative_error=float(abs(fine['work'][-1]-gc)/gc))
        _gate(gates, name+'_energy_refinement', errors[-1] < errors[0]*.4 and errors[-1]/gc < 5e-4,
              max_residuals_n_per_mm=errors, finest_over_coarsest=errors[-1]/max(errors[0], 1e-30))
        traces[name] = fine

    vertices = np.array([[0., 0.], [.5, 0.], [.2, 0.], [.2, .15], [.2, -.15],
                         [.5, .7], [0., 0.], [0., 1.07]])*np.array([df, df/beta])
    paths = [integrate_path(law, _polyline(vertices, n), deadline) for n in subdivisions]
    cases['nonproportional_unload_reverse_reload'] = [_path_summary(p, n) for p, n in zip(paths, subdivisions)]
    traces['nonproportional_unload_reverse_reload'] = paths[-1]
    fine = paths[-1]
    errors = [p['max_absolute_energy_residual'] for p in paths]
    _gate(gates, 'nonproportional_history_irreversibility',
          np.min(np.diff(fine['history'])) >= -1e-14 and np.min(np.diff(fine['damage'])) >= -1e-14
          and np.min(np.diff(fine['dissipated'])) >= -1e-14,
          minimum_dissipation_increment_n_per_mm=float(np.min(np.diff(fine['dissipated']))))
    _gate(gates, 'nonproportional_energy_refinement', errors[-1] < errors[0]*.4 and errors[-1]/gc < 5e-4,
          max_residuals_n_per_mm=errors, finest_over_coarsest=errors[-1]/max(errors[0], 1e-30))
    _gate(gates, 'nonproportional_final_fracture_work', abs(fine['work'][-1]-gc)/gc < 5e-4
          and abs(fine['dissipated'][-1]-gc) < 1e-12)
    # The first unload/reload loop remains below the previous maximum; D must stay fixed.
    n = subdivisions[-1]
    frozen = fine['dissipated'][n:4*n+1]
    _gate(gates, 'unload_and_direction_reversal_do_not_create_damage', np.ptp(frozen) < 1e-14,
          dissipation_range_n_per_mm=float(np.ptp(frozen)))

    closure = integrate_path(law, _polyline([[0., 0.], [-df, 0.], [0., 0.]], 64), deadline)
    _gate(gates, 'pure_compression_no_damage_or_cohesive_penalty',
          np.max(closure['damage']) == 0. and np.max(np.abs(closure['traction'])) == 0.
          and np.max(closure['dissipated']) == 0.,
          contact_scope='Compression must be carried by the host contact law; contact is not solved here')
    closed_shear = law.response(-.2*df, .6*df/beta, 0.)
    _gate(gates, 'closure_does_not_suppress_shear_failure',
          closed_shear['damage'] > 0. and closed_shear['traction'][0] == 0.
          and closed_shear['traction'][1] > 0.)
    peak_n = law.response(law.peak_opening_mm, 0., 0.)
    peak_s = law.response(0., law.peak_opening_mm/beta, 0.)
    _gate(gates, 'independent_normal_and_shear_strengths',
          abs(peak_n['traction'][0]-law.normal_strength_mpa) < 1e-12
          and abs(peak_s['traction'][1]-law.shear_strength_mpa) < 1e-12)
    derivatives = _finite_difference_checks(law, gates, deadline)
    series = [_series_path(law, n, deadline) for n in subdivisions]
    series_errors = [s['max_absolute_energy_residual_n_mm'] for s in series]
    _gate(gates, 'stable_series_equilibrium_and_analytic_solution',
          all(s['maximum_force_residual_n'] < 1e-10 and s['maximum_analytic_opening_error_mm'] < 1e-10
              and s['minimum_total_energy_second_derivative_n_per_mm'] > 0. for s in series),
          maximum_force_residual_n=max(s['maximum_force_residual_n'] for s in series),
          maximum_analytic_opening_error_mm=max(s['maximum_analytic_opening_error_mm'] for s in series))
    _gate(gates, 'stable_series_external_work_refinement',
          series_errors[-1] < series_errors[0]*.4 and series_errors[-1]/gc < 5e-4,
          max_residuals_n_mm=series_errors)
    series_summary = [{k: v for k, v in s.items() if not isinstance(v, np.ndarray)} for s in series]
    return dict(passed=all(g['passed'] for g in gates), gates=gates, paths=cases,
                finite_differences=derivatives, stable_series=series_summary), traces, series[-1]


def _render(output, report, traces=None, series=None):
    image = ''
    if traces:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        for name in ('mode_I', 'mode_II', 'mixed_30', 'mixed_65'):
            path = traces[name]
            axes[0, 0].plot(np.linalg.norm(path['jumps'], axis=1), np.linalg.norm(path['traction'], axis=1), label=name)
            rows = report['verification']['paths'][name]
            axes[0, 1].loglog([r['subdivisions_per_leg'] for r in rows],
                             [r['max_absolute_energy_residual_n_per_mm'] for r in rows], 'o-', label=name)
        axes[0, 0].set(xlabel='Local separation magnitude (mm)', ylabel='Traction magnitude (MPa)')
        axes[0, 0].legend()
        axes[0, 1].set(xlabel='Subdivisions per path leg', ylabel='Max work-balance residual (N/mm)')
        axes[0, 1].legend()
        path = traces['nonproportional_unload_reverse_reload']
        axes[1, 0].plot(path['work'], label='Independently integrated work')
        axes[1, 0].plot(path['stored']+path['dissipated'], '--', label='Stored + damage dissipation')
        axes[1, 0].set(xlabel='Accepted local sample', ylabel='Energy per reference area (N/mm)')
        axes[1, 0].legend()
        axes[1, 1].plot(series['displacement'], series['force'])
        axes[1, 1].set(xlabel='Prescribed series displacement (mm)', ylabel='Spring / cohesive force (N)',
                       title='Stable one-DOF verification, area = 1 mm$^2$')
        fig.suptitle('Theory verification examples; no cutting data or physical onset prediction')
        fig.tight_layout(); fig.savefig(output/'verification.svg'); plt.close(fig)
        image = '<img src="verification.svg" alt="Constitutive paths, energy convergence, and stable equilibrium">'
    rows = ''.join('<tr><td>'+html.escape(g['name'])+'</td><td>'+('PASS' if g['passed'] else 'FAIL')+'</td></tr>'
                   for g in report.get('verification', {}).get('gates', []))
    text = ('<!doctype html><meta charset="utf-8"><title>Mixed-mode cohesive verification</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px}'
            'img{max-width:100%}td,th{padding:5px 16px;border-bottom:1px solid #ddd}'
            'pre{white-space:pre-wrap;background:#f3f5f6;padding:16px}</style>'
            '<h1>混合模式内聚模型：独立数值验证</h1>'
            '<p>参数仅作理论验证，未使用实验力数据或反演。本报告验证局部本构、共转界面、'
            '能量积分与稳定的一自由度平衡；不构成切割起裂深度预测或完整切割模型的验证。</p>'
            '<p>本律采用共同模式断裂能 Gc；法向与剪切强度可不同，但 Ks = Kn (Ts/Tn)²。'
            '这是明示的材料限制。压缩接触需由耦合模型承担。</p>'
            '<p>卸载、方向反转、闭合与失效均按历史验证。三个离散尺度的外功由牵引独立积分，'
            '未把能量余项改称耗散。所有 gate 必须通过才报告 passed。</p>'
            '<pre>'+html.escape(json.dumps({k: report.get(k) for k in ('status', 'passed', 'error', 'parameters', 'elapsed_s')},
                                          ensure_ascii=False, indent=2))+'</pre>'+image+
            '<table><thead><tr><th>检验</th><th>结果</th></tr></thead><tbody>'+rows+'</tbody></table>'
            '<p><a href="verification.json">参数、阈值、逐尺度指标与源码哈希 JSON</a></p>')
    (output/'index.html').write_text(text, encoding='utf-8')


def run(output, seconds=120.):
    if not np.isfinite(seconds) or not 1 <= seconds <= 300:
        raise ValueError('seconds must be in [1,300]')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+seconds
    project = Path(__file__).resolve().parents[1]
    sources = ['prototype/cohesive_mixed.py', 'tools/verify_cohesive_mixed.py']
    report = dict(schema='softcut-mixed-cohesive-verification/1', status='running', passed=False,
                  parameter_status='theory_verification_only', no_inverse_fitting=True,
                  experimental_inputs_used=[], physical_onset_depth_prediction=None,
                  started_utc=datetime.now(timezone.utc).isoformat(), interpreter=sys.executable,
                  execution_budget_s=float(seconds), parameters=MixedModeLaw(**EXAMPLE_PARAMETERS).to_dict(),
                  source_sha256={path: hashlib.sha256((project/path).read_bytes()).hexdigest() for path in sources},
                  budget_scope='Local constitutive/element checks, stable series solve, rendering and serialization',
                  limitations=['Common mode-I/mode-II toughness; no independent GIIc.',
                               'Prescribed local interface directions; no crack path prediction.',
                               'Host contact, bulk viscoelasticity and dynamic fracture are not exercised.',
                               'Stable spring verification does not validate a cutting model or fix its branch jumps.'])
    traces, series = None, None
    try:
        verification, traces, series = verify(deadline=deadline)
        final_hashes = {path: hashlib.sha256((project/path).read_bytes()).hexdigest() for path in sources}
        _gate(verification['gates'], 'numerical_sources_unchanged_during_verification',
              report['source_sha256'] == final_hashes)
        verification['passed'] = all(gate['passed'] for gate in verification['gates'])
        report['source_sha256_at_verification_end'] = final_hashes
        report.update(verification=verification, passed=verification['passed'],
                      status='passed' if verification['passed'] else 'failed_gates')
        _tick(deadline)
        report['elapsed_s'] = time.monotonic()-started
        _render(output, report, traces, series)
        _tick(deadline)
    except Exception as error:
        report.update(status='budget_exceeded' if isinstance(error, TimeoutError) else 'failed',
                      passed=False, error=f'{type(error).__name__}: {error}')
        _render(output, report)
    report['elapsed_s'] = time.monotonic()-started
    (output/'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return 0 if report['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New directory; existing results are never overwritten')
    parser.add_argument('--seconds', type=float, default=120., help='Whole verification budget, 1 to 300 seconds')
    arguments = parser.parse_args(argv)
    return run(arguments.output, arguments.seconds)


if __name__ == '__main__':
    raise SystemExit(main())
