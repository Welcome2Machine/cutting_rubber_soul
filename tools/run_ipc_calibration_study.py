"""Bounded intact IPC qualification; optional screenshot errors, no automatic fit.

Five independent OFAT paths test local computability before inverse modelling.
The 3 mm scientific target is unchanged. A passed study still needs a mesh audit.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_support import source_hashes, write_json
from tools.run_ipc_queue import (run_jobs, hidden_options, process_identity,
                                 mechanical_completion, now)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def force_points(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = ('depth_mm', 'force_n', 'lower_n', 'upper_n')
        if reader.fieldnames != list(fields):
            raise ValueError('Force CSV columns must be depth_mm,force_n,lower_n,upper_n')
        rows = [{k: float(row[k]) for k in fields} for row in reader]
    if (not rows or any(not all(math.isfinite(v) for v in row.values()) or row['depth_mm'] < 0
                        or not row['lower_n'] <= row['force_n'] <= row['upper_n'] for row in rows)
            or len({r['depth_mm'] for r in rows}) != len(rows)):
        raise ValueError('Force readings must be finite, unique by depth, and within reading bounds')
    return rows


def force_loss(rows, observations):
    # Collapse only floating-point depth representation noise, never resample.
    by_depth = {round(row['depth_mm'], 12): row['fz_N'] for row in rows}
    errors = []
    for point in observations:
        prediction = by_depth.get(round(point['depth_mm'], 12))
        if prediction is None:
            return None  # An unobserved depth is never interpolated for this qualification.
        outside = max(point['lower_n']-prediction, 0., prediction-point['upper_n'])
        errors.append(dict(depth_mm=point['depth_mm'], predicted_force_N=prediction,
                           error_N=prediction-point['force_n'], outside_reading_range_N=outside))
    return dict(rmse_N=math.sqrt(sum(r['error_N']**2 for r in errors)/len(errors)),
                outside_range_count=sum(r['outside_reading_range_N'] > 0 for r in errors),
                outside_range_rmse_N=math.sqrt(sum(r['outside_reading_range_N']**2 for r in errors)/len(errors)),
                points=errors, scaling_applied=False)


def prepare(output, *, depth=1., seconds=900., threads=2, g0=.66,
            g_fraction=.1, mu0=.35, mu_step=.05, force_points_path=None,
            linear_solver='direct'):
    output = Path(output).resolve()
    if (not all(math.isfinite(v) for v in (depth, seconds, g0, g_fraction, mu0, mu_step))
            or not 0 < depth <= 1 or seconds <= 0 or g0 <= 0
            or not 0 < g_fraction < 1 or not 0 < mu_step <= mu0
            or type(threads) is not int or threads < 1):
        raise ValueError('Require 0 < depth <= 1, finite positive budget/G/threads, '
                         '0 < G fraction < 1, and 0 < mu step <= mu0')
    if linear_solver not in ('direct', 'reused-lu'):
        raise ValueError('Linear solver must be direct or reused-lu')
    if output.exists():
        raise ValueError('Output must not exist; qualification runs are never overwritten')
    if not (0 < g0*(1-g_fraction) < g0 < g0*(1+g_fraction) < math.inf
            and 0 <= mu0-mu_step < mu0 < mu0+mu_step < math.inf):
        raise ValueError('Perturbations must produce finite, distinct parameter candidates')
    baseline = read(ROOT/'prototype/ipc_baseline.json')
    if baseline['model']['poisson_ratio'] != .49:
        raise ValueError('Qualification requires unchanged nu=0.49')
    if force_points_path is not None:
        force_points(force_points_path)  # Validate before writing any output.
    output.mkdir(parents=True)
    frozen = output/'source'
    model_hashes = source_hashes(snapshot=frozen)
    hashes = dict(model_hashes)
    for name in ('prototype/ipc_baseline.json', 'tools/visualize_ipc.py', 'tools/run_ipc_queue.py',
                 'tools/run_ipc_calibration_study.py', 'requirements.txt'):
        target = frozen/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
        hashes[name] = digest(target)
    model_hashes['tools/visualize_ipc.py'] = hashes['tools/visualize_ipc.py']
    config = {k: v for k, v in baseline['model'].items() if k != 'mesh_size_mm'}
    config.update(inclination_deg=60., friction_coefficient=mu0, shear_modulus_n_mm2=g0,
                  ogden_alpha=2.)
    candidates = [('base', g0, mu0), ('g_minus', g0*(1-g_fraction), mu0),
                  ('g_plus', g0*(1+g_fraction), mu0), ('mu_minus', g0, mu0-mu_step),
                  ('mu_plus', g0, mu0+mu_step)]
    jobs = []
    common = ['--case', 'oblique', '--element', 'p1', '--inclination', '60',
              '--tip-radius', '.02', '--local-spacing', '.05', '--surface-spacing', '.005',
              '--fine-halfwidth', '3', '--nose-facets', '64', '--dhat', '.0025',
              '--increment', '.01', '--depth', str(depth), '--seconds', str(seconds),
              '--threads', str(threads), '--max-iterations', '500', '--hessian-mode', 'regularized',
              '--linear-solver', linear_solver, '--ogden-alpha', '2',
              '--contact-formulation', 'improved-max', '--track-tip-contact', '--no-visualize']
    for name, shear, friction in candidates:
        folder = output/name
        command = [sys.executable, '-B', '-u', '-m', 'prototype.ipc_reference',
                   '--output', str(folder), '--shear-modulus', str(shear),
                   '--friction', str(friction), *common]
        jobs.append(dict(name=name, angle=60, friction=friction, shear_modulus_mpa=shear,
                         output=str(folder), command=command, status='pending'))
    state = dict(status='pending', created=now(), source_root=str(frozen), source_sha256=hashes,
                 model_source_sha256=model_hashes, python=sys.executable, threads=threads,
                 target_mm=depth, main_target_mm=3., solve_budget_s=seconds,
                 linear_solver=linear_solver, contact_formulation='improved-max',
                 process_timeout_s=seconds+300, native_python_path=str(ROOT/'.runtime/ipc-python'),
                 jobs=jobs, base_config=config, baseline_config_sha256=hashes['prototype/ipc_baseline.json'],
                 parameters=dict(g0_mpa=g0, g_fraction=g_fraction, mu0=mu0, mu_step=mu_step),
                 meaning='bounded numerical qualification, no experiment fitted; mesh audit still required',
                 comparison_command=[sys.executable, '-B', str(frozen/'tools/run_ipc_calibration_study.py'),
                                     '--output', str(output), '--summarize'])
    if force_points_path is not None:
        local = output/'force_points.csv'
        shutil.copy2(force_points_path, local)
        state['force_points'] = dict(source_path=str(Path(force_points_path).resolve()), path=str(local),
                                     sha256=digest(local), provenance='manual screenshot readings; not raw data or confidence intervals')
    write_json(output/'queue.json', state)
    return state


def audit_settings(report, state, job):
    if report.get('status') not in ('running', 'completed', 'failed'):
        raise ValueError('Unknown solver report status')
    expected = dict(state['base_config'], shear_modulus_n_mm2=job['shear_modulus_mpa'],
                    friction_coefficient=job['friction'])
    expected.setdefault('ogden_alpha', 2.)
    actual = dict(report['config'])
    actual.setdefault('ogden_alpha', 2.)
    if actual != expected:
        raise ValueError('Changed physical config outside declared G/mu candidates')
    if report.get('contact_formulation', 'improved-max') != state.get('contact_formulation', 'improved-max'):
        raise ValueError('Changed setting: contact_formulation')
    expected_fields = dict(case='oblique', target_mm=state['target_mm'], loading_increment_mm=.01,
                           dhat_mm=.0025, barrier_stiffness_mpa=6.6, force_tolerance_n=1e-4,
                           max_inner_iterations=500, hessian_mode='regularized',
                           linear_solver=state.get('linear_solver', 'reused-lu'),
                           baseline_config_sha256=state['baseline_config_sha256'])
    for key, value in expected_fields.items():
        if report.get(key) != value:
            raise ValueError('Changed setting: '+key)
    friction_settings = (dict(initial_metric_shift=.1, model_error_fraction=.25,
                              pseudo_time_growth=1.2, max_tip_radius_fraction=.5,
                              max_model_retries=18, rejection_metric_shift_floor=.1)
                         if job['friction'] else None)
    if (report.get('friction_solver') != ('coupled_pseudo_transient' if job['friction'] else None)
            or report.get('friction_solver_settings') != friction_settings):
        raise ValueError('Changed friction solver settings')
    if report['mesh_settings'] != dict(bulk_spacing_mm=1., local_spacing_mm=.05,
                                       surface_spacing_mm=.005, fine_halfwidth_mm=3.):
        raise ValueError('Changed mesh settings')
    for key, value in dict(displacement_element='P1, fixed mesh', tip_radius_mm=.02,
                           nose_facets=64, fracture=False,
                           material='existing finite-strain plane-strain elastic law',
                           friction_smoothing_mm=1e-5).items():
        if report['simplifications'].get(key) != value:
            raise ValueError('Changed simplification: '+key)
    if report['source_sha256'] != state['model_source_sha256']:
        raise ValueError('Case source fingerprints differ from frozen study')


def accepted_rows(report, folder, target):
    """No interpolation, extrapolation, or replacement of absent contact by zero."""
    import numpy as np
    rows, geometry, prior = [], None, None
    for index, step in enumerate(report['steps']):
        values = [float(step[k]) for k in ('depth_mm', 'reaction_fz_n', 'residual_n',
                                          'min_j', 'minimum_distance_mm', 'friction_work_n_mm')]
        depth, force, residual, jacobian, gap, work = values
        if (not all(math.isfinite(v) for v in values) or depth > target+1e-9
                or abs(depth-min(index*.01, target)) > 1e-9
                or index > math.ceil(target/.01-1e-12)
                or not 0 <= residual <= 1e-4 or jacobian <= 0 or gap <= 0 or work < -1e-12):
            raise ValueError('Invalid or noncontiguous accepted state')
        with np.load(Path(folder)/'states'/f'{index:04d}.npz', allow_pickle=False) as saved:
            signature = hashlib.sha256()
            for key in ('reference_xy', 'triangles', 'body_nodes'):
                array = saved[key]
                if not np.isfinite(array).all():
                    raise ValueError('Nonfinite reference geometry')
                signature.update(str((array.dtype.str, array.shape)).encode())
                signature.update(array.tobytes())
            current = signature.hexdigest()
            if geometry is not None and current != geometry:
                raise ValueError('Reference geometry changed within accepted path')
            geometry = current
            reference = saved['reference_xy']
            positions, previous = saved['positions'], saved['previous_positions']
            body_nodes = int(saved['body_nodes'])
            triangles = saved['triangles']
            if (reference.ndim != 2 or reference.shape[1] != 2
                    or positions.shape != reference.shape or previous.shape != reference.shape
                    or not 3 <= body_nodes < len(reference)
                    or triangles.ndim != 2 or triangles.shape[1] != 3
                    or not np.issubdtype(triangles.dtype, np.integer)
                    or np.any(triangles < 0) or np.any(triangles >= body_nodes)):
                raise ValueError('Invalid checkpoint shapes/connectivity')
            if (not math.isfinite(float(saved['depth_mm']))
                    or abs(float(saved['depth_mm'])-depth) > 1e-9
                    or not np.isfinite(positions).all() or not np.isfinite(previous).all()):
                raise ValueError('Invalid accepted checkpoint')
            if not np.array_equal(previous, reference if prior is None else prior):
                raise ValueError('Broken accepted friction history chain')
            angle = math.radians(report['config']['inclination_deg'])
            bottom = np.abs(reference[:body_nodes] @ np.array([math.sin(angle), math.cos(angle)])) < 1e-10
            if (not np.any(bottom) or not np.allclose(positions[:body_nodes][bottom],
                                                    reference[:body_nodes][bottom], rtol=0, atol=1e-12)):
                raise ValueError('Fixed bottom moved in accepted checkpoint')
            blade = reference[body_nodes:].copy()
            start = -2*report['dhat_mm']/max(math.cos(angle), .1)
            blade[:, 1] -= depth-start
            if not np.allclose(positions[body_nodes:], blade, rtol=0, atol=1e-12):
                raise ValueError('Prescribed blade motion differs from accepted depth')
            prior = positions.copy()
        tip = step.get('tip_contact') or {}
        status = tip.get('status', 'missing')
        offset = None
        if status == 'unique':
            gap = tip.get('gap_mm')
            if gap is None or not math.isfinite(float(gap)) or not 0 <= float(gap) < report['dhat_mm']:
                status = 'invalid_observation'
            else:
                offset = float(tip['reference_offset_mm'])
                if not math.isfinite(offset):
                    raise ValueError('Nonfinite unique contact observation')
        if 'depth_mm' in tip and (not math.isfinite(float(tip['depth_mm']))
                                 or abs(float(tip['depth_mm'])-depth) > 1e-9):
            raise ValueError('Contact observation depth mismatch')
        rows.append(dict(depth_mm=depth, fz_N=force, tip_reference_offset_mm=offset,
                         tip_status=status, residual_N=residual, min_J=jacobian))
    return rows, geometry


def write_csv(path, fields, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def termination_category(job, report, qualification):
    """Separate computational budget stops from numerical or evidence failures.

    Called only after report/checkpoint validation. Invalid evidence is handled
    separately by summarize, even when the queue also records a process timeout.
    """
    if job.get('status') == 'timed_out':
        return 'process_timeout'
    if qualification == 'passed':
        return 'completed'
    if qualification == 'pending':
        return ('running' if job.get('status') == 'running'
                or report.get('status') == 'running' else 'pending')
    if report.get('status') == 'failed':
        error = str(report.get('error') or job.get('error') or '')
        if error.startswith('TimeoutError:') and any(
                phrase in error.lower() for phrase in ('wall budget', 'wall time', 'wall-time', 'deadline')):
            return 'solver_wall_budget'
        return 'numerical_failure'
    if report.get('status') == 'completed':
        # An incomplete claimed completion, or a process failure after a
        # completed report, must not be interpreted as a solver convergence result.
        return ('process_failure' if job.get('status') == 'failed'
                else 'invalid_evidence')
    return 'process_failure'


def summarize(output):
    output = Path(output).resolve()
    state = read(output/'queue.json')
    for name, value in state['source_sha256'].items():
        if digest(Path(state['source_root'])/name) != value:
            raise ValueError('Frozen source changed: '+name)
    observations = None
    if state.get('force_points'):
        data = state['force_points']
        if digest(data['path']) != data['sha256']:
            raise ValueError('Frozen force readings changed')
        observations = force_points(data['path'])
    cases, series, records = [], {}, []
    common_geometry, common_solver = None, None
    for job in state['jobs']:
        folder = Path(job['output'])
        item = dict(name=job['name'], qualification='pending', accepted_states=0,
                    last_depth_mm=None, numerical_complete=False,
                    termination_category='running' if job.get('status') == 'running' else 'pending',
                    job_status=job.get('status'), exit_code=job.get('exit_code'),
                    process_error=job.get('error'))
        if observations is not None:
            item['force_loss'] = None
        try:
            if not (folder/'result.json').is_file():
                if job['status'] not in ('pending', 'running'):
                    raise ValueError('Terminal job has no result report')
                cases.append(item)
                continue
            report = read(folder/'result.json')
            audit_settings(report, state, job)
            rows, geometry = accepted_rows(report, folder, state['target_mm'])
            if geometry is not None:
                if common_geometry is not None and geometry != common_geometry:
                    raise ValueError('Reference geometry differs across candidates')
                common_geometry = geometry
            # A zero-mu candidate legitimately has no friction equation; the
            # remaining solver/binary provenance must agree between all cases.
            solver = {k: report[k] for k in ('ipc_binary_sha256', 'ipc_version')}
            if common_solver is not None and solver != common_solver:
                raise ValueError('IPC runtime differs across candidates')
            common_solver = solver
            complete = mechanical_completion(report, folder, state['target_mm'])
            terminal_failure = job['status'] not in ('pending', 'running', 'completed')
            if complete and not terminal_failure:
                status = 'passed'
            elif report.get('status') in ('failed', 'completed') or terminal_failure:
                status = 'failed'
            else:
                status = 'pending'
            item.update(qualification=status, numerical_complete=complete and not terminal_failure,
                        accepted_states=len(rows), last_depth_mm=rows[-1]['depth_mm'] if rows else None,
                        report_status=report.get('status'), error=report.get('error'),
                        termination_category=termination_category(job, report, status))
            if observations is not None and status == 'passed':
                item['force_loss'] = force_loss(rows, observations)
            series[job['name']] = rows
            records.extend(dict(case=job['name'], shear_modulus_mpa=job['shear_modulus_mpa'],
                                friction_coefficient=job['friction'], **row) for row in rows)
        except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError, zipfile.BadZipFile) as exc:
            item.update(qualification='failed', error=f'{type(exc).__name__}: {exc}',
                        termination_category='invalid_evidence')
        cases.append(item)
    passed = len(cases) == 5 and all(c['qualification'] == 'passed' for c in cases)
    status = 'passed' if passed else ('failed' if any(c['qualification'] == 'failed' for c in cases) else 'pending')
    failure_counts = {}
    for case in cases:
        if case['qualification'] == 'failed':
            category = case['termination_category']
            failure_counts[category] = failure_counts.get(category, 0)+1
    sensitivity = []
    if passed:
        p = state['parameters']
        for index, base in enumerate(series['base']):
            entry = dict(depth_mm=base['depth_mm'])
            for parameter, minus, plus, denominator in (
                    ('G', 'g_minus', 'g_plus', 2*p['g0_mpa']*p['g_fraction']),
                    ('mu', 'mu_minus', 'mu_plus', 2*p['mu_step'])):
                low, high = series[minus][index], series[plus][index]
                if low['depth_mm'] != base['depth_mm'] or high['depth_mm'] != base['depth_mm']:
                    raise ValueError('Central differences require identical accepted depths')
                entry['dFz_d'+parameter] = (high['fz_N']-low['fz_N'])/denominator
                a, b = low['tip_reference_offset_mm'], high['tip_reference_offset_mm']
                entry['dTipOffset_d'+parameter] = (None if a is None or b is None
                    or base['tip_reference_offset_mm'] is None else (b-a)/denominator)
            sensitivity.append(entry)
    write_csv(output/'accepted_prefixes.csv',
              ['case', 'shear_modulus_mpa', 'friction_coefficient', 'depth_mm', 'fz_N',
               'tip_reference_offset_mm', 'tip_status', 'residual_N', 'min_J'], records)
    write_csv(output/'sensitivities.csv', ['depth_mm', 'dFz_dG', 'dTipOffset_dG',
                                         'dFz_dmu', 'dTipOffset_dmu'], sensitivity)
    result = dict(qualification=status, target_mm=state['target_mm'], main_target_mm=3., cases=cases,
                  failure_counts=failure_counts,
                  termination_interpretation=('Budget limits mark incomplete computational attempts, '
                                              'not physical absence of equilibrium.'),
                  sensitivities=sensitivity, experiment_fitted=False, calibration_ready=False,
                  mesh_check='not established; required before inference',
                  meaning='accepted-prefix diagnostics; failed paths are not completed inverse samples',
                  units=dict(dFz_dG='N/MPa', dTipOffset_dG='mm/MPa', dFz_dmu='N', dTipOffset_dmu='mm'),
                  tip_meaning='original-surface coordinate migration, not accumulated tangential slip')
    if observations is not None:
        result['force_points'] = state['force_points']
    write_json(output/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    for name in ('detach', 'worker', 'summarize'):
        mode.add_argument('--'+name, action='store_true')
    for name, default in (('depth', 1.), ('seconds', 900.), ('g0', .66),
                          ('g-fraction', .1), ('mu0', .35), ('mu-step', .05)):
        parser.add_argument('--'+name, type=float, default=default)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--linear-solver', choices=('direct', 'reused-lu'), default='direct',
                        help='P1 defaults to direct; LU reuse has no demonstrated P1 benefit')
    parser.add_argument('--force-points', type=Path, help='Optional local screenshot readings; no automatic fit')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize:
        print(json.dumps(dict(qualification=summarize(output)['qualification'])))
        return
    state = (read(output/'queue.json') if args.worker else prepare(
        output, depth=args.depth, seconds=args.seconds, threads=args.threads, g0=args.g0,
        g_fraction=args.g_fraction, mu0=args.mu0, mu_step=args.mu_step, force_points_path=args.force_points,
        linear_solver=args.linear_solver))
    if state['status'] != 'pending':
        parser.error('Queue already started; refusing duplicate execution')
    for name, value in state['source_sha256'].items():
        if digest(Path(state['source_root'])/name) != value:
            parser.error('Frozen source changed: '+name)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([state['source_root'], state['native_python_path']])
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = str(state['threads'])
    if args.detach:
        command = [sys.executable, '-B', '-u', str(output/'source/tools/run_ipc_calibration_study.py'),
                   '--output', str(output), '--worker']
        with (output/'queue.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=state['source_root'], env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log, **hidden_options())
        write_json(output/'launch.json', dict(pid=process.pid, process_identity=process_identity(process.pid),
                                              command=command, started=now()))
        print(json.dumps(dict(queue_pid=process.pid, output=str(output))))
    else:
        run_jobs(output, state, env)


if __name__ == '__main__':
    main()
