"""Explicit bounded real-blade zero-gap dynamic control, no experimental inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from numerical_runtime import runtime_entry, add_thread_arguments, thread_arguments
from run_support import source_hashes, write_json
from prototype.zero_gap_cut import ZeroGapCutProblem, EnergyResolutionRequired
from prototype.ipc_reference import LoadStepFailure
from step_control import execute_step, failure_kind, physical_cutback, reduced_increment


def problem(config):
    allowed = {'schema', 'target_mm', 'increment_mm', 'minimum_increment_mm',
        'density_kg_m3', 'strength_mpa', 'toughness_n_mm', 'seam_spacing_mm',
        'local_spacing_mm', 'surface_spacing_mm', 'coarse_spacing_mm',
        'nose_facets', 'damage_enabled', 'parameter_status', 'nonlinear_solver',
        'energy_tolerance', 'attempt_seconds', 'dynamic_cutback', 'friction_velocity_epsilon_mm_s',
        'energy_normalization', 'energy_absolute_tolerance_n_mm', 'iteration_target', 'cohesive_unloading',
        'time_integrator', 'barrier_quadrature', 'barrier_quadrature_rule', 'insertion_rule', 'nucleation_length_mm',
        'kinematic_theta', 'contact_activation_mm', 'blade_surface_spacing_mm',
        'fracture_criterion', 'ffm_virtual_seconds', 'ffm_friction_epsilon_mm', 'ffm_max_candidates',
        'ffm_virtual_solver', 'ffm_ccd_tolerance', 'ffm_chord_max_queries', 'ffm_chord_partition', 'ffm_recovery_attempts'}
    if set(config)-allowed or config['schema'] != 'softcut-zero-gap-dynamics/1':
        raise ValueError('Unknown zero-gap configuration')
    if not 0 < config['target_mm'] <= 2 or not 0 < config['minimum_increment_mm'] <= config['increment_mm']:
        raise ValueError('Invalid travel/increments')
    if config['parameter_status'] != 'independent_demonstration_not_measured':
        raise ValueError('This control has no independently measured parameter qualification')
    for key in ('attempt_seconds', 'iteration_target'):
        if key in config and (isinstance(config[key], bool) or not np.isfinite(config[key]) or config[key] <= 0):
            raise ValueError(f'{key} must be finite and positive')
    if 'dynamic_cutback' in config and type(config['dynamic_cutback']) is not bool:
        raise ValueError('dynamic_cutback must be boolean')
    integrator = config.get('time_integrator', 'backward_euler')
    extra = {}
    cls = ZeroGapCutProblem
    if integrator == 'discrete_gradient':
        from prototype.energy_cut import EnergyCutProblem
        cls = EnergyCutProblem
        extra = {key:config[key] for key in ('barrier_quadrature', 'barrier_quadrature_rule', 'insertion_rule', 'nucleation_length_mm', 'kinematic_theta') if key in config}
    elif integrator != 'backward_euler' or any(key in config for key in ('barrier_quadrature', 'barrier_quadrature_rule', 'insertion_rule', 'nucleation_length_mm', 'kinematic_theta')):
        raise ValueError('Unknown integrator or options not supported by backward Euler')
    criterion = config.get('fracture_criterion', 'local_or_neighbourhood')
    if criterion == 'finite_increment':
        if config.get('cohesive_unloading','secant') != 'secant':
            raise ValueError('Finite-increment frozen virtual potential requires secant unloading')
        if config.get('insertion_rule', 'local_strength') != 'local_strength':
            raise ValueError('Finite-increment criterion cannot combine a second insertion gate')
        from prototype.finite_increment_cut import FiniteIncrementEnergyCutProblem, FiniteIncrementBackwardCutProblem
        cls = FiniteIncrementEnergyCutProblem if integrator == 'discrete_gradient' else FiniteIncrementBackwardCutProblem
        extra.update({key:config[key] for key in ('ffm_virtual_seconds', 'ffm_friction_epsilon_mm',
            'ffm_max_candidates', 'ffm_virtual_solver', 'ffm_ccd_tolerance', 'ffm_chord_max_queries', 'ffm_chord_partition', 'ffm_recovery_attempts') if key in config})
    elif criterion != 'local_or_neighbourhood' or any(key.startswith('ffm_') for key in config):
        raise ValueError('Unknown fracture criterion or unbound virtual options')
    return cls(**extra, density_kg_m3=config['density_kg_m3'],
        strength_mpa=config['strength_mpa'], toughness_n_mm=config['toughness_n_mm'],
        seam_spacing_mm=config['seam_spacing_mm'], damage_enabled=config['damage_enabled'],
        nonlinear_solver=config.get('nonlinear_solver', 'ptc'),
        energy_tolerance=config.get('energy_tolerance'),
        energy_normalization=config.get('energy_normalization', 'increment_work'),
        energy_absolute_tolerance=config.get('energy_absolute_tolerance_n_mm', 0.),
        cohesive_unloading=config.get('cohesive_unloading', 'secant'),
        friction_velocity_epsilon=config.get('friction_velocity_epsilon_mm_s', .0002),
        maxwell_shear=1.24, relaxation_s=1.7, loading_rate=.2,
        bulk_modulus=23.34333333333331, shear_modulus=.14, mu=.35, dhat=config.get('contact_activation_mm', .0025),
        local_spacing=config['local_spacing_mm'], surface_spacing=config['surface_spacing_mm'],
        coarse_spacing=config['coarse_spacing_mm'], nose_facets=config['nose_facets'],
        blade_spacing=config.get('blade_surface_spacing_mm'),
        predictor='secant', max_iterations=500, hessian_mode='regularized')


def hashes():
    value = source_hashes()
    for name in ('numerical_runtime.py', 'virtual_path_geometry.py', 'virtual_path_probe.py',
                 'virtual_friction_work.py', 'virtual_constrained_root.py', 'step_control.py'):
        value[name] = hashlib.sha256((Path(__file__).parents[1]/name).read_bytes()).hexdigest()
    value['tools/run_zero_gap_cut.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    baseline = Path(__file__).parents[1]/'prototype/ipc_baseline.json'
    value['prototype/ipc_baseline.json'] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    return value


def execution_identity():
    """Commit is provenance; exact source hashes also cover uncommitted research."""
    identity = dict(host=socket.gethostname(), git_commit=None, git_dirty=None,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'))
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=5, check=True)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=5, check=True)
        identity.update(git_commit=head.stdout.strip(), git_dirty=bool(dirty.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


def accepted_step_growth(config, record):
    """Suggest the next step, never change the acceptance conditions.

    Iteration effort may cap growth, but cannot lock a resolved step forever
    at target iterations. Energy error can still reduce a successful step.
    Rejected steps use the separate force/geometry/energy cutback policy.
    """
    growth = 1.5
    if config.get('iteration_target'):
        growth = min(growth, max(1.05,
            np.sqrt(config['iteration_target']/max(1, record['iterations']))))
    if (config.get('energy_tolerance') is not None and
            config.get('energy_normalization') == 'mechanical_reservoir'):
        ratio = record['energy_reservoir_error_ratio']
        growth = min(growth, max(.5, .9/np.sqrt(max(ratio, 1e-12))))
    return float(growth)


def snapshot(p, x, old_released, depth):
    return dict(positions=x, increment=p.step_increment(x), previous_positions=p.previous, velocity=p.velocity,
        previous_velocity=p.previous_velocity, material_metric=p.material_metric,
        previous_material_metric=p.previous_material_metric, history=p.history,
        previous_history=p.previous_history, released=p.released,
        previous_released=old_released, active_contact=p.active_contact,
        material_dt_s=p.material_dt, depth_mm=depth, reference_xy=p.rest,
        triangles=p.mesh.triangles, mass=p.mass,
        previous_barrier_energy_n_mm=p.previous_barrier_energy)


def audit(p, state, rec):
    p.previous = state['previous_positions'].copy()
    p.previous_velocity = state['previous_velocity'].copy()
    p.previous_material_metric = state['previous_material_metric'].copy()
    p.previous_history = state['previous_history'].copy()
    p.material_dt = float(state['material_dt_s'])
    p.released = state['released'].copy()
    p.rebuild_contact()
    p.friction = p.ipc.FrictionPotential(p.friction_velocity_epsilon*p.material_dt)
    p.previous_barrier_energy = float(state['previous_barrier_energy_n_mm'])
    p.active_contact = state['active_contact'].copy()
    p.set_free()
    x = state['positions']
    p.set_primary_increment(x, state['increment'])
    if not np.array_equal(p.rest, state['reference_xy']) or not np.array_equal(p.mesh.triangles, state['triangles']):
        raise RuntimeError('Audit geometry mismatch')
    if (not np.array_equal(x[p.mesh.bottom_nodes], p.rest[p.mesh.bottom_nodes])
            or not np.allclose(x[p.n:], p.prescribed_blade_positions(float(state['depth_mm'])), rtol=0, atol=1e-11)):
        raise RuntimeError('Audit prescribed boundary mismatch')
    if np.any(state['previous_released'] & ~p.released):
        raise RuntimeError('Audit healing detected')
    fresh = p.acceptance(x, float(state['depth_mm']), 0, 0)
    np.testing.assert_allclose(p.material_response(x)['metric'], state['material_metric'], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(p.cohesive_response(x)['history'], state['history'], rtol=0, atol=1e-12)
    expected_v = p.endpoint_velocity(x)
    np.testing.assert_allclose(expected_v, state['velocity'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fresh['reaction_fz_n'], rec['reaction_fz_n'], rtol=1e-9, atol=1e-8)
    for key, val in rec['energy'].items():
        np.testing.assert_allclose(fresh['energy'][key], val, rtol=1e-9, atol=1e-9)
    # The prescribed-motion predictor and Newton CCD do not by themselves prove
    # that the straight path between accepted end states is collision free.
    linear_path_safe = p.accepted_chord_is_safe(x)
    return dict(depth_mm=fresh['depth_mm'], passed=True, residual_n=fresh['residual_n'],
                min_j=fresh['min_j'], accepted_chord_ccd_safe=bool(linear_path_safe))


def report(out, result):
    rows = result['steps']
    max_res = max((r['residual_n'] for r in rows), default=None)
    energy = [r['energy'] for r in rows]
    w = sum(abs(e['external_work_n_mm']) for e in energy)
    absolute_defect = sum(abs(e['physical_balance_defect_n_mm']) for e in energy)
    result['summary'] = dict(last_depth_mm=rows[-1]['depth_mm'] if rows else None,
        accepted_states=len(rows), max_residual_n=max_res,
        absolute_physical_defect_over_absolute_work=absolute_defect/w if w > 0 else None,
        max_algebraic_balance_residual_n_mm=max((abs(e['algebraic_balance_residual_n_mm']) for e in energy), default=None),
        accepted_chord_ccd_verified=bool(rows) and len(result['audits'])==len(rows)
            and all(a['accepted_chord_ccd_safe'] for a in result['audits']),
        energy_qualified=False, temporal_convergence_verified=False,
        offcentre_or_oblique_qualified=False, free_path_qualified=False)
    write_json(out/'result.json', result)
    table = ''.join('<tr>'+''.join(f'<td>{r[k]:.7g}</td>' for k in ('depth_mm','reaction_fz_n','residual_n','released_nodes','fully_failed_nodes'))+
        f"<td>{r['energy']['physical_balance_defect_n_mm']:.5g}</td></tr>" for r in rows)
    text = f'''<!doctype html><meta charset="utf-8"><title>零间隙二维切割对照</title>
    <style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:24px;color:#172b36}}td,th{{padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap}}.note{{background:#fff0d5;padding:18px}}</style>
    <h1>零间隙、裂面接触、二维惯性：实际刀具切割对照</h1>
    <p class="note">30×5×20 mm，刀具居中 s=15 mm；与旧基线 s=7 mm 不同。0°镜像对称、预设直裂路，不能预测斜切、非对称分叉或混合模态滑动。独立示例强度、断裂能和密度，未读取实验力。</p>
    <p>未断连接为精确约束；裂面为零间隙单边接触；体内两个方向具有真实单位的质量和惯性。时间算法：{html.escape(result['config'].get('time_integrator', 'backward_euler'))}。离散梯度分支采用定末变形Maxwell分裂更新与区间平均反力；时间、接触求积和分裂误差仍需验证。</p>
    <p>数值完成不等于失稳已分辨。能量代数恒等式闭合不等于物理收支收敛；质量未放大，数值耗散与材料耗散分别报告。2 mm 是刀具行程，并非切穿 5 mm 厚试样。</p>
    <pre>{html.escape(json.dumps(result['summary'], ensure_ascii=False, indent=2))}</pre>
    <p>状态：{html.escape(result['status'])}。<a href="result.json">完整结果、输入与源码哈希</a></p>
    <table><tr><th>深度 mm</th><th>Fz N</th><th>动量残差 N</th><th>释放节点</th><th>全失效节点</th><th>物理能量缺口 N mm</th></tr>{table}</table>'''
    (out/'index.html').write_text(text, encoding='utf-8')


@runtime_entry
def run(config, out, seconds, *, runtime=None):
    if not np.isfinite(seconds) or seconds < 30 or seconds > 3600:
        raise ValueError('Explicit budget must be 30..3600 s')
    out.mkdir(parents=True, exist_ok=False)
    from prototype.ipc_reference import toolkit
    toolkit().set_num_threads(runtime['requested']['ipc_threads'])
    started = time.monotonic()
    initial_hashes = hashes()
    result = dict(schema='softcut-zero-gap-result/1', config=config,
        config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        started_utc=datetime.now(timezone.utc).isoformat(), interpreter=sys.executable,
        budget_s=seconds, native_ipc_threads=runtime['ipc_threads_effective'],
        blas_threads=runtime['requested']['blas_threads'], thread_runtime=runtime,
        source_hashes=initial_hashes, steps=[], attempts=[], audits=[],
        no_experimental_force_inputs=True, geometry='centred symmetric 30x5x20 mm; NOT s=7 baseline',
        status='running')
    result['execution_identity'] = execution_identity()
    result['step_control'] = 'typed_failure_same_step_virtual_recovery_v2'
    write_json(out/'manifest.json', result)
    deadline = started+seconds
    solve_deadline = deadline-max(15., min(180., .2*seconds))
    p = problem(config)
    import scipy
    native = Path(p.ipc.__file__)
    result['runtime'] = dict(numpy=np.__version__, scipy=scipy.__version__, ipctk=p.ipc.__version__,
        ipctk_binary=str(native), ipctk_sha256=hashlib.sha256(native.read_bytes()).hexdigest())
    result['mesh'] = dict(half_nodes=p.n, half_triangles=len(p.mesh.triangles),
        full_reference_volume_mm3=2*float(p.material_integration_weights().sum()),
        full_mass_n_s2_mm=2*float(p.mass.sum()), gap_mm=0., blade_s_mm=p.config.drop_s_mm)
    x, depth = p.rest.copy(), p.start
    increment = config['increment_mm']
    result['next_increment_mm'] = increment
    try:
        while depth < config['target_mm']-1e-12:
            if time.monotonic() >= solve_deadline:
                raise TimeoutError('Reserved audit budget reached')
            result['next_increment_mm'] = increment
            target = min(config['target_mm'], depth+increment)
            old_released = p.released.copy()
            attempt_start = time.monotonic()
            try:
                value, rec = execute_step(p, x, depth, target, min(solve_deadline,
                    attempt_start+config.get('attempt_seconds', seconds)), solve_deadline)
            except (LoadStepFailure, TimeoutError) as exc:
                failure = (p.write_failure_evidence(out, len(result['attempts']))
                    if hasattr(p, 'write_failure_evidence') else None)
                kind = failure_kind(exc, solve_deadline)
                result['attempts'].append(dict(from_mm=depth, target_mm=target,
                    failure_kind=kind, physical_dt_s=(target-depth)/p.loading_rate,
                    physical_cutback_allowed=physical_cutback(kind),
                    status='rejected', error=str(exc), elapsed_s=time.monotonic()-attempt_start,
                    virtual_failure=failure))
                if (not physical_cutback(kind) or target-depth <= config['minimum_increment_mm']+4*np.spacing(max(1., abs(depth)))):
                    raise
                increment = reduced_increment(p, config, target-depth, kind)
                continue
            x, depth = value, target
            if hasattr(p, 'write_step_evidence'):
                p.write_step_evidence(out, len(result['steps']), rec)
            state = snapshot(p, x, old_released, depth)
            filename = f'state_{len(result["steps"]):05d}.npz'
            np.savez_compressed(out/filename, **state)
            rec.update(state_file=filename, state_sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest())
            result['steps'].append(rec)
            print('ACCEPTED '+json.dumps({k:rec[k] for k in ('depth_mm','reaction_fz_n','residual_n','released_nodes','fully_failed_nodes')}), flush=True)
            write_json(out/'progress.json', dict(depth_mm=depth, accepted=len(result['steps']), elapsed_s=time.monotonic()-started))
            growth = accepted_step_growth(config, rec)
            increment = max(config['minimum_increment_mm'], min(config['increment_mm'], increment*growth))
            result['next_increment_mm'] = increment
        result['status'] = 'completed_2mm_control' if config['target_mm']==2 else 'completed_short_control'
    except (TimeoutError, LoadStepFailure, RuntimeError, FloatingPointError) as exc:
        kind = failure_kind(exc, solve_deadline)
        result.update(status='budget_exhausted' if kind == 'global_compute_budget_exhausted' else 'failed',
            error=str(exc), failure_kind=kind, next_increment_mm=increment,
            timeout_scope='global_loading' if kind == 'global_compute_budget_exhausted' else kind)
        if hasattr(p, 'trial'):
            np.savez_compressed(out/'failed_trial.npz', positions=getattr(p, 'failed_positions', p.trial),
                increment=getattr(p, 'failed_increment', p.step_increment(p.trial)), previous_positions=p.previous,
                previous_material_metric=p.previous_material_metric, material_dt_s=p.material_dt,
                previous_velocity=p.previous_velocity, previous_history=p.previous_history,
                released=getattr(p, 'failed_released', p.released),
                active_contact=getattr(p, 'failed_active_contact', p.active_contact),
                previous_barrier_energy_n_mm=p.previous_barrier_energy,
                failed_stage=getattr(p, 'failed_stage', 'before_step'),
                target_depth_mm=target)
    try:
        fresh = problem(config)
        previous_state = None
        for rec in result['steps']:
            if time.monotonic() >= deadline:
                raise TimeoutError('Fresh audit budget')
            with np.load(out/rec['state_file']) as state:
                if previous_state is not None:
                    for a, b in [('previous_positions','positions'), ('previous_velocity','velocity'),
                                 ('previous_material_metric','material_metric'), ('previous_history','history'),
                                 ('previous_released','released')]:
                        np.testing.assert_array_equal(state[a], previous_state[b])
                    expected_dt = (float(state['depth_mm'])-float(previous_state['depth_mm']))/.2
                    np.testing.assert_allclose(float(state['material_dt_s']), expected_dt, rtol=0, atol=1e-12)
                result['audits'].append(audit(fresh, state, rec))
                previous_state = {key:state[key].copy() for key in state.files}
        result['accepted_prefix_audited'] = bool(result['steps'])
    except (RuntimeError, AssertionError, TimeoutError) as exc:
        result['accepted_prefix_audited'] = False
        result['audit_error'] = str(exc)
    result['source_unchanged'] = hashes()==initial_hashes
    result['elapsed_s'] = time.monotonic()-started
    report(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seconds', required=True, type=float)
    add_thread_arguments(parser)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    result = run(config, args.out, args.seconds, **thread_arguments(args))
    write_json(args.out/'result.json', result)
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
