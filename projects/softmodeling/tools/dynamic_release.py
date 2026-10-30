"""Bounded real-state conditional release and independent saved-path audit.

This does not resume a production path or decide initiation. Input is a checked
but fracture-undecided real trial. Output remains a separate conditional branch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from numerical_runtime import runtime_entry, add_thread_arguments, thread_arguments
from run_support import write_json
from . import run_zero_gap_cut as driver
from .frozen_scene import check_archived_equations, digest
from .diagnose_finite_increment import restore_failed_seed
from prototype.ipc_reference import LoadStepFailure
from step_control import execute_step, failure_kind, physical_cutback
from stiff_contact import install as install_integrator, momentum_ledger as impulse_ledger
from release_dynamics import (activate, check_energy_gate, instantaneous_friction,
    contact_velocity, independent_friction_work, DEFECT_KEYS)


def sources():
    root = Path(__file__).resolve().parents[1]
    result = driver.hashes()
    for name in ('release_dynamics.py', 'tools/dynamic_release.py', 'tools/frozen_scene.py',
                 'tools/diagnose_finite_increment.py', 'stiff_contact.py', 'endpoint_friction.py'):
        result[name] = digest(root/name)
    return result


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def conditional_seed(directory, raw):
    """An explicitly audited same-source conditional parent, never production."""
    if raw['source_hashes'] != sources() or not raw.get('source_unchanged'):
        raise ValueError('Conditional parent requires identical numerical source')
    if raw['status'] != 'completed_conditional_window' or not raw['steps']:
        raise ValueError('Conditional parent window is incomplete')
    inputs = {str(directory/'result.json'): digest(directory/'result.json')}
    for suffix in ('audit', 'contact_work'):
        path = directory.with_name(directory.name+'.'+suffix+'.json')
        report = json.loads(path.read_text())
        if not report['passed'] or report['result_sha256'] != inputs[str(directory/'result.json')]:
            raise ValueError('Conditional parent needs bound force/history and independent contact-work audits')
        inputs[str(path)] = digest(path)
    rec = raw['steps'][-1]; path = directory/rec['state_file']
    if digest(path) != rec['state_sha256']: raise ValueError('Conditional parent state changed')
    inputs[str(path)] = digest(path); state = arrays(path)
    method = raw['settings'].get('integrator', 'midpoint')
    p = install_integrator(driver.problem(raw['config']), method)
    checked = driver.audit(p, state, rec)
    if not checked['accepted_chord_ccd_safe']: raise ValueError('Conditional parent CCD failed')
    gate = check_energy_gate(p, rec['energy'])
    for attr in ('velocity', 'history', 'material_metric'):
        setattr(p, attr, state[attr].copy())
    info = dict(inputs_sha256=inputs, real_trial_record=rec, physical_energy_gate=gate,
        seed_kind='audited_conditional_parent', seed_integrator=method,
        real_trial_depth_mm=float(state['depth_mm']),
        qualification='same-source conditional branch only; not certified initiation or production continuation',
        production_history_committed=False)
    return p, state['positions'].copy(), float(state['depth_mm']), state, info, raw['config']


def real_seed(directory, seed_kind='trial', accepted_index=-1):
    if seed_kind not in ('trial', 'accepted'):
        raise ValueError('Choose the undecided real trial or the last accepted state')
    directory = Path(directory).resolve()
    raw = json.loads((directory/'result.json').read_text())
    if raw.get('schema') == 'softcut-conditional-dynamic-release/1':
        if seed_kind != 'accepted' or accepted_index != -1:
            raise ValueError('Conditional parent must explicitly use its last accepted state')
        return conditional_seed(directory, raw)
    provenance = check_archived_equations(raw, Path(__file__).resolve().parents[1])
    inputs = {str(directory/'result.json'): digest(directory/'result.json')}
    # The parent accepted prefix must already have both independent audits.
    for suffix in ('path_audit', 'finite_audit'):
        path = directory.with_name(directory.name+'.'+suffix+'.json')
        audit = json.loads(path.read_text())
        if not audit['passed'] or audit['result_sha256'] != digest(directory/'result.json'):
            raise ValueError('Parent audit is missing, failed or bound to another result')
        inputs[str(path)] = digest(path)
    if seed_kind == 'trial' and accepted_index != -1:
        raise ValueError('Trial seed must use the last accepted anchor')
    if not -len(raw['steps']) <= accepted_index < len(raw['steps']):
        raise ValueError('Accepted seed index outside parent path')
    seed_record = raw['steps'][accepted_index]
    previous_path = directory/seed_record['state_file']
    if digest(previous_path) != seed_record['state_sha256']:
        raise ValueError('Parent last accepted state hash differs')
    old = arrays(previous_path)
    inputs[str(previous_path)] = digest(previous_path)
    if accepted_index == -1:
        trial = arrays(directory/'failed_trial.npz')
        for a, b in (('previous_positions', 'positions'), ('previous_velocity', 'velocity'),
                     ('previous_material_metric', 'material_metric'), ('previous_history', 'history')):
            np.testing.assert_array_equal(trial[a], old[b])
        np.testing.assert_array_equal(trial['released'], old['released'])
        inputs[str(directory/'failed_trial.npz')] = digest(directory/'failed_trial.npz')
    p = driver.problem(raw['config'])
    if seed_kind == 'trial':
        x, depth = restore_failed_seed(p, trial)
    else:
        driver.audit(p, old, seed_record)
        x, depth = old['positions'].copy(), float(old['depth_mm'])
    _, g, _, _ = p.evaluate(x, p.lag(x), tangent=False)
    p.update_constraint_free(x, g)
    record = p.acceptance(x, depth, 0, 0)
    if not p.accepted_chord_is_safe(x):
        raise ValueError('Real trial temporal CCD failed')
    gate = check_energy_gate(p, record['energy'])
    # Derive a conditional endpoint with the original physical update, without
    # committing anything to the source path or certifying its FFM decision.
    incoming = np.zeros(p.total)
    incoming[p.surface_dofs] = p.friction.gradient(p.lag(x), p.cmesh, p.step_increment(x)[p.surface])
    p.velocity = p.endpoint_velocity(x)
    p.history = p.cohesive_response(x)['history'].copy()
    p.material_metric = p.material_response(x)['metric'].copy()
    if seed_kind == 'accepted':
        # Verify the update, then inherit the saved committed arrays bitwise.
        for attr, key in (('velocity', 'velocity'), ('history', 'history'),
                          ('material_metric', 'material_metric')):
            np.testing.assert_allclose(getattr(p, attr), old[key], rtol=1e-10, atol=1e-11)
            setattr(p, attr, old[key].copy())
    instant = instantaneous_friction(p, x, contact_velocity(p, p.velocity))
    state = driver.snapshot(p, x, p.released.copy(), depth)
    info = dict(inputs_sha256=inputs, provenance=provenance, real_trial_record=record,
        physical_energy_gate=gate, parent_last_accepted_depth_mm=raw['steps'][-1]['depth_mm'],
        real_trial_depth_mm=depth, inherited_physical_dt_s=p.material_dt,
        incoming_average_friction_norm_n=float(np.sqrt(2)*np.linalg.norm(incoming)),
        endpoint_instantaneous_friction_norm_n=float(np.sqrt(2)*np.linalg.norm(instant)),
        interval_average_to_instantaneous_difference_n=float(np.sqrt(2)*np.linalg.norm(incoming-instant)),
        seed_kind=seed_kind, accepted_index=accepted_index, seed_integrator='midpoint',
        qualification=('conditional real trial; finite-increment decision still unknown' if seed_kind == 'trial'
            else 'committed real state; subsequent candidate activation is conditional, not certified initiation'),
        production_history_committed=False)
    return p, x, depth, state, info, raw['config']


def summarize(result):
    rows = result['steps']
    energies = [r['energy'] for r in rows]
    keys = ('external_work_n_mm', 'bulk_stored_delta_n_mm', 'cohesive_stored_delta_n_mm',
        'barrier_delta_n_mm', 'kinetic_delta_n_mm', 'fracture_dissipation_n_mm',
        'friction_work_n_mm', 'viscous_endpoint_dissipation_n_mm', 'physical_balance_defect_n_mm')
    result['summary'] = dict(accepted_steps=len(rows), reached_time_s=rows[-1]['local_time_s'] if rows else 0.,
        ledger_n_mm={k: sum(e[k] for e in energies) for k in keys}, auxiliary_work_n_mm=0.,
        max_residual_n=max((r['residual_n'] for r in rows), default=None),
        peak_kinetic_n_mm=max((r['energy']['kinetic_energy_n_mm'] for r in rows), default=None),
        candidate_dissipation_n_mm=rows[-1]['candidate_dissipation_n_mm'] if rows else 0.,
        max_candidate_opening_mm=max((r['candidate_opening_mm'] for r in rows), default=0.),
        maximum_speed_mm_s=max((r['maximum_speed_mm_s'] for r in rows), default=0.),
        actual_dt_range_s=[min((r['dt_s'] for r in rows), default=0.), max((r['dt_s'] for r in rows), default=0.)],
        numerical_ledger_n_mm={k: sum(e.get(k, 0.) for e in energies) for k in (*DEFECT_KEYS, 'friction_time_defect_n_mm')},
        absolute_numerical_defect_n_mm=sum(sum(abs(e.get(k, 0.)) for k in (*DEFECT_KEYS, 'friction_time_defect_n_mm')) for e in energies),
        initiation_qualified=False, complete_cutting_qualified=False, event_energy_converged=False)


@runtime_entry
def run(directory, out, dt, duration, seconds, *, release=True, attempt_seconds=30., seed_kind='trial', accepted_index=-1, integrator='midpoint', minimum_dt=None, candidate_index=48, runtime=None):
    if not (np.isfinite([dt, duration, seconds, attempt_seconds]).all() and
            0 < dt <= duration <= .025 and 1 <= seconds <= 3600 and 1 <= attempt_seconds <= 90):
        raise ValueError('Require 0 < dt <= duration <= .025 s and bounded compute budgets')
    minimum_dt = dt/128 if minimum_dt is None else minimum_dt
    if not np.isfinite(minimum_dt) or not 0 < minimum_dt <= dt:
        raise ValueError('Require 0 < minimum_dt <= dt; it is a stopping bound, not an acceptance gate')
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started+seconds
    source = sources()
    p, x, depth, seed, info, config = real_seed(directory, seed_kind, accepted_index)
    seed_path = out/'seed.npz'; np.savez_compressed(seed_path, **seed)
    original_released = p.released.copy()
    if type(candidate_index) is not int or not 0 <= candidate_index < len(p.seam):
        raise ValueError('Explicit candidate seam index outside geometry')
    event = activate(p, x, [candidate_index] if release else [])
    install_integrator(p, integrator)
    result = dict(schema='softcut-conditional-dynamic-release/1', status='running',
        source_hashes=source, execution_identity=driver.execution_identity(), interpreter=sys.executable,
        thread_runtime=runtime, config=config, seed=info, seed_sha256=digest(seed_path),
        activation=event, settings=dict(dt_max_s=dt, duration_s=duration, attempt_seconds=attempt_seconds,
            seconds=seconds, release_candidate=release, candidate_seam_index=candidate_index, seed_kind=seed_kind, accepted_index=accepted_index, integrator=integrator,
            minimum_dt_s=minimum_dt,
            blade_policy='continue original speed', physical_mass_unchanged=True),
        steps=[], attempts=[])
    write_json(out/'manifest.json', result)
    start_depth = depth; local_time = 0.; next_dt = dt
    try:
        while local_time < duration-1e-14:
            if time.monotonic() >= deadline:
                raise TimeoutError('Conditional process global budget')
            target_time = min(duration, local_time+next_dt)
            target_depth = start_depth+p.loading_rate*target_time
            old_released = p.released.copy(); before = time.monotonic()
            unchanged = [p.velocity.copy(), p.history.copy(), p.material_metric.copy(), p.released.copy()]
            try:
                value, rec = execute_step(p, x, depth, target_depth,
                    min(deadline, before+attempt_seconds), deadline)
            except (LoadStepFailure, TimeoutError) as exc:
                kind = failure_kind(exc, deadline)
                for expected, actual in zip(unchanged, (p.velocity, p.history, p.material_metric, p.released)):
                    np.testing.assert_array_equal(expected, actual)
                result['attempts'].append(dict(local_time_s=local_time, target_time_s=target_time,
                    dt_s=(target_depth-depth)/p.loading_rate, failure_kind=kind, error=str(exc),
                    rollback_verified=True, elapsed_s=time.monotonic()-before))
                if physical_cutback(kind) and next_dt > minimum_dt:
                    next_dt = max(minimum_dt, next_dt/2); continue
                raise
            x, depth, local_time = value, target_depth, target_time
            q = p.cohesive_response(x, p.history)
            rec.update(local_time_s=local_time, dt_s=p.material_dt,
                candidate_opening_mm=float(2*(x[p.seam[candidate_index], 0]-p.config.drop_s_mm)),
                candidate_dissipation_n_mm=float(p.seam_area[candidate_index]*q['dissipated'][candidate_index]*p.released[candidate_index]),
                maximum_speed_mm_s=float(np.linalg.norm(p.velocity[:p.n], axis=1).max()))
            impulse_fields, rec['impulse'] = impulse_ledger(p, x)
            state = driver.snapshot(p, x, old_released, depth)
            state.update(impulse_fields)
            name = f'state_{len(result["steps"]):05d}.npz'
            np.savez_compressed(out/name, **state)
            rec.update(state_file=name, state_sha256=digest(out/name))
            result['steps'].append(rec)
            write_json(out/'progress.json', dict(local_time_s=local_time, steps=len(result['steps']),
                kinetic_n_mm=rec['energy']['kinetic_energy_n_mm'], opening_mm=rec['candidate_opening_mm']))
            if len(result['steps']) % 20 == 0:
                print(json.dumps(dict(time_s=local_time, steps=len(result['steps']), dt_s=p.material_dt)), flush=True)
            next_dt = min(dt, next_dt*1.25)
        result['status'] = 'completed_conditional_window'
    except (TimeoutError, LoadStepFailure, RuntimeError, ValueError, FloatingPointError) as exc:
        result.update(status='stopped', error=str(exc), failure_kind=failure_kind(exc, deadline))
        if hasattr(p, 'failed_positions'):
            np.savez_compressed(out/'failed_trial.npz', positions=p.failed_positions,
                increment=p.failed_increment, previous_positions=p.previous,
                previous_velocity=p.previous_velocity, previous_history=p.previous_history,
                previous_material_metric=p.previous_material_metric, material_dt_s=p.material_dt,
                released=p.failed_released, active_contact=p.failed_active_contact,
                previous_barrier_energy_n_mm=p.previous_barrier_energy, target_depth_mm=target_depth)
    result.update(elapsed_s=time.monotonic()-started, source_unchanged=source == sources())
    summarize(result); write_json(out/'result.json', result)
    return result


@runtime_entry
def inspect(directory, seconds, *, runtime=None):
    if not np.isfinite(seconds) or not 1 <= seconds <= 600:
        raise ValueError('Independent audit must be bounded by 600 seconds')
    started = time.monotonic(); deadline = started+seconds
    directory = Path(directory).resolve()
    raw = json.loads((directory/'result.json').read_text())
    source = sources()
    report = dict(schema='softcut-conditional-release-audit/1', passed=False,
        result_sha256=digest(directory/'result.json'), source_hashes=source,
        producer_source_matches_current=source == raw['source_hashes'], audits=[],
        physical_continuation_permitted=False, thread_runtime=runtime)
    try:
        if source != raw['source_hashes']:
            raise ValueError('Conditional producer source differs')
        if digest(directory/'seed.npz') != raw['seed_sha256']:
            raise ValueError('Seed hash differs')
        for name, value in raw['seed']['inputs_sha256'].items():
            if digest(name) != value: raise ValueError('Original scene input changed')
        previous = arrays(directory/'seed.npz')
        p = driver.problem(raw['config'])
        if raw['seed'].get('seed_integrator', 'midpoint') != 'midpoint':
            install_integrator(p, raw['seed']['seed_integrator'])
        driver.audit(p, previous, raw['seed']['real_trial_record'])
        p.velocity = previous['velocity'].copy(); p.history = previous['history'].copy()
        p.material_metric = previous['material_metric'].copy()
        candidate = raw['settings']['candidate_seam_index']
        event = activate(p, previous['positions'], [candidate] if raw['settings']['release_candidate'] else [])
        np.testing.assert_allclose(event['friction_topology_force_jump_n'],
            raw['activation']['friction_topology_force_jump_n'], atol=1e-10)
        install_integrator(p, raw['settings'].get('integrator', 'midpoint'))
        previous['released'] = p.released.copy()
        for rec in raw['steps']:
            if time.monotonic() >= deadline: raise TimeoutError('Independent audit budget')
            path = (directory/rec['state_file']).resolve()
            if path.parent != directory or digest(path) != rec['state_sha256']:
                raise ValueError('State path/hash differs')
            state = arrays(path)
            for a, b in (('previous_positions', 'positions'), ('previous_velocity', 'velocity'),
                         ('previous_material_metric', 'material_metric'), ('previous_history', 'history'),
                         ('previous_released', 'released')):
                np.testing.assert_array_equal(state[a], previous[b])
            np.testing.assert_array_equal(state['released'], previous['released'])
            np.testing.assert_array_equal(state['mass'], p.mass)
            actual_dt = (float(state['depth_mm'])-float(previous['depth_mm']))/p.loading_rate
            np.testing.assert_allclose(float(state['material_dt_s']), actual_dt, rtol=0, atol=1e-14)
            audit = driver.audit(p, state, rec)
            if not audit['accepted_chord_ccd_safe']: raise ValueError('Temporal chord CCD failed')
            check_energy_gate(p, rec['energy'])
            impulse_fields, impulse = impulse_ledger(p, state['positions'])
            if 'impulse' in rec:
                for key, value in impulse_fields.items():
                    np.testing.assert_allclose(value, state[key], rtol=1e-10, atol=1e-14)
            audit['impulse'] = impulse
            quadrature = independent_friction_work(p, state['previous_positions'], state['positions'],
                state['increment'], deadline)
            audit.update(local_time_s=rec['local_time_s'], friction=quadrature,
                producer_friction_work_n_mm=rec['energy']['friction_work_n_mm'],
                friction_comparison_scope='integral on primary displacement chord versus producer time rule; difference includes temporal error',
                friction_quadrature_difference_n_mm=quadrature['work_n_mm']-rec['energy']['friction_work_n_mm'])
            report['audits'].append(audit); previous = state
        report['passed'] = bool(raw['steps'])
    except (AssertionError, RuntimeError, ValueError, TimeoutError, OSError) as exc:
        report['error'] = str(exc)
    report.update(elapsed_s=time.monotonic()-started, source_unchanged=source == sources())
    report['passed'] &= report['source_unchanged']
    report['all_friction_quadratures_converged'] = bool(report['audits']) and all(
        r['friction']['converged'] for r in report['audits'])
    report['summed_friction_quadrature_difference_n_mm'] = sum(r['friction_quadrature_difference_n_mm'] for r in report['audits'])
    report['summed_friction_empirical_uncertainty_n_mm'] = sum(r['friction']['empirical_uncertainty_n_mm'] for r in report['audits'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'audit'])
    parser.add_argument('directory', type=Path); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--dt', type=float, default=2e-6); parser.add_argument('--duration', type=float, default=2e-4)
    parser.add_argument('--no-release-control', action='store_true')
    parser.add_argument('--seed', choices=['trial', 'accepted'], default='trial',
        help='Explicit conditional real trial or bitwise inherited last committed state')
    parser.add_argument('--accepted-index', type=int, default=-1, help='Explicit earlier audited parent state; no history interpolation')
    parser.add_argument('--candidate-index', type=int, default=48, help='Conditional activation only; not an initiation selector')
    parser.add_argument('--integrator', choices=['midpoint', 'endpoint_be', 'endpoint_friction'], default='midpoint')
    parser.add_argument('--minimum-dt', type=float, help='Explicit physical cutback stopping bound; default dt/128')
    parser.add_argument('--attempt-seconds', type=float, default=30.)
    add_thread_arguments(parser); args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Preserve old evidence; choose a new output')
    if args.mode == 'run':
        report = run(args.directory, args.out, args.dt, args.duration, args.seconds,
            release=not args.no_release_control, attempt_seconds=args.attempt_seconds,
            seed_kind=args.seed, accepted_index=args.accepted_index, candidate_index=args.candidate_index, integrator=args.integrator,
            minimum_dt=args.minimum_dt, **thread_arguments(args))
        # runtime_entry finalizes counters after return.
        write_json(args.out/'result.json', report)
    else:
        report = inspect(args.directory, args.seconds, **thread_arguments(args))
        write_json(args.out, report)
    print(json.dumps({k:v for k,v in report.items() if k in ('status','error','passed','summary')}), flush=True)


if __name__ == '__main__': main()
