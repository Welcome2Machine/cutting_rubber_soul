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
from release_dynamics import (activate, check_energy_gate, instantaneous_friction,
    contact_velocity, independent_friction_work)


def sources():
    root = Path(__file__).resolve().parents[1]
    result = driver.hashes()
    for name in ('release_dynamics.py', 'tools/dynamic_release.py', 'tools/frozen_scene.py',
                 'tools/diagnose_finite_increment.py'):
        result[name] = digest(root/name)
    return result


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def real_seed(directory):
    directory = Path(directory).resolve()
    raw = json.loads((directory/'result.json').read_text())
    provenance = check_archived_equations(raw, Path(__file__).resolve().parents[1])
    inputs = {str(directory/'result.json'): digest(directory/'result.json')}
    # The parent accepted prefix must already have both independent audits.
    for suffix in ('path_audit', 'finite_audit'):
        path = directory.with_name(directory.name+'.'+suffix+'.json')
        audit = json.loads(path.read_text())
        if not audit['passed'] or audit['result_sha256'] != digest(directory/'result.json'):
            raise ValueError('Parent audit is missing, failed or bound to another result')
        inputs[str(path)] = digest(path)
    previous_path = directory/raw['steps'][-1]['state_file']
    if digest(previous_path) != raw['steps'][-1]['state_sha256']:
        raise ValueError('Parent last accepted state hash differs')
    old = arrays(previous_path)
    trial = arrays(directory/'failed_trial.npz')
    for a, b in (('previous_positions', 'positions'), ('previous_velocity', 'velocity'),
                 ('previous_material_metric', 'material_metric'), ('previous_history', 'history')):
        np.testing.assert_array_equal(trial[a], old[b])
    np.testing.assert_array_equal(trial['released'], old['released'])
    for path in (previous_path, directory/'failed_trial.npz'):
        inputs[str(path)] = digest(path)
    p = driver.problem(raw['config'])
    x, depth = restore_failed_seed(p, trial)
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
    instant = instantaneous_friction(p, x, contact_velocity(p, p.velocity))
    state = driver.snapshot(p, x, p.released.copy(), depth)
    info = dict(inputs_sha256=inputs, provenance=provenance, real_trial_record=record,
        physical_energy_gate=gate, parent_last_accepted_depth_mm=raw['steps'][-1]['depth_mm'],
        real_trial_depth_mm=depth, inherited_physical_dt_s=p.material_dt,
        incoming_average_friction_norm_n=float(np.sqrt(2)*np.linalg.norm(incoming)),
        endpoint_instantaneous_friction_norm_n=float(np.sqrt(2)*np.linalg.norm(instant)),
        interval_average_to_instantaneous_difference_n=float(np.sqrt(2)*np.linalg.norm(incoming-instant)),
        qualification='conditional real trial; finite-increment decision still unknown',
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
        initiation_qualified=False, complete_cutting_qualified=False, event_energy_converged=False)


@runtime_entry
def run(directory, out, dt, duration, seconds, *, release=True, attempt_seconds=30., runtime=None):
    if not (np.isfinite([dt, duration, seconds, attempt_seconds]).all() and
            0 < dt <= duration <= .025 and 1 <= seconds <= 3600 and 1 <= attempt_seconds <= 90):
        raise ValueError('Require 0 < dt <= duration <= .025 s and bounded compute budgets')
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started+seconds
    source = sources()
    p, x, depth, seed, info, config = real_seed(directory)
    seed_path = out/'seed.npz'; np.savez_compressed(seed_path, **seed)
    original_released = p.released.copy()
    event = activate(p, x, [48] if release else [])
    result = dict(schema='softcut-conditional-dynamic-release/1', status='running',
        source_hashes=source, execution_identity=driver.execution_identity(), interpreter=sys.executable,
        thread_runtime=runtime, config=config, seed=info, seed_sha256=digest(seed_path),
        activation=event, settings=dict(dt_max_s=dt, duration_s=duration, attempt_seconds=attempt_seconds,
            seconds=seconds, release_candidate=release, candidate_seam_index=48,
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
                if physical_cutback(kind) and next_dt > dt/128:
                    next_dt /= 2; continue
                raise
            x, depth, local_time = value, target_depth, target_time
            q = p.cohesive_response(x, p.history)
            rec.update(local_time_s=local_time, dt_s=p.material_dt,
                candidate_opening_mm=float(2*(x[p.seam[48], 0]-p.config.drop_s_mm)),
                candidate_dissipation_n_mm=float(p.seam_area[48]*q['dissipated'][48]*p.released[48]),
                maximum_speed_mm_s=float(np.linalg.norm(p.velocity[:p.n], axis=1).max()))
            state = driver.snapshot(p, x, old_released, depth)
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
    except (TimeoutError, LoadStepFailure, RuntimeError, ValueError) as exc:
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
        driver.audit(p, previous, raw['seed']['real_trial_record'])
        p.velocity = previous['velocity'].copy(); p.history = previous['history'].copy()
        p.material_metric = previous['material_metric'].copy()
        event = activate(p, previous['positions'], [48] if raw['settings']['release_candidate'] else [])
        np.testing.assert_allclose(event['friction_topology_force_jump_n'],
            raw['activation']['friction_topology_force_jump_n'], atol=1e-10)
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
            quadrature = independent_friction_work(p, state['previous_positions'], state['positions'],
                state['increment'], deadline)
            audit.update(local_time_s=rec['local_time_s'], friction=quadrature,
                midpoint_friction_work_n_mm=rec['energy']['friction_work_n_mm'],
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
    parser.add_argument('--attempt-seconds', type=float, default=30.)
    add_thread_arguments(parser); args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Preserve old evidence; choose a new output')
    if args.mode == 'run':
        report = run(args.directory, args.out, args.dt, args.duration, args.seconds,
            release=not args.no_release_control, attempt_seconds=args.attempt_seconds, **thread_arguments(args))
        # runtime_entry finalizes counters after return.
        write_json(args.out/'result.json', report)
    else:
        report = inspect(args.directory, args.seconds, **thread_arguments(args))
        write_json(args.out, report)
    print(json.dumps({k:v for k,v in report.items() if k in ('status','error','passed','summary')}), flush=True)


if __name__ == '__main__': main()
