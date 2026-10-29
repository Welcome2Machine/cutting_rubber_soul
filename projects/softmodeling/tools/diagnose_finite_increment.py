"""Bounded reconstruction of a failed physical seed and its virtual candidates.

No physical loading or continuation is performed. Unknown candidates stay
unknown; diagnostic equilibria never replace accepted fields or histories.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from run_support import write_json
from .run_zero_gap_cut import problem, hashes
from .audit_finite_increment import stored_equilibrium
from prototype.finite_increment_cut import FrozenVirtualCrack, finite_patches, coupled_margin
from prototype.ipc_reference import LoadStepFailure
from prototype.force_globalization import solve


def restore_failed_seed(p, fields):
    """Use the failed attempt's actual previous state, dt and primary increment."""
    for attr, key in [('previous', 'previous_positions'),
                      ('previous_velocity', 'previous_velocity'),
                      ('previous_material_metric', 'previous_material_metric'),
                      ('previous_history', 'previous_history'),
                      ('released', 'released'), ('active_contact', 'active_contact')]:
        setattr(p, attr, fields[key].copy())
    p.material_dt = float(fields['material_dt_s'])
    if not np.isfinite(p.material_dt) or p.material_dt <= 0:
        raise ValueError('Failed attempt has invalid physical dt')
    p.previous_barrier_energy = float(fields['previous_barrier_energy_n_mm'])
    p.rebuild_contact()
    p.set_free()
    p.friction = p.ipc.FrictionPotential(p.friction_velocity_epsilon*p.material_dt)
    x = fields['positions'].copy()
    p.set_primary_increment(x, fields['increment'])
    depth = float(fields['target_depth_mm'])
    if (not np.array_equal(x[p.mesh.bottom_nodes], p.rest[p.mesh.bottom_nodes])
            or not np.allclose(x[p.n:], p.prescribed_blade_positions(depth), rtol=0, atol=1e-11)):
        raise ValueError('Failed seed boundary mismatch')
    return x, depth


def virtual_state(v, x):
    """Report the ORIGINAL equation even after unsuccessful solver continuation."""
    if np.any(v.virtual_solver_load):
        raise ValueError('Diagnostic cannot certify an artificial load')
    v.set_primary_increment(x, x-v.previous)
    _, g, _, j = v.evaluate(x, v.lag(x), tangent=False)
    v.update_constraint_free(x, g)
    return dict(residual_n=float(np.sqrt(2)*np.linalg.norm(g[v.free])),
                min_j=float(j), chord_safe=bool(v.accepted_chord_is_safe(x)),
                stored_n_mm=float(v.stored(x)),
                opening_max_mm=float(np.max(2*(x[v.seam, 0]-v.config.drop_s_mm))))


def probe_equilibrium(v, seed, depth, deadline, mode):
    """Diagnostic directions only; certify the original virtual equation last.

    The production operator/source and the physical seed/history are unchanged.
    These probes do not authorize loading or insertion, even when a root exists.
    """
    friction, lag, tangent, maximum = v.friction, v.lag, v.coupled_friction_tangent, v.max_iterations
    value, phases = seed.copy(), []
    v.probe_phases = phases
    try:
        if mode == 'wide_smoothing':
            factors = (1e6, 1e5, 1e4, 1e3, 100., 10., 1.)
            for index, factor in enumerate(factors):
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError('Virtual smoothing probe budget')
                stage_deadline = now+(deadline-now)/(len(factors)-index)
                v.friction = friction if factor == 1 else v.ipc.FrictionPotential(v.virtual_friction_epsilon*factor)
                v.set_primary_increment(value, value-v.previous)
                start = value.copy()
                phase = dict(factor=factor, accepted=False)
                phases.append(phase)
                try:
                    value, iterations = solve(v, start.copy(), depth, min(stage_deadline, now+5.))
                except (LoadStepFailure, TimeoutError):
                    v.set_primary_increment(start, start-v.previous)
                    value, iterations = v.solve_friction(start.copy(), depth, stage_deadline)
                phase.update(accepted=True, iterations=iterations)
        elif mode in ('lagged_friction', 'lagged_newton'):
            for outer in range(100):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Lagged virtual probe budget')
                # The inner solve uses a fixed tangential basis/normal load.
                # Its Hessian must not include a derivative of the frozen lag.
                frozen = lag(value)
                v.lag = lambda x: frozen
                v.coupled_friction_tangent = lambda x, l, h: h
                v.set_primary_increment(value, value-v.previous)
                phase = dict(outer=outer, inner_solved=False)
                phases.append(phase)
                try:
                    value, iterations = v.solve_friction(value.copy(), depth, min(deadline, time.monotonic()+10.))
                except (LoadStepFailure, TimeoutError):
                    if mode != 'lagged_newton':
                        raise
                    value, iterations = v.trial.copy(), None
                v.lag, v.coupled_friction_tangent = lag, tangent
                q = virtual_state(v, value)
                phase.update(inner_solved=iterations is not None, iterations=iterations, **q)
                if q['residual_n'] <= 1e-4:
                    break
                if mode == 'lagged_newton':
                    # Correct the predictor with the ORIGINAL coupled force.
                    # Keep every local segment and the full anchor chord safe.
                    v.set_primary_increment(value, value-v.previous)
                    try:
                        value, iterations = solve(v, value.copy(), depth, min(deadline, time.monotonic()+10.))
                        phase['corrector_solved'] = True
                        phase['corrector_state'] = virtual_state(v, value)
                        if phase['corrector_state']['residual_n'] <= 1e-4:
                            break
                    except (LoadStepFailure, TimeoutError) as exc:
                        phase.update(corrector_solved=False, corrector_error=str(exc))
                        value = v.trial.copy()
            else:
                raise LoadStepFailure('Lagged virtual probe iteration limit')
        else:
            raise ValueError('Unknown diagnostic equilibrium probe')
    finally:
        v.friction, v.lag, v.coupled_friction_tangent, v.max_iterations = friction, lag, tangent, maximum
    g, residual = stored_equilibrium(v, value)
    return value, g, residual


def inspect(directory, output, seconds, candidate=None, probe='production'):
    if not np.isfinite(seconds) or not 1 <= seconds <= 600:
        raise ValueError('Explicit diagnostic budget must be 1..600 seconds')
    if probe not in ('production', 'wide_smoothing', 'lagged_friction', 'lagged_newton') or (probe != 'production' and candidate is None):
        raise ValueError('A diagnostic solver probe requires one explicit candidate')
    directory, output = Path(directory).resolve(), Path(output).resolve()
    result_path = directory/'result.json'
    raw = json.loads(result_path.read_text(encoding='utf-8'))
    source = hashes()
    if source != raw['source_hashes'] or not raw.get('source_unchanged'):
        raise ValueError('Failed-seed reconstruction requires unchanged producer source')
    if raw['config'].get('fracture_criterion') != 'finite_increment':
        raise ValueError('Not a finite-increment case')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+seconds
    failed = directory/'failed_trial.npz'
    report = dict(schema='softcut-finite-failure-diagnostic/1', loading_started=False,
        physical_history_committed=False, reconstructs_supplied_failed_attempt_only=True,
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        failed_sha256=hashlib.sha256(failed.read_bytes()).hexdigest(),
        source_hashes=source, diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        budget_s=seconds, equilibrium_probe=probe, candidates=[], completed=False, intact_certified=False)
    try:
        p = problem(raw['config'])
        p.ipc.set_num_threads(2)
        with np.load(failed, allow_pickle=False) as data:
            fields = {k: data[k].copy() for k in data.files}
        x, depth = restore_failed_seed(p, fields)
        _, g, _, j = p.evaluate(x, p.lag(x), tangent=False)
        p.update_constraint_free(x, g)
        residual = float(np.sqrt(2)*np.linalg.norm(g[p.free]))
        report['physical_seed'] = dict(depth_mm=depth, dt_s=p.material_dt,
            residual_n=residual, min_j=float(j), chord_safe=p.accepted_chord_is_safe(x),
            failed_stage=str(fields['failed_stage']))
        if residual > 1e-4 or j <= 0 or not report['physical_seed']['chord_safe']:
            raise ValueError('Failed physical seed is not equilibrated/admissible; virtual gate not isolated')
        empty = np.zeros_like(p.released)
        metric = p.material_response(x)['metric'].copy()
        history = p.cohesive_response(x)['history'].copy()
        traction = -g.reshape(-1, 2)[p.seam, 0]/p.seam_area
        base = FrozenVirtualCrack(p, x, metric, history, p.released, empty, p.ffm_epsilon)
        before, bg, _ = base.equilibrium(x, depth, min(deadline, time.monotonic()+p.ffm_virtual_seconds))
        report['virtual_before'] = virtual_state(base, before)
        stress = np.minimum(traction, -bg.reshape(-1, 2)[p.seam, 0]/p.seam_area)
        patches = finite_patches(stress, p.released, p.rest[p.seam, 1], p.seam_area, p.strength)
        report['eligible_candidates'] = [ids.tolist() for ids in patches]
        if candidate is not None:
            patches = [ids for ids in patches if ids.tolist() == candidate]
            if not patches:
                raise ValueError('Requested candidate is not strength eligible')
        np.savez_compressed(output/'context.npz', seed_positions=x, seed_increment=fields['increment'],
            before_positions=before, frozen_metric=metric, frozen_history=history,
            previous_released=p.released, physical_traction=traction, stress=stress, depth_mm=depth)
        for index, ids in enumerate(patches):
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic budget reached')
            removed = empty.copy()
            removed[ids] = True
            post = FrozenVirtualCrack(p, before, metric, history, p.released, removed, p.ffm_epsilon)
            check = dict(seam_indices=ids.tolist(), reference_y_mm=p.rest[p.seam[ids], 1].tolist(),
                status='unknown', stress_min_mpa=float(stress[ids].min()))
            begin = time.monotonic()
            try:
                if probe == 'production':
                    after, _, _ = post.equilibrium(before, depth,
                        min(deadline, begin+p.ffm_virtual_seconds))
                else:
                    after, _, _ = probe_equilibrium(post, before, depth, deadline, probe)
                stored_equilibrium(post, after)
                d5, d9, d17 = [post.friction_path_work(before, after, n) for n in (5, 9, 17)]
                ok, margin = coupled_margin(stress[ids], p.strength,
                    base.stored(before)-post.stored(after), d17,
                    p.toughness*float(p.seam_area[ids].sum()), 2*abs(d17-d9))
                check.update(status='equilibrated', **virtual_state(post, after),
                    coupled_condition_passed=ok, refined_energy_margin_n_mm=margin,
                    friction_work_orders_5_9_17_n_mm=[d5, d9, d17])
            except (LoadStepFailure, TimeoutError, ValueError) as exc:
                after = getattr(post, 'trial', before).copy()
                check.update(error=str(exc), **virtual_state(post, after))
            evidence = output/f'candidate_{index:03d}.npz'
            np.savez_compressed(evidence, positions=after, removed=removed)
            check.update(elapsed_s=time.monotonic()-begin, evidence_file=evidence.name,
                probe_phases=getattr(post, 'probe_phases', []),
                evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
            report['candidates'].append(check)
            write_json(output/'diagnostic.json', report)
        report['completed'] = True
        report['all_candidates_equilibrated'] = all(c['status'] == 'equilibrated' for c in report['candidates'])
    except (ValueError, RuntimeError, AssertionError, TimeoutError) as exc:
        report['error'] = str(exc)
    report.update(source_unchanged=source == hashes(), elapsed_s=time.monotonic()-started)
    write_json(output/'diagnostic.json', report)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Finite crack failure diagnostic</title><h1>Failed seed and virtual candidates</h1>'
        '<p>Diagnostic only. Unknown candidates do not certify an intact state; '
        'no physical loading or history commit.</p><pre>'+html.escape(json.dumps(report, indent=2))+
        '</pre><a href="diagnostic.json">JSON evidence</a>', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--candidate', type=int, nargs='+', help='Only this exact contiguous seam patch')
    parser.add_argument('--probe', choices=['production', 'wide_smoothing', 'lagged_friction', 'lagged_newton'], default='production',
                        help='Diagnostic root directions only; requires --candidate for a non-production probe')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = inspect(args.directory, args.output, args.seconds, args.candidate, args.probe)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}, indent=2))


if __name__ == '__main__':
    main()
