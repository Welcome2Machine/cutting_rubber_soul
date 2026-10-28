"""Read-only, authenticated per-increment energy screening of saved cutting.

No fresh FE solve. Existing audit hashes bind records and checkpoints; this
postprocessor cannot replace that audit, identify missing energy as kinetic
energy, or qualify physical fracture evolution from a scalar remainder.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.run_cohesive_cut import energy_ledger


def screen(rows, *, global_tolerance=.02, local_tolerance=.05, absolute_floor=1e-8):
    if not rows:
        return dict(available=False, numerical_screen_passed=False, increments=[])
    out = []
    terms = ['external_work_n_mm', 'equilibrium_stored_change_n_mm',
             'nonequilibrium_stored_change_n_mm', 'damageable_cohesive_stored_change_n_mm',
             'compression_stored_change_n_mm', 'viscous_dissipation_n_mm',
             'fracture_dissipation_n_mm', 'friction_work_n_mm', 'ipc_barrier_stored_change_n_mm']
    for row in rows:
        if not np.isfinite([row[k] for k in terms+['unaccounted_energy_n_mm']]).all():
            raise ValueError('Nonfinite energy ledger')
        scale = max(sum(abs(row[k]) for k in terms), absolute_floor)
        relative = abs(row['unaccounted_energy_n_mm'])/scale
        out.append(dict(row, absolute_activity_n_mm=scale, relative_local_remainder=relative,
                        local_screen_passed=relative <= local_tolerance))
    signed = sum(r['unaccounted_energy_n_mm'] for r in out)
    absolute = sum(abs(r['unaccounted_energy_n_mm']) for r in out)
    scale = max(sum(abs(r['external_work_n_mm']) for r in out), absolute_floor)
    return dict(available=True, signed_remainder_n_mm=signed, absolute_remainder_sum_n_mm=absolute,
                absolute_work_sum_n_mm=scale, signed_relative_remainder=abs(signed)/scale,
                absolute_relative_remainder=absolute/scale,
                maximum_local_relative_remainder=max(r['relative_local_remainder'] for r in out),
                failed_local_increments=sum(not r['local_screen_passed'] for r in out),
                largest_increment=max(out, key=lambda r: abs(r['unaccounted_energy_n_mm'])),
                numerical_screen_passed=(absolute/scale <= global_tolerance and all(r['local_screen_passed'] for r in out)),
                policy=dict(global_absolute_relative_tolerance=global_tolerance, local_activity_relative_tolerance=local_tolerance,
                            absolute_floor_n_mm=absolute_floor, kind='predeclared numerical screening, not experimental uncertainty'),
                increments=out)


def audit(result):
    folder = Path(result).resolve()
    report = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    evidence = json.loads((folder/'audit.json').read_text(encoding='utf-8'))
    if (not evidence.get('passed') or not evidence.get('cohesive_history_chain_passed')
            or not report.get('verification', {}).get('accepted_prefix_audited')):
        raise ValueError('Requires successful independent saved-path audit')
    steps, rows = report['steps'], evidence['rows']
    if not steps or len(steps) != len(rows):
        raise ValueError('Audit coverage differs from accepted records')
    inputs = {}
    for i, (step, row) in enumerate(zip(steps, rows)):
        digest = hashlib.sha256(json.dumps(step, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        path = folder/'states'/f'{i:04d}.npz'
        state_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (not row.get('passed') or row.get('accepted_record_sha256') != digest
                or row.get('checkpoint_sha256') != state_digest):
            raise ValueError(f'Accepted record/checkpoint authentication failed at {i}')
        inputs[str(path)] = state_digest
    for name in ('result.json', 'audit.json'):
        inputs[str(folder/name)] = hashlib.sha256((folder/name).read_bytes()).hexdigest()
    ledger = energy_ledger(steps)
    return dict(schema='softcut-fracture-energy-screen/1', result=str(folder),
                inputs_sha256=inputs, authenticated_states=len(steps),
                first_depth_mm=steps[0]['depth_mm'], last_depth_mm=steps[-1]['depth_mm'],
                completed=report.get('completed', False), target_mm=report['target_mm'],
                kinetic_energy_computed=False, physical_evolution_qualified=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                interpretation='Remainder is unresolved work/quadrature/evolution error; never relabeled as fracture, friction or kinetic energy',
                **screen(ledger))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = audit(a.result)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k: value.get(k) for k in ('authenticated_states', 'last_depth_mm', 'absolute_relative_remainder',
                     'maximum_local_relative_remainder', 'numerical_screen_passed', 'physical_evolution_qualified')}))


if __name__ == '__main__':
    main()
