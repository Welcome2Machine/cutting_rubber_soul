"""Sensitivity of accepted crack decisions to tighter virtual force solves.

Frozen material/history, physical seed and model are unchanged. This is an
independent accuracy control, not physical continuation or a source-hash bypass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numerical_runtime import numerical_runtime, add_thread_arguments, thread_arguments
from prototype.finite_increment_cut import FrozenVirtualCrack
from prototype.force_globalization import solve
from prototype.ipc_reference import LoadStepFailure
from run_support import write_json
from virtual_friction_work import interval_decision
from .audit_finite_increment import audit_evidence, stored_equilibrium
from .frozen_scene import check_archived_equations
from .run_zero_gap_cut import problem, hashes
from .virtual_work_audit import audit_work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=480.)
    add_thread_arguments(parser)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 500:
        raise ValueError('Accuracy audit requires a budget in 1..500 seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    raw = json.loads((args.case/'result.json').read_text())
    source = hashes(); deadline = time.monotonic()+args.seconds
    identity = check_archived_equations(raw, Path(__file__).resolve().parents[1])
    with numerical_runtime(**thread_arguments(args)) as runtime:
        report = dict(schema='softcut-virtual-force-accuracy/1', source_identity=identity,
            source_hashes=source, runtime=runtime, target_full_force_n=1e-8,
            loading_started=False, physical_history_committed=False, checks=[],
            result_sha256=hashlib.sha256((args.case/'result.json').read_bytes()).hexdigest(),
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        p = problem(raw['config'])
        def load(path, expected):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Archived input hash differs')
            with np.load(path, allow_pickle=False) as z:
                return {k:z[k].copy() for k in z.files}
        for step in raw['steps']:
            for check in step.get('finite_increment_checks', []):
                if not check['accepted']:
                    continue
                state = load(args.case/step['state_file'], step['state_sha256'])
                fields = load(args.case/check['evidence_file'], check['evidence_sha256'])
                original = audit_evidence(p, state, check, fields, deadline)
                rec = dict(depth_mm=step['depth_mm'], seam_indices=check['seam_indices'],
                    original=original, status='unknown', evidence_sha256=check['evidence_sha256'])
                report['checks'].append(rec)
                try:
                    empty = np.zeros_like(fields['removed'])
                    base = FrozenVirtualCrack(p, fields['pre_anchor'], fields['frozen_metric'],
                        fields['frozen_history'], fields['previous_released'], empty, p.ffm_epsilon)
                    def tighten(v, x):
                        v.force_tolerance_n = 1e-8/np.sqrt(2)
                        v.max_iterations = 12
                        v.set_primary_increment(x, x-v.previous)
                        value, _ = solve(v, x.copy(), step['depth_mm'], min(deadline, time.monotonic()+30))
                        gradient, norm = stored_equilibrium(v, value)
                        if norm > 1e-8:
                            raise ValueError('Tighter force target not reached')
                        return value, gradient, norm
                    before, gradient, r0 = tighten(base, fields['before_positions'])
                    post = FrozenVirtualCrack(p, before, fields['frozen_metric'], fields['frozen_history'],
                        fields['previous_released'], fields['removed'], p.ffm_epsilon)
                    after, _, r1 = tighten(post, fields['after_positions'])
                    work = audit_work(post, before, after, deadline)
                    decision = interval_decision(base.stored(before)-post.stored(after),
                        check['required_fracture_work_n_mm'], work['work_n_mm'],
                        work['empirical_uncertainty_n_mm'], work['converged'])
                    stress = np.minimum(fields['physical_traction'], -gradient.reshape(-1, 2)[p.seam, 0]/p.seam_area)
                    rec.update(status=decision['status'] if np.min(stress[fields['removed']])>=p.strength else 'strength_failed',
                        tightened_pre_residual_n=r0, tightened_post_residual_n=r1,
                        tightened_work=work, tightened_energy_interval=decision,
                        original_lower_margin_n_mm=check['energy_margin_n_mm'],
                        pre_endpoint_max_change_mm=float(np.max(np.abs(before-fields['before_positions']))),
                        post_endpoint_max_change_mm=float(np.max(np.abs(after-fields['after_positions']))))
                    np.savez_compressed(args.output/f'polished_{len(report["checks"]):03d}.npz', before=before, after=after)
                except (LoadStepFailure, TimeoutError, ValueError) as exc:
                    rec['error'] = str(exc)
                write_json(args.output/'audit.json', report)
                if time.monotonic() >= deadline:
                    report['budget_exhausted'] = True
                    break
            if time.monotonic() >= deadline:
                break
    report['source_unchanged'] = source == hashes()
    report['all_accepted_decisions_stable'] = (len(report['checks']) == sum(c['accepted'] for s in raw['steps'] for c in s.get('finite_increment_checks', []))
        and all(c['status']=='accepted' for c in report['checks']) and report['source_unchanged'])
    write_json(args.output/'audit.json', report)
    print(json.dumps(dict(stable=report['all_accepted_decisions_stable'], checks=[dict(depth=c['depth_mm'], ids=c['seam_indices'], status=c['status'], margin=c.get('tightened_energy_interval')) for c in report['checks']])))


if __name__ == '__main__':
    main()
