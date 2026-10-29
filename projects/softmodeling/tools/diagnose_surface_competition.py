"""Check fixed surface/interior candidates on an archived physical seed.

This is an equation-compatible diagnostic, never a physical continuation or a
change to candidate ordering. Each candidate has independent objects/histories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numerical_runtime import numerical_runtime, add_thread_arguments, thread_arguments
from prototype.finite_increment_cut import FrozenVirtualCrack
from prototype.ipc_reference import LoadStepFailure
from run_support import write_json
from virtual_friction_work import virtual_work, interval_decision
from .frozen_scene import load_scene
from .diagnose_finite_increment import virtual_state
from .run_zero_gap_cut import hashes
from .virtual_work_audit import audit_work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', type=int, nargs='+', default=[50, 48])
    parser.add_argument('--root-seconds', type=float, default=120.)
    add_thread_arguments(parser)
    args = parser.parse_args()
    if not 1 <= len(args.candidates) <= 2 or not 1 <= args.root_seconds <= 150:
        raise ValueError('At most two independent roots, each at most 150 seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    source = hashes()
    with numerical_runtime(**thread_arguments(args)) as runtime:
        p, base, archived, before, _, scene = load_scene(args.case, 'production')
        _, physical_g, _, _ = p.evaluate(base.previous, p.lag(base.previous), tangent=False)
        _, virtual_g, _, _ = base.evaluate(before, base.lag(before), tangent=False)
        stress = np.minimum(-physical_g.reshape(-1, 2)[p.seam, 0]/p.seam_area,
                            -virtual_g.reshape(-1, 2)[p.seam, 0]/p.seam_area)
        y = p.rest[p.seam, 1]
        face_area = np.diff(y)*p.config.width_mm
        old_support = p.released[:-1] | p.released[1:]
        report = dict(schema='softcut-surface-candidate-diagnostic/1', scene=scene,
            runtime=runtime, source_hashes=source, candidates=[],
            diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            physical_insertion_permitted=False, original_candidate_order_unchanged=True,
            interpretation='Support union area is a geometry diagnostic, not an alternative fracture cost. Existing nodal and face quadrature histories are not interchangeable.')
        for index in args.candidates:
            if not 0 <= index < len(p.seam) or p.released[index]:
                raise ValueError('A virgin seam candidate is required')
            removed = np.zeros_like(p.released); removed[index] = True
            v = FrozenVirtualCrack(p, before, archived.previous_material_metric,
                archived.previous_history, p.released, removed, p.ffm_epsilon)
            support = removed[:-1] | removed[1:]
            rec = dict(seam_index=index, reference_y_mm=float(y[index]),
                stress_mpa=float(stress[index]), strength_mpa=p.strength,
                nodal_tributary_area_mm2=float(p.seam_area[index]),
                p1_support_area_mm2=float(face_area[support].sum()),
                additional_support_union_area_mm2=float(face_area[support & ~old_support].sum()),
                required_fracture_work_n_mm=float(p.toughness*p.seam_area[index]),
                status='unknown')
            report['candidates'].append(rec); start = time.monotonic()
            try:
                after, _, _ = v.equilibrium(before, scene['physical_seed']['depth_mm'], start+args.root_seconds)
                rec['root_elapsed_s'] = time.monotonic()-start
                rec['endpoint'] = virtual_state(v, after)
                release = base.stored(before)-v.stored(after)
                rec['released_stored_energy_n_mm'] = release
                # Even a rigorous energy-only rejection is accompanied by the
                # actual integral here, to distinguish it from a fabricated zero.
                work = virtual_work(v, before, after, time.monotonic()+45)
                independent = audit_work(v, before, after, time.monotonic()+45)
                rec.update(production_work=work, independent_work=independent)
                if work['work_n_mm'] is not None:
                    rec['energy_interval'] = interval_decision(release,
                        rec['required_fracture_work_n_mm'], work['work_n_mm'],
                        work['empirical_uncertainty_n_mm'], work['converged'])
                    rec['work_agreement'] = abs(work['work_n_mm']-independent['work_n_mm']) <= (
                        work['empirical_uncertainty_n_mm']+independent['empirical_uncertainty_n_mm']+1e-7)
                    rec['status'] = rec['energy_interval']['status'] if independent['converged'] and rec['work_agreement'] else 'unknown'
                rec['energy_insufficient_even_with_zero_friction'] = release < rec['required_fracture_work_n_mm']
            except (LoadStepFailure, TimeoutError, ValueError) as exc:
                after = v.trial.copy()
                rec.update(error=str(exc), endpoint=virtual_state(v, after))
            path = args.output/f'candidate_{index}.npz'
            np.savez_compressed(path, before=before, after=after,
                physical_seed=base.previous, seed_increment=p.step_increment(base.previous),
                frozen_metric=archived.previous_material_metric, frozen_history=archived.previous_history,
                previous_released=p.released, removed=removed)
            rec.update(evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                elapsed_s=time.monotonic()-start)
            write_json(args.output/'diagnostic.json', report)
    report['source_unchanged'] = source == hashes()
    write_json(args.output/'diagnostic.json', report)
    print(json.dumps([dict(index=c['seam_index'], status=c['status'], residual=c['endpoint']['residual_n'],
        energy_insufficient_even_with_zero_friction=c.get('energy_insufficient_even_with_zero_friction')) for c in report['candidates']]))


if __name__ == '__main__':
    main()
