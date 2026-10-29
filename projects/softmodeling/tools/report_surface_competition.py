"""Read-only geometry and candidate-cost report for saved finite-crack paths."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from run_support import write_json
from .run_zero_gap_cut import problem
from .frozen_scene import check_archived_equations


def report(directory, output):
    output.mkdir(parents=True, exist_ok=False)
    raw = json.loads((directory/'result.json').read_text())
    identity = check_archived_equations(raw, Path(__file__).resolve().parents[1])
    p = problem(raw['config'])
    reference = p.rest[p.seam, 1]
    face_area = np.diff(reference)*p.config.width_mm
    final_opening = []
    records, candidates = [], []
    for step in raw['steps']:
        path = directory/step['state_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != step['state_sha256']:
            raise ValueError('State checksum mismatch')
        with np.load(path, allow_pickle=False) as z:
            history, released = z['history'], z['released']
            opening = 2*(z['positions'][p.seam, 0]-p.config.drop_s_mm)
        failed = released & (history >= 2*p.toughness/p.strength)
        connected = 0.
        for i in range(len(reference)-2, -1, -1):
            if not (failed[i] and failed[i+1]):
                break
            connected += reference[i+1]-reference[i]
        records.append(dict(depth_mm=step['depth_mm'], top_released=bool(released[-1]),
            top_history_mm=float(history[-1]), top_opening_mm=float(opening[-1]),
            released_indices=np.flatnonzero(released).tolist(), fully_failed_indices=np.flatnonzero(failed).tolist(),
            nodal_released_area_mm2=float(p.seam_area[released].sum()),
            p1_support_union_area_mm2=float(face_area[released[:-1] | released[1:]].sum()),
            surface_connected_fully_failed_length_mm=float(connected)))
        final_opening.append(opening)
        for check in step.get('finite_increment_checks', []):
            if check['accepted'] or check['seam_indices'] in ([len(reference)-1], [len(reference)-3]):
                candidates.append(dict(depth_mm=step['depth_mm'], seam_indices=check['seam_indices'],
                    stress_min_mpa=check['stress_min_mpa'], accepted=check['accepted'],
                    released_energy_n_mm=check['virtual_stored_before_n_mm']-check['virtual_stored_after_n_mm'],
                    required_n_mm=check['required_fracture_work_n_mm'],
                    lower_margin_n_mm=check['energy_margin_n_mm'],
                    tributary_area_mm2=check['reference_tributary_area_mm2'],
                    additional_support_union_area_mm2=check['newly_exposed_support_area_mm2']))
    result = dict(schema='softcut-surface-competition-report/1', source_identity=identity,
        result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        loading_started=False, physical_history_committed=False, records=records, candidates=candidates,
        interpretation='Postprocessing of saved producer decisions. Support union is geometric area, not a replacement for nodal fracture work. These data do not prove that face discretization alone will open a surface channel.',
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output/'report.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    for index in (len(reference)-1, len(reference)-3):
        rows = [c for c in candidates if c['seam_indices'] == [index]]
        axes[0].plot([c['depth_mm'] for c in rows], [c['stress_min_mpa'] for c in rows], label=f'node {index}')
        axes[1].plot([c['depth_mm'] for c in rows], [c['released_energy_n_mm']-c['required_n_mm'] for c in rows], label=f'node {index}: before friction')
        axes[1].plot([c['depth_mm'] for c in rows], [c['lower_margin_n_mm'] for c in rows], '--', label=f'node {index}: lower margin')
    axes[0].axhline(p.strength, color='k', ls=':');axes[0].set(ylabel='Minimum pre-state traction / MPa')
    axes[1].axhline(0, color='k', ls=':');axes[1].set(ylabel='Candidate energy margin / N mm')
    axes[2].plot(final_opening[-1], reference, 'o-', markersize=3);axes[2].set(xlabel='Final physical seam opening / mm', ylabel='Reference height / mm')
    for ax in axes[:2]:
        ax.set(xlabel='Absolute travel / mm');ax.legend(fontsize=8);ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(output/'competition.png', dpi=150);plt.close(fig)
    output.joinpath('index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Surface competition</title><body><h1>Surface and internal candidates</h1><p>Absolute travel, no alignment or smoothing. A geometric support area is not an interchangeable fracture cost. No full-cut qualification follows from this diagnostic.</p><img src="competition.png" width="800"><p><a href="report.json">JSON and provenance</a></p></body></html>')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    q = report(args.case, args.output)
    print(json.dumps(q['records'][-1]))


if __name__ == '__main__':
    main()
