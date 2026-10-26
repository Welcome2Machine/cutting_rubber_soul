"""Paired contact-address estimates from saved tracks and explicit depth cutoffs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
import html
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote

import cv2
import numpy as np

from src.io import ROOT, HIDDEN, sha, load_npz, read_json, clean, write_json
from src.analysis.dual_contact import (
    flattened_input, RepairedDualContactSolver, ObservationRecoveryContactSolver,
    ConvergedContactSolver, REPAIRED_METHODS, RECOVERY_METHODS,
)
from src.analysis.contact_transport_filter import WINDOWED_STATES
from src.analysis.contact_table_summary import (
    summarize_curve, extrapolate_cutoff, interpolate_internal_gaps,
)
from src.workflows import contact_inputs

ESTIMATORS = {'bounded': RepairedDualContactSolver, 'recovery': ObservationRecoveryContactSolver,
              'converged': ConvergedContactSolver}


def methods_for(estimator):
    return REPAIRED_METHODS if estimator == 'bounded' else RECOVERY_METHODS


def load_cases(manifest=None, *, names=None, source=None, measurements=None, stop_mm=None, stop_map=None):
    """Normalize manifests without inferring fracture depth from a filename."""
    base = Path(manifest).resolve().parent if manifest else ROOT
    rows = read_json(manifest)['cases'] if manifest else [dict(name=name) for name in names or []]
    overrides = dict(stop_map or {})
    selected = set(names) if names else None
    known = [row['name'] for row in rows]
    if not rows or len(set(known)) != len(known):
        raise ValueError('A nonempty, unique case list is required')
    if selected is not None and selected-set(known):
        raise ValueError('Unknown case names: '+str(sorted(selected-set(known))))
    if set(overrides)-set(known):
        raise ValueError('Depth overrides contain unknown cases')
    cases = []
    for row in rows:
        name = row['name']
        if selected is not None and name not in selected:
            continue
        if not name or name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError('Case names must be single directory names')
        case = dict(row)
        cutoff = dict(row.get('cutoff') or row.get('declared_cutoff') or {})
        explicit = overrides.get(name, stop_mm)
        depth = explicit if explicit is not None else cutoff.get('depth_mm')
        if depth is None and 'depth_um' in cutoff:
            depth = float(cutoff['depth_um'])/1000.
        if depth is None or not np.isfinite(float(depth)) or float(depth) <= 0:
            raise ValueError(f'{name}: explicit positive cutoff required; fracture depth is not guessed')
        if explicit is None and 'depth_um' in cutoff and not np.isclose(float(depth)*1000, cutoff['depth_um']):
            raise ValueError(name+': inconsistent cutoff units')
        if explicit is not None:
            cutoff = dict(source='explicit_cli_cutoff')
        case['cutoff'] = dict(cutoff, depth_mm=float(depth), depth_um=float(depth)*1000)
        for key, override in (('tracks_root', source), ('measurements_root', measurements)):
            value = override if override is not None else row.get(key)
            if value is None:
                raise ValueError(f'{name}: {key} is required')
            path = Path(value)
            case[key] = str((path if path.is_absolute() or override is not None else base/path).resolve())
        track, measure = Path(case['tracks_root'])/name, Path(case['measurements_root'])/name
        required = [track/file for file in ('tracking/tracks.npz', 'geometry.npz', 'geometry.json',
            'plan.json', 'normalized_lossless.mkv', 'queries.npz', 'frame_mapping.npz')]
        required += [measure/file for file in ('measurement_support.npz', 'blade_contact_geometry.npz',
            'initial_surface_extended.npz', 'motion_measurements.npz')]
        case['missing_prerequisites'] = [str(path) for path in required if not path.is_file()]
        cases.append(case)
    return cases


def contact_review(case, folder):
    """Review the contact zero from fixed-ID motion; preserve original source files."""
    from src.analysis.contact_zero_audit import audit_contact
    from src.analysis.material_coordinate_contact import MaterialCoordinateContactEstimator
    source = Path(case['tracks_root'])/case['name']
    measure = Path(case['measurements_root'])/case['name']
    meta = read_json(source/'geometry.json')
    q, tr, g, clock, ref = [load_npz(path) for path in (source/'queries.npz', source/'tracking/tracks.npz',
        source/'geometry.npz', source/'frame_mapping.npz', measure/'initial_surface_extended.npz')]
    mapping = MaterialCoordinateContactEstimator(q['seed_points'], q['point_ids'], ref, 1.)
    review, signals = audit_contact(clock['physical_time_s'], tr['positions'], tr['supported'],
        q['seed_points'], g['tips_px'], g['supported'], mapping, meta['contact_frame'], meta['reversal_frame'])
    old = meta['contact_frame']
    selected = review['candidate_frame'] if review['accepted'] else old
    stroke = g['tips_px'][meta['reversal_frame'], 1]-g['tips_px'][selected, 1]
    if not np.isfinite(stroke) or stroke <= 0:
        raise ValueError('Contact review cannot establish a positive calibrated stroke')
    scale = 3000./stroke
    review.update(selected_contact_frame=int(selected), old_scale_um_per_px=3000./meta['contact_to_reversal_px'],
        candidate_scale_um_per_px=float(scale), declared_fracture_cutoff_um=case['cutoff']['depth_um'],
        declared_fracture_cutoff_changed=False, calibration_status='motion_onset_candidate_not_independent_ground_truth')
    write_json(folder/'contact_audit.json', review)
    np.savez_compressed(folder/'contact_signals.npz', **signals)
    return review


@contextmanager
def candidate_contact_zero(case, review):
    """Worker-local adaptation, always restored even when input preparation fails."""
    from src.workflows import motion
    from src.analysis.loading_blade_reference import prepare_loading_blade_reference
    original = motion.data
    def read(name):
        data = original(name)
        contact, reversal = review['selected_contact_frame'], data['plan']['reversal_frame']
        frames = np.arange(contact, reversal+1)
        geometry = load_npz(data['source']/'geometry.npz')
        folder = Path(case['measurements_root'])/name
        blade, relative = load_npz(folder/'blade_contact_geometry.npz'), load_npz(folder/'motion_measurements.npz')
        scale = review['candidate_scale_um_per_px']
        data.update(plan=dict(data['plan'], contact_frame=contact), frames=frames, um_per_px=scale,
            gap=blade['signed_blade_gap_px'][frames].copy(),
            material=(blade['material_pixel'] & ~blade['on_blade'])[frames].copy(),
            legacy=relative['group_y_median'][frames].copy(),
            original_depth=(geometry['tips_px'][frames, 1]-geometry['tips_px'][contact, 1])*scale/1000,
            reference=prepare_loading_blade_reference(geometry['tips_px'][frames], geometry['supported'][frames], data['times'][frames]))
        return data
    motion.data = read
    try:
        yield
    finally:
        motion.data = original


def solve_prepared(data, geometry, origin, motion, estimator='bounded', progress=None):
    solver = ESTIMATORS[estimator](flattened_input(data), geometry, origin, motion)
    results, reports = solver.calculate(progress)
    if estimator == 'converged':
        guide = results['mmt']['transport_search_reference_s']
        results['mmt'], reports['mmt'] = solver.mixed_model_converged(guide)
    return results, reports


def comparison_plot(case, results, folder, estimator='bounded'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for ax, method in zip(axes, methods_for(estimator)):
        data = results[method.key]
        ax.plot(data['depth_mm'], data['offset_um'], color=method.color, lw=2, label=method.english_name)
        ax.fill_between(data['depth_mm'], data['offset_um']-data['sigma_um'],
            data['offset_um']+data['sigma_um'], color=method.color, alpha=.18)
        completed = interpolate_internal_gaps(data['depth_mm'], data['offset_um'],
            eligible=(data['depth_mm'] >= 0) & (data['depth_mm'] <= case['cutoff']['depth_mm']+1e-9))
        drawn = False
        for gap in completed['gaps']:
            if gap['filled']:
                ends = [gap['left_index'], gap['right_index']]
                ax.plot(data['depth_mm'][ends], data['offset_um'][ends], color='#167e9b', ls='--', lw=1.8,
                    label='Conditional interpolation' if not drawn else '_nolegend_')
                drawn = True
        ax.axhline(0, color='.7', lw=.7)
        ax.set(ylabel='Surface address shift (um)', xlim=(0, case['cutoff']['depth_mm']))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Indentation from first contact (mm)')
    fig.suptitle(case['name']+' | supplied cutoff; shading = model sensitivity')
    fig.savefig(folder/'comparison.png', dpi=140)
    plt.close(fig)


def report_curves(case, folder, results=None, estimator='bounded'):
    """Derive traceable display-only interpolation without changing solver arrays."""
    if results is None:
        stored = read_json(folder/'complete.json')
        results = {row.get('method_key', row['method'].split('-')[0].lower()):
                   load_npz(folder/(row['method']+'.npz')) for row in stored['methods']}
        estimator = stored.get('configuration', {}).get('estimator', 'bounded')
    summaries, records = [], []
    for key, arrays in results.items():
        identifier = str(arrays['method_identifier'].item())
        source = folder/(identifier+'.npz')
        fingerprint = sha(source)
        depth, offset = arrays['depth_mm'], arrays['offset_um']
        completion = interpolate_internal_gaps(depth, offset,
            eligible=(depth >= 0) & (depth <= case['cutoff']['depth_mm']+1e-9))
        gaps = completion.pop('gaps')
        filled = completion['estimate_kind'] == 2
        for gap in gaps:
            left, right = gap['left_index'], gap['right_index']
            gap.update(left_frame=int(arrays['frames'][left]), right_frame=int(arrays['frames'][right]),
                duration_s=float(arrays['times_s'][right]-arrays['times_s'][left]),
                uses_later_anchor_within_cutoff=True, interpolation_error_quantified=False)
        np.savez_compressed(folder/(identifier+'.report.npz'), **completion, depth_mm=depth,
            frames=arrays['frames'], times_s=arrays['times_s'], source_sha256=np.array(fingerprint),
            original_sigma_um=arrays['sigma_um'], completed_sigma_um=np.where(filled, np.nan, arrays['sigma_um']))
        summaries.append(dict(method=identifier, method_key=key, source_sha256=fingerprint, gaps=gaps,
            endpoint_um=clean(offset[-1]), actual_endpoint_depth_um=float(depth[-1]*1000),
            endpoint_state=WINDOWED_STATES[int(arrays['transport_state'][-1])],
            available_fraction=float(np.mean(arrays['transport_offset_available'])),
            **summarize_curve(depth, completion['offset_completed_um'], completion['estimate_kind'], case['cutoff']['depth_um']),
            **extrapolate_cutoff(arrays, case['cutoff']['depth_um'])))
        for i, frame in enumerate(arrays['frames']):
            records.append(dict(method=identifier, source_frame=int(frame), depth_mm=float(depth[i]),
                original_offset_um=clean(offset[i]), completed_offset_um=clean(completion['offset_completed_um'][i]),
                estimate_kind=int(completion['estimate_kind'][i]), gap_id=int(completion['interpolation_gap_id'][i]),
                left_anchor_index=int(completion['left_anchor_index'][i]), right_anchor_index=int(completion['right_anchor_index'][i])))
        if sha(source) != fingerprint:
            raise RuntimeError('Solver result changed while reporting: '+str(source))
    write_json(folder/'report.json', dict(case=case['name'], methods=summaries,
        curves_have_no_endpoint_extrapolation=True, source_solver_results_unchanged=True,
        interpolation_error_quantified=False))
    with (folder/'report.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    export_curves(folder, results, estimator)
    comparison_plot(case, results, folder, estimator)
    return summaries


def export_curves(folder, results, estimator='bounded'):
    methods = methods_for(estimator)
    with (folder/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_frame', 'depth_mm', 'physical_time_s']+[method.identifier+'_'+key
            for method in methods for key in ('offset_um', 'model_sensitivity_um', 'state', 'motion_used', 'address_used')])
        reference = results['mmt']
        for i, frame in enumerate(reference['frames']):
            row = [int(frame), float(reference['depth_mm'][i]), float(reference['times_s'][i])]
            for method in methods:
                data = results[method.key]
                state = 'imposed contact zero' if i == 0 else WINDOWED_STATES[int(data['transport_state'][i])]
                row += [clean(data['offset_um'][i]), clean(data['sigma_um'][i]), state,
                        bool(data['transport_velocity_used'][i]), bool(data['transport_absolute_used'][i])]
            writer.writerow(row)


def calculate(case, out, *, estimator='bounded', audit_contact_zero=False, overwrite=False):
    started = time.perf_counter()
    folder = Path(out)/case['name']
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder/'complete.json'
    configuration = dict(estimator=estimator, audit_contact_zero=audit_contact_zero,
        tracks_root=case['tracks_root'], measurements_root=case['measurements_root'], cutoff=case['cutoff'])
    if marker.exists() and not overwrite:
        previous = read_json(marker)
        if previous.get('configuration') != configuration:
            raise RuntimeError('Existing result has a different recipe or input; use a separate --out or --overwrite')
        if all((folder/(m.identifier+'.npz')).is_file() for m in methods_for(estimator)):
            for path, expected in previous.get('source_sha256', {}).items():
                if sha(ROOT/path) != expected:
                    raise RuntimeError('Saved inputs changed; use a separate output or --overwrite')
            return previous
    if case['missing_prerequisites']:
        raise FileNotFoundError('Missing saved inputs: '+str(case['missing_prerequisites']))
    if marker.exists():
        marker.unlink()  # An interrupted overwrite must not retain an old success marker.
    cv2.setNumThreads(1)
    if audit_contact_zero:
        review = contact_review(case, folder)
        with candidate_contact_zero(case, review):
            data, geometry, origin, motion, metadata, guard = contact_inputs.prepare_guarded(case, folder, out)
        guard.update(contact_zero_audit=review,
            calibration_changed=review['selected_contact_frame'] != review['old_contact_frame'])
        write_json(folder/'blade_reference_guard.json', guard)
    else:
        data, geometry, origin, motion, metadata, guard = contact_inputs.prepare_guarded(case, folder, out)
    results, reports = solve_prepared(data, geometry, origin, motion, estimator,
        lambda key, iteration, frame, total: print(f'{case["name"]}: {key} {iteration} {frame}/{total}', flush=True))
    for method in methods_for(estimator):
        np.savez_compressed(folder/(method.identifier+'.npz'), **results[method.key])
    summaries = report_curves(case, folder, results, estimator)
    for method, summary in zip(methods_for(estimator), summaries):
        write_json(folder/(method.identifier+'.json'), dict(summary=summary, **reports[method.key]), compact=True)
    for path, expected in metadata['provenance'].items():
        if sha(ROOT/path) != expected:
            raise RuntimeError('Read-only input changed: '+path)
    if float(np.max(data['depth'])) > case['cutoff']['depth_mm']+1e-9:
        raise AssertionError('Prepared samples exceed supplied cutoff')
    report = dict(case=case['name'], configuration=configuration, declared_cutoff=case['cutoff'],
        cutoff=metadata['cutoff'], methods=summaries, blade_guard=guard, input_video=guard['source'],
        source_sha256=metadata['provenance'], seconds=time.perf_counter()-started,
        optical_flow_recomputed=False, post_cutoff_analysis_samples=0,
        physical_accuracy_validated=False, interpretation='Original-surface address estimate; not automatic fracture detection',
        code_sha256={str(path.relative_to(ROOT)): sha(path) for path in
            (Path(__file__), ROOT/'src/analysis/dual_contact.py', ROOT/'src/analysis/contact_transport_filter.py')})
    write_json(marker, report)
    return report


def write_index(out, cases, states):
    rows, cards, summaries = [], [], []
    def number(value):
        return 'Unavailable' if value is None else f'{value:.1f}'
    for case in cases:
        name, folder = case['name'], out/case['name']
        report = read_json(folder/'complete.json') if (folder/'complete.json').is_file() else None
        if report and (folder/'report.json').is_file():
            report = dict(report, methods=read_json(folder/'report.json')['methods'])
        escaped, link = html.escape(name), quote(name)
        cells = []
        for method in ('mmt', 'obt'):
            result = next((m for m in report['methods'] if
                m.get('method_key', m['method'].split('-')[0].lower()) == method), {}) if report else {}
            row = dict(case=name, cutoff_um=case['cutoff']['depth_um'], **result)
            row['method'] = method
            summaries.append(row)
            cells.append(f'<td>{number(result.get("last_estimate_um"))}</td><td>{number(result.get("last_estimate_depth_um"))}</td>'
                         f'<td>{number(result.get("tail_estimate_um"))}</td>')
        rows.append(f'<tr><th><a href="#{link}">{escaped}</a></th><td>{case["cutoff"]["depth_um"]:g}</td>'
                    +''.join(cells)+f'<td>{html.escape(states[name]["stage"])}</td></tr>')
        if report:
            video = '<video controls preload="metadata" src="'+link+'/repaired_contact.mp4"></video>' if (folder/'repaired_contact.mp4').exists() else ''
            cards.append(f'<section id="{link}"><h2>{escaped}</h2>{video}<img src="{link}/comparison.png" alt="Slip curves">'
                         f'<p><a href="{link}/comparison.csv">Solver data</a> · <a href="{link}/report.csv">Interpolation data</a> · '
                         f'<a href="{link}/report.json">Gap provenance</a> · <a href="{link}/complete.json">Calculation</a></p></section>')
    write_json(out/'summary.json', summaries)
    columns = ['case', 'method', 'cutoff_um', 'last_estimate_um', 'last_estimate_depth_um', 'tail_estimate_um', 'tail_estimate_kind']
    with (out/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    headings = '<th>Case</th><th>Cutoff (um)</th>'+''.join(f'<th>{method} {label}</th>' for method in ('MMT', 'OBT')
        for label in ('last slip (um)', 'last depth (um)', 'conditional cutoff estimate (um)'))+'<th>Status</th>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Contact sliding</title>'
        '<style>body{font:16px system-ui;margin:2rem;background:#102029;color:#e8eff2}table{border-collapse:collapse}'
        'td,th{padding:.6rem;border:1px solid #65757d}a{color:#83dcec}img,video{display:block;max-width:100%;width:1100px}'
        'section{margin-top:2rem}</style><h1>Contact sliding</h1><p>Two distinct estimators; conditional material-address inference. '
        'Last available estimate and endpoint extrapolation are reported separately. Solid lines: original estimates; '
        'blue dashed lines: conditional internal interpolation. Gaps receive no added uncertainty band. '
        'Shading shows original model sensitivity, not accuracy.</p>'
        '<table><tr>'+headings+'</tr>'+''.join(rows)+'</table>'+''.join(cards), encoding='utf-8')


def worker_command(args, name, out):
    command = [sys.executable, '-u', '-m', 'src.workflows.sliding', '--manifest', str(out/'manifest.json'),
        '--names', name, '--out', str(out), '--worker', '--estimator', args.estimator]
    for flag in ('render', 'overwrite', 'audit_contact_zero'):
        if getattr(args, flag):
            command.append('--'+flag.replace('_', '-'))
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--measurements', type=Path)
    parser.add_argument('--stop-mm', type=float)
    parser.add_argument('--stop-map', type=Path, help='JSON object mapping case names to cutoff depths in mm')
    parser.add_argument('--out', type=Path, default=ROOT/'results/current_sliding')
    parser.add_argument('--estimator', choices=tuple(ESTIMATORS), default='bounded')
    parser.add_argument('--jobs', type=int, choices=range(1, 9), default=2)
    for flag in ('worker', 'render', 'overwrite', 'audit-contact-zero', 'inventory-only', 'report-only'):
        parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args(argv)
    if args.report_only and args.worker:
        parser.error('--report-only cannot run a calculation worker')
    if args.report_only and any(value is not None for value in
            (args.stop_mm, args.stop_map, args.source, args.measurements)):
        parser.error('--report-only preserves the saved manifest inputs and cutoff definitions')
    if args.report_only and args.manifest is None:
        args.manifest = args.out.resolve()/'manifest.json'
    if args.audit_contact_zero and args.estimator == 'bounded':
        parser.error('--audit-contact-zero is available for recovery/converged only')
    try:
        cases = load_cases(args.manifest, names=args.names, source=args.source, measurements=args.measurements,
            stop_mm=args.stop_mm, stop_map=read_json(args.stop_map) if args.stop_map else None)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    out = args.out.resolve()
    for case in cases:
        for key in ('tracks_root', 'measurements_root'):
            if out == Path(case[key]).resolve():
                parser.error('Choose an output root separate from saved inputs')
    if args.worker:
        if len(cases) != 1:
            parser.error('One case per worker')
        calculate(cases[0], out, estimator=args.estimator, audit_contact_zero=args.audit_contact_zero, overwrite=args.overwrite)
        if args.render:
            from src.workflows.sliding_render import render
            render(out/cases[0]['name'], maximum_depth_mm=cases[0]['cutoff']['depth_mm'], overwrite=args.overwrite)
        return
    out.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        write_json(out/'manifest.json', dict(cases=cases, estimator=args.estimator, audit_contact_zero=args.audit_contact_zero))
    states = {case['name']: dict(stage='missing_inputs' if case['missing_prerequisites'] else 'pending') for case in cases}
    def snapshot(stage):
        write_json(out/'status.json', dict(stage=stage, supervisor_pid=os.getpid(), updated=time.time(), cases=states))
        write_index(out, cases, states)
    if args.inventory_only or args.report_only:
        for case in cases:
            if (out/case['name']/'complete.json').is_file():
                states[case['name']]['stage'] = 'complete'
                if args.report_only:
                    report_curves(case, out/case['name'])
        if args.report_only:
            write_index(out, cases, states)
        else:
            snapshot('inventory')
        print(out/'index.html')
        return
    write_json(out/'session_evidence.json', dict(python=sys.executable, optical_flow_recomputed=False,
        environment_exception='Existing documented AcademicForge virtual environment; no new dependencies',
        formal_artifacts=['manifest.json', 'status.json', 'index.html', 'summary.csv', 'summary.json',
            '*/complete.json', '*/MMT-*.npz', '*/OBT-*.npz', '*/comparison.csv', '*/comparison.png',
            '*/report.csv', '*/report.json', '*/*.report.npz']))
    def worker(case):
        folder = out/case['name']
        folder.mkdir(exist_ok=True)
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with (folder/'run.log').open('ab') as log:
            return subprocess.run(worker_command(args, case['name'], out), cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, creationflags=HIDDEN).returncode
    snapshot('running')
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(worker, case): case['name'] for case in cases if not case['missing_prerequisites']}
        for future in as_completed(futures):
            name = futures[future]
            try:
                code = future.result()
                states[name] = dict(stage='complete' if code == 0 else 'failed', returncode=code)
            except Exception as error:
                states[name] = dict(stage='failed', reason=str(error))
            snapshot('running')
    failed = any(state['stage'] != 'complete' for state in states.values())
    snapshot('incomplete' if failed else 'complete')
    print(out/'index.html', flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
