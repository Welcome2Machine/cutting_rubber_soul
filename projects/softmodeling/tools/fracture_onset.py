"""Nonlocal tensile-event hypotheses from every qualified intact saved state.

This is read-only postprocessing. No damage is inserted and no histories reset.
The event criterion is explicitly independent of the coupled cohesive model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.spatial import cKDTree

from workbench.results import saved_state_fields, _qualified_prefix, _digest
from tools.fracture_labels import load_labels


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def neighborhood_operator(centres, areas, length_mm, *, max_pairs=12000000):
    """Compact radial kernel in the reference plane; reference volume weighting.

    Constant reference width cancels. Averaging is truncated and renormalized
    at free boundaries; this is a specified hypothesis, not an internal length
    identified from measurements or a proof of mesh convergence.
    """
    centres, areas = np.asarray(centres, float), np.asarray(areas, float)
    if (centres.ndim != 2 or centres.shape[1] != 2 or areas.shape != (len(centres),)
            or not len(areas) or not np.isfinite(centres).all() or not np.isfinite(areas).all()
            or np.any(areas <= 0) or not np.isfinite(length_mm) or length_mm <= 0):
        raise ValueError('Finite centres, positive areas and physical length required')
    tree = cKDTree(centres)
    pairs = int(tree.count_neighbors(tree, length_mm))
    if pairs > max_pairs:
        raise ValueError(f'Neighborhood would require {pairs} pairs; reduce length or use a justified mesh')
    distances = tree.sparse_distance_matrix(tree, length_mm, output_type='coo_matrix')
    weights = (1-distances.data/length_mm)**2 * areas[distances.col]
    matrix = csr_matrix((weights, (distances.row, distances.col)), shape=(len(areas), len(areas)))
    normalizer = np.asarray(matrix.sum(axis=1)).ravel()
    return diags(1/normalizer) @ matrix


def first_crossing(rows, threshold, *, prefix_complete=True):
    """First accepted-state bracket; no interpolation is claimed as equilibrium."""
    if threshold is None:
        return dict(status='missing_threshold', calibrated=False, depth_interval_mm=None)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Threshold must be finite and positive')
    for i, row in enumerate(rows):
        if row['q_inplane_MPa'] >= threshold:
            previous = rows[i-1] if i else None
            return dict(status='crossed' if previous else 'already_above_at_first_qualified_state',
                        calibrated=False, threshold_MPa=float(threshold),
                        depth_interval_mm=[previous['depth_mm'] if previous else None, row['depth_mm']],
                        force_bracket_N=[previous['fz_N'] if previous else None, row['fz_N']],
                        state_indices=[previous['index'] if previous else None, row['index']],
                        location_reference_mm=row['location_reference_mm'],
                        local_normal=row['local_normal'],
                        interpretation='first sampled criterion crossing, not observed fracture or a cohesive onset')
    return dict(status='not_triggered_in_qualified_range' if prefix_complete else 'censored_by_qualification',
                calibrated=False, threshold_MPa=float(threshold), depth_interval_mm=None,
                last_qualified_depth_mm=rows[-1]['depth_mm'] if rows else None)


def analyze_case(source, output, *, length_mm=.15, threshold=None, seconds=300,
                 tip_radius_mm=3., clamp_exclusion_mm=.3):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('Analysis output must be separate from saved source cases')
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('A positive wall-clock budget is required')
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
        raise ValueError('Threshold must be finite and positive')
    if not np.isfinite([tip_radius_mm, clamp_exclusion_mm]).all() or tip_radius_mm <= 0 or clamp_exclusion_mm < 0:
        raise ValueError('Invalid reference search-domain dimensions')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = json.loads((source/'result.json').read_text(encoding='utf-8'))
    if report.get('schema') == 'softcut-cohesive-result/1' or any('fracture' in step for step in report.get('steps', [])):
        raise ValueError('Intact onset diagnosis cannot consume a coupled fracture history')
    audit = json.loads((source/'audit.json').read_text(encoding='utf-8'))
    prefix, reason = _qualified_prefix(report, audit)
    if not prefix:
        raise ValueError('No continuously qualified states; onset analysis is unavailable')
    signature = report['signature']
    material, config = signature['material'], signature['config']
    theta = np.deg2rad(float(config['inclination_deg']))
    es, en = np.array([np.cos(theta), -np.sin(theta)]), np.array([np.sin(theta), np.cos(theta)])
    # Same material-coordinate mapping as model_config.local_to_lab.
    from prototype.model_config import local_to_lab, FEMConfig
    cfg = FEMConfig(**config)
    tip_reference = local_to_lab(np.array([[cfg.drop_s_mm, cfg.height_mm]]), cfg)[0]
    rows, hashes = [], {}
    geometry = None
    field_paths = []
    snapshot_indices = set(np.linspace(0, prefix-1, min(5, prefix)).round().astype(int))
    for i in range(prefix):
        if time.monotonic()-started > seconds:
            raise TimeoutError(f'Saved-state analysis exceeded {seconds}s at index {i}; no completion is claimed')
        path = source/'states'/f'{i:04d}.npz'
        sha = _digest(path)
        if sha != audit['rows'][i]['checkpoint_sha256']:
            raise ValueError(f'Checkpoint hash differs from audit: {path}')
        hashes[path.name] = sha
        with np.load(path, allow_pickle=False) as state:
            n = int(state['body_nodes'])
            X, x, tri = state['reference_xy'][:n], state['positions'][:n], state['triangles']
            fields, energy, areas = saved_state_fields(X, x, tri, state['material_metric'], material,
                                                       include_mechanics=True)
            step = report['steps'][i]
            if not np.isclose(float(state['depth_mm']), step['depth_mm'], rtol=0, atol=1e-12):
                raise ValueError('Accepted depth/checkpoint mismatch')
            if geometry is None:
                geometry = (X.copy(), tri.copy())
                centres = X[tri].mean(axis=1)
                # Search all elements for diagnostics; also a prescribed tip domain
                # excluding the fixed boundary. Never choose a mask by FE error.
                bottom_level = float(np.min(X @ en))
                tip_mask = ((np.linalg.norm(centres-tip_reference, axis=1) <= tip_radius_mm)
                            & (centres @ en > bottom_level+clamp_exclusion_mm))
                if not tip_mask.any():
                    raise ValueError('Reference tip search domain contains no elements')
                operator = neighborhood_operator(centres, areas, length_mm)
            elif not np.array_equal(X, geometry[0]) or not np.array_equal(tri, geometry[1]):
                raise ValueError('Saved reference topology changed inside intact history')
            if not np.isclose(float(config['width_mm'])*(areas @ energy),
                              step['material']['stored_energy_n_mm'], rtol=1e-8, atol=1e-9):
                raise ValueError('Recovered stored energy differs from accepted history')
            if not np.isclose(fields['J'].min(), step['min_j'], rtol=1e-9, atol=1e-10):
                raise ValueError('Recovered J differs from accepted history')
            q = operator @ fields['inplane_tensile_MPa']
            q3 = operator @ fields['tensile_MPa']
            winner = int(np.argmax(np.where(tip_mask, q, -np.inf)))
            global_winner = int(np.argmax(q))
            outplane = np.maximum(fields['cauchy_MPa'][:, 2, 2], 0)
            qout = operator @ outplane
            current = x[tri].mean(axis=1)
            row = dict(index=i, depth_mm=float(step['depth_mm']),
                       time_s=step['material'].get('time_s'), dt_s=step['material'].get('dt_s'),
                       fz_N=float(step['reaction_fz_n']),
                       fy_N=step.get('reaction_x_n'), q_inplane_MPa=float(q[winner]),
                       q_3d_MPa=float(np.max(q3[tip_mask])),
                       q_outplane_MPa=float(np.max(qout[tip_mask])),
                       q_global_inplane_MPa=float(q[global_winner]),
                       global_winner_outside_tip_domain=bool(not tip_mask[global_winner]),
                       element=winner, location_reference_mm=centres[winner].tolist(),
                       location_current_mm=current[winner].tolist(),
                       local_normal=fields['inplane_normal'][winner].tolist(),
                       local_max_stretch=float(fields['principal_stretch'][winner, -1]),
                       min_J=float(fields['J'].min()), residual_N=float(step['residual_n']),
                       equilibrium_energy_Nmm=float(config['width_mm'])*(areas @ fields['equilibrium_density_MPa']),
                       nonequilibrium_energy_Nmm=float(config['width_mm'])*(areas @ fields['nonequilibrium_density_MPa']),
                       viscous_dissipation_Nmm=step['material'].get('cumulative_viscous_dissipation_n_mm'),
                       friction_work_increment_Nmm=step.get('friction_work_n_mm'))
            rows.append(row)
            crossing_now = (threshold is not None and q[winner] >= threshold and
                            not any(v['q_inplane_MPa'] >= threshold for v in rows[:-1]))
            if i in snapshot_indices or crossing_now:
                field_path = output/f'fields_{i:04d}.npz'
                np.savez_compressed(field_path, reference_xy=X, positions=x, triangles=tri,
                                    material_metric=state['material_metric'], q_inplane_MPa=q, q_3d_MPa=q3,
                                    tip_mask=tip_mask, **fields)
                field_paths.append(field_path.name)
        if i % 25 == 0:
            print(f'ONSET {source.name}: state {i+1}/{prefix}', flush=True)
    complete = prefix == len(report['steps']) and bool(report.get('completed'))
    result = dict(schema_version='softcut-onset/1', source=str(source),
                  angle_deg=float(config['inclination_deg']), material=material,
                  parameter_status='unidentified' if threshold is None else 'demonstration_unvalidated',
                  criterion=dict(name='reference_neighborhood_positive_inplane_principal_Cauchy',
                      length_mm=length_mm, threshold_MPa=threshold, kernel='(1-r/ell)^2 for r<ell',
                      domain='reference element centroids within tip radius and above clamp exclusion',
                      tip_radius_mm=tip_radius_mm, clamp_exclusion_mm=clamp_exclusion_mm,
                      boundary_normalization='truncated support, normalized by reference volume',
                      directions='in-plane crack normals only; local eigenvector is diagnostic, not averaged orientation'),
                  qualification=dict(states=prefix, complete_path=complete, stop_reason=reason,
                                     scope='prior audit reused, every checkpoint hash/J/stored energy rechecked'),
                  event=first_crossing(rows, threshold, prefix_complete=complete), rows=rows,
                  field_files=field_paths,
                  provenance=dict(result_sha256=_digest(source/'result.json'), audit_sha256=_digest(source/'audit.json'),
                                  checkpoints_sha256=hashes,
                                  sources_sha256={str(p.relative_to(Path(__file__).parents[1])): _digest(p)
                                      for p in [Path(__file__), Path(__file__).parents[1]/'workbench/results.py']}),
                  limitations=['Intact-state diagnostic only; no crack insertion or energy criterion',
                      'No physical onset labels verified; force changes and group cutoffs remain proxies',
                      'Material, contact, length and strength not independently calibrated',
                      '2D surface comparison and large local compression remain model limitations',
                      'Reported time uses solver physical increments; optical synchronization is unverified'])
    write_json(output/'onset.json', result)
    with (output/'onset.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    render_case(result, output)
    return result


def render_case(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = result['rows']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    depth = [v['depth_mm'] for v in r]
    for name, label in [('q_inplane_MPa', 'tip: in-plane'), ('q_3d_MPa', 'tip: 3D'),
                        ('q_global_inplane_MPa', 'whole body: in-plane')]:
        ax[0].plot(depth, [v[name] for v in r], label=label)
    if result['criterion']['threshold_MPa'] is not None:
        ax[0].axhline(result['criterion']['threshold_MPa'], color='k', linestyle='--', label='example threshold')
    ax[0].set(xlabel='Vertical travel (mm)', ylabel='Averaged positive stress (MPa)')
    ax[0].legend(fontsize=8)
    ax[1].plot(depth, [v['fz_N'] for v in r]); ax[1].set(xlabel='Vertical travel (mm)', ylabel='Raw simulated Fz (N)')
    fig.tight_layout(); fig.savefig(output/'curves.png', dpi=150); plt.close(fig)
    # Show the actual first crossing if present, never silently show the final
    # intact state beneath a heading that implies this is an onset field.
    indices = result['event'].get('state_indices')
    field_index = indices[-1] if indices else r[-1]['index']
    field_name = f'fields_{field_index:04d}.npz'
    field_depth = next(v['depth_mm'] for v in r if v['index'] == field_index)
    field_role = 'first sampled threshold crossing' if indices else 'last qualified state'
    with np.load(output/field_name, allow_pickle=False) as field:
        fig, ax = plt.subplots(figsize=(9, 4))
        X = field['positions']; tri = field['triangles']
        col = ax.tripcolor(X[:,0], X[:,1], tri, facecolors=field['q_inplane_MPa'], shading='flat')
        ax.set_aspect('equal'); ax.set(xlabel='laboratory horizontal (mm)', ylabel='laboratory vertical (mm)')
        row = next(v for v in r if v['index'] == field_index)
        ax.plot(*row['location_current_mm'], marker='x', color='red', ms=9, mew=2)
        ax.set_title(f'{field_role}: d={field_depth:.4f} mm; demonstration criterion')
        fig.colorbar(col, ax=ax, label='Nonlocal in-plane positive stress (MPa)')
        fig.tight_layout(); fig.savefig(output/'field.png', dpi=150); plt.close(fig)
    title = f"{result['angle_deg']:g}° 起裂候选指标"
    body = f'''<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:35px auto;line-height:1.6}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:#f3f5f7;padding:18px}}</style>
<h1>{title}</h1><p>完整体保存态后处理；阈值与长度尺度未完成实验标定。指标首次越阈不等于可见裂纹或CZM损伤。</p>
<p><a href="onset.json">JSON与来源</a> · <a href="onset.csv">CSV全历史</a></p>
<pre>{html.escape(json.dumps(result['event'],ensure_ascii=False,indent=2))}</pre>
<img src="curves.png"><p>下图：{html.escape(field_role)}，行程 {field_depth:.4f} mm；红叉为预设搜索域内的候选位置。此处仍是完整体，没有插入裂纹。</p><img src="field.png"><h2>适用范围</h2>
<ul>{''.join('<li>'+html.escape(x)+'</li>' for x in result['limitations'])}</ul>'''
    (output/'index.html').write_text(body, encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--length-mm', type=float, default=.15)
    parser.add_argument('--threshold-mpa', type=float)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--labels', type=Path, default=Path(__file__).parents[1]/'configs/fracture_labels.example.json',
                        help='Evidence inventory only; no automatic trial/depth registration or calibration')
    args = parser.parse_args(argv)
    labels = load_labels(args.labels)
    if len(set(p.resolve() for p in args.case)) != len(args.case):
        parser.error('Duplicate source cases')
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    start = time.monotonic()
    for index, case in enumerate(args.case):
        remaining = args.seconds-(time.monotonic()-start)
        result = analyze_case(case, args.output/f'case_{index}_{case.name}', length_mm=args.length_mm,
                              threshold=args.threshold_mpa, seconds=remaining)
        results.append(dict(angle_deg=result['angle_deg'], event=result['event'],
                            report=f'case_{index}_{case.name}/index.html', source=str(case.resolve())))
    eligible = [label for label in labels['labels'] if label['kind'] == 'verified_observation'
                and label['event_kind'] == 'physical_onset' and label['split'] != 'unused']
    evidence = dict(catalogue_source=str(args.labels.resolve()), catalogue_sha256=_digest(args.labels),
                    verified_physical_onset_labels=len(eligible),
                    comparison_status='no_verified_labels' if not eligible else 'explicit_trial_and_depth_registration_required',
                    calibration_performed=False, validation_performed=False,
                    note='Angle alone does not register a trial, a depth reference or an event definition.',
                    labels=labels['labels'])
    write_json(args.output/'experimental_evidence.json', evidence)
    write_json(args.output/'onset_summary.json', dict(schema_version='softcut-onset-summary/1',
               shared_length_mm=args.length_mm, shared_threshold_MPa=args.threshold_mpa,
               calibration='not_performed', experimental_comparison=evidence, cases=results))
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>多角度起裂候选诊断</h1>'
        '<p>共享示例阈值，尚未实验标定。当前起裂标签不足，未执行物理验证或强行触发裂纹。</p><ul>'
        + ''.join(f'<li><a href="{html.escape(r["report"],quote=True)}">{r["angle_deg"]:g}° — {html.escape(r["event"]["status"])}</a></li>' for r in results)
        + f'</ul><p>可用于物理起裂评价的已核实标签：{len(eligible)}；未执行标定或留出验证。标签证据与试次身份保持原样，未按角度自动配对。</p>'
        + '<a href="experimental_evidence.json">实验标签证据与缺项</a> · <a href="onset_summary.json">机器摘要</a>', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output.resolve()), cases=len(results))))


if __name__ == '__main__':
    main()
