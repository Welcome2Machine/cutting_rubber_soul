"""Experimental motion features and conditional P1 saved-state comparison.

No equilibrium solve, force/video synchronization, fracture classification or
parameter fitting. Outputs are local scientific data, excluded from publishing.
The conditional image mapping must be explicitly enabled; it is not calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ('near_left', 'near_right', 'far_left', 'far_right')
MILESTONES = ('contact_estimate', 'est_0.25mm', 'est_0.5mm', 'est_0.75mm', 'est_1mm')


def checked(root, relative, limit=512 * 1024**2):
    root = Path(root).resolve()
    path = root / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f'Not a regular input under its declared root: {relative}')
    if any(p.is_symlink() for p in path.parents if p != root.parent):
        raise ValueError('Symlink input ancestor')
    if path.stat().st_size > limit:
        raise ValueError('Input exceeds bounded analysis size')
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def fixed_cohort(table, frame_ids):
    """A fixed survivor set; never join new reference segments to initial motion."""
    columns = ('frame_id', 'point_id', 'reference_frame_id', 'valid', 'x_px', 'y_px')
    if not set(columns).issubset(table.columns):
        raise ValueError('Missing initial-identity trajectory columns')
    if table.duplicated(['frame_id', 'point_id']).any():
        raise ValueError('Ambiguous point identity at a frame')
    sets = []
    for frame in frame_ids:
        q = table[(table.frame_id == frame) & (table.valid == 1)
                  & (table.reference_frame_id == 0)]
        q = q[np.isfinite(q[['x_px', 'y_px']]).all(axis=1)]
        sets.append(set(q.point_id))
    ids = sorted(set.intersection(*sets))
    if not ids:
        raise ValueError('No continuous initial-reference cohort')
    return ids


def region_summary(displacement, regions):
    result = {}
    for name in REGIONS:
        values = np.asarray(displacement)[np.asarray(regions) == name]
        values = values[np.isfinite(values).all(axis=1)]
        result[name] = dict(count=len(values), median=np.median(values, axis=0).tolist()
                            if len(values) else [None, None])
    return result


def region_contrasts(summary, component):
    result = {}
    for name, a, b in [('left', 'near_left', 'far_left'),
                       ('right', 'near_right', 'far_right')]:
        va, vb = summary[a]['median'][component], summary[b]['median'][component]
        result[name] = None if va is None or vb is None else float(va - vb)
    return result


def validate_image_estimates(table):
    """Explicit manual screenshot readings; ranges are visual, not confidence intervals."""
    columns = ['depth_mm', 'force_n', 'lower_n', 'upper_n']
    if not set(columns).issubset(table.columns):
        raise ValueError('Expected manual image depth/force/range columns; sensor channels are excluded')
    points = table[columns].copy().sort_values('depth_mm')
    if len(points) < 2 or not np.isfinite(points.to_numpy(dtype=float)).all():
        raise ValueError('Finite screenshot readings at two or more depths required')
    if (not np.all(np.diff(points.depth_mm) > 0) or (points.depth_mm < 0).any()
            or (points.lower_n < 0).any() or (points.lower_n > points.force_n).any()
            or (points.force_n > points.upper_n).any() or points.lower_n.iloc[-1] <= 0):
        raise ValueError('Invalid visual ranges or duplicate screenshot depths')
    return points


def chord_summary(reference, current, pairs):
    """Nonlocal point-pair length ratio, invariant to common rigid image motion.

    This is a chord stretch, not a point strain or a principal stretch. Repeated
    far-point endpoints make samples correlated; quantiles are spatial spread.
    """
    a, b = np.asarray(pairs, dtype=int).T
    denominator = np.linalg.norm(np.asarray(reference)[a] - np.asarray(reference)[b], axis=1)
    if np.any(denominator <= 0):
        raise ValueError('Distinct material endpoints required')
    ratio = np.linalg.norm(np.asarray(current)[a] - np.asarray(current)[b], axis=1) / denominator
    if not np.isfinite(ratio).all():
        raise ValueError('Chord endpoints must remain observed')
    return dict(count=len(ratio), median=float(np.median(ratio)),
                q25=float(np.quantile(ratio, .25)), q75=float(np.quantile(ratio, .75)))


def initial_near_far_pairs(points, regions, shared):
    from scipy.spatial import cKDTree
    result = {}
    for side in ('left', 'right'):
        near = np.flatnonzero((regions == 'near_' + side) & shared)
        far = np.flatnonzero((regions == 'far_' + side) & shared)
        if not len(near) or not len(far):
            raise ValueError('Both initial regions required for chord observations')
        distance, index = cKDTree(points[far]).query(points[near])
        if np.any(distance <= 0):
            raise ValueError('Coincident near/far initial points')
        result[side] = np.column_stack([near, far[index]])
    return result


class P1Sampler:
    """Reference-mesh interpolation; points outside remain unobserved."""
    def __init__(self, xy, triangles):
        from matplotlib.tri import Triangulation
        self.xy = np.asarray(xy, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        if (self.xy.ndim != 2 or self.xy.shape[1] != 2
                or self.triangles.ndim != 2 or self.triangles.shape[1] != 3
                or not np.isfinite(self.xy).all()
                or self.triangles.min() < 0 or self.triangles.max() >= len(self.xy)):
            raise ValueError('Invalid P1 reference mesh')
        self.finder = Triangulation(self.xy[:, 0], self.xy[:, 1], self.triangles).get_trifinder()

    def locate(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Finite 2D query points required')
        ids = np.asarray(self.finder(points[:, 0], points[:, 1]))
        valid = ids >= 0
        weights = np.full((len(points), 3), np.nan)
        vertices = self.xy[self.triangles[ids[valid]]]
        matrix = np.stack([vertices[:, 1] - vertices[:, 0],
                           vertices[:, 2] - vertices[:, 0]], axis=2)
        uv = np.linalg.solve(matrix, (points[valid] - vertices[:, 0])[..., None])[..., 0]
        weights[valid] = np.column_stack([1 - uv.sum(axis=1), uv])
        return ids, weights

    def sample(self, nodal, location):
        ids, weights = location
        nodal = np.asarray(nodal)
        if nodal.ndim != 2 or nodal.shape[0] != len(self.xy) or not np.isfinite(nodal).all():
            raise ValueError('Finite nodal vector field on reference mesh required')
        result = np.full((len(ids), nodal.shape[1]), np.nan)
        valid = ids >= 0
        result[valid] = np.einsum('ni,nij->nj', weights[valid], nodal[self.triangles[ids[valid]]])
        return result


def conditional_reference_points(points, metadata, config, scale_factor=1., shift_s=0., pixel_aspect=1.):
    """Assume isotropic pixel scale and project inferred tip on initial top line.

    This maps initial material positions, not current contact or crack positions.
    It aligns the initial surface, not the independently unknown lab camera pose.
    """
    if not np.isfinite(pixel_aspect) or pixel_aspect <= 0:
        raise ValueError('Positive conditional x/y pixel metric ratio required')
    metric = np.array([pixel_aspect, 1.])
    es = np.asarray(metadata['reference']['initial_surface_direction_image'], dtype=float)
    if es.shape != (2,) or not np.isclose(np.linalg.norm(es), 1., atol=1e-6):
        raise ValueError('Unit image surface direction required')
    es = es * metric
    es /= np.linalg.norm(es)
    en = np.array([es[1], -es[0]])
    center = np.asarray(metadata['reference']['initial_surface_center_image_px']) * metric
    tip = np.asarray(metadata['reference']['estimated_contact_blade_source_xy_px']) * metric
    anchor = center + np.dot(tip - center, es) * es
    scale = float(metadata['coordinates']['conditional_image_scale_mm_px']) * scale_factor
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive conditional scale required')
    local = (points * metric - anchor) @ np.stack([es, en]).T * scale
    local += [config['drop_s_mm'] + shift_s, config['height_mm']]
    theta = np.deg2rad(config['inclination_deg'])
    basis = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return local @ basis, basis, np.stack([es, en]), scale






def load_state(run, index, expected_depth):
    path = checked(run, f'states/{index:04d}.npz', 64 * 1024**2)
    with np.load(path, allow_pickle=False) as z:
        needed = ('reference_xy', 'triangles', 'positions', 'depth_mm')
        if not set(needed).issubset(z.files):
            raise ValueError('Unsupported state schema')
        state = {key: z[key].copy() for key in needed}
    if not np.isclose(float(state['depth_mm']), expected_depth, atol=1e-10, rtol=0):
        raise ValueError('Report/checkpoint depth mismatch')
    return state, path


def analyze_case(package, snapshots, case, runs, conditional_map, sources, pixel_aspect=1.):
    if '/' in case or '\\' in case or '..' in case:
        raise ValueError('Unsafe case identifier')
    meta_path = checked(package, f'{case}/metadata.json', 2 * 1024**2)
    frame_path = checked(package, f'{case}/frames.csv', 8 * 1024**2)
    audit_path = checked(snapshots, f'{case}_audit.json', 2 * 1024**2)
    track_path = checked(snapshots, f'{case}_selected_frames.csv', 16 * 1024**2)
    m = json.loads(meta_path.read_text(encoding='utf-8'))
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    # Certify cache is still from this exact source, without re-running optical flow.
    for name in ('metadata.json', 'frames.csv', 'tracks.csv'):
        path = checked(package, f'{case}/{name}')
        sha = digest(path)
        if sha != audit['manifest'][name]['sha256']:
            raise ValueError('Stale trajectory cache; re-audit changed source first')
        sources[str(path)] = sha
    sources[str(track_path)] = digest(track_path)
    sources[str(audit_path)] = digest(audit_path)
    table = pd.read_csv(track_path)
    frames = pd.read_csv(frame_path).set_index('frame_id')
    milestone = {v['label']: v for v in audit['milestones']}
    fids = [int(milestone[label]['frame_id']) for label in MILESTONES]
    if not np.all(np.diff(fids) > 0):
        raise ValueError('Milestones must form a loading time prefix')
    if not (frames.loc[fids].geometry_supported == 1).all():
        raise ValueError('Unsupported blade geometry at milestone')
    cohort = fixed_cohort(table, [0] + fids)
    initial = table[(table.frame_id == 0) & table.point_id.isin(cohort)].set_index('point_id').loc[cohort]
    contact = table[(table.frame_id == fids[0]) & table.point_id.isin(cohort)].set_index('point_id').loc[cohort]
    p0 = initial[['x_px', 'y_px']].to_numpy()
    pc = contact[['x_px', 'y_px']].to_numpy()
    regions = initial.roi_ref.to_numpy()
    es = np.asarray(m['reference']['initial_surface_direction_image'])
    image_basis = np.stack([es, [es[1], -es[0]]])
    blade0 = frames.loc[fids[0], ['blade_tip_x_px', 'blade_tip_y_px']].to_numpy(dtype=float)
    observations = []
    for fid in fids:
        now = table[(table.frame_id == fid) & table.point_id.isin(cohort)].set_index('point_id').loc[cohort]
        du = now[['x_px', 'y_px']].to_numpy() - pc
        db = frames.loc[fid, ['blade_tip_x_px', 'blade_tip_y_px']].to_numpy(dtype=float) - blade0
        observations.append(dict(frame_id=fid, conditional_depth_mm=float(frames.loc[fid, 'depth_video_estimate_mm']),
                                 blade_xy_px=db, blade_local_px=db @ image_basis.T,
                                 image=region_summary(du, regions),
                                 local=region_summary(du @ image_basis.T, regions)))
    rec = dict(case_id=case, data_role=m.get('data_role'), cohort_count=len(cohort),
               interpretation='Candidate intact window; every fracture_state remains unknown',
               experimental=observations, comparison={},
               initial_surface_angle_image_deg=m['reference']['initial_surface_angle_image_deg'],
               force_source='User force-depth screenshots only; exchange force-channel CSV excluded')
    if runs and not conditional_map:
        raise ValueError('Saved-state comparison requires explicit --conditional-map acknowledgement')
    for label, run in runs.items():
        report_path = checked(run, 'result.json', 64 * 1024**2)
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if not report.get('completed') or report['simplifications']['fracture'] is not False:
            raise ValueError('Only completed intact saved histories are supported')
        sources[str(report_path)] = digest(report_path)
        depths = np.array([s['depth_mm'] for s in report['steps']])
        if not np.all(np.diff(depths) > 0) or abs(depths[0]) > 1e-10:
            raise ValueError('Expected ordered accepted history starting at depth zero')
        base, state_path = load_state(run, 0, depths[0])
        sources[str(state_path)] = digest(state_path)
        sampler = P1Sampler(base['reference_xy'], base['triangles'])
        mapped, basis, metric_basis, _ = conditional_reference_points(p0, m, report['config'], pixel_aspect=pixel_aspect)
        metric = np.array([pixel_aspect, 1.])
        location = sampler.locate(mapped)
        # Explicit engineering perturbations, not measured confidence limits.
        locations = []
        for sf in (.95, 1., 1.05):
            for shift in (-.1, 0., .1):
                points, _, _, _ = conditional_reference_points(p0, m, report['config'], sf, shift, pixel_aspect)
                loc = sampler.locate(points)
                locations.append((sf, shift, loc))
        shared = np.logical_and.reduce([loc[0] >= 0 for _, _, loc in locations])
        pairs = initial_near_far_pairs(p0, regions, shared)
        baseline_points = sampler.sample(base['positions'], location)
        points = []
        for item in observations:
            index = int(np.argmin(abs(depths - item['conditional_depth_mm'])))
            if abs(depths[index] - item['conditional_depth_mm']) > .005:
                raise ValueError('No close actual accepted depth; temporal interpolation prohibited')
            state, path = load_state(run, index, depths[index])
            sources[str(path)] = digest(path)
            if not (np.array_equal(state['reference_xy'], base['reference_xy'])
                    and np.array_equal(state['triangles'], base['triangles'])):
                raise ValueError('Changing reference mesh requires a different observation operator')
            field = (state['positions'] - base['positions']) @ basis.T
            sampled = sampler.sample(field, location)
            current_fe_points = sampler.sample(state['positions'], location)
            summary = region_summary(sampled[shared], regions[shared])
            alternatives = []
            for sf, shift, loc in locations:
                values = sampler.sample(field, loc)
                base_alt = sampler.sample(base['positions'], loc)
                now_alt = sampler.sample(state['positions'], loc)
                alternatives.append(dict(scale_factor=sf, anchor_shift_s_mm=shift,
                                         local=region_summary(values[shared], regions[shared]),
                                         chords={side: chord_summary(base_alt, now_alt, edges)
                                                 for side, edges in pairs.items()}))
            exp = dict(item)
            now = table[(table.frame_id == item['frame_id']) & table.point_id.isin(cohort)].set_index('point_id').loc[cohort]
            exp['common_mapped_local_px'] = region_summary(
                (((now[['x_px', 'y_px']].to_numpy() - pc) * metric) @ metric_basis.T)[shared], regions[shared])
            exp['chords'] = {side: chord_summary(pc * metric, now[['x_px', 'y_px']].to_numpy() * metric, edges)
                             for side, edges in pairs.items()}
            points.append(dict(experimental=exp, actual_fe_depth_mm=float(depths[index]),
                               fe_fz_n=report['steps'][index]['reaction_fz_n'], local=summary,
                               chords={side: chord_summary(baseline_points, current_fe_points, edges)
                                       for side, edges in pairs.items()},
                               engineering_mapping_perturbations=alternatives))
        final_blade = (observations[-1]['blade_xy_px'] * metric) @ metric_basis.T
        rec['comparison'][label] = dict(config=report['config'], mapped_inside_count=int((location[0] >= 0).sum()),
            common_mapping_count=int(shared.sum()), region_counts={name: int(((regions == name) & shared).sum()) for name in REGIONS},
            experiment_blade_local_unit=(final_blade / np.linalg.norm(final_blade)).tolist(),
            fe_blade_local_unit=[float(np.sin(np.deg2rad(report['config']['inclination_deg']))),
                                 float(-np.cos(np.deg2rad(report['config']['inclination_deg'])))],
            mapping_status='Conditional surface alignment; lab/camera pose and mechanical zero unconfirmed; not FE validation',
            conditional_pixel_metric_xy=metric.tolist(),
            perturbation_status='±5% isotropic scale and ±.1 mm surface anchor are engineering probes, not uncertainty bounds',
            chord_definition='Each surviving initial near point connected to its nearest surviving initial far point, endpoints fixed at frame 0',
            chord_pair_point_ids={side: [[cohort[a], cohort[b]] for a, b in edges] for side, edges in pairs.items()},
            milestones=points)
    return rec


def plots(out, records, force_reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(records), 2, figsize=(12, 4.3 * len(records)), squeeze=False, layout='constrained')
    for row, rec in enumerate(records):
        for side, color in [('left', '#0072B2'), ('right', '#D55E00')]:
            obs = rec['experimental']
            x = np.array([a['blade_xy_px'][1] for a in obs])
            y = np.array([region_contrasts(a['local'], 1)[side] for a in obs], dtype=float)
            axes[row, 0].plot(x / x[-1], y / y[-1], marker='o', color=color, label=f'Image {side}')
            for label, comparison in rec['comparison'].items():
                pts = comparison['milestones']
                fe = np.array([region_contrasts(a['local'], 1)[side] for a in pts], dtype=float)
                e = np.array([region_contrasts(a['experimental']['common_mapped_local_px'], 1)[side] for a in pts], dtype=float)
                style = '--' if label == 'alpha2' else ':'
                axes[row, 1].plot([a['actual_fe_depth_mm'] for a in pts], fe / fe[-1],
                                  linestyle=style, color=color, label=f'{label} {side}')
                if label == next(iter(rec['comparison'])):
                    axes[row, 1].plot([a['experimental']['conditional_depth_mm'] for a in pts], e / e[-1],
                                      marker='o', color=color, label=f'Image common {side}')
        for column, ax in enumerate(axes[row]):
            ax.plot([0, 1], [0, 1], color='.6', lw=.8, label='Linear reference')
            ax.set(title=rec['case_id'] + (' : fixed survivor cohort' if column == 0 else ' : CONDITIONAL spatial mapping'),
                   xlabel='Blade travel / final travel' if column == 0 else 'FE or conditional image depth (mm)',
                   ylabel='Near-minus-far normal motion / final value')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle('Five actual image milestones / saved accepted FE states; no temporal interpolation\nShape comparison, not lab registration, force-video alignment or experimental validation')
    fig.savefig(out / 'motion_comparison.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(len(records), 2, figsize=(12, 4.1 * len(records)), squeeze=False, layout='constrained')
    for row, rec in enumerate(records):
        for column, side in enumerate(('left', 'right')):
            ax = axes[row, column]
            for label, comparison in rec['comparison'].items():
                points = comparison['milestones']
                ax.plot([a['actual_fe_depth_mm'] for a in points], [a['chords'][side]['median'] for a in points],
                        '--' if label == 'alpha2' else ':', label=label)
                if label == next(iter(rec['comparison'])):
                    ax.plot([a['experimental']['conditional_depth_mm'] for a in points],
                            [a['experimental']['chords'][side]['median'] for a in points], 'o-', label='Conditional image pairs')
            ax.axhline(1., color='.6', lw=.7)
            ax.set(title=f"{rec['case_id']} : {side} initial near/far pairs", xlabel='FE / conditional image depth (mm)',
                   ylabel='Median chord length / contact-image length')
            ax.grid(alpha=.2)
            ax.legend()
    aspects = sorted({c['conditional_pixel_metric_xy'][0] for r in records for c in r['comparison'].values()})
    fig.suptitle('Fixed material endpoint pairs: common translation/rotation cannot change chord length\n'
                 f'Conditional x/y pixel metric = {aspects}; nonlocal stretch, not strain/stress or fracture identification')
    fig.savefig(out / 'chord_comparison.png', dpi=170)
    plt.close(fig)
    if force_reference is not None:
        points = pd.DataFrame(force_reference['points'])
        final = float(points.force_n.iloc[-1])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
        for label, curve in force_reference['models'].items():
            depth = np.array([s['depth_mm'] for s in curve])
            force = np.array([s['reaction_fz_n'] for s in curve])
            axes[0].plot(depth, force, label=f'FE {label}')
            axes[1].plot(depth, force / force[-1], label=f'FE {label}')
        errors = np.stack([points.force_n - points.lower_n, points.upper_n - points.force_n])
        axes[0].errorbar(points.depth_mm, points.force_n, yerr=errors, fmt='ko', capsize=4,
                         label='Image estimates / visual ranges')
        # Range ratios conservatively use independent displayed bounds; not a CI.
        norm = points.force_n.to_numpy() / final
        low = points.lower_n.to_numpy() / float(points.upper_n.iloc[-1])
        high = points.upper_n.to_numpy() / float(points.lower_n.iloc[-1])
        low[-1] = high[-1] = 1.  # same endpoint divided by itself, not two independent values
        axes[1].errorbar(points.depth_mm, norm, yerr=[norm-low, high-norm], fmt='ko', capsize=4,
                         label='Image estimates / range ratios')
        for ax in axes:
            ax.set_xlabel('Downward depth (mm)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].set_ylabel('Fz (N)')
        axes[1].set_ylabel('Fz / force at final shown depth')
        fig.suptitle('User force-depth image estimates vs actual accepted FE states\n'
                     'Manual visual ranges, not raw samples or confidence intervals; no fit / no optical-flow synchronization')
        fig.savefig(out / 'force_image_comparison.png', dpi=170)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exchange-root', type=Path, required=True)
    parser.add_argument('--snapshot-root', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--alpha2-run', type=Path)
    parser.add_argument('--alpha1-run', type=Path)
    parser.add_argument('--conditional-map', action='store_true')
    parser.add_argument('--pixel-aspect', type=float, default=1.,
                        help='Explicit conditional x/y image metric probe; never a calibrated camera transform')
    parser.add_argument('--force-image-estimates', type=Path, help='Local manual force-depth readings from user images; never exchange sensor CSV')
    parser.add_argument('--force-image-provenance', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {str(Path(__file__).resolve()): digest(__file__)}
    runs = {k: v.resolve() for k, v in [('alpha2', args.alpha2_run), ('alpha1', args.alpha1_run)] if v}
    records = [analyze_case(args.exchange_root, args.snapshot_root, case, runs, args.conditional_map, sources, args.pixel_aspect)
               for case in args.cases]
    force_reference = None
    if args.force_image_estimates:
        if not args.force_image_provenance:
            raise ValueError('Screenshot estimates require explicit provenance')
        image_path = checked(args.force_image_estimates.parent, args.force_image_estimates.name, 1024**2)
        provenance_path = checked(args.force_image_provenance.parent, args.force_image_provenance.name, 1024**2)
        points = validate_image_estimates(pd.read_csv(image_path))
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        if provenance.get('reading_csv_sha256') and provenance['reading_csv_sha256'] != digest(image_path):
            raise ValueError('Screenshot reading file does not match its provenance')
        sources[str(image_path.resolve())] = digest(image_path)
        sources[str(provenance_path.resolve())] = digest(provenance_path)
        force_reference = dict(kind='Manual readings from user screenshots; not sensor CSV or raw experimental samples',
                               points=points.to_dict(orient='records'), provenance=provenance,
                               models={label: [dict(depth_mm=s['depth_mm'], reaction_fz_n=s['reaction_fz_n'])
                                               for s in json.loads(checked(run, 'result.json').read_text(encoding='utf-8'))['steps']
                                               if s['depth_mm'] <= points.depth_mm.iloc[-1] + 1e-10]
                                       for label, run in runs.items()})
    result = dict(scope='Experimental feature screening and conditional saved-state P1 comparison; no new solve or fit',
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        source_sha256=sources, cases=records, force_reference=force_reference,
        excluded_source="Exchange V01/V02/V03 and original sensor CSV: user instruction",
        limitations=['Screenshot readings have visual ranges, not statistical confidence intervals', 'No force-video synchronization',
                     'No certified intact time interval', 'Free visible side is not a central plane-strain slice',
                     'Cohort spatial spread is not measurement uncertainty', 'Region differences cancel translation but not rotation or perspective'])
    (args.out / 'analysis.json').write_text(json.dumps(native(result), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    plots(args.out, records, force_reference)
    print(json.dumps(dict(case_count=len(records), force_source='user screenshots only', new_equilibrium_solves=0)))


if __name__ == '__main__':
    main()
