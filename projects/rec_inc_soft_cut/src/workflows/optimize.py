"""Reproducible research candidates; never overwrite or promote production tracks.

Synthetic image warps provide geometric truth. Real-video pilots provide image
consistency and observability diagnostics only, not physical accuracy labels.
The existing tracker is the initializer, not ground truth for the candidate.
"""
from __future__ import annotations

import argparse
import html
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from src.io import ROOT, clean, load_npz, read_json, sha, write_json
from src.tracking.material_patch_tracker import MaterialPatchTracker
from src.tracking.finite_deformation import FiniteDeformationConfig, fit_material_mesh
from src.analysis.surface_observability import SurfaceObservabilityConfig, contact_from_material_surface


def depth_prefix(frames, tips, scale_um_per_px, stop_mm):
    """Dense loading prefix ending BEFORE first crossing; never drop fluctuations."""
    frames, tips = np.asarray(frames), np.asarray(tips, float)
    if (frames.ndim != 1 or len(frames) < 2 or tips.shape != (len(frames), 2)
            or not np.isfinite(scale_um_per_px) or scale_um_per_px <= 0
            or not np.isfinite(stop_mm) or stop_mm <= 0):
        raise ValueError('Need finite dense registered geometry and a positive cutoff/scale')
    depth = (tips[:, 1]-tips[0, 1])*scale_um_per_px/1000
    crossed = np.flatnonzero(depth > stop_mm)
    end = int(crossed[0]) if len(crossed) else len(depth)
    checked = min(end+1, len(frames))  # the crossing itself must be supported geometrically
    if not np.isfinite(tips[:checked]).all() or not np.all(np.diff(frames[:checked]) == 1):
        raise ValueError('Need finite dense registered geometry through the first cutoff crossing')
    if end < 2:
        raise ValueError('Cutoff leaves insufficient loading observations')
    return frames[:end], tips[:end], depth[:end]


def aligned_rows(source_frames, requested):
    source = np.asarray(source_frames)
    if source.ndim != 1 or np.any(np.diff(source) <= 0):
        raise ValueError('Source frame mapping is not strictly ordered')
    rows = np.searchsorted(source, requested)
    if np.any(rows >= len(source)) or not np.array_equal(source[rows], requested):
        raise ValueError('Requested frame absent from saved source mapping')
    return rows


def spread_subset(points, maximum):
    """Deterministic farthest-point subset; selection never uses fit accuracy."""
    p = np.asarray(points, float)
    if maximum < 6 or not np.isfinite(p).all():
        raise ValueError('At least six finite mesh nodes are required')
    if len(p) <= maximum:
        return np.arange(len(p))
    first = int(np.argmin(np.linalg.norm(p-np.mean(p, axis=0), axis=1)))
    selected = [first]
    distance = np.sum((p-p[first])**2, axis=1)
    for _ in range(maximum-1):
        distance[selected] = -1
        j = int(np.argmax(distance))
        selected.append(j)
        distance = np.minimum(distance, np.sum((p-p[j])**2, axis=1))
    return np.asarray(selected)


def _path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def inspect_config(config):
    FiniteDeformationConfig(**config.get('candidate', {}))
    SurfaceObservabilityConfig(**config.get('surface', {}))
    if not 6 <= int(config.get('maximum_nodes', 100)) <= 1000:
        raise ValueError('Pilot maximum_nodes must be between 6 and 1000')
    if int(config.get('reference_lag_frames', 12)) < 1:
        raise ValueError('Reference must precede current frame')
    if float(config.get('near_tip_radius_px', 100)) <= 0:
        raise ValueError('Near-tip radius must be positive')
    for key, default in [('pilot_depths_mm', [.25, .5, .75, 1.]),
                         ('surface_pilot_depths_mm', [.75, 1.])]:
        depths = np.asarray(config.get(key, default), float)
        if (depths.ndim != 1 or not len(depths) or not np.isfinite(depths).all()
                or np.any(depths < 0) or (key == 'pilot_depths_mm' and np.any(depths == 0))):
            raise ValueError('Pilot depths must be finite and positive; surface reference may be zero')
    models = config.get('surface_models', ['affine', 'surface-quadratic'])
    if not models or len(set(models)) != len(models):
        raise ValueError('Surface models must be a nonempty unique list')
    for model in models:
        SurfaceObservabilityConfig(**dict(config.get('surface', {}), model=model))
    names, inspected = set(), []
    for case in config.get('cases', []):
        name = case['name']
        if not name or name in ('.', '..') or Path(name).name != name or '/' in name or '\\' in name or name in names:
            raise ValueError('Unique local case stems required')
        names.add(name)
        if case.get('role') not in ('development_only', 'held_out_validation'):
            raise ValueError('Declare development or held-out role')
        if not np.isfinite(case['stop_mm']) or case['stop_mm'] <= 0:
            raise ValueError('Explicit positive analysis cutoff required')
        td, md, prep = (_path(case[k]) for k in ('tracks', 'measurements', 'prepared'))
        files = [td/'tracking/tracks.npz', td/'queries.npz', td/'plan.json',
                 td/'frame_mapping.npz', td/'normalized_lossless.mkv',
                 md/'measurement_support.npz', md/'initial_surface_extended.npz',
                 prep/'input.npz', prep/'blade_reference_guard.npz', _path(case['contact_audit'])]
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise FileNotFoundError('Missing pilot inputs: '+', '.join(missing))
        inspected.append(dict(name=name, role=case['role'], paths=files))
    return inspected


def _frame(video, frame):
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = cap.read()
        if not ok:
            raise ValueError(f'Cannot decode frame {frame}: {video}')
        return image
    finally:
        cap.release()


def _mask(folder, name, frame):
    p = folder/'segmentation'/name/'semantic_masks'/f'f{int(frame):06d}.png'
    mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise ValueError(f'Missing exact-frame segmentation: {p}')
    return mask


def _save_result(path, result, extra=None):
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    arrays.update(extra or {})
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    write_json(path.with_suffix('.json'), {k: v for k, v in result.items()
                                         if not isinstance(v, np.ndarray)})


def _preview(image, before, after, tip, filename, surfaces=(None, None), supports=(None, None),
             titles=('Saved baseline', 'Research candidate (not ground truth)')):
    canvases = []
    for (title, p, color), surface, supported in zip(
            [(titles[0], before, (220, 220, 220)),
             (titles[1], after, (250, 220, 40))], surfaces, supports):
        view = image.copy()
        points_layer = view.copy()
        for i in np.flatnonzero(np.isfinite(p).all(1)):
            observed = supported is None or supported[i]
            cv2.circle(points_layer, tuple(np.rint(p[i]).astype(int)), 1,
                       color if observed else (145, 145, 145), -1 if observed else 1, cv2.LINE_AA)
        view = cv2.addWeighted(points_layer, .55, view, .45, 0)
        cv2.drawMarker(view, tuple(np.rint(tip).astype(int)), (0, 180, 255), cv2.MARKER_CROSS, 18)
        if surface is not None:
            for a, b, valid in zip(surface['mapped_xy_px'][:-1], surface['mapped_xy_px'][1:], surface['segment_valid']):
                if valid and np.isfinite([a, b]).all():
                    cv2.line(view, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), (210, 110, 220), 1)
        x, y = np.rint(tip).astype(int)
        radius = 160
        lo = np.maximum([x-radius, y-radius], 0)
        hi = np.minimum([x+radius, y+radius], [image.shape[1], image.shape[0]])
        crop = view[lo[1]:hi[1], lo[0]:hi[0]]
        crop = cv2.resize(crop, (480, 480))
        canvas = np.full((522, 480, 3), 22, np.uint8)
        canvas[42:] = crop
        cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .44, (230, 230, 230), 1)
        canvases.append(canvas)
    cv2.imwrite(str(filename), np.hstack(canvases))


def synthetic_benchmark(out, config):
    """Known image deformation, never fitted to microscopy target offsets."""
    rng = np.random.default_rng(config.get('seed', 20261024))
    image = cv2.GaussianBlur(rng.uniform(.05, .95, (180, 220)).astype(np.float32), (3, 3), .5)
    yy, xx = np.meshgrid(np.arange(45, 141, 16), np.arange(55, 166, 16), indexing='ij')
    points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    full = np.ones(image.shape, bool)
    theta = np.deg2rad(18)
    scenarios = [
        ('translation', np.eye(2), np.array([4., -2.]), 'none'),
        ('stretch_shear', np.array([[1.35, .16], [-.06, .8]]), np.array([-32., 18.]), 'none'),
        ('rotation', np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([32., -24.]), 'none'),
        ('shadow_gain', np.array([[1.12, .1], [.03, .9]]), np.array([-15., 8.]), 'shadow'),
        ('moving_glint', np.array([[1.1, .03], [-.02, .94]]), np.array([-5., 5.]), 'glint'),
        ('occlusion', np.array([[1.15, .06], [-.01, .9]]), np.array([-12., 8.]), 'occlusion'),
    ]
    output = []
    for name, affine, offset, interference in scenarios:
        tracker = MaterialPatchTracker(image, full, points)
        began = time.perf_counter()
        for k in range(1, 13):
            f = k/12
            A = np.eye(2)+(affine-np.eye(2))*f
            target = cv2.warpAffine(image, np.c_[A, offset*f], image.shape[::-1])
            mask = full.copy()
            blade = np.zeros_like(mask)
            if interference == 'shadow':
                target *= (.55+.35*np.linspace(0, 1, image.shape[1]))[None, :]
                target += .04
            elif interference == 'glint':
                cv2.line(target, (80+k, 35), (88+k, 145), .99, 4)
            elif interference == 'occlusion':
                blade[60:120, 95:126] = True
                target[blade] = .98
                mask[blade] = False
            baseline = tracker.step(target, mask, blade_mask=blade)
        baseline_seconds = time.perf_counter()-began
        truth = points@affine.T+offset
        began = time.perf_counter()
        candidate = fit_material_mesh(image, target, points, baseline['positions'],
            baseline['supported'], material_mask=(full, mask), blade_mask=blade,
            config=FiniteDeformationConfig(**config.get('candidate', {})))
        record = dict(scenario=name, ground_truth='known affine image warp',
                      baseline_seconds_12_steps=baseline_seconds,
                      candidate_seconds_single_refinement=time.perf_counter()-began,
                      comparison_scope='Same baseline initializer; candidate is an optional final-image mesh refinement')
        for label, result in [('baseline', baseline), ('candidate', candidate)]:
            error = np.linalg.norm(result['positions']-truth, axis=1)
            ok = result['supported']
            record[label] = dict(supported_fraction=float(ok.mean()),
                epe_all_finite_px=float(np.mean(error[np.isfinite(error)])),
                epe_supported_px=float(np.mean(error[ok])) if ok.any() else None,
                p95_supported_px=float(np.percentile(error[ok], 95)) if ok.any() else None,
                false_supported_over_1px=int(np.sum(ok & (error > 1.))))
        record['candidate_accepted'] = bool(candidate['accepted'])
        record['optimizer_converged'] = bool(candidate['optimizer_converged'])
        common = baseline['supported'] & candidate['supported']
        record['common_supported_count'] = int(common.sum())
        record['common_supported_epe_px'] = {label: float(np.mean(np.linalg.norm(
            result['positions'][common]-truth[common], axis=1))) if common.any() else None
            for label, result in [('baseline', baseline), ('candidate', candidate)]}
        record['holdout'] = candidate['holdout']
        _save_result(out/f'synthetic_{name}', candidate,
                     dict(truth=truth, reference_points=points, baseline_positions=baseline['positions'],
                          baseline_supported=baseline['supported']))
        output.append(record)
        print(f'Synthetic {name}: {clean(record)}', flush=True)
    write_json(out/'synthetic_summary.json', output)
    return output


def pilot_case(case, config, out):
    """Image-only local refinements, with unchanged source positions and cutoffs."""
    dest = out/case['name']
    dest.mkdir()
    td, md, prep = (_path(case[k]) for k in ('tracks', 'measurements', 'prepared'))
    guard = load_npz(prep/'blade_reference_guard.npz')
    with np.load(prep/'input.npz', allow_pickle=False) as z:
        scale = float(z['scale'])
    frames, tips, depth = depth_prefix(guard['source_frames'], guard['tips_xy'], scale, case['stop_mm'])
    audit = read_json(_path(case['contact_audit']))
    if not audit.get('accepted') or audit['selected_contact_frame'] != int(frames[0]):
        raise ValueError('Prepared contact zero and accepted audit disagree')
    tracks = load_npz(td/'tracking/tracks.npz')
    support_data = load_npz(md/'measurement_support.npz')
    surface = load_npz(md/'initial_surface_extended.npz')['boundary_xy']
    if not np.array_equal(tracks['point_ids'], support_data['point_ids']):
        raise ValueError('Measurement and tracking IDs disagree')
    support = support_data['supported'][aligned_rows(support_data['frame_indices'], frames)]
    rows = aligned_rows(tracks['source_frame_indices'], frames)
    positions = tracks['positions'][rows]
    seeds, ids = tracks['seed_points'], tracks['point_ids']
    records = []
    mask_files = set()
    used = set()
    for requested in config['pilot_depths_mm']:
        if requested > case['stop_mm']:
            continue
        crossing = np.flatnonzero(depth > requested)
        i = int(crossing[0]-1) if len(crossing) else len(depth)-1
        if i <= 0 or i in used:
            continue
        used.add(i)
        j = max(0, i-int(config['reference_lag_frames']))
        f, ref = int(frames[i]), int(frames[j])
        if guard['state'][i] != 2 or guard['state'][j] != 2:
            raise ValueError('Pilot needs observed registered blade coordinates')
        before, after = positions[j], positions[i]
        near = np.linalg.norm(after-tips[i], axis=1) <= config['near_tip_radius_px']
        eligible = np.flatnonzero(near & np.isfinite(before).all(1) & np.isfinite(after).all(1))
        if len(eligible) < 6:
            raise ValueError('Too few finite near-tip material nodes')
        selected = eligible[spread_subset(before[eligible], int(config['maximum_nodes']))]
        a, b = (_frame(td/'normalized_lossless.mkv', v) for v in (ref, f))
        ma, mb = (_mask(td, case['name'], v) for v in (ref, f))
        mask_files.update(td/'segmentation'/case['name']/'semantic_masks'/f'f{v:06d}.png' for v in (ref, f))
        observed = support[i, selected] & support[j, selected]
        began = time.perf_counter()
        candidate = fit_material_mesh(a, b, before[selected], after[selected], observed,
            material_mask=(ma == 1, mb == 1), blade_mask=(ma == 2, mb == 2),
            config=FiniteDeformationConfig(**config.get('candidate', {})))
        seconds = time.perf_counter()-began
        _save_result(dest/f'frame_{f:06d}_mesh', candidate,
                     dict(point_ids=ids[selected], reference_positions=before[selected],
                          baseline_positions=after[selected], source_frame=np.array(f), reference_frame=np.array(ref)))
        # Non-mesh nodes are untouched. A subset refinement cannot be presented
        # as a full-video retrack. Model completion never becomes an observation.
        revised = after.copy()
        revised[selected] = candidate['positions']
        qualified = support[i].copy()
        qualified[selected] &= candidate['supported']
        base_contact = new_contact = dict(state='not_run', surface=None)
        if config.get('pilot_surface_mapping', True):
            base_contact = contact_from_material_surface(seeds, after, support[i], surface, tips[i],
                point_ids=ids, blade_mask=mb == 2, observation_mask=mb == 1,
                config=SurfaceObservabilityConfig(**config.get('surface', {})))
            new_contact = contact_from_material_surface(seeds, revised, qualified, surface, tips[i],
                point_ids=ids, blade_mask=mb == 2, observation_mask=mb == 1,
                config=SurfaceObservabilityConfig(**config.get('surface', {})))
        for label, result in [('baseline', base_contact), ('candidate', new_contact)]:
            if result['surface'] is not None:
                _save_result(dest/f'frame_{f:06d}_{label}_surface', result['surface'])
            write_json(dest/f'frame_{f:06d}_{label}_contact.json',
                       {k: v for k, v in result.items() if k != 'surface'})
        record = dict(case=case['name'], role=case['role'], frame=f, reference_frame=ref,
            requested_depth_mm=requested, actual_depth_mm=float(depth[i]),
            scale_um_per_px=scale, scale_status='user approximate 3 mm stroke, motion-onset contact candidate',
            selected_nodes=len(selected), total_ids=len(ids), input_observed_fraction=float(observed.mean()),
            candidate_supported_fraction=float(candidate['supported'].mean()),
            candidate_seconds=seconds, candidate_accepted=bool(candidate['accepted']),
            optimizer_converged=bool(candidate['optimizer_converged']),
            total_triangles=len(candidate['triangles']),
            eligible_triangles=int(candidate['eligible_triangles'].sum()),
            image_eligible_triangles=int(candidate['image_eligible_triangles'].sum()),
            image_covered_nodes=int(len(np.unique(candidate['triangles'][candidate['image_eligible_triangles']]))),
            eligible_minimum_jacobian=float(np.min(candidate['jacobian'][candidate['eligible_triangles']]))
                if candidate['eligible_triangles'].any() else None,
            deformation_reference='Incremental map from reference_frame, not cumulative initial deformation',
            observation_mask_basis='Semantic material mask; texture visibility not independently verified',
            candidate_holdout=candidate['holdout'],
            displacement_change_median_px=float(np.median(np.linalg.norm(candidate['positions']-after[selected], axis=1))),
            contact={label: {k: v for k, v in result.items() if k not in ('surface',)}
                     for label, result in [('baseline', base_contact), ('candidate', new_contact)]},
            physical_accuracy='unknown; image residual and tracking agreement are not truth',
            future_images_read=False, previous_flow_historical_dependencies_unchanged=True)
        _preview(b, after[selected], revised[selected], tips[i], dest/f'frame_{f:06d}.png',
                 (base_contact['surface'], new_contact['surface']),
                 (support[i, selected], candidate['supported']))
        records.append(record)
        write_json(dest/'summary.json', records)
        print(f'{case["name"]}: frame {f}, {depth[i]:.4f} mm, {seconds:.2f}s, accepted={candidate["accepted"]}', flush=True)
    fingerprints = {str(p): sha(p) for p in [td/'tracking/tracks.npz', md/'measurement_support.npz',
        md/'initial_surface_extended.npz', td/'queries.npz', td/'plan.json', td/'frame_mapping.npz',
        prep/'input.npz', prep/'blade_reference_guard.npz', _path(case['contact_audit']), *sorted(mask_files)]}
    write_json(dest/'provenance.json', dict(case=case, source_sha256=fingerprints,
        baseline_unchanged=True, immutable_ids=True, promotes_candidate=False,
        analysis_cutoff_mm=case['stop_mm'], contact_frame=int(frames[0]),
        endpoint_frame=int(frames[-1]), endpoint_estimate_mm=float(depth[-1])))
    return records


def report(out, results):
    rows = []
    for s in results.get('synthetic', []):
        fmt = lambda x: 'unavailable' if x is None else f'{x:.3f}'
        pair = lambda a, b: f'{fmt(a)} / {fmt(b)}'
        rows.append(f'<tr><td>{html.escape(s["scenario"])}</td>'
                    f'<td>{pair(s["baseline"]["epe_all_finite_px"], s["candidate"]["epe_all_finite_px"])}</td>'
                    f'<td>{pair(s["common_supported_epe_px"]["baseline"], s["common_supported_epe_px"]["candidate"])}</td>'
                    f'<td>{pair(s["baseline"]["supported_fraction"], s["candidate"]["supported_fraction"])}</td>'
                    f'<td>{s["candidate_accepted"]}</td></tr>')
    body = '<h1>Material tracking optimization pilot</h1><p>Baseline preserved. No candidate promoted. Synthetic EPE has known image-warp truth; real-video residuals do not establish physical accuracy. Previews show a selected node subset; hollow gray circles are unsupported.</p>'
    body += '<p>Each pair: baseline / candidate. Coverage and common-point error are reported together to expose selection effects.</p><table><tr><th>Synthetic case</th><th>All-node EPE (px)</th><th>Common-supported EPE (px)</th><th>Supported fraction</th><th>Numerically accepted</th></tr>'+''.join(rows)+'</table>'
    for r in results.get('pilot', []):
        name, f = r['case'], r['frame']
        body += f'<h2>{html.escape(name)} | {r["actual_depth_mm"]:.3f} mm</h2><p>Local mesh: {r["selected_nodes"]} nodes. Fit acceptance: {r["candidate_accepted"]}. Physical accuracy unknown.</p>'
        body += f'<img src="{html.escape(name)}/frame_{f:06d}.png" width="960"><p><a href="{html.escape(name)}/summary.json">Diagnostics and conditional surface-address candidates</a></p>'
    for r in results.get('surface', []):
        body += f'<h2>{html.escape(r["case"])} | {r["actual_depth_mm"]:.3f} mm</h2><p>Same saved trajectories, independent surface models. Addresses are not automatically slip or observed contact.</p>'
        directory = html.escape(r.get('relative_directory', r['case']))
        body += '<table><tr><th>Model</th><th>State</th><th>Surface arc coverage</th><th>Tip gap (px)</th></tr>'
        for label in r.get('models', ['affine', 'surface_quadratic']):
            item = r[label]
            diag = item.get('diagnostics', {})
            gap = diag.get('nearest_valid_surface_tip_gap_px')
            body += (f'<tr><td><a href="{directory}/frame_{r["frame"]:06d}_{label}_contact.json">{html.escape(label)}</a></td>'
                     f'<td>{html.escape(item["state"])}</td><td>{diag.get("valid_surface_arc_fraction", 0):.1%}</td>'
                     f'<td>{gap if gap is not None else "unavailable"}</td></tr>')
        body += f'</table><img src="{directory}/frame_{r["frame"]:06d}_surface.png" width="960">'
    for mode in results.get('features', {}):
        body += f'<h2>{html.escape(mode)}</h2><a href="{mode}/index.html">Same inputs: image-feature ablation</a>'
    if results.get('feature_comparison'):
        body += '<p>Pairs: legacy / material-masked. Feature residuals use different normalization and cannot be compared as physical accuracy.</p><table><tr><th>Known warp</th><th>All-node EPE (px)</th><th>All-node P95 (px)</th><th>Common-point EPE (px)</th><th>Support</th><th>Supported error &gt; 1 px (count)</th></tr>'
        for r in results['feature_comparison']:
            cells = [html.escape(r['scenario'])]
            for key in ('epe_all_px', 'p95_all_px', 'epe_common_px', 'supported_fraction', 'false_supported_over_1px'):
                cells.append(' / '.join('unavailable' if r[m][key] is None else f'{r[m][key]:.4f}'
                                        for m in ('legacy', 'material-masked')))
            body += '<tr>'+''.join(f'<td>{c}</td>' for c in cells)+'</tr>'
        body += '</table>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Optimization pilot</title><style>body{font:16px Arial;margin:32px;max-width:1200px;background:#142028;color:#e4edf0}table{border-collapse:collapse}td,th{border:1px solid #647078;padding:10px}a{color:#87d9e4}img{max-width:100%}</style>'+body, encoding='utf-8')


def surface_diagnostics(mapped, tip, semantic_mask):
    """Geometry consistency only: semantic contours are not independent truth."""
    xy, valid = mapped['mapped_xy_px'], mapped['segment_valid']
    ds = np.diff(mapped['s_px'])
    gap = None
    if valid.any():
        a, b = xy[:-1][valid], xy[1:][valid]
        v = b-a
        u = np.clip(np.sum((tip-a)*v, axis=1)/np.maximum(np.sum(v*v, axis=1), 1.e-20), 0, 1)
        gap = float(np.min(np.linalg.norm(a+u[:, None]*v-tip, axis=1)))
    material = (semantic_mask == 1).astype(np.uint8)
    boundary = material-cv2.erode(material, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(1-boundary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    def distances(points):
        p = np.asarray(points)
        p = p[np.isfinite(p).all(1)]
        p = np.rint(p).astype(int)
        p = p[(p[:, 0] >= 0) & (p[:, 0] < distance.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < distance.shape[0])]
        return distance[p[:, 1], p[:, 0]] if boundary.any() else np.array([])
    near = xy[np.linalg.norm(xy-tip, axis=1) <= 100]
    residual, tip_residual = distances(near), distances(np.asarray(tip)[None])
    return dict(valid_surface_arc_fraction=float(ds[valid].sum()/ds.sum()),
        nearest_valid_surface_tip_gap_px=gap,
        near_tip_mapped_nodes=len(near),
        surface_to_semantic_edge_median_px=float(np.median(residual)) if len(residual) else None,
        surface_to_semantic_edge_p95_px=float(np.percentile(residual, 95)) if len(residual) else None,
        tip_to_semantic_edge_px=float(tip_residual[0]) if len(tip_residual) else None,
        contour_evidence='same segmentation pipeline; not independent image or physical ground truth')


def feature_comparison(out, runs):
    """Compare the SAME truth, initializers and nodes, including rejected ones."""
    records = []
    for item in runs['legacy']['synthetic']:
        name = item['scenario']
        values = {mode: load_npz(out/mode/f'synthetic_{name}.npz') for mode in runs}
        a, b = values['legacy'], values['material-masked']
        for key in ('truth', 'reference_points', 'baseline_positions', 'baseline_supported'):
            if not np.array_equal(a[key], b[key]):
                raise ValueError('Feature comparison requires identical truth and baseline inputs')
        common = a['supported'] & b['supported']
        record = dict(scenario=name, common_supported_count=int(common.sum()))
        for mode, z in values.items():
            error = np.linalg.norm(z['positions']-z['truth'], axis=1)
            record[mode] = dict(epe_all_px=float(error.mean()), p95_all_px=float(np.percentile(error, 95)),
                epe_common_px=float(error[common].mean()) if common.any() else None,
                supported_fraction=float(z['supported'].mean()),
                false_supported_over_1px=int(np.sum(z['supported'] & (error > 1))))
        records.append(record)
    return records


def surface_benchmark(case, config, out):
    """Separate mapping-model failure from image absence using the same tracks."""
    dest = out/case['name']
    dest.mkdir()
    td, md, prep = (_path(case[k]) for k in ('tracks', 'measurements', 'prepared'))
    guard = load_npz(prep/'blade_reference_guard.npz')
    with np.load(prep/'input.npz', allow_pickle=False) as z:
        scale = float(z['scale'])
    frames, tips, depth = depth_prefix(guard['source_frames'], guard['tips_xy'], scale, case['stop_mm'])
    audit = read_json(_path(case['contact_audit']))
    if not audit.get('accepted') or int(frames[0]) != audit['selected_contact_frame']:
        raise ValueError('Surface benchmark requires the same audited contact reference')
    tracks = load_npz(td/'tracking/tracks.npz')
    support = load_npz(md/'measurement_support.npz')
    if not np.array_equal(tracks['point_ids'], support['point_ids']):
        raise ValueError('Surface benchmark material ID mismatch')
    surface = load_npz(md/'initial_surface_extended.npz')['boundary_xy']
    records, used, masks = [], set(), []
    for target in config.get('surface_pilot_depths_mm', [.75, 1.]):
        if not 0 <= target <= case['stop_mm']:
            continue
        crossed = np.flatnonzero(depth > target)
        i = 0 if target == 0 else int(crossed[0]-1) if len(crossed) else len(depth)-1
        if i < 0 or i in used:
            continue
        used.add(i)
        f = int(frames[i])
        if guard['state'][i] != 2:
            raise ValueError('Surface benchmark tip is not observed/registered')
        row = aligned_rows(tracks['source_frame_indices'], [f])[0]
        support_row = aligned_rows(support['frame_indices'], [f])[0]
        mask = _mask(td, case['name'], f)
        masks.append(td/'segmentation'/case['name']/'semantic_masks'/f'f{f:06d}.png')
        record = dict(case=case['name'], role=case['role'], frame=f, actual_depth_mm=float(depth[i]),
                      fixed_tracking_input=True, physical_accuracy='unknown', point_count=len(tracks['point_ids']),
                      requested_depth_mm=target, models=[], scale_um_per_px=scale)
        mapped_models = {}
        for model in config.get('surface_models', ['affine', 'surface-quadratic']):
            label = model.replace('-', '_')
            record['models'].append(label)
            options = dict(config.get('surface', {}), model=model)
            began = time.perf_counter()
            result = contact_from_material_surface(tracks['seed_points'], tracks['positions'][row],
                support['supported'][support_row], surface, tips[i], point_ids=tracks['point_ids'],
                blade_mask=mask == 2, observation_mask=mask == 1, config=SurfaceObservabilityConfig(**options))
            _save_result(dest/f'frame_{f:06d}_{label}_surface', result['surface'])
            contact = {k: v for k, v in result.items() if k != 'surface'}
            write_json(dest/f'frame_{f:06d}_{label}_contact.json', contact)
            reasons, counts = np.unique(result['surface']['reason'], return_counts=True)
            mapped_models[label] = result['surface']
            record[label] = dict(state=result['state'], evidence=result['evidence'],
                candidates=result['candidates'], runtime_s=time.perf_counter()-began,
                valid_surface_nodes=int(np.isfinite(result['surface']['mapped_xy_px']).all(1).sum()),
                reason_counts=dict(zip(reasons, counts)), configuration=asdict(SurfaceObservabilityConfig(**options)),
                diagnostics=surface_diagnostics(result['surface'], tips[i], mask))
        labels = record['models']
        current = tracks['positions'][row]
        near = np.linalg.norm(current-tips[i], axis=1) <= 100
        image = _frame(td/'normalized_lossless.mkv', f)
        _preview(image, current[near], current[near], tips[i], dest/f'frame_{f:06d}_surface.png',
                 (mapped_models[labels[0]], mapped_models[labels[-1]]),
                 (support['supported'][support_row, near],)*2,
                 titles=(labels[0]+' | same saved IDs', labels[-1]+' | conditional surface'))
        records.append(record)
        write_json(dest/'summary.json', records)
        print(f'Surface {case["name"]} {depth[i]:.4f} mm: '+
              ' / '.join(f'{label}={record[label]["state"]}' for label in labels), flush=True)
    evidence = [td/'tracking/tracks.npz', md/'measurement_support.npz', md/'initial_surface_extended.npz',
                prep/'input.npz', prep/'blade_reference_guard.npz', _path(case['contact_audit']), *masks]
    write_json(dest/'provenance.json', dict(case=case, source_sha256={str(p): sha(p) for p in evidence},
        numerical_thresholds_relaxed=False, microscopy_target_used=False,
        observation_mask_basis='Semantic mask; photometric visibility not independently verified',
        arc_length_reference='First point of ORIGINAL surface, not P0; subtract a separately supported P0 to obtain slip',
        input_tracking_recomputed=False, historical_full_track_and_calibration_dependencies_unchanged=True))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/optimization.example.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--stage', choices=('synthetic', 'pilot', 'surface', 'features', 'reliability', 'identity', 'all'), default='all')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    config = read_json(args.config)
    inspected = inspect_config(config) if args.stage != 'synthetic' else []
    if args.stage == 'reliability':
        from .reliability import validate_config
        validate_config(config)
    if args.stage == 'identity':
        from .reliability import validate_identity_config
        validate_identity_config(config)
    FiniteDeformationConfig(**config.get('candidate', {}))
    if args.check_only:
        print(clean(dict(stage=args.stage, cases=inspected, default_tracker_changed=False)))
        return 0
    if args.out is None:
        parser.error('--out is required for computation')
    out = args.out.resolve()
    if not any(out.is_relative_to((ROOT/folder).resolve()) for folder in ('results', 'runs')):
        raise ValueError('Write research outputs only below repository results/ or runs/')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory; existing evidence is never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[2]
    write_json(out/'run_config.json', config)
    write_json(out/'execution.json', dict(python=sys.executable, stage=args.stage,
        candidate_config=asdict(FiniteDeformationConfig(**config.get('candidate', {}))),
        surface_config=asdict(SurfaceObservabilityConfig(**config.get('surface', {}))),
        optical_flow_default_changed=False, source_results_modified=False, model_training=False,
        input_config_sha256=sha(args.config), candidate_is_research_only=True,
        source_sha256={str(p.relative_to(source_root)): sha(p) for p in
            [source_root/'src/workflows/optimize.py', source_root/'src/tracking/finite_deformation.py',
             source_root/'src/analysis/surface_observability.py']}))
    if args.stage in ('reliability', 'identity'):
        from .reliability import run, run_identity
        execution = read_json(out/'execution.json')
        execution['source_sha256'].update({str(p.relative_to(source_root)): sha(p) for p in
            [source_root/'src/workflows/reliability.py', source_root/'src/tracking/material_patch_tracker.py',
             source_root/'src/tracking/neural_candidates.py', source_root/'src/analysis/blade_geometry_review.py']})
        write_json(out/'execution.json', execution)
        if args.stage == 'identity':
            execution['source_sha256']['src/tracking/identity_chain.py'] = sha(source_root/'src/tracking/identity_chain.py')
            write_json(out/'execution.json', execution)
        (run_identity if args.stage == 'identity' else run)(config, out)
        print(f'Report: {out / "index.html"}', flush=True)
        return 0
    results = dict(synthetic=[], pilot=[], surface=[], promoted_to_default=False)
    if args.stage in ('synthetic', 'all'):
        results['synthetic'] = synthetic_benchmark(out, config)
    if args.stage in ('pilot', 'all'):
        for case in config['cases']:
            results['pilot'].extend(pilot_case(case, config, out))
    if args.stage in ('surface', 'all'):
        surface_out = out if args.stage == 'surface' else out/'surface_curve_ablation'
        surface_out.mkdir(exist_ok=True)
        for case in config['cases']:
            records = surface_benchmark(case, config, surface_out)
            for record in records:
                record['relative_directory'] = (surface_out/record['case']).relative_to(out).as_posix()
            results['surface'].extend(records)
    if args.stage == 'features':
        results['features'] = {}
        for mode in ('legacy', 'material-masked'):
            variant = dict(config, candidate=dict(config.get('candidate', {}), feature_mode=mode),
                           pilot_surface_mapping=False)
            folder = out/mode
            folder.mkdir()
            write_json(folder/'run_config.json', variant)
            run = dict(synthetic=synthetic_benchmark(folder, variant), pilot=[], promoted_to_default=False)
            for case in config['cases']:
                run['pilot'].extend(pilot_case(case, variant, folder))
            write_json(folder/'summary.json', run)
            report(folder, run)
            results['features'][mode] = run
        results['feature_comparison'] = feature_comparison(out, results['features'])
    write_json(out/'summary.json', results)
    report(out, results)
    print(f'Report: {out / "index.html"}', flush=True)
    return 0
