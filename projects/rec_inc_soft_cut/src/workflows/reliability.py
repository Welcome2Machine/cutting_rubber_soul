"""Bounded, reproducible evaluation of the four material-observability branches.

All inputs remain immutable. Network visibility, repeated image agreement and
mechanical regularity have different meanings; none is physical ground truth.
"""
from __future__ import annotations

import html
import shutil
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from src.io import ROOT, load_npz, read_json, sha, write_json
from src.tracking.finite_deformation import FiniteDeformationConfig, fit_material_mesh
from src.tracking.material_patch_tracker import verify_material_candidates
from .optimize import (_frame, _mask, _path, _preview, _save_result,
                       aligned_rows, depth_prefix, spread_subset)


def validate_config(config):
    names = config.get('network_backends', [])
    if (not names or len(set(names)) != len(names)
            or any(n not in ('cotracker3', 'tapnextpp') for n in names)):
        raise ValueError('Explicit unique supported neural candidate backends required')
    for key in ('identity_agreement_px', 'original_surface_band_px', 'mechanics_weight'):
        if not np.isfinite(config.get(key, np.nan)) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key, lo, hi in [('network_crop_side_px', 64, 2048), ('maximum_identity_points', 6, 1000)]:
        v = config.get(key, np.nan)
        if not np.isfinite(v) or int(v) != v or not lo <= v <= hi:
            raise ValueError(f'{key} must be an integer in [{lo},{hi}]')
    ratios = np.asarray(config.get('mechanics_bulk_shear_ratios', []), float)
    if ratios.ndim != 1 or not len(ratios) or not np.isfinite(ratios).all() or np.any(ratios <= 2/3):
        raise ValueError('Explicit positive finite bulk/shear sensitivity ratios > 2/3 required')
    if config['reference_lag_frames'] > 63:
        raise ValueError('Candidate window is limited to 64 frames')


def identity_consensus(original, recent, baseline_supported, tolerance_px=1.5):
    """Two reference checks of the SAME ID, never nearest-neighbour relabelling."""
    if not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError('Positive finite reference agreement tolerance required')
    if not np.array_equal(original['point_ids'], recent['point_ids']):
        raise ValueError('Reference checks have different material identities')
    ids = np.asarray(original['point_ids'])
    baseline_supported = np.asarray(baseline_supported, bool)
    if (ids.ndim != 1 or len(np.unique(ids)) != len(ids) or baseline_supported.shape != ids.shape
            or any(np.asarray(r['positions']).shape != (len(ids), 2)
                   or np.asarray(r['accepted']).shape != ids.shape for r in (original, recent))):
        raise ValueError('Unique IDs and aligned N,2 positions / N support arrays required')
    agreement = np.linalg.norm(original['positions']-recent['positions'], axis=1)
    both = original['accepted'] & recent['accepted'] & (agreement <= tolerance_px)
    return dict(point_ids=original['point_ids'], positions=original['positions'],
                identity_consistent=both, recovered=both & ~np.asarray(baseline_supported, bool),
                recent_only=recent['accepted'] & ~both, reference_disagreement_px=agreement,
                evidence='Double-reference image consistency, not independently observed material truth')


def _window(video, first, last, lower, upper):
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (not np.isfinite([first, last]).all() or first < 0 or last <= first
            or int(first) != first or int(last) != last
            or lower.shape != (2,) or upper.shape != (2,)
            or not np.isfinite([lower, upper]).all() or np.any(lower < 0)
            or np.any(upper <= lower) or np.any(lower != np.floor(lower))
            or np.any(upper != np.floor(upper))):
        raise ValueError('Ordered integer frames and positive in-image crop required')
    first, last, lower, upper = int(first), int(last), lower.astype(int), upper.astype(int)
    cap = cv2.VideoCapture(str(video))
    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for f in range(first, last+1):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode candidate window frame {f}')
            if np.any(upper > image.shape[1::-1]):
                raise ValueError('Network crop exceeds image bounds')
            frames.append(cv2.cvtColor(image[lower[1]:upper[1], lower[0]:upper[0]], cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(frames)


def _identity(a, recent_image, b, ma, mr, mb, seeds, previous, after, affine, relative,
              ids, observed, proposals, boundary_aware, tolerance, *, boundary_options=None):
    options = dict(boundary_options or {})
    erosion = options.pop('blade_core_erosion_px', None)
    common = dict(point_ids=ids, current_blade=mb == 2, boundary_aware=boundary_aware, **options)
    original_extra, recent_extra = {}, {}
    if erosion is not None:
        common['current_blade_core'] = _blade_core(mb, erosion)
        original_extra['reference_blade_core'] = _blade_core(ma, erosion)
        recent_extra['reference_blade_core'] = _blade_core(mr, erosion)
    original = verify_material_candidates(a, b, seeds, proposals, ma == 1, mb == 1,
        reference_blade=ma == 2, initial_affine=affine, **common, **original_extra)
    recent = verify_material_candidates(recent_image, b, previous, proposals, mr == 1, mb == 1,
        reference_blade=mr == 2, initial_affine=relative, **common, **recent_extra)
    result = identity_consensus(original, recent, observed, tolerance)
    summary = dict(selected=len(ids), baseline_supported=int(observed.sum()),
        original_accepted=int(original['accepted'].sum()), recent_accepted=int(recent['accepted'].sum()),
        identity_consistent=int(result['identity_consistent'].sum()), recovered=int(result['recovered'].sum()),
        ambiguous=int(np.sum(original['ambiguous'] | recent['ambiguous'])),
        recent_only=int(result['recent_only'].sum()), boundary_aware=boundary_aware,
        supported_identity_lost=int(np.sum(observed & ~result['identity_consistent'])),
        median_position_change_px=float(np.median(np.linalg.norm(result['positions']-after, axis=1))),
        physical_accuracy='unknown; original/recent checks share texture and initializer')
    return result, original, recent, summary


def evaluate_case(case, config, out, networks):
    dest = out/case['name']
    dest.mkdir()
    td, md, prep = (_path(case[k]) for k in ('tracks', 'measurements', 'prepared'))
    guard, tracks = load_npz(prep/'blade_reference_guard.npz'), load_npz(td/'tracking/tracks.npz')
    observations = load_npz(md/'measurement_support.npz')
    surface = load_npz(md/'initial_surface_extended.npz')['boundary_xy']
    with np.load(prep/'input.npz', allow_pickle=False) as z:
        scale = float(z['scale'])
    frames, tips, depths = depth_prefix(guard['source_frames'], guard['tips_xy'], scale, case['stop_mm'])
    audit = read_json(_path(case['contact_audit']))
    if not audit.get('accepted') or audit['selected_contact_frame'] != int(frames[0]):
        raise ValueError('Prepared contact zero and accepted audit disagree')
    if not np.array_equal(tracks['point_ids'], observations['point_ids']):
        raise ValueError('Measurement IDs do not match the immutable track IDs')
    if int(tracks['source_frame_indices'][0]) != 0 or not np.allclose(
            tracks['positions'][0], tracks['seed_points'], atol=1e-4):
        raise ValueError('Initial stress-free assumption requires seed frame 0, not an incremental map')
    rows = aligned_rows(tracks['source_frame_indices'], frames)
    supported = observations['supported'][aligned_rows(observations['frame_indices'], frames)]
    seeds, ids = tracks['seed_points'], tracks['point_ids']
    surface_distance = cKDTree(surface).query(seeds)[0]
    video = td/'normalized_lossless.mkv'
    initial_image, initial_mask = _frame(video, 0), _mask(td, case['name'], 0)
    records, masks_used, frames_used = [], {0}, {0}
    for requested in config['pilot_depths_mm']:
        cross = np.flatnonzero(depths > requested)
        i = int(cross[0]-1) if len(cross) else len(depths)-1
        if requested > case['stop_mm'] or i < config['reference_lag_frames']:
            continue
        j = i-int(config['reference_lag_frames'])
        f, ref = int(frames[i]), int(frames[j])
        if guard['state'][i] != 2 or guard['state'][j] != 2:
            raise ValueError('Evaluation endpoints require registered, observed blade geometry')
        previous, after = tracks['positions'][rows[j]], tracks['positions'][rows[i]]
        near = (np.linalg.norm(after-tips[i], axis=1) <= config['near_tip_radius_px'])
        finite = np.isfinite(previous).all(1) & np.isfinite(after).all(1)
        # Selection never requires current-frame support: missing IDs may recover.
        eligible = np.flatnonzero(near & finite & supported[j]
            & (surface_distance <= config['original_surface_band_px']))
        selected = eligible[spread_subset(seeds[eligible], config['maximum_identity_points'])] if len(eligible) else eligible
        recent_image, image = _frame(video, ref), _frame(video, f)
        recent_mask, mask = _mask(td, case['name'], ref), _mask(td, case['name'], f)
        masks_used.update((ref, f)); frames_used.update(range(ref, f+1))
        record = dict(case=case['name'], role=case['role'], frame=f, reference_frame=ref,
            requested_depth_mm=requested, actual_depth_mm=float(depths[i]),
            selected_near_surface=len(selected), eligible_near_surface=len(eligible), total_ids=len(ids),
            selection='Fixed original-surface band, current tip neighbourhood, recent supported query; current support NOT required',
            calibration='Existing zero and approximate 3 mm calibration preserved; geometry audit unresolved',
            identity={}, mechanics=[], promotes_to_default=False)
        if len(selected):
            affine = tracks['cumulative_affine'][rows[i], selected]
            previous_affine = tracks['cumulative_affine'][rows[j], selected]
            relative = np.full_like(affine, np.nan)
            invertible = np.isfinite(previous_affine).all((1, 2)) & (np.abs(np.linalg.det(previous_affine)) > .01)
            relative[invertible] = affine[invertible] @ np.linalg.inv(previous_affine[invertible])
            args = (initial_image, recent_image, image, initial_mask, recent_mask, mask,
                    seeds[selected], previous[selected], after[selected], affine, relative, ids[selected], supported[i, selected])
            variants = [('legacy_boundary', after[selected][None], False),
                        ('masked_boundary', after[selected][None], True)]
            side = int(config['network_crop_side_px'])
            center = np.median(np.vstack((previous[selected], after[selected], tips[i])), axis=0)
            lower = np.maximum(np.rint(center-side/2).astype(int), 0)
            upper = np.minimum(lower+side, np.array(image.shape[1::-1]))
            lower = np.maximum(upper-side, 0)
            if np.any(previous[selected] < lower) or np.any(previous[selected] > upper-1):
                raise ValueError('Network crop excludes a selected reference point')
            window = _window(video, ref, f, lower, upper)
            for name, network in networks.items():
                result = network.track(window, np.c_[np.zeros(len(selected)), previous[selected]-lower],
                    point_ids=ids[selected], frame_indices=np.arange(ref, f+1), cutoff_frame=f)
                result['positions'] += lower
                _save_result(dest/f'f{f:06d}_{name}_proposal', result,
                             dict(crop_lower_xy=lower, crop_upper_xy=upper))
                proposal = result['positions'][-1].copy()
                proposal[~result['visibility'][-1]] = np.nan
                # Keep the saved proposal as a competitor: two good but distinct
                # solutions must be ambiguous, not silently replaced by a network.
                variants.append((name, np.stack((after[selected], proposal)), True))
                record[name] = result['metadata']
            for name, proposals, aware in variants:
                started = time.perf_counter()
                result, original, recent, summary = _identity(*args, proposals, aware, config['identity_agreement_px'])
                summary['seconds'] = time.perf_counter()-started
                for suffix, value in [('identity', result), ('original_check', original), ('recent_check', recent)]:
                    _save_result(dest/f'f{f:06d}_{name}_{suffix}', value)
                record['identity'][name] = summary
                _preview(image, after[selected], result['positions'], tips[i], dest/f'f{f:06d}_{name}.png',
                    supports=(supported[i, selected], result['identity_consistent']),
                    titles=('Saved support (not ground truth)', f'{name}: double-reference consistency'))
        else:
            record['identity_unavailable_reason'] = 'No recently supported original-surface query in the neighbourhood'
        # Mechanical sensitivity uses INITIAL coordinates and all finite nearby
        # nodes, not the thin surface band. Unknown boundary tractions are excluded.
        eligible = np.flatnonzero(near & finite & np.isfinite(seeds).all(1))
        if len(eligible) >= 6:
            nodes = eligible[spread_subset(seeds[eligible], config['maximum_nodes'])]
            observed = supported[i, nodes] & tracks['supported'][0, nodes]
            for ratio in [None, *config['mechanics_bulk_shear_ratios']]:
                name = 'kinematic_baseline' if ratio is None else f'equilibrium_Kmu_{ratio:g}'
                options = dict(config.get('candidate', {}), feature_mode='material-masked',
                    equilibrium_weight=0 if ratio is None else config['mechanics_weight'],
                    equilibrium_bulk_shear_ratio=10 if ratio is None else ratio,
                    equilibrium_reference='stress-free-assumed')
                started = time.perf_counter()
                fit = fit_material_mesh(initial_image, image, seeds[nodes], after[nodes], observed,
                    material_mask=(initial_mask == 1, mask == 1), blade_mask=(initial_mask == 2, mask == 2),
                    config=FiniteDeformationConfig(**options))
                _save_result(dest/f'f{f:06d}_{name}', fit, dict(point_ids=ids[nodes], initial_xy=seeds[nodes], baseline_xy=after[nodes]))
                record['mechanics'].append(dict(variant=name, seconds=time.perf_counter()-started,
                    nodes=len(nodes), accepted=fit['accepted'], supported=int(fit['supported'].sum()),
                    holdout=fit['holdout'], equilibrium_gap=fit.get('equilibrium_gap'),
                    median_correction_px=float(np.median(np.linalg.norm(fit['positions']-after[nodes], axis=1))),
                    reference='Frame 0 is ASSUMED stress-free; plane-strain Neo-Hookean sensitivity, not inferred stress'))
        else:
            record['mechanics_unavailable_reason'] = 'Fewer than six finite near-tip nodes'
        records.append(record)
        write_json(dest/'summary.json', records)
        print(f'{case["name"]} {depths[i]:.3f} mm: '+str(record['identity']), flush=True)
    paths = [td/'tracking/tracks.npz', md/'measurement_support.npz', md/'initial_surface_extended.npz',
             prep/'input.npz', prep/'blade_reference_guard.npz', td/'frame_mapping.npz', td/'plan.json', video]
    paths.extend(td/'segmentation'/case['name']/'semantic_masks'/f'f{f:06d}.png' for f in sorted(masks_used))
    write_json(dest/'provenance.json', dict(case=case, source_sha256={str(p): sha(p) for p in paths},
        new_images_read_source_frames=sorted(frames_used), old_tracks_may_contain_historical_future_dependencies=True,
        model_training=False, microscopy_targets_used=False, source_results_modified=False,
        geometry_audit=str(_path(config['geometry_evidence'])/case['name']/'summary.json')))
    return records


def run(config, out):
    """One entry point, exact same windows/IDs and gates for both neural models."""
    validate_config(config)
    from src.tracking.neural_candidates import CoTrackerCandidates, TAPNextCandidates
    from src.analysis.blade_geometry_review import audit_contact_case
    geometry = []
    (out/'geometry').mkdir()
    for case in config['cases']:
        existing = _path(config['geometry_evidence'])/case['name']/'summary.json'
        if existing.is_file():
            provenance = read_json(existing.with_name('provenance.json'))
            for path, expected in provenance['source_sha256'].items():
                if sha(path) != expected:
                    raise ValueError(f'Geometry audit input changed: {path}')
            summary = read_json(existing)
            if summary['case'] != case['name']:
                raise ValueError('Geometry audit belongs to a different case')
            audit_out = out/'geometry'/case['name']
            audit_out.mkdir()
            for filename in ('summary.json', 'provenance.json', 'arrays.npz', 'geometry_review.png'):
                shutil.copyfile(existing.with_name(filename), audit_out/filename)
            # Preserve the separate formal audit, including its input hashes.
            geometry.append(dict(case=case['name'], source=str(existing), sha256=sha(existing),
                                 summary=summary, provenance=provenance))
        else:
            summary = audit_contact_case(case, out/'geometry'/case['name'])
            geometry.append(dict(case=case['name'], summary=summary))
    constructors = {'cotracker3': CoTrackerCandidates, 'tapnextpp': TAPNextCandidates}
    names = config['network_backends']
    if not names or len(set(names)) != len(names) or any(n not in constructors for n in names):
        raise ValueError('Explicit unique supported neural candidate backends required')
    networks = {name: constructors[name](asset_dir=ROOT/'models/point_tracking'/name) for name in names}
    results = dict(geometry=geometry, cases=[], promotes_to_default=False,
        scientific_scope='Historical development cases. Physical accuracy unknown; no microscopy target fitting.')
    for case in config['cases']:
        results['cases'].extend(evaluate_case(case, config, out, networks))
        write_json(out/'summary.json', results)
    body = '<h1>Material identity and contact reliability</h1><p>Development evidence. Production unchanged. Hollow points failed identity checks. Network visibility and mechanical regularity are not material truth.</p>'
    body += '<h2>Contact-zero crosscheck</h2><table><tr><th>Case</th><th>Audit</th><th>Existing zero</th></tr>'
    for g in geometry:
        s = g['summary']
        body += f'<tr><td><a href="geometry/{g["case"]}/geometry_review.png">{html.escape(g["case"])}</a></td><td>{html.escape(str(s["state"]))}</td><td>{s["existing_contact_frame"]}</td></tr>'
    body += '</table>'
    for r in results['cases']:
        stem = f'{r["case"]}/f{r["frame"]:06d}'
        body += f'<h2>{html.escape(r["case"])} | {r["actual_depth_mm"]:.3f} mm</h2><table><tr><th>Candidate</th><th>Original check</th><th>Recent check</th><th>Both / all selected</th><th>Recovered IDs</th><th>Ambiguous</th></tr>'
        for name, s in r['identity'].items():
            body += f'<tr><td>{name}</td><td>{s["original_accepted"]}</td><td>{s["recent_accepted"]}</td><td>{s["identity_consistent"]} / {s["selected"]}</td><td>{s["recovered"]}</td><td>{s["ambiguous"]}</td></tr>'
        body += '</table>'
        for name in r['identity']:
            body += f'<details><summary>{name}</summary><img src="{stem}_{name}.png" width="960"></details>'
        body += '<table><tr><th>Mechanics sensitivity</th><th>Fit accepted</th><th>Supported nodes</th><th>Interior nodes</th><th>Image holdout RMS before / after</th><th>Median correction px</th></tr>'
        for s in r['mechanics']:
            gap, holdout = s['equilibrium_gap'], s['holdout']
            internal = gap['interior_node_count'] if gap['enabled'] else 'disabled'
            before, after = (holdout[k] for k in ('before_normalized_rms', 'after_normalized_rms'))
            values = 'unavailable' if before is None or after is None else f'{before:.3f} / {after:.3f}'
            body += f'<tr><td>{s["variant"]}</td><td>{s["accepted"]}</td><td>{s["supported"]}/{s["nodes"]}</td><td>{internal}</td><td>{values}</td><td>{s["median_correction_px"]:.3f}</td></tr>'
        body += f'</table><p><a href="{r["case"]}/summary.json">Residuals, assumptions, times and evidence counts</a></p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Material reliability</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#222}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}img{max-width:100%}</style>'+body, encoding='utf-8')
    return results


def _blade_core(semantic, erosion):
    """An explicitly conditional eroded semantic core, not blade ground truth."""
    return cv2.erode((semantic == 2).astype(np.uint8),
                    np.ones((2*int(erosion)+1, 2*int(erosion)+1), np.uint8)).astype(bool)


def identity_cohort(seeds, surface, contact_tip, config, regression_ids, point_ids):
    """Fixed initial geometry plus declared historical regression IDs; no support filter."""
    seeds, point_ids = np.asarray(seeds), np.asarray(point_ids)
    if (seeds.shape != (len(point_ids), 2) or not np.isfinite(seeds).all()
            or len(np.unique(point_ids)) != len(point_ids)
            or not np.isin(regression_ids, point_ids).all()):
        raise ValueError('Finite initial coordinates, unique IDs and known regression IDs required')
    band = cKDTree(surface).query(seeds)[0] <= config['original_surface_band_px']
    near = np.linalg.norm(seeds-contact_tip, axis=1) <= config['initial_neighbourhood_px']
    regression = np.isin(point_ids, regression_ids)
    selected = np.flatnonzero((band & near) | regression)
    if not len(selected):
        raise ValueError('Initial geometry leaves an empty identity cohort')
    return selected


def validate_identity_config(config):
    from src.tracking.identity_chain import IdentityChainConfig
    IdentityChainConfig(**config.get('identity_chain', {}))
    for key in ('initial_neighbourhood_px', 'identity_boundary_tolerance_px',
                'original_surface_band_px', 'identity_agreement_px'):
        if not np.isfinite(config.get(key, np.nan)) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if config['identity_boundary_tolerance_px'] > 2:
        raise ValueError('Boundary uncertainty band must not exceed 2 native pixels')
    for key, lo, hi in [('identity_network_crop_side_px', 128, 1024),
                        ('identity_network_max_frames', 2, 2000),
                        ('identity_blade_core_erosion_px', 0, 2)]:
        v = config.get(key, np.nan)
        if not np.isfinite(v) or int(v) != v or not lo <= v <= hi:
            raise ValueError(f'Invalid {key}')
    if config.get('identity_prefix_stride') != 1:
        raise ValueError('Identity benchmark requires complete consecutive source frames')
    for case in config.get('cases', []):
        p = _path(config['regression_evidence'])/case['name']/'summary.json'
        records = read_json(p)
        if (not isinstance(records, list) or not records
                or any(r.get('case', case['name']) != case['name']
                       or r.get('reference_frame', -1) < 0
                       or r.get('frame', -1) <= r.get('reference_frame', -1) for r in records)):
            raise ValueError('Regression evidence must contain matching ordered case endpoints')


def _boundary_regression(case, config, out, tracks, observations, surface_frames, tips):
    """Exact old 24-ID endpoints, separating mask gating from physical identity loss."""
    from src.tracking.identity_chain import _relative_map
    old = _path(config['regression_evidence'])/case['name']
    td = _path(case['tracks'])
    records, failed_ids = [], []
    initial_image, ma = _frame(td/'normalized_lossless.mkv', 0), _mask(td, case['name'], 0)
    for endpoint in read_json(old/'summary.json'):
        f, ref = endpoint['frame'], endpoint['reference_frame']
        stem = f'f{f:06d}'
        identity = load_npz(old/f'{stem}_masked_boundary_identity.npz')
        ids = identity['point_ids']
        selected = np.array([np.flatnonzero(tracks['point_ids'] == v)[0] for v in ids])
        ri, ti = aligned_rows(tracks['source_frame_indices'], [ref, f])
        oi = aligned_rows(observations['frame_indices'], [f])[0]
        observed = observations['supported'][oi, selected]
        failed_ids.extend(ids[~observed].tolist())
        previous, after = tracks['positions'][ri, selected], tracks['positions'][ti, selected]
        affine = tracks['cumulative_affine'][ti, selected]
        relative, _ = _relative_map(affine, tracks['cumulative_affine'][ri, selected])
        a, b = (_frame(td/'normalized_lossless.mkv', x) for x in (ref, f))
        mr, mb = (_mask(td, case['name'], x) for x in (ref, f))
        base_args = (initial_image, a, b, ma, mr, mb, tracks['seed_points'][selected],
                     previous, after, affine, relative, ids, observed)
        record = dict(frame=f, reference_frame=ref, depth_mm=endpoint['actual_depth_mm'],
                      point_count=len(ids), measurement_unsupported=int((~observed).sum()),
                      raw_tracker_unsupported=int((~tracks['supported'][ti, selected]).sum()), variants={})
        for name in ('masked_boundary', 'cotracker3', 'tapnextpp'):
            proposals = after[None]
            if name != 'masked_boundary':
                neural = load_npz(old/f'{stem}_{name}_proposal.npz')
                if not np.array_equal(neural['point_ids'], ids):
                    raise ValueError('Regression network candidates have different IDs')
                candidate = neural['positions'][-1].copy()
                candidate[~neural['visibility'][-1]] = np.nan
                proposals = np.stack((after, candidate))
            for boundary in ('eroded', 'one_sided'):
                opts = {} if boundary == 'eroded' else dict(boundary_mode='one-sided',
                    center_tolerance_px=config['identity_boundary_tolerance_px'],
                    blade_core_erosion_px=config['identity_blade_core_erosion_px'])
                result, original, recent, summary = _identity(*base_args, proposals, True,
                    config['identity_agreement_px'], boundary_options=opts)
                label = f'{name}_{boundary}'
                summary['raw_tracker_recovered'] = int(np.sum(result['identity_consistent']
                    & ~tracks['supported'][ti, selected]))
                record['variants'][label] = summary
                for suffix, value in [('identity', result), ('original', original), ('recent', recent)]:
                    _save_result(out/f'{stem}_{label}_{suffix}', value)
                if name == 'tapnextpp':
                    tip = tips[aligned_rows(surface_frames, [f])[0]]
                    _preview(b, after, result['positions'], tip, out/f'{stem}_{boundary}.png',
                        supports=(observed, result['identity_consistent']),
                        titles=('Saved measurement support', f'{boundary}: initial + recent image checks'))
        records.append(record)
    write_json(out/'summary.json', records)
    return records, np.unique(failed_ids)


def identity_case(case, config, out, network):
    from src.tracking.identity_chain import IdentityChainConfig, InitialRootedIdentityChain
    dest = out/case['name']; dest.mkdir()
    regression = dest/'boundary_regression'; regression.mkdir()
    td, md, prep = (_path(case[k]) for k in ('tracks', 'measurements', 'prepared'))
    tracks = load_npz(td/'tracking/tracks.npz')
    observations = load_npz(md/'measurement_support.npz')
    if not np.array_equal(tracks['point_ids'], observations['point_ids']):
        raise ValueError('Material identities differ between tracking and measurements')
    guard = load_npz(prep/'blade_reference_guard.npz')
    scale = float(load_npz(prep/'input.npz')['scale'])
    frames, tips, depths = depth_prefix(guard['source_frames'], guard['tips_xy'], scale, case['stop_mm'])
    audit = read_json(_path(case['contact_audit']))
    if not audit.get('accepted') or audit['selected_contact_frame'] != int(frames[0]):
        raise ValueError('Prepared zero and audit disagree')
    surface = load_npz(md/'initial_surface_extended.npz')['boundary_xy']
    regressions, failed_ids = _boundary_regression(case, config, regression, tracks, observations, frames, tips)
    selected = identity_cohort(tracks['seed_points'], surface, tips[0], config, failed_ids, tracks['point_ids'])
    seeds, ids = tracks['seed_points'][selected], tracks['point_ids'][selected]
    all_frames = np.arange(int(frames[-1])+1)
    if len(all_frames) > config['identity_network_max_frames']:
        raise ValueError('Full prefix exceeds declared network frame limit before decoding')
    rows = aligned_rows(tracks['source_frame_indices'], all_frames)
    if not np.allclose(tracks['positions'][rows[0], selected], seeds, atol=1e-4):
        raise ValueError('Network and identity chain require actual initial seed coordinates')
    first = _frame(td/'normalized_lossless.mkv', 0)
    # Crop determined only from initial seeds and calibration, never from later
    # saved trajectories. Its downward margin covers the specified 1 mm field.
    side = config['identity_network_crop_side_px']
    center = np.mean([seeds.min(0), seeds.max(0)], axis=0)+[0, min(80., case['stop_mm']*500/scale)]
    upper = np.minimum(np.maximum(np.rint(center-side/2).astype(int), 0)+side, first.shape[1::-1])
    lower = np.maximum(upper-side, 0)
    if np.any(seeds < lower) or np.any(seeds > upper-1):
        raise ValueError('Initial query outside fixed network crop')
    window = _window(td/'normalized_lossless.mkv', 0, int(frames[-1]), lower, upper)
    neural = network.track(window, np.c_[np.zeros(len(ids)), seeds-lower], point_ids=ids,
        frame_indices=all_frames, cutoff_frame=int(frames[-1]))
    neural['positions'] += lower
    _save_result(dest/'initial_tapnextpp', neural, dict(crop_lower_xy=lower, crop_upper_xy=upper, seed_xy=seeds))
    del window
    initial_mask = _mask(td, case['name'], 0)
    chain_options = dict(config.get('identity_chain', {}),
                         center_tolerance_px=config['identity_boundary_tolerance_px'],
                         agreement_px=config['identity_agreement_px'])
    variants = {name: InitialRootedIdentityChain(first, initial_mask == 1, seeds,
        point_ids=ids, frame_index=0, cutoff_frame=int(frames[-1]), blade_mask=initial_mask == 2,
        blade_core_mask=_blade_core(initial_mask, config['identity_blade_core_erosion_px']),
        config=IdentityChainConfig(**chain_options)) for name in ('rooted_saved_proposal', 'rooted_initial_network')}
    output = {name: [] for name in variants}
    geometry_frames = {v['frame'] for v in regressions}
    cap = cv2.VideoCapture(str(td/'normalized_lossless.mkv'))
    started = time.perf_counter()
    try:
        ok, _ = cap.read()
        if not ok:
            raise ValueError('Cannot decode initial frame')
        for f in all_frames[1:]:
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode identity frame {f}')
            mask = _mask(td, case['name'], f)
            old = tracks['positions'][rows[f], selected]
            proposal = neural['positions'][f].copy()
            proposal[~neural['visibility'][f]] = np.nan
            for name, chain in variants.items():
                candidates = old[None] if name == 'rooted_saved_proposal' else np.stack((old, proposal))
                result = chain.step(image, mask == 1, frame_index=int(f), proposals=candidates,
                    blade_mask=mask == 2, blade_core_mask=_blade_core(mask, config['identity_blade_core_erosion_px']))
                output[name].append(result)
                if f in geometry_frames:
                    _save_result(dest/f'f{f:06d}_{name}', result)
                    _preview(image, old, result['positions'], tips[aligned_rows(frames, [f])[0]],
                        dest/f'f{f:06d}_{name}.png', supports=(tracks['supported'][rows[f], selected], result['rooted_supported']),
                        titles=('Saved raw tracker support', f'{name}: conditional rooted paths'))
            if f % 100 == 0:
                print(f'{case["name"]}: initial-rooted frame {f}/{all_frames[-1]}', flush=True)
    finally:
        cap.release()
    summary = dict(case=case['name'], role=case['role'], frames=int(len(all_frames)), selected=len(ids),
        stop_mm=case['stop_mm'], last_depth_mm=float(depths[-1]), first_contact_frame=int(frames[0]),
        boundary_regression=regressions, full_prefix_seconds=time.perf_counter()-started,
        network=neural['metadata'], variants={}, physical_accuracy='not established; historical development set',
        selection='Initial surface band and initial contact neighbourhood, plus declared regression failures; no recent/current support selection')
    indentation = np.full(len(all_frames), np.nan)
    indentation[frames] = depths
    archived = {}
    for name, results in output.items():
        arrays = {key: np.stack([r[key] for r in results]) for key, value in results[0].items()
                  if isinstance(value, np.ndarray) and key != 'point_ids'}
        arrays.update(point_ids=ids, source_frames=all_frames[1:], seed_xy=seeds,
                      indentation_mm=indentation[1:],
                      saved_raw_supported=tracks['supported'][rows[1:]][:, selected],
                      initial_supported=variants[name].initial_supported,
                      initial_failure_flags=variants[name].initial_failure_flags)
        _save_result(dest/name, dict(arrays,
            failure_flag_meanings=results[0]['failure_flag_meanings'],
            reference_axis=['recent', 'keyframe', 'root'],
            semantics=results[0]['semantics']))
        loading = all_frames[1:] >= frames[0]
        summary['variants'][name] = dict(loading_point_times=int(loading.sum()*len(ids)),
            rooted_supported=int(arrays['rooted_supported'][loading].sum()),
            anchor_confirmed=int(arrays['root_confirmed'][loading].sum()),
            initial_supported=int(arrays['initial_supported'].sum()),
            raw_supported=int(arrays['saved_raw_supported'][loading].sum()),
            raw_unsupported_but_rooted=int(np.sum(arrays['rooted_supported'][loading]
                & ~arrays['saved_raw_supported'][loading])),
            raw_supported_but_rejected=int(np.sum(~arrays['rooted_supported'][loading]
                & arrays['saved_raw_supported'][loading])),
            endpoint_supported=int(arrays['rooted_supported'][-1].sum()),
            endpoint_raw_supported=int(arrays['saved_raw_supported'][-1].sum()))
        for key in ('recovered', 'gap_age', 'cumulative_cycle_error_px', 'cycle_budget_exceeded'):
            if key in arrays:
                summary['variants'][name][key+'_sum'] = float(np.nansum(arrays[key][loading]))
        archived[name] = arrays
    saved, proposed = (archived[n]['rooted_supported'][loading] for n in variants)
    summary['network_increment'] = dict(additional_point_times=int(np.sum(proposed & ~saved)),
        lost_point_times=int(np.sum(saved & ~proposed)),
        meaning='Conditional admission changes on the same fixed cohort; not physical accuracy')
    _identity_plot(archived, dest/'support_curves.png', case['name'])
    write_json(dest/'summary.json', summary)
    sources = [td/'tracking/tracks.npz', td/'normalized_lossless.mkv', td/'frame_mapping.npz',
        md/'measurement_support.npz', md/'initial_surface_extended.npz', prep/'blade_reference_guard.npz',
        prep/'input.npz', _path(case['contact_audit'])]
    write_json(dest/'provenance.json', dict(source_sha256={str(p): sha(p) for p in sources},
        mask_sha256={str(int(f)): sha(td/'segmentation'/case['name']/'semantic_masks'/f'f{f:06d}.png') for f in all_frames},
        initial_query_frame=0, new_image_last_frame=int(frames[-1]), source_trajectories_unchanged=True,
        historical_scale_zero_and_saved_prior_have_prior_full_video_dependencies=True))
    return summary


def _identity_plot(variants, path, title):
    """Show raw counts against measured depth, without smoothing or gap filling."""
    from matplotlib.figure import Figure
    figure = Figure(figsize=(10, 8), layout='constrained')
    axes = figure.subplots(3, 1, sharex=True)
    first = next(iter(variants.values()))
    loading = np.isfinite(first['indentation_mm'])
    depth = first['indentation_mm'][loading]
    axes[0].plot(depth, first['saved_raw_supported'][loading].mean(1)*100,
                 color='0.5', lw=1, label='Saved tracker support')
    for (name, data), color in zip(variants.items(), ('#d18a24', '#267c9a')):
        label = 'Saved proposals' if name == 'rooted_saved_proposal' else 'Saved + initial-query TAPNext++'
        axes[0].plot(depth, data['rooted_supported'][loading].mean(1)*100,
                     color=color, lw=1, label=label)
        checked = data['reference_frames'][loading, 2, 0] >= 0
        axes[1].plot(depth[checked], data['root_confirmed'][loading][checked].mean(1)*100,
                     color=color, lw=1, marker='.', markersize=3, label=label)
        if 'cycle_budget_exceeded' in data:
            axes[2].plot(depth, data['cycle_budget_exceeded'][loading].mean(1)*100,
                         color=color, lw=1, label=label)
    for ax, ylabel in zip(axes, ('Rooted image support (%)', 'Scheduled initial checks (%)',
                                'Cumulative-cycle warning (%)')):
        ax.set(ylabel=ylabel, ylim=(-2, 102))
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel('Indentation from existing contact zero (mm)')
    figure.suptitle(f'{title} | fixed initial IDs; conditional image evidence')
    figure.savefig(path, dpi=150)


def run_identity(config, out):
    from src.tracking.neural_candidates import TAPNextCandidates
    validate_identity_config(config)
    model = TAPNextCandidates(ROOT/'models/point_tracking/tapnextpp',
                              maximum_frames=config['identity_network_max_frames'])
    report = dict(cases=[], promotes_to_default=False, physical_accuracy_not_established=True)
    for case in config['cases']:
        report['cases'].append(identity_case(case, config, out, model))
        write_json(out/'summary.json', report)
    body = '<h1>Surface support and initial-rooted identity</h1><p>All IDs selected before tracking. Full frame-0 prefixes; no reference is updated from unsupported predictions. Rooted paths are conditional image evidence, not physical identity truth.</p>'
    for case in report['cases']:
        name = case['case']
        body += f'<h2>{name}</h2><p>{case["selected"]} fixed IDs, {case["frames"]} consecutive frames through {case["last_depth_mm"]:.3f} mm.</p><table><tr><th>Method</th><th>Loading supported / total</th><th>Direct root confirmations</th><th>Cycle-risk point-times</th><th>Endpoint supported</th></tr>'
        for method, s in case['variants'].items():
            body += f'<tr><td>{method}</td><td>{s["rooted_supported"]}/{s["loading_point_times"]}</td><td>{s["anchor_confirmed"]}</td><td>{int(s.get("cycle_budget_exceeded_sum", 0))}</td><td>{s["endpoint_supported"]}/{case["selected"]}</td></tr>'
        body += f'</table><p><a href="{name}/summary.json">Counts, limitations and boundary ablations</a></p><img src="{name}/support_curves.png" width="960">'
        for endpoint in case['boundary_regression']:
            f = endpoint['frame']
            body += f'<h3>{endpoint["depth_mm"]:.3f} mm</h3>'
            for variant in case['variants']:
                body += f'<details><summary>{variant}</summary><img src="{name}/f{f:06d}_{variant}.png" width="960"></details>'
            body += f'<details><summary>Same historical IDs: one-sided boundary regression</summary><img src="{name}/boundary_regression/f{f:06d}_one_sided.png" width="960"></details>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Surface identity</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse}img{max-width:100%}</style>'+body, encoding='utf8')
    return report
