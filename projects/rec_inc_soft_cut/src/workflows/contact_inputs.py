"""Shared immutable input preparation for the paired contact estimators."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import cv2
import numpy as np
from src.io import ROOT, load_npz as load, sha, write_json
from src.workflows import motion as saved
from src.analysis.blade_contact_geometry import blade_contact_geometry
from src.analysis.contact_motion_evidence import supported_local_slopes
from src.analysis.boundary_material_contact import extract_boundary_path
from src.analysis.contour_geometry import ContourGeometryTracker
from src.analysis.loading_blade_reference import restrict_loading_case, guarded_prefix
from src.analysis.fixed_surface_origin import FixedSurfaceOriginTracker
from src.analysis.validated_material_motion import validated_local_slopes

SOURCE = ROOT/'results/current_material_loading'
MEASUREMENTS = ROOT/'results/current_contact_measurements'
OUT = ROOT/'results/current_sliding'
DEFAULT_CUTOFFS = {f'0930{kind}a{number}': depth for number in (1, 2)
                   for kind, depth in (('l30', 1.), ('l60', 1.4), ('n60', 1.5))}


def sample_rows(depth, preview=False):
    finite = np.flatnonzero(np.isfinite(depth))
    if len(finite) < 2:
        raise ValueError('At least two finite registered blade depths are required')
    rows = {int(finite[0]), int(finite[-1])} if preview else set(range(0, len(depth), 2)) & set(finite)
    rows.add(int(finite[-1]))
    rows.update(int(finite[np.argmin(abs(depth[finite]-d))]) for d in (.1, .3, .5, 1., 2.))
    return np.asarray(sorted(rows), int)


def tip_slopes(tips, times, half=.075):
    """Local fixed-tip regression; endpoint one-sided windows are declared."""
    v, sigma, one_sided = np.full_like(tips, np.nan), np.full(len(tips), np.nan), np.zeros(len(tips), bool)
    for i, t in enumerate(times):
        keep = abs(times-t) <= half
        if keep.sum() < 5: continue
        dt = times[keep]-t
        design = np.column_stack((np.ones(len(dt)), dt))
        beta = np.linalg.lstsq(design, tips[keep], rcond=None)[0]
        error = np.sqrt(np.mean(np.sum((design@beta-tips[keep])**2, axis=1)))
        v[i] = beta[1]
        sigma[i] = max(.25, error)/max(np.sqrt(np.sum((dt-dt.mean())**2)), 1e-8)
        one_sided[i] = not ((dt < 0).any() and (dt > 0).any())
    return v, sigma, one_sided


def prepare(name, preview=False, stop_depth_mm=None):
    saved.SOURCE = SOURCE
    saved.MEASUREMENTS = MEASUREMENTS
    case = saved.data(name)
    cutoff = None
    if stop_depth_mm is not None:
        geometry = saved.load_npz(case['source']/'geometry.npz')
        case = restrict_loading_case(case, geometry['tips_px'], geometry['supported'], stop_depth_mm)
        cutoff = case['cutoff']
    source, measure = case['source'], saved.MEASUREMENTS/name
    query = saved.load_npz(source/'queries.npz')
    initial = saved.load_npz(measure/'initial_surface_extended.npz')
    frames, tips = case['frames'], case['reference']['tips_xy']
    scale = float(case['um_per_px'])
    depth = (tips[:,1]-tips[0,1])*scale/1000.
    rows = sample_rows(depth, preview)
    if cutoff is not None:
        unavailable = ~np.isfinite(tips).all(1)
        cutoff['unavailable_blade_reference_source_frames'] = frames[unavailable].tolist()
        cutoff['unavailable_blade_reference_loading_samples'] = int(unavailable.sum())
        cutoff['unavailable_reference_policy'] = 'No contact observation at missing tip coordinates; temporal process bridges the gap'
    source_paths = [source/'tracking/tracks.npz', source/'queries.npz', source/'frame_mapping.npz',
                    source/'geometry.npz', source/'plan.json', source/'geometry.json',
                    measure/'measurement_support.npz', measure/'initial_surface_extended.npz',
                    measure/'blade_contact_geometry.npz']
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    repair, estimated = saved.prepare_current_masks(case, name)
    # Explicitly remove blade/background observations from temporal regressions.
    case['support'][frames] &= case['material']
    case['edges'][frames] &= case['support'][frames]
    case['edges'][frames[1:]] &= case['support'][frames[:-1]]
    nt = len(case['positions'])
    slopes = supported_local_slopes(case['positions'], case['times'][:nt], case['support'],
        half_window_s=.075, maximum_gap_s=.04, minimum_samples=5, minimum_span_s=.025,
        maximum_fit_rms=2., motion_step_valid=case['edges'])
    # Correlated tracking residuals: retain the position floor and do not divide
    # by sqrt(point count) here. Subsequent spatial model uses a common floor.
    velocity_sigma = np.maximum(.25, slopes['fit_rms'])/np.maximum(slopes['fit_span_s'], .025)*np.sqrt(12.)
    bv, bs, one_sided = tip_slopes(tips, case['times'][frames])
    branches, support, conditional, uncertainty, reports, paths = [], [], [], [], [], []
    mask_hash = hashlib.sha256()
    for row in rows:
        frame = int(frames[row])
        labels, est = repair.read(frame)
        report = repair.diagnostics[frame]
        geometry = blade_contact_geometry(labels, case['positions'][frame], tips[row])
        branches.append(geometry['routing_branch'])
        support.append(case['support'][frame] & geometry['material_pixel'] & ~geometry['on_blade'])
        cond = bool(case['reference']['state'][row] != 2 or est or report['reason'] != 'original_tip_mask_available')
        conditional.append(cond)
        uncertainty.append(4. if cond else 2.)
        reports.append(dict(report))
        paths.append(extract_boundary_path(labels,tips[row],np.asarray(initial['tangent'])))
        mask_hash.update(labels.tobytes())
    return dict(case=case, query=query, initial=initial, rows=rows, frames=frames[rows],
        times=case['times'][frames[rows]], tips=tips[rows], depth=depth[rows], scale=scale,
        positions=case['positions'][frames[rows]], support=np.asarray(support),
        branches=np.asarray(branches), velocity=slopes['slope'][frames[rows]],
        velocity_valid=slopes['valid'][frames[rows]], velocity_sigma=velocity_sigma[frames[rows]],
        blade_velocity=bv[rows], blade_sigma=bs[rows], tip_one_sided=one_sided[rows],
        conditional=np.asarray(conditional), uncertainty=np.asarray(uncertainty),
        provenance=provenance, mask_digest=mask_hash.hexdigest(), mask_reports=reports,paths=paths,
        cutoff=cutoff)


def prepare_cached(name,preview=False,stop_depth_mm=None):
    """Cache only immutable derived observations, never modify saved tracks."""
    folder=OUT/name
    cache=folder/('input_preview.npz' if preview else 'input.npz')
    meta_path=cache.with_suffix('.json')
    settings = dict(source=str(SOURCE.resolve()), measurements=str(MEASUREMENTS.resolve()),
                    stop_depth_mm=stop_depth_mm, schema='contact_input_depth_prefix_v1')
    if cache.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('settings') != settings:
            raise RuntimeError('Cached preparation settings differ; use a new output directory')
        if any(sha(ROOT/p)!=h for p,h in meta['provenance'].items()):
            raise RuntimeError('Cached source fingerprint no longer matches')
        a=saved.load_npz(cache)
        if not np.isfinite(a['depth']).all() or not np.isfinite(a['tips']).all():
            raise RuntimeError('Cached inputs include missing blade references; rebuild preparation in a fresh cache')
        data={k:v for k,v in a.items() if not k.startswith(('query__','initial__','path__'))}
        data['case']={'ids':a['query__point_ids']}
        data['query']={k.removeprefix('query__'):v for k,v in a.items() if k.startswith('query__')}
        data['initial']={k.removeprefix('initial__'):v for k,v in a.items() if k.startswith('initial__')}
        data.update(provenance=meta['provenance'],mask_digest=meta['mask_digest'],mask_reports=meta['mask_reports'],
                    cutoff=meta.get('cutoff'))
        data['paths']=[]
        for i in range(len(data['frames'])):
            start,end=a['path__offsets'][i:i+2]
            data['paths'].append(None if end==start else dict(xy=a['path__xy'][start:end],arc=a['path__arc'][start:end],
                tip_gap=float(a['path__gap'][i]),tip_projection=a['path__projection'][i]))
        return data
    data=prepare(name,preview,stop_depth_mm)
    arrays={k:v for k,v in data.items() if isinstance(v,(np.ndarray,float,int))}
    arrays.update({f'query__{k}':v for k,v in data['query'].items()})
    arrays.update({f'initial__{k}':v for k,v in data['initial'].items()})
    xy=[p['xy'] if p else np.empty((0,2)) for p in data['paths']]
    arc=[p['arc'] if p else np.empty(0) for p in data['paths']]
    arrays.update(path__offsets=np.r_[0,np.cumsum([len(v) for v in xy])],path__xy=np.concatenate(xy),path__arc=np.concatenate(arc),
        path__gap=np.asarray([p['tip_gap'] if p else np.nan for p in data['paths']]),
        path__projection=np.asarray([p['tip_projection'] if p else [np.nan,np.nan] for p in data['paths']]))
    np.savez_compressed(cache,**arrays)
    write_json(meta_path,dict(settings=settings,**{k:data[k] for k in ('provenance','mask_digest','mask_reports','cutoff')}))
    return data


def cutoffs_for(names, default=None, overrides=()):
    result = {name: default if default is not None else DEFAULT_CUTOFFS.get(name) for name in names}
    seen = set()
    for item in overrides:
        name, separator, value = item.partition('=')
        if not separator or name not in result or name in seen:
            raise ValueError('Use unique --stop-map NAME=MM values for selected cases')
        result[name] = float(value)
        seen.add(name)
    if any(value is None or not np.isfinite(value) or not 0 < value <= 3 for value in result.values()):
        raise ValueError('Each case needs a declared cutoff in (0,3] mm; use --stop-mm or --stop-map')
    return result


def prepare_case(name, args, folder, cutoff):
    """Build all prerequisites from tracks; no dated results are required."""
    global SOURCE, MEASUREMENTS, OUT
    SOURCE, MEASUREMENTS, OUT = args.source, args.measurements, args.out
    prepare_cached(name, stop_depth_mm=cutoff)
    d = load(folder/'input.npz')
    metadata = json.loads((folder/'input.json').read_text(encoding='utf-8'))
    saved.SOURCE, saved.MEASUREMENTS = args.source, args.measurements
    case = saved.data(name)
    reference = load(args.source/name/'geometry.npz')
    case = restrict_loading_case(case, reference['tips_px'], reference['supported'], cutoff)
    repair, _ = saved.prepare_current_masks(case, name)
    frames = case['frames']
    case['support'][frames] &= case['material']
    case['edges'][frames] &= case['support'][frames]
    case['edges'][frames[1:]] &= case['support'][frames[:-1]]
    slopes = validated_local_slopes(case['positions'], case['times'], case['support'],
                                   motion_step_valid=case['edges'])
    motion = {key: value[d['frames']] for key, value in slopes.items()
              if isinstance(value, np.ndarray) and value.shape[:1] == (len(case['times']),)}
    motion.update(frames=d['frames'], times_s=d['times'], depth_mm=d['depth'])
    np.savez_compressed(folder/'temporal_motion.npz', **motion)

    initial = {key.removeprefix('initial__'): value for key, value in d.items() if key.startswith('initial__')}
    geometry_tracker = ContourGeometryTracker()
    origin_tracker = FixedSurfaceOriginTracker(d['query__seed_points'], d['query__point_ids'],
                                              initial, float(initial['contact_s_px']))
    records, origin_records, hidden = [], [], []
    video = args.source/name/'normalized_lossless.mkv'
    cap = cv2.VideoCapture(str(video))
    frame = -1
    mask_digest = hashlib.sha256()
    try:
        for i, target in enumerate(d['frames']):
            while frame < int(target):
                ok, image = cap.read()
                frame += 1
                if not ok:
                    raise OSError(f'Missing normalized frame {target}: {video}')
            labels, estimated = repair.read(int(target))
            mask_digest.update(labels.tobytes())
            g = geometry_tracker.estimate(image, labels, d['tips'][i], initial['tangent'],
                previous_points=None if i == 0 else d['positions'][i-1], current_points=d['positions'][i],
                common_supported=None if i == 0 else d['support'][i-1]&d['support'][i],
                geometry_conditional=bool(d['conditional'][i] or estimated))
            if len(g['path_xy']) < 2:
                raise RuntimeError(f'No contour path in {name} frame {target}')
            records.append(g)
            origin_records.append(origin_tracker.step(d['positions'][i], d['support'][i], d['tips'][i],
                labels, g['path_xy'], g['path_arc'], float(d['times'][i]), d['branches'][i]))
            xy = np.rint(np.nan_to_num(d['positions'][i], nan=-1.e6)).astype(int)
            inside = (xy[:, 0] >= 0)&(xy[:, 0] < labels.shape[1])&(xy[:, 1] >= 0)&(xy[:, 1] < labels.shape[0])
            on_blade = ~inside
            on_blade[inside] = labels[xy[inside, 1], xy[inside, 0]] == 2
            hidden.append(on_blade)
            if i % 80 == 0:
                print(f'{name}: shared geometry {i+1}/{len(d["frames"])}', flush=True)
    finally:
        cap.release()
    g = dict(frames=d['frames'], C_xy=np.array([r['contact_xy'] for r in records]),
        C_state=np.array([r['contact_state'] for r in records]), C_gap_px=np.array([r['tip_gap_px'] for r in records]),
        on_blade=np.array(hidden))
    for key, prefix in (('closed_xy', 'closed'), ('path_xy', 'path')):
        values = [r[key] for r in records]
        g[prefix+'_offsets'] = np.r_[0, np.cumsum([len(value) for value in values])]
        g[prefix+'_xy'] = np.concatenate(values)
    for key in ('closed_state', 'closed_edge_valid', 'path_state', 'path_arc'):
        g[key] = np.concatenate([r[key] for r in records])
    g['path_edge_valid'] = np.concatenate([np.r_[r['path_edge_valid'], False] for r in records])
    mapping = dict(xy='p0_xy', raw_xy='p0_raw_xy', state='p0_state', on_boundary='p0_on_boundary',
        sigma_px='p0_sigma_px', hidden_by_blade='p0_hidden', geometry_ambiguous='p0_geometry_ambiguous',
        contour_arc_px='p0_arc_px', age_since_observation_s='p0_observation_age_s',
        geometry_correction_px='p0_geometry_correction_px', fit_rms_px='p0_fit_rms_px', holdout_px='p0_holdout_px')
    origin = {new: np.array([record[old] for record in origin_records]) for old, new in mapping.items()}
    origin['frames'] = d['frames']
    np.savez_compressed(folder/'geometry.npz', **g)
    np.savez_compressed(folder/'origin.npz', **origin)
    write_json(folder/'shared_preparation.json', dict(cutoff=metadata['cutoff'],
        geometry_config=asdict(geometry_tracker.config), origin_config=asdict(origin_tracker.config),
        mask_sha256=mask_digest.hexdigest(), no_post_cutoff_analysis_samples=True,
        original_flow_unchanged=True))
    return d, g, origin, motion, metadata


def prepare_guarded(case, folder, out):
    global restrict_loading_case
    guards = []
    def restrict(*args, **kwargs):
        result = guarded_prefix(*args, **kwargs)
        guards.append(result['reference'])
        return result
    # Isolated worker-only dependency injection; frozen module files unchanged.
    previous_restrict = restrict_loading_case
    restrict_loading_case = restrict
    args = SimpleNamespace(source=Path(case['tracks_root']),
                           measurements=Path(case['measurements_root']), out=out)
    try:
        d, g, p, m, metadata = prepare_case(case['name'], args, folder, case['cutoff']['depth_mm'])
    finally:
        restrict_loading_case = previous_restrict
    guard = guards[-1]
    indices = np.searchsorted(guard['source_frames'], d['frames'])
    np.testing.assert_array_equal(guard['source_frames'][indices], d['frames'])
    d['blade_observation_qualified'] = guard['analysis_qualified'][indices]
    d['blade_reference_state'] = guard['state'][indices]
    bad = ~d['blade_observation_qualified']
    d['conditional'] |= bad
    d['uncertainty'][bad] = np.maximum(d['uncertainty'][bad], 8.)
    g['C_state'][bad], g['C_xy'][bad] = 0, np.nan
    p['p0_state'][bad], p['p0_on_boundary'][bad] = 4, False
    for i in np.flatnonzero(bad):
        lo, hi = g['path_offsets'][i:i+2]
        g['path_edge_valid'][lo:hi] = False
    for filename, arrays in (('input.npz', d), ('geometry.npz', g), ('origin.npz', p), ('temporal_motion.npz', m)):
        np.savez_compressed(folder/filename, **arrays)
    np.savez_compressed(folder/'blade_reference_guard.npz',
        **{key: value for key, value in guard.items() if isinstance(value, np.ndarray)})
    report = dict(guard['metadata'], source=str(args.source/case['name']),
        old_contact_source=str(folder), new_image_localization_performed=False,
        calibration_changed=False, original_results_used_for_fitting=False,
        maximum_allowed_source_frame=int(guard['source_frames'][-1]),
        post_cutoff_guard_samples=0, affected_contact_samples=int(bad.sum()))
    write_json(folder/'blade_reference_guard.json', report)
    return d, g, p, m, metadata, report
