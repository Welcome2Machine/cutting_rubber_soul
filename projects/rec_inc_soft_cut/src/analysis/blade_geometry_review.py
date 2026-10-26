"""Reviewable image-supported blade geometry; never a crack-onset detector.

Uses saved semantic masks, with a raw bright-shaft fallback when a mask has lost
the blade. Top/shaft connectivity and shape exclude the camera watermark. No
trajectory model, forced monotonicity or optical-flow/model inference is used.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ContactGeometryConfig:
    """Pixel-domain audit tolerances, not calibrated measurement uncertainty."""
    tip_agreement_px: float = 6.0
    contact_distance_px: float = 3.0
    clear_distance_px: float = 8.0
    sustained_frames: int = 3
    minimum_material_points: int = 12
    grid_enabled: bool = True

    def __post_init__(self):
        values = (self.tip_agreement_px, self.contact_distance_px, self.clear_distance_px)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Geometry tolerances must be finite and positive')
        if self.clear_distance_px <= self.contact_distance_px:
            raise ValueError('Clearance must exceed the contact distance tolerance')
        if (self.sustained_frames < 2 or self.minimum_material_points < 3
                or int(self.sustained_frames) != self.sustained_frames
                or int(self.minimum_material_points) != self.minimum_material_points):
            raise ValueError('Need integer sustained-frame and material-support counts')


def _material_boundary_measure(mask, gray, tips):
    """Signed distance to semantic material boundary and raw contrast diagnostic.

    The contrast is an image check at a mask-selected location, not an
    independent material-boundary detector. Positive distance means outside.
    """
    material = (mask == 1).astype(np.uint8)
    boundary = material & (1-cv2.erode(material, np.ones((3, 3), np.uint8)))
    yy, xx = np.nonzero(boundary)
    points = np.column_stack((xx, yy))
    distances = np.full(len(tips), np.nan)
    nearest = np.full((len(tips), 2), np.nan)
    contrast = np.full(len(tips), np.nan)
    if not len(points):
        return distances, nearest, contrast
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), .7)
    magnitude = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0),
                              cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    for i, tip in enumerate(tips):
        if not np.isfinite(tip).all():
            continue
        x, y = np.rint(tip).astype(int)
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            continue
        distance = np.linalg.norm(points-tip, axis=1)
        j = int(np.argmin(distance))
        distances[i] = distance[j]*(-1 if material[y, x] else 1)
        nearest[i] = points[j]
        bx, by = points[j]
        contrast[i] = magnitude[by, bx]
    return distances, nearest, contrast


def audit_blade_contact(frames, registered_tips, read_frame, read_mask, *,
                        shaft_x, shaft_width, material_displacement=None,
                        material_supported=None, times_s=None, config=None):
    """Cross-check contact without changing saved tips, depth, or material IDs.

    ``frames`` are contiguous source frame numbers. Callbacks return uint8
    grayscale/BGR frames and same-coordinate labels (1 material, 2 blade).
    Optional displacement is T x N x 2 for FIXED near-surface material IDs,
    with independently removed far-field translation; support is T x N and
    aligned physical ``times_s`` is required when using motion evidence. No
    changing-neighbour median, forecast point or nearest blade pixel qualifies
    as material motion evidence. Results are a diagnostic dict plus NPZ-ready
    arrays. Candidate frame ranges are observation brackets, not confidence
    intervals. Semantic and image apex share pixels and are not independent
    statistical measurements. No contact is inferred solely from a tiny gap.
    """
    from .contact_zero_audit import hinge_onset, sustained_first

    config = config or ContactGeometryConfig()
    frames = np.asarray(frames)
    registered = np.asarray(registered_tips, float)
    if (frames.ndim != 1 or len(frames) < config.sustained_frames
            or not np.issubdtype(frames.dtype, np.integer) or np.any(np.diff(frames) != 1)
            or registered.shape != (len(frames), 2)
            or not np.isfinite([shaft_x, shaft_width]).all() or shaft_width <= 0):
        raise ValueError('Need contiguous integer frames, aligned tips, and a finite shaft corridor')
    if (material_displacement is None) != (material_supported is None):
        raise ValueError('Material displacement and its support must be supplied together')
    if material_displacement is not None:
        times_s = np.asarray(times_s, float)
        if (times_s.shape != frames.shape or not np.isfinite(times_s).all()
                or np.any(np.diff(times_s) <= 0)):
            raise ValueError('Material response audit requires aligned increasing physical times_s')
    n = len(frames)
    semantic, image = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    gaps, nearest, contrast = np.full((n, 3), np.nan), np.full((n, 3, 2), np.nan), np.full((n, 3), np.nan)
    shape = None
    for i, frame in enumerate(frames):
        gray, mask = np.asarray(read_frame(int(frame))), np.asarray(read_mask(int(frame)))
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if (gray.dtype != np.uint8 or mask.ndim != 2 or gray.shape != mask.shape
                or (shape is not None and gray.shape != shape)):
            raise ValueError('Frames and masks must share fixed image coordinates; frame dtype uint8')
        shape = gray.shape
        half = max(85, int(np.ceil(3.2*shaft_width)))
        x0, x1 = max(0, int(shaft_x)-half), min(mask.shape[1], int(shaft_x)+half+1)
        component = _component_tip(mask[:, x0:x1] == 2, x0, shaft_x, shaft_width)
        if component is not None:
            semantic[i] = component[0]
        image[i], _, _ = detect_blade(np.zeros_like(mask), gray, shaft_x, shaft_width,
                                      grid_enabled=config.grid_enabled)
        gaps[i], nearest[i], contrast[i] = _material_boundary_measure(
            mask, gray, [registered[i], semantic[i], image[i]])

    agreement = np.linalg.norm(semantic-image, axis=1)
    qualified = np.isfinite(gaps[:, 1:]).all(1) & (agreement <= config.tip_agreement_px)
    close = qualified & (np.max(abs(gaps[:, 1:]), axis=1) <= config.contact_distance_px)
    clear = qualified & (np.min(gaps[:, 1:], axis=1) > config.clear_distance_px)
    first = sustained_first(close, config.sustained_frames)
    geometric = dict(available=False, reason='no_persistent_geometric_proximity')
    if first is not None:
        preceding = np.flatnonzero(clear[:first])
        begin = int(preceding[-1]) if len(preceding) else 0
        geometric = dict(available=True, candidate_frame=int(frames[first]),
                         frame_range=[int(frames[begin]), int(frames[first])],
                         left_censored=not len(preceding),
                         reason='sustained_two_apex_proximity', contact_observed=False)
    motion = dict(accepted=False, reason='material_response_not_supplied')
    signals = np.full((n, 2), np.nan)
    counts = np.zeros(n, int)
    if material_displacement is not None:
        delta, supported = np.asarray(material_displacement, float), np.asarray(material_supported, bool)
        if delta.ndim != 3 or delta.shape[0] != n or delta.shape[2] != 2 or supported.shape != delta.shape[:2]:
            raise ValueError('Material displacement T x N x 2 and support T x N must align')
        # A common cohort prevents support loss alone from moving the median.
        # This is an offline audit of the supplied short window, not a causal
        # contact detector; report the cohort instead of hiding its attrition.
        valid = supported & np.isfinite(delta).all(2)
        cohort = valid.all(0)
        valid &= cohort[None, :]
        counts = valid.sum(1)
        for i in np.flatnonzero(counts):
            signals[i] = np.median(delta[i, valid[i]], axis=0)
        # Evaluate material evidence even when the two apex estimates disagree;
        # otherwise a geometry conflict masquerades as absent material motion.
        motion = hinge_onset(times_s, signals, -gaps[:, 2],
                             np.isfinite(gaps[:, 2]) & (counts >= config.minimum_material_points))
        motion['fixed_common_cohort_indices'] = np.flatnonzero(cohort).tolist()
        motion['offline_window_cohort_selection'] = True
        motion['gap_basis'] = 'negative image-apex distance to current semantic material boundary, not a fixed reference plane'
        for key in ('frame', 'threshold_frame'):
            if key in motion:
                motion[key] = int(frames[motion[key]])
        for key in ('baseline_frames', 'fit_frames'):
            if key in motion:
                motion[key] = [int(frames[k]) for k in motion[key]]
    overlap = bool(geometric['available'] and not geometric['left_censored']
                   and motion['accepted']
                   and geometric['frame_range'][0] <= motion['frame'] <= geometric['frame_range'][1])
    summary = dict(schema='contact_geometry_crosscheck_1', configuration=asdict(config),
                   geometry=geometric, material_response=motion,
                   state='cross_evidence_candidate' if overlap else 'unresolved',
                   contact_observed=False, depth_modified=False, tip_modified=False,
                   proximity_is_not_contact_proof=True,
                   candidate_frame_range=geometric.get('frame_range') if overlap else None,
                   tip_columns=['registered', 'semantic_apex', 'image_contrast_apex'],
                   material_motion_requires_fixed_ids_and_far_field_correction=True,
                   image_contrast_is_not_independent_material_boundary=True)
    arrays = dict(source_frames=frames, registered_tip_px=registered.copy(),
                  semantic_tip_px=semantic, image_tip_px=image,
                  tip_agreement_px=agreement, signed_material_gap_px=gaps,
                  nearest_material_boundary_px=nearest, boundary_image_gradient=contrast,
                  registered_semantic_delta_px=registered-semantic,
                  registered_image_delta_px=registered-image,
                  geometric_qualified=qualified, geometric_proximity=close,
                  geometric_clear=clear, material_displacement_px=signals,
                  material_supported_count=counts)
    return summary, arrays


def audit_contact_case(case, out, *, before_frames=80, after_frames=50):
    """Reproduce a local geometry audit from an existing optimization case.

    Input conventions are shared with ``softcut optimize``. Raw source video
    and exact-frame semantic masks are read; no segmentation or flow is rerun.
    """
    from src.io import load_npz, read_json, sha, write_json

    def path(value):
        p = Path(value)
        return p if p.is_absolute() else ROOT/p

    if min(before_frames, after_frames) < 1:
        raise ValueError('Both contact-audit windows must be positive')
    td, md, prep = (path(case[k]) for k in ('tracks', 'measurements', 'prepared'))
    audit_path = path(case['contact_audit'])
    audit, plan = read_json(audit_path), read_json(td/'plan.json')
    if not audit.get('accepted'):
        raise ValueError('Existing zero must be explicitly accepted before cross-audit')
    center = int(audit['selected_contact_frame'])
    geometry = load_npz(td/'geometry.npz')
    guard = load_npz(prep/'blade_reference_guard.npz')
    tracks, support = load_npz(td/'tracking/tracks.npz'), load_npz(md/'measurement_support.npz')
    surface = load_npz(md/'initial_surface_extended.npz')
    if not np.array_equal(tracks['point_ids'], support['point_ids']):
        raise ValueError('Tracking and measurement IDs differ')
    frames = np.arange(max(0, center-int(before_frames)), center+int(after_frames)+1)
    def rows(source):
        source = np.asarray(source)
        row = np.searchsorted(source, frames)
        if np.any(row >= len(source)) or not np.array_equal(source[row], frames):
            raise ValueError('Missing exact-frame source data in audit window')
        return row
    p = tracks['positions'][rows(tracks['source_frame_indices'])]
    ok = support['supported'][rows(support['frame_indices'])]
    q, ids = tracks['seed_points'], tracks['point_ids']
    tangent, normal, boundary = surface['tangent'], surface['inward_normal'], surface['boundary_xy']
    station = q@tangent
    boundary_station = boundary@tangent
    order = np.argsort(boundary_station)
    depth = q@normal-np.interp(station, boundary_station[order], (boundary@normal)[order])
    # Saved surface_s has a separate origin; never compare it directly with
    # the absolute image dot-product coordinate q @ tangent.
    surface_s = np.asarray(surface['surface_s_px'])
    if np.any(np.diff(surface_s) <= 0):
        raise ValueError('Initial surface material coordinates must be ordered')
    contact_station = float(np.interp(float(surface['contact_s_px']), surface_s, boundary_station))
    near = (abs(station-contact_station) <= 80) & (depth >= -3) & (depth <= 8)
    # Fixed far-field cohort is selected geometrically, never by low motion.
    far = (abs(station-contact_station) >= 180) & (depth >= 0)
    far &= (ok & np.isfinite(p).all(2)).all(0)
    translation = np.full((len(frames), 2), np.nan)
    if far.sum() >= 20:
        translation = np.median((p-q)[:, far], axis=1)
    delta = (p-q-translation[:, None])[:, near]
    video = path(plan['video'])
    cap = cv2.VideoCapture(str(video))
    images, masks, sources = {}, {}, [td/'plan.json', td/'geometry.npz',
        td/'tracking/tracks.npz', td/'frame_mapping.npz', md/'measurement_support.npz',
        md/'initial_surface_extended.npz', prep/'blade_reference_guard.npz', audit_path, video]
    mapping = load_npz(td/'frame_mapping.npz')
    mapping_rows = rows(mapping['frame_indices'])
    try:
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) <= frames[-1]:
            raise ValueError('Raw compressed source video is shorter than audit window')
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
        for frame in frames:
            success, image = cap.read()
            if not success:
                raise ValueError(f'Cannot decode exact source frame {frame}')
            images[int(frame)] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask_path = td/'segmentation'/case['name']/'semantic_masks'/f'f{frame:06d}.png'
            masks[int(frame)] = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if masks[int(frame)] is None or masks[int(frame)].shape != image.shape[:2]:
                raise ValueError('Raw compressed frame and semantic coordinates differ')
            sources.append(mask_path)
    finally:
        cap.release()
    shaft_x, shaft_width, shaft_evidence = estimate_initial_shaft_adaptive(frames, masks.__getitem__)
    registered = geometry['tips_px'][frames]
    result, arrays = audit_blade_contact(frames, registered, images.__getitem__, masks.__getitem__,
        shaft_x=shaft_x, shaft_width=shaft_width, material_displacement=delta,
        material_supported=ok[:, near], times_s=mapping['physical_time_s'][mapping_rows], config=ContactGeometryConfig(
            grid_enabled=bool(plan.get('input_profile', {}).get('grid_enabled', True))))
    overlap_frames = guard['source_frames']
    overlap_delta = geometry['tips_px'][overlap_frames]-guard['tips_xy']
    overlap_delta = np.linalg.norm(overlap_delta[np.isfinite(overlap_delta).all(1)], axis=1)
    arrays.update(near_point_ids=ids[near], far_point_ids=ids[far],
        far_field_translation_px=translation, registered_source_supported=geometry['supported'][frames],
        acquisition_source_frames=mapping['source_indices'][mapping_rows],
        physical_time_s=mapping['physical_time_s'][mapping_rows])
    i = center-int(frames[0])
    result.update(case=case['name'], existing_contact_frame=center,
        window_source_frames=[int(frames[0]), int(frames[-1])],
        fixed_near_surface_ids=int(near.sum()), fixed_far_field_ids=int(far.sum()),
        near_band_selection='initial normal depth [-3,8] px, tangent distance <=80 px from P0',
        far_field_selection='initial tangent distance >=180 px, fixed support throughout window',
        shaft=shaft_evidence, registered_tip_source='tracks/geometry.npz tips_px; no source switching',
        guard_overlap_finite_frames=len(overlap_delta),
        guard_overlap_tip_max_difference_px=float(overlap_delta.max()) if len(overlap_delta) else None,
        saved_zero_signed_material_gaps_px=arrays['signed_material_gap_px'][i].tolist(),
        saved_zero_semantic_image_apex_disagreement_px=float(arrays['tip_agreement_px'][i]),
        raw_frame_shape=list(images[center].shape), frame_mapping_keys=list(mapping),
        frame_coordinate='compressed source-video frame; acquisition mapping retained in arrays.npz',
        calibration_status='pixel-domain cross-audit; no force or external length calibration',
        future_frames_used_for_offline_audit=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'arrays.npz', **arrays)
    write_json(out/'summary.json', result)
    write_json(out/'provenance.json', dict(source_sha256={str(p): sha(p) for p in sources},
        source_code_sha256=sha(Path(__file__)), sources_unchanged=True,
        microscope_targets_used=False, before_frames=before_frames, after_frames=after_frames))
    montage = []
    chosen = sorted(set([int(frames[0]), max(int(frames[0]), center-20), center,
                         min(int(frames[-1]), center+20), int(frames[-1])]))
    for frame in chosen:
        j = frame-int(frames[0])
        view = cv2.cvtColor(images[frame], cv2.COLOR_GRAY2BGR)
        material = (masks[frame] == 1).astype(np.uint8)
        view[(material-cv2.erode(material, np.ones((3, 3), np.uint8))) > 0] = (90, 180, 90)
        for tip, color in zip([registered[j], arrays['semantic_tip_px'][j], arrays['image_tip_px'][j]],
                              [(240, 220, 40), (240, 100, 210), (30, 200, 250)]):
            if np.isfinite(tip).all():
                cv2.drawMarker(view, tuple(np.rint(tip).astype(int)), color, cv2.MARKER_CROSS, 14, 1)
        x, y = np.rint(registered[i]).astype(int)
        crop = view[max(0, y-80):min(view.shape[0], y+100), max(0, x-90):min(view.shape[1], x+90)]
        tile = np.full((300, 300, 3), 20, np.uint8)
        tile[40:] = cv2.resize(crop, (300, 260))
        cv2.putText(tile, f'{case["name"]} | frame {frame}', (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .44, (240, 240, 240), 1)
        cv2.putText(tile, 'cyan=saved pink=mask gold=image', (5, 34), cv2.FONT_HERSHEY_SIMPLEX, .40, (240, 240, 240), 1)
        montage.append(tile)
    cv2.imwrite(str(out/'geometry_review.png'), np.hstack(montage))
    return result


def _initial_shaft_candidate(mask):
    """One geometrically qualified shaft vote, or None, from one frame."""
    count,labels,stats,_=cv2.connectedComponentsWithStats((mask==2).astype('uint8'),8)
    candidates=[]
    for index in range(1,count):
        x,y,w,h,area=stats[index]
        if y>100 or h<50 or area<150:continue
        row_centers=[];row_widths=[]
        for row in range(max(45,y),min(120,y+h),5):
            xx=np.flatnonzero(labels[row]==index)
            if len(xx) and 5<=np.ptp(xx)+1<=80:
                row_centers.append(float(np.median(xx)));row_widths.append(float(np.ptp(xx)+1))
        if len(row_centers)<3:continue
        width=float(np.median(row_widths))
        if h<1.5*width or np.ptp(row_centers)>max(12,width):continue
        candidates.append((h/width,float(np.median(row_centers)),width))
    return max(candidates)[1:] if candidates else None


def _shaft_vote_estimate(votes):
    if len(votes)<3:raise ValueError('Cannot establish image-reviewable initial shaft corridor')
    centers,widths=np.asarray(votes,float).T
    center=float(np.median(centers));deviation=np.abs(centers-center)
    keep=deviation<=max(8.,3*float(np.median(deviation)))
    return float(np.median(centers[keep])),float(np.median(widths[keep]))


def estimate_initial_shaft_adaptive(indices, read_mask, initial_limit=120, consecutive=5):
    """Preserve early calibration; recover late blade entry from stable votes.

    This only locates the search corridor. Contact and stroke endpoints still
    require separate image observations. A missed/unstable vote resets the run;
    a whole-video median could incorrectly mix the blade with later reflections.
    """
    indices=np.asarray(indices,int)
    if consecutive<3 or np.any(np.diff(indices)<=0):
        raise ValueError('Need ordered unique frames and at least three shaft votes')
    early={int(i):_initial_shaft_candidate(read_mask(int(i))) for i in indices if i<initial_limit}
    votes=[vote for vote in early.values() if vote is not None]
    if len(votes)>=3:
        center,width=_shaft_vote_estimate(votes)
        witnesses=[dict(frame=i,center_px=v[0],width_px=v[1]) for i,v in early.items() if v is not None]
        mode='original_early_window'
    else:
        run=[]
        for i in indices:
            i=int(i)
            vote=early[i] if i in early else _initial_shaft_candidate(read_mask(i))
            if vote is None:
                run=[]
                continue
            run.append(dict(frame=i,center_px=vote[0],width_px=vote[1]))
            run=run[-consecutive:]
            if len(run)<consecutive:continue
            centers=np.array([v['center_px'] for v in run])
            widths=np.array([v['width_px'] for v in run])
            if np.ptp(centers)<=max(8.,.6*float(np.median(widths))) and widths.max()/widths.min()<=1.6:
                center,width=_shaft_vote_estimate(list(zip(centers,widths)))
                witnesses=run
                mode='delayed_entry_stable_run'
                break
        else:
            raise ValueError('Cannot establish image-reviewable shaft: no stable observed entry')
    return center,width,dict(mode=mode,initial_limit_frames=initial_limit,
        initial_valid_votes=len(votes),required_stable_votes=consecutive,
        shaft_x_px=center,shaft_width_px=width,witnesses=witnesses,
        contact_assigned_from_shaft_entry=False)


def _component_tip(binary, x_offset, shaft_x, shaft_width, raw=False):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype('uint8'), 8)
    candidates = []
    for label in range(1, n):
        x, y, width, height, area = stats[label]
        if height < max(50, 1.5*shaft_width) or area < 250:
            continue
        yy, xx = np.nonzero(labels == label)
        bottom = yy.max()
        if bottom >= binary.shape[0]-2:
            # The point is outside/occluded by the image boundary, not an
            # observable tip. This also rejects a full-height camera grid line.
            continue
        end = yy >= bottom-3
        tip = np.array([np.median(xx[end])+x_offset, bottom], float)
        # The pointed end, unlike the bright housing/watermark, must be narrow.
        tail_width = np.median([np.ptp(xx[yy == row])+1 for row in range(max(y, bottom-12), bottom+1)])
        if tail_width > 1.6*shaft_width or abs(tip[0]-shaft_x) > 3*shaft_width:
            continue
        # A later start is allowed when the upper blade is optically blurred,
        # provided a long, narrow shaft is independently present in the image.
        if y > 100 and height < 5*shaft_width:
            continue
        upper = yy < min(y+40, bottom-20)
        upper_x = float(np.median(xx[upper])+x_offset) if upper.any() else tip[0]
        if abs(tip[0]-upper_x) > max(1.8*shaft_width, 65):
            continue
        score = height - .3*abs(tip[0]-shaft_x)
        # Before contact the real shaft enters through the top of the image.
        # A longer disconnected bright material streak below it is not a
        # better blade merely because its connected component is taller.
        if raw and y<=40 and abs(upper_x-shaft_x)<=max(12,shaft_width):score+=10000
        candidates.append((score, tip, dict(y_start=int(y), height=int(height),
                           tail_width_px=float(tail_width), area=int(area))))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1:]


def detect_blade(mask, gray, shaft_x, shaft_width, *, grid_enabled=True):
    """Return (tip, source, diagnostics), or (NaNs, 0, reason).

    ``shaft_x`` identifies the physical shaft corridor, NOT a previous tip. A
    genuine supported candidate can move within this wide corridor, including
    reversals or a new position far from a previous-frame estimate. Source 1 is
    semantic shape support, 2 independent image shaft contrast support.
    """
    half = max(85, int(np.ceil(3.2*shaft_width)))
    x0, x1 = max(0, int(shaft_x)-half), min(mask.shape[1], int(shaft_x)+half+1)
    semantic_candidate = _component_tip(mask[:, x0:x1] == 2, x0, shaft_x, shaft_width)
    # A horizontal top-hat removes the broad material/housing brightness while
    # retaining the thin bright steel blade. It does not invent missing pixels.
    if grid_enabled and gray.shape == (768, 1024):
        from ..tracking.fixed_grid import FixedGridProcessor,grid_0915_config
        # Repair only the known recorded lines. Erasing a five-pixel strip
        # through a coincident blade can destroy the apex or split the shaft.
        image_gray=FixedGridProcessor(grid_0915_config())(gray)
    else:image_gray=gray
    roi = image_gray[:, x0:x1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(2*shaft_width+1)|1, 1))
    contrast = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
    binary = ((contrast >= 20) & (roi >= 75)).astype('uint8')
    binary[:35] = 0
    # Bridge at most 2-pixel image grid/quantisation interruptions; record this
    # morphology in metadata. A long missing shaft is never interpolated here.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), 'uint8'))
    candidate = _component_tip(binary, x0, shaft_x, shaft_width, raw=True)
    if semantic_candidate is not None:
        if candidate is not None:
            image_tip = candidate[0]
            sx, sy = np.rint(semantic_candidate[0]).astype(int)
            rx, ry = np.rint(image_tip).astype(int)
            # A bright shaft may end above a semantic extension into a dark
            # slit. Validate this from the current image, without a previous
            # tip or an assumed velocity. Do not apply it to a dim/blurred shaft.
            shaft = gray[max(35,ry-45):max(36,ry-15), max(0,rx-8):min(gray.shape[1],rx+9)]
            endpoint = gray[max(0,sy-3):min(gray.shape[0],sy+4), max(0,sx-3):min(gray.shape[1],sx+4)]
            if (sy-ry > max(10, .4*shaft_width) and shaft.size and endpoint.size
                    and np.percentile(shaft, 70) > 140 and np.percentile(endpoint, 70) < 125
                    and np.percentile(shaft, 70)-np.percentile(endpoint, 70) > 65):
                candidate[1]['semantic_overrun_px'] = int(sy-ry)
                return image_tip, 2, candidate[1]
            if (ry-sy > max(15, .4*shaft_width) and abs(rx-sx) < shaft_width):
                # Opposite disagreement: the semantic mask may truncate a
                # blurred apex, while the raw component may include material
                # glare. Neither endpoint is automatically ground truth. Keep
                # this as missing rather than fabricate a loading trajectory.
                return np.array([np.nan, np.nan]), 0, {
                    'reason': 'ambiguous_semantic_image_endpoint',
                    'semantic_image_disagreement_px': float(ry-sy),
                    'semantic_tip_candidate': semantic_candidate[0].tolist(),
                    'image_tip_candidate': image_tip.tolist()}
        return semantic_candidate[0], 1, semantic_candidate[1]
    if candidate is not None:
        return candidate[0], 2, candidate[1]
    return np.array([np.nan, np.nan]), 0, {'reason': 'no_supported_shaft_component'}


def isolated_temporal_outliers(tips, supported, threshold_px=12.):
    """Reject isolated one-frame spikes, not sustained moves or reversals."""
    result = np.zeros(len(tips), bool)
    for i in range(1, len(tips)-1):
        if not supported[i-1:i+2].all():
            continue
        a, b, c = tips[i-1:i+2]
        if (np.linalg.norm(c-a) <= 4 and min(np.linalg.norm(b-a), np.linalg.norm(b-c)) > threshold_px):
            result[i] = True
    return result


def fill_short_gaps(tips, supported, max_gap=2):
    """Explicit linear fill only between supported endpoints; support stays False."""
    filled = np.asarray(tips, float).copy()
    interpolated = np.zeros(len(tips), bool)
    i = 0
    while i < len(tips):
        if supported[i]:
            i += 1
            continue
        j = i
        while j < len(tips) and not supported[j]:
            j += 1
        if i > 0 and j < len(tips) and j-i <= max_gap:
            # No bridging a large physical jump or loss of view.
            if np.linalg.norm(tips[j]-tips[i-1]) <= 3*(j-i+1)+2:
                for k in range(i, j):
                    fraction = (k-i+1)/(j-i+1)
                    filled[k] = (1-fraction)*tips[i-1]+fraction*tips[j]
                    interpolated[k] = True
        i = j
    return filled, interpolated


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _review_images(root, name, out, tips, support, source, contact, reversal, line):
    frames = sorted(set([0, max(0, contact-16), max(0, contact-8), contact,
                         contact+8, contact+16, 812, reversal, min(reversal+200, len(tips)-1), len(tips)-1]))
    cap = cv2.VideoCapture(str(root/'data/videos/0923'/f'{name}.mp4'))
    tiles = []
    review = out/'geometry_review'
    review.mkdir(exist_ok=True)
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if not ok:
            continue
        tip = tips[frame]
        annotated = image.copy()
        if np.isfinite(tip).all():
            cv2.drawMarker(annotated, tuple(np.rint(tip).astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 15, 1)
        center, tangent = np.asarray(line['center']), np.asarray(line['tangent'])
        cv2.line(annotated, tuple(np.rint(center-900*tangent).astype(int)),
                 tuple(np.rint(center+900*tangent).astype(int)), (255, 130, 0), 1)
        cv2.putText(annotated, f'{name} f{frame} source={source[frame]} supported={int(support[frame])}',
                    (15, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(review/f'f{frame:06d}_marked.jpg'), annotated)
        if np.isfinite(tip).all():
            x, y = np.rint(tip).astype(int)
            crop = annotated[max(0,y-80):min(image.shape[0],y+90), max(0,x-100):min(image.shape[1],x+100)]
        else:
            crop = annotated
        tile = cv2.resize(crop, (320, 272))
        cv2.rectangle(tile, (0, 0), (320, 24), (0, 0, 0), -1)
        cv2.putText(tile, f'f{frame}: source {source[frame]}', (6,17), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255),1)
        tiles.append(tile)
    cap.release()
    while len(tiles) % 5:
        tiles.append(np.zeros((272, 320, 3), 'uint8'))
    cv2.imwrite(str(out/'geometry_review.jpg'), np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,len(tiles),5)]))


def build_geometry(name, out_dir, root=None):
    """Build saved geometry for one existing video without model inference."""
    root = Path(root) if root is not None else ROOT
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((root/'results/20260925_six_contact_offsets/measurements.json').read_text(encoding='utf-8'))[name]['geometry']
    line = previous['surfaces'][0]
    width = float(previous['width_calibration']['median_width_px'])
    mask_dir = root/'results/20260924_0923_native_grid/segmentation'/name/'semantic_masks'
    mask0 = cv2.imread(str(mask_dir/'f000000.png'), 0)
    near = np.where((mask0 == 2) & (np.indices(mask0.shape)[0] >= 40) & (np.indices(mask0.shape)[0] < 90))
    shaft_x = float(np.median(near[1]))
    cap = cv2.VideoCapture(str(root/'data/videos/0923'/f'{name}.mp4'))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_tips = np.full((count, 2), np.nan)
    source = np.zeros(count, 'uint8')
    gap = np.full(count, np.nan)
    semantic_overrun = np.zeros(count, np.float32)
    endpoint_disagreement = np.zeros(count, np.float32)
    shape_height = np.zeros(count, np.float32)
    frame = 0
    while frame < count:
        ok, image = cap.read()
        if not ok:
            break
        mask = cv2.imread(str(mask_dir/f'f{frame:06d}.png'), 0)
        if mask is not None:
            tip, src, diag = detect_blade(mask, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), shaft_x, width)
            raw_tips[frame], source[frame] = tip, src
            shape_height[frame] = diag.get('height', 0)
            semantic_overrun[frame] = diag.get('semantic_overrun_px', 0)
            endpoint_disagreement[frame] = diag.get('semantic_image_disagreement_px', 0)
            if src:
                # Local nearest-mask gap, not a physical contact ground truth.
                x, y = np.rint(tip).astype(int)
                crop = mask[max(0,y-130):min(mask.shape[0],y+130), max(0,x-130):min(mask.shape[1],x+130)]
                yy, xx = np.nonzero(crop == 1)
                if len(xx):
                    gap[frame] = np.min(np.hypot(xx+max(0,x-130)-tip[0], yy+max(0,y-130)-tip[1]))
        frame += 1
    cap.release()
    supported = source > 0
    spikes = isolated_temporal_outliers(raw_tips, supported)
    supported[spikes] = False
    reject = np.where(source == 0, 1, 0).astype('uint8')
    reject[spikes] = 2
    reject[endpoint_disagreement > 0] = 3
    tips = raw_tips.copy()
    tips[~supported] = np.nan
    tips, interpolated = fill_short_gaps(tips, supported)
    # Sustained proximity rejects a single noisy boundary pixel. This remains
    # segmentation-based contact geometry, whose uncertainty is shown on image.
    contact_old = int(previous['contact_frame'])
    contact_candidates = {}
    for threshold in (1., 2., 4., 6.):
        test = supported & (gap <= threshold)
        candidates = [f for f in range(max(0,contact_old-80), min(count-3,contact_old+80)) if test[f:f+3].all()]
        contact_candidates[str(threshold)] = candidates[0] if candidates else None
    contact = contact_candidates['2.0']
    if contact is None:
        contact = contact_old
    early = contact_candidates['6.0']
    late = contact_candidates['1.0']
    contact_range = [min(contact, early) if early is not None else contact-8,
                     max(contact, late) if late is not None else contact+8]
    valid_ids = np.flatnonzero(supported)
    reversal = int(valid_ids[np.argmax(tips[valid_ids, 1])])
    travel = float(tips[reversal, 1]-tips[contact, 1])
    scale = 3000./travel
    entry = scale*(tips[:,1]-tips[contact,1])
    source_change = np.r_[False, source[1:] != source[:-1]]
    metadata = dict(name=name, display_name=previous['display_name'], frames=count,
        contact_frame=int(contact), contact_frame_range=contact_range,
        contact_frame_status='segmentation_proximity_candidate_requires_image_review',
        contact_range_kind='1_to_6_px_gap_parameter_sensitivity_not_statistical_confidence',
        gap_threshold_first_sustained_frames=contact_candidates,
        reversal_frame=reversal, scale_um_per_px=scale, travel_px=travel,
        scale_method='approximate_3000_um_contact_to_reversal_vertical_travel',
        width_scale_um_per_px=245./width, scale_calibration_relative_difference=float(scale/(245./width)-1),
        surface_center=line['center'], surface_tangent=line['tangent'], surface_angle_deg=line['angle_deg'],
        surface_status='existing_initial_straight_surface_fit_not_deformed_contact_surface',
        shaft_center_x_px=shaft_x, shaft_width_px=width,
        source_transition_frames=np.flatnonzero(source_change).tolist(),
        semantic_dark_tail_overrun_corrected_frames=np.flatnonzero(semantic_overrun > 0).tolist(),
        ambiguous_semantic_image_endpoint_frames=np.flatnonzero(endpoint_disagreement > 0).tolist(),
        support_counts=dict(semantic=int(np.sum(supported & (source == 1))), raw_image=int(np.sum(supported & (source == 2))),
                            directly_supported=int(supported.sum()), interpolated=int(interpolated.sum()), missing=int(np.sum(~np.isfinite(tips).all(axis=1))),
                            isolated_spikes_rejected=int(spikes.sum())),
        source_codes={'0':'unsupported', '1':'semantic_shaft_shape', '2':'raw_bright_shaft_top_hat'},
        reject_codes={'0':'none','1':'no_physical_shaft_component','2':'isolated_temporal_spike','3':'ambiguous_semantic_image_endpoint'},
        interpolation='Only <=2 frame gaps with supported nearby endpoints; supported remains false.',
        detection_note='Wide physical-shaft corridor, pointed-end width and long/top-connected component. Raw fallback uses horizontal bright ridge and 3x3 closing; known 1024x768 camera vertical-grid columns +/-2 pixels excluded. Image-bottom truncation is unsupported. No fixed tip path or monotonicity.',
        limitations=['Masks and image threshold do not provide metrological tip certainty.',
                     'Entry is blade motion in image coordinates; bulk-corrected relative approach must be computed separately.',
                     'Contact range is threshold sensitivity, not a bound on true contact time.',
                     'Semantic and raw threshold tips can differ by a few pixels; source transitions are flagged and should not define blade acceleration.',
                     'Late withdrawal mask may extend into a dark slit; a high-contrast raw endpoint replaces that overrun only with current-image evidence.',
                     'A raw endpoint substantially below the semantic endpoint can mean a truncated blurred blade or connected material glare; it is marked unsupported, not forced onto a monotonic path.',
                     'Supported means an accepted image/mask geometry candidate, not validated metrological certainty.',
                     'No physical crack initiation is inferred by this module.'])
    np.savez_compressed(out/'geometry.npz', frame_indices=np.arange(count), tips_px=tips,
        raw_tips_px=raw_tips, supported=supported, interpolated=interpolated,
        gap_px=gap, entry_image_um=entry, source_code=source, reject_code=reject,
        component_height_px=shape_height, source_change=source_change,
        semantic_dark_tail_overrun_px=semantic_overrun,
        semantic_image_disagreement_px=endpoint_disagreement)
    _review_images(root, name, out, tips, supported, source, contact, reversal, line)
    (out/'geometry.json').write_text(json.dumps(_plain(metadata), ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    parser.add_argument('--out', type=Path, default=ROOT/'results/20260929_recovery_pilot_v2')
    args = parser.parse_args()
    for name in args.names:
        m = build_geometry(name, args.out/name)
        print(json.dumps({k:m[k] for k in ['name','contact_frame','contact_frame_range','reversal_frame','scale_um_per_px','support_counts']}), flush=True)
