"""Loading-only surface-address estimates from fixed original material IDs.

The tip marker is the registered blade tip. The other marker is drawn ONLY on
the initial image. A material-block inverse map may be projected onto that
initial surface when strict surface compatibility fails; this is an explicit
conditional estimate, not proof that the original free surface touches the tip.
Freshly reseeded short-time image points are never inputs to this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .contact_material_mapping import _hull_distance, fit_local


@dataclass(frozen=True)
class SurfaceProjectionConfig:
    current_radius_px: float = 80.
    reference_patch_radii_px: tuple[float, ...] = (36., 54., 78.)
    minimum_points: int = 8
    maximum_condition: float = 20.
    maximum_cv_px: float = 12.
    maximum_rms_px: float = 12.
    maximum_current_extrapolation_px: float = 20.
    maximum_reference_extrapolation_px: float = 30.
    maximum_nearest_distance_px: float = 30.
    maximum_competing_address_px: float = 60.
    tip_sensitivity_px: float = 2.


def _field(data, key, length, default=np.nan, shape=()):
    return np.asarray(data[key]) if key in data else np.full((length, *shape), default)


class FixedMaterialSurfaceProjector:
    """Use a locally supported fixed-ID deformation map, then label-project.

    Unlike the strict surface estimator, original normal depth is not restricted
    to the first few layers. The normal offset is retained, so a deeply interior
    inverse point cannot silently become a confirmed surface contact.
    """
    def __init__(self, seed_points, point_ids, surface_reference, um_per_px, config=None):
        self.reference = np.asarray(seed_points, float).copy()
        self.ids = np.asarray(point_ids, np.int64).copy()
        if (self.reference.ndim != 2 or self.reference.shape[1] != 2
                or self.ids.shape != (len(self.reference),)
                or not np.isfinite(self.reference).all()
                or len(np.unique(self.ids)) != len(self.ids)):
            raise ValueError('Finite original coordinates and unique fixed material IDs are required')
        self.boundary = np.asarray(surface_reference['boundary_xy'], float).copy()
        self.addresses = np.asarray(surface_reference['surface_s_px'], float).copy()
        self.tangent = np.asarray(surface_reference['tangent'], float).copy()
        self.tangent /= np.linalg.norm(self.tangent)
        self.normal = np.asarray(surface_reference['inward_normal'], float).copy()
        self.normal /= np.linalg.norm(self.normal)
        self.axes = np.column_stack([self.tangent, self.normal])
        if (self.boundary.shape != (len(self.addresses), 2)
                or len(self.addresses) < 2 or np.any(np.diff(self.addresses) <= 0)
                or not np.isfinite(self.boundary).all() or abs(np.linalg.det(self.axes)) < .95):
            raise ValueError('A finite ordered initial surface and orthogonal axes are required')
        self.along = self.boundary @ self.tangent
        if np.any(np.diff(self.along) <= 0):
            raise ValueError('Initial surface must be ordered along its tangent')
        self.scale = float(um_per_px)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Pixel calibration must be positive')
        self.contact_s = float(surface_reference['contact_s_px'])
        self.config = config or SurfaceProjectionConfig()

    def surface_point(self, address):
        if not self.addresses[0] <= address <= self.addresses[-1]:
            return np.full(2, np.nan)
        return np.array([np.interp(address, self.addresses, self.boundary[:, d]) for d in range(2)])

    def estimate_frame(self, current_points, supported, tip_xy, *, branch_labels=None, blade_mask=None,
                       previous=None):
        current = np.asarray(current_points, float)
        valid = np.asarray(supported, bool).copy()
        tip = np.asarray(tip_xy, float)
        unknown = dict(state=0, reason='no_supported_fixed_material_patch')
        if current.shape != self.reference.shape or valid.shape != self.ids.shape or tip.shape != (2,):
            raise ValueError('Fixed material coordinate and support shapes do not agree')
        if not np.isfinite(tip).all():
            return dict(unknown, reason='unsupported_blade_tip')
        cfg = self.config
        distance = np.linalg.norm(current - tip, axis=1)
        valid &= np.isfinite(current).all(1) & (distance <= cfg.current_radius_px)
        branches = np.ones(len(valid), int) if branch_labels is None else np.asarray(branch_labels, int)
        if branches.shape != valid.shape:
            raise ValueError('Branch labels must retain fixed material identity order')
        valid &= branches != 0
        if blade_mask is not None:
            mask = np.asarray(blade_mask)
            blade = mask if mask.dtype == bool else mask == 2
            pixel = np.rint(np.nan_to_num(current, nan=-10000)).astype(int)
            inside = ((pixel[:, 0] >= 0) & (pixel[:, 1] >= 0)
                      & (pixel[:, 0] < blade.shape[1]) & (pixel[:, 1] < blade.shape[0]))
            good = np.flatnonzero(inside)
            valid &= inside
            valid[good] &= ~blade[pixel[good, 1], pixel[good, 0]]
        candidates, failures = [], []
        for branch in np.unique(branches[valid]):
            order = np.flatnonzero(valid & (branches == branch))
            order = order[np.argsort(distance[order])]
            centers = []
            for index in order:
                if not centers or np.min(np.linalg.norm(self.reference[centers] - self.reference[index], axis=1)) >= 18:
                    centers.append(int(index))
                if len(centers) == 3:
                    break
            for center in centers:
                reference_distance = np.linalg.norm(self.reference - self.reference[center], axis=1)
                for radius in cfg.reference_patch_radii_px:
                    chosen = valid & (branches == branch) & (reference_distance <= radius)
                    ids = np.flatnonzero(chosen)
                    if len(ids) < cfg.minimum_points or distance[ids].min() > cfg.maximum_nearest_distance_px:
                        continue
                    origin = np.mean(self.reference[ids], axis=0)
                    weights = np.exp(-.5 * (distance[ids] / 35.) ** 2
                                     -.5 * (reference_distance[ids] / (.65 * radius)) ** 2)
                    # A compressed dense patch must not outweigh sparse original
                    # material solely because its initialization was denser.
                    cells = np.floor(self.reference[ids] / 12.).astype(int)
                    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
                    weights /= counts[inverse]
                    fit = fit_local(self.reference[ids], current[ids], weights, origin,
                                    self.axes, check_cv=True)
                    if not fit['valid'] or np.linalg.cond(fit['F']) > cfg.maximum_condition:
                        failures.append('noninvertible_or_ill_conditioned_material_block'); continue
                    if fit['cv'] > cfg.maximum_cv_px or fit['rms'] > cfg.maximum_rms_px:
                        failures.append('material_block_residual_exceeds_projection_limit'); continue
                    inverse_map = np.linalg.inv(fit['F'])
                    original = origin + inverse_map @ (tip - fit['xy'])
                    current_extrap = float(_hull_distance(current[ids], tip))
                    reference_extrap = float(_hull_distance(self.reference[ids], original))
                    if (current_extrap > cfg.maximum_current_extrapolation_px
                            or reference_extrap > cfg.maximum_reference_extrapolation_px):
                        failures.append('material_block_extrapolation_exceeds_projection_limit'); continue
                    along = float(original @ self.tangent)
                    if not self.along[0] <= along <= self.along[-1]:
                        failures.append('outside_visible_initial_surface'); continue
                    address = float(np.interp(along, self.along, self.addresses))
                    foot = self.surface_point(address)
                    normal_offset = float((original-foot) @ self.normal)
                    sensitivity = max(3., float(np.linalg.norm(self.tangent @ inverse_map))
                                      * (cfg.tip_sensitivity_px + fit['cv']) + .25*abs(normal_offset))
                    score = float(fit['cv'] + .08*distance[ids].min() + .05*current_extrap)
                    candidates.append(dict(state=2, reason='conditional_fixed_material_block_projection',
                        reference_s_px=address, reference_marker_xy=foot, raw_inverse_material_xy=original,
                        s_um=(address-self.contact_s)*self.scale,
                        low_um=(address-self.contact_s-sensitivity)*self.scale,
                        high_um=(address-self.contact_s+sensitivity)*self.scale,
                        normal_offset_px=normal_offset, fit_residual_px=fit['rms'],
                        heldout_residual_px=fit['cv'], map_condition=float(np.linalg.cond(fit['F'])),
                        current_extrapolation_px=current_extrap, reference_extrapolation_px=reference_extrap,
                        support_ids=self.ids[ids], branch_label=int(branch), score=score))
        if not candidates:
            if failures:
                unknown['reason'] = max(dict.fromkeys(failures), key=failures.count)
            return unknown
        best = min(candidates, key=lambda c: c['score'])
        comparable = [c for c in candidates if c['score'] <= best['score']+3.]
        spread = float(np.ptp([c['reference_s_px'] for c in comparable]))
        independent_best = best
        if previous is not None and previous.get('state', 0) > 0:
            old_address = previous['s_um'] / self.scale + self.contact_s
            old_ids = np.asarray(previous.get('support_ids', []), np.int64)
            old_branch = previous.get('branch_label', 0)
            def continuity_score(candidate):
                overlap = len(np.intersect1d(old_ids, candidate['support_ids']))
                shared = overlap / max(1, min(len(old_ids), len(candidate['support_ids'])))
                return (candidate['score'] + .06*min(60., abs(candidate['reference_s_px']-old_address))
                        + .75*(old_branch != candidate['branch_label']) + .75*(1-shared))
            best = min(comparable, key=continuity_score)
        # Choose one actual supported material map, never average different
        # branches into a fictitious material identity. Disagreement between
        # individually qualified maps widens a CONDITIONAL interval; it does
        # not manufacture an observation or silently remove its uncertainty.
        result = dict(best)
        result.update(low_um=min(c['low_um'] for c in comparable),
                      high_um=max(c['high_um'] for c in comparable),
                      candidate_count=len(candidates), candidate_spread_px=spread,
                      projection_ambiguous=spread > cfg.maximum_competing_address_px,
                      independent_best_address_um=independent_best['s_um'],
                      continuity_selected=best is not independent_best)
        if result['projection_ambiguous']:
            result['reason'] = 'conditional_competing_material_block_projections'
        return result


def build_loading_surface_sliding(saved_address, frame_indices, tip_xy, *, surface_reference,
                                  um_per_px, seed_points=None, point_ids=None, positions=None,
                                  supported=None, branch_labels=None, blade_masks=None,
                                  tip_state=None, blade_geometry_uncertain=None,
                                  contact_frame=None, reference_frame_index=0, config=None):
    """Restore strict estimates and fill with qualified fixed-block projections.

    Sequence inputs are already sliced to loading. ``blade_masks`` receives the
    loading ROW index. Saved estimates align by source frame labels, not assumed
    NPZ row indices. Callers may omit trajectory inputs for strict-only reuse.
    """
    frames = np.asarray(frame_indices, np.int64)
    tips = np.asarray(tip_xy, float)
    count = len(frames)
    if frames.shape != (count,) or tips.shape != (count, 2) or np.any(np.diff(frames) <= 0):
        raise ValueError('Increasing loading source frames and one tip per frame are required')
    old_frames = np.asarray(saved_address['frame_indices'], np.int64)
    if len(np.unique(old_frames)) != len(old_frames):
        raise ValueError('Saved frame labels must be unique')
    lookup = {int(f): i for i, f in enumerate(old_frames)}
    if any(int(f) not in lookup for f in frames):
        raise ValueError('Every loading frame must be present in saved address data')
    rows = np.array([lookup[int(f)] for f in frames])
    contact_frame = int(frames[0] if contact_frame is None else contact_frame)
    tip_states = np.full(count, 2, np.uint8) if tip_state is None else np.asarray(tip_state)
    uncertain = np.zeros(count, bool) if blade_geometry_uncertain is None else np.asarray(blade_geometry_uncertain, bool)
    if tip_states.shape != (count,) or uncertain.shape != (count,):
        raise ValueError('Tip and blade state arrays must match loading frames')
    boundary = np.asarray(surface_reference['boundary_xy'], float)
    stations = np.asarray(surface_reference['surface_s_px'], float)
    data = dict(frame_indices=frames.copy(), current_marker_xy=tips.copy(),
        state=np.zeros(count, np.uint8), strict_state=np.zeros(count, np.uint8),
        migration_um=np.full(count, np.nan), interval_low_um=np.full(count, np.nan),
        interval_high_um=np.full(count, np.nan), address_um=np.full(count, np.nan),
        address_low_um=np.full(count, np.nan), address_high_um=np.full(count, np.nan),
        address_initial_xy=np.full((count, 2), np.nan), raw_inverse_material_xy=np.full((count, 2), np.nan),
        normal_offset_px=np.full(count, np.nan), fit_residual_px=np.full(count, np.nan),
        heldout_residual_px=np.full(count, np.nan), map_condition=np.full(count, np.nan),
        current_extrapolation_px=np.full(count, np.nan),
        reference_extrapolation_px=np.full(count, np.nan),
        candidate_spread_px=np.full(count, np.nan), projection_ambiguous=np.zeros(count, bool),
        independent_best_address_um=np.full(count, np.nan), continuity_selected=np.zeros(count, bool),
        support_count=np.zeros(count, np.int32), branch_label=np.zeros(count, np.int16),
        evidence_kind=np.full(count, 'unknown', dtype='U64'),
        reason=np.full(count, 'unknown', dtype='U128'),
        strict_reason=np.asarray(saved_address['reason'])[rows].astype('U128'),
        strict_address_um=np.full(count, np.nan),
        initial_surface_xy=boundary.copy(), initial_surface_s_px=stations.copy(),
        initial_contact_xy=np.full(2, np.nan), reference_frame_index=np.array(reference_frame_index),
        original_surface_identity_proven=np.zeros(count, bool),
        diagnostic_saved_raw_projection_um=_field(saved_address, 'raw_reference_s_um', len(old_frames))[rows].copy())
    projector = None
    if positions is not None:
        points = np.asarray(positions)
        support = np.asarray(supported, bool)
        if points.shape[:2] != support.shape or len(points) != count:
            raise ValueError('Loading positions and support have inconsistent shape')
        projector = FixedMaterialSurfaceProjector(seed_points, point_ids, surface_reference, um_per_px, config)
        if points.shape[1:] != projector.reference.shape:
            raise ValueError('Current points must preserve the original fixed-ID count and order')
    records = []
    for row, old in enumerate(rows):
        record = dict(state=0, reason='no_supported_surface_address')
        old_tip = _field(saved_address, 'current_marker_xy', len(old_frames), shape=(2,))[old]
        same_tip = np.isfinite(old_tip).all() and np.linalg.norm(old_tip-tips[row]) <= .25
        old_state = int(saved_address['state'][old])
        if tip_states[row] <= 0 or not np.isfinite(tips[row]).all():
            record['reason'] = 'unsupported_blade_tip'
        elif old_state > 0 and same_tip and np.isfinite(saved_address['s_um'][old]):
            state = 4 if old_state == 2 and tip_states[row] == 2 and not uncertain[row] else 2
            record = dict(state=state, reason='saved_surface_compatible_material_map',
                s_um=float(saved_address['s_um'][old]), low_um=float(saved_address['low_um'][old]),
                high_um=float(saved_address['high_um'][old]),
                reference_marker_xy=np.asarray(saved_address['reference_marker_xy'][old]),
                normal_offset_px=float(_field(saved_address, 'surface_normal_offset_px', len(old_frames))[old]),
                fit_residual_px=float(_field(saved_address, 'fit_residual_px', len(old_frames))[old]),
                heldout_residual_px=float(_field(saved_address, 'heldout_residual_px', len(old_frames))[old]),
                branch_label=int(_field(saved_address, 'branch_label', len(old_frames), 0)[old]),
                raw_inverse_material_xy=_field(saved_address, 'raw_inverse_material_xy', len(old_frames), shape=(2,))[old])
            offsets = saved_address.get('support_id_offsets')
            if offsets is not None:
                record['support_ids'] = saved_address['support_ids_flat'][offsets[old]:offsets[old+1]]
            data['strict_state'][row] = state
            data['strict_address_um'][row] = record['s_um']
        elif projector is not None:
            record = projector.estimate_frame(points[row], support[row], tips[row],
                branch_labels=None if branch_labels is None else branch_labels[row],
                blade_mask=None if blade_masks is None else blade_masks(row),
                previous=records[-1] if records and frames[row] == frames[row-1]+1 else None)
        else:
            record['reason'] = str(saved_address['reason'][old]) if same_tip else 'tip_changed_requires_fixed_material_reestimate'
        data['state'][row] = record['state']
        data['reason'][row] = record['reason']
        if record['state']:
            data['address_um'][row] = record['s_um']
            data['address_low_um'][row] = record['low_um']
            data['address_high_um'][row] = record['high_um']
            data['address_initial_xy'][row] = record['reference_marker_xy']
            data['support_count'][row] = len(record.get('support_ids', []))
            data['evidence_kind'][row] = ('surface_compatible' if data['strict_state'][row]
                                          else 'material_block_projected_surface_address')
            for key in ('normal_offset_px', 'fit_residual_px', 'heldout_residual_px', 'map_condition',
                        'raw_inverse_material_xy', 'branch_label', 'current_extrapolation_px',
                        'reference_extrapolation_px', 'candidate_spread_px', 'projection_ambiguous',
                        'independent_best_address_um', 'continuity_selected'):
                if key in record:
                    data[key][row] = record[key]
        records.append(record)
    anchors = np.flatnonzero(frames == contact_frame)
    anchor = int(anchors[0]) if len(anchors) else None
    anchor_valid = anchor is not None and data['state'][anchor] > 0
    if anchor_valid:
        initial = data['address_um'][anchor]
        data['migration_um'] = data['address_um'] - initial
        data['interval_low_um'] = data['address_low_um'] - data['address_high_um'][anchor]
        data['interval_high_um'] = data['address_high_um'] - data['address_low_um'][anchor]
        data['initial_contact_xy'] = data['address_initial_xy'][anchor].copy()
        # The first-contact datum is its own reference, exactly zero by definition.
        data['interval_low_um'][anchor] = data['interval_high_um'][anchor] = 0.
    else:
        data['state'][:] = 0
        data['reason'][:] = 'initial_contact_address_unavailable_no_slip_baseline'
    ids = [np.asarray(r.get('support_ids', []), np.int64) for r in records]
    data['support_id_offsets'] = np.r_[0, np.cumsum([len(x) for x in ids])]
    data['support_ids_flat'] = np.concatenate(ids) if any(map(len, ids)) else np.empty(0, np.int64)
    data['metadata'] = dict(schema='loading_surface_sliding_v1',
        implementation_version='qualified_fixed_id_continuity_v1',
        state_labels={0: 'unknown', 2: 'conditional', 4: 'surface_supported'},
        frames=count, strict_frames=int(np.sum(data['strict_state'] > 0)),
        conditional_projection_frames=int(np.sum((data['state'] > 0) & (data['strict_state'] == 0))),
        unknown_frames=int(np.sum(data['state'] == 0)), first_contact_baseline_available=bool(anchor_valid),
        ambiguous_projection_frames=int(np.sum(data['projection_ambiguous'])),
        continuity_selected_frames=int(np.sum(data['continuity_selected'])),
        um_per_px=float(um_per_px), contact_frame=contact_frame,
        current_marker='Registered blade tip only; never a distant tracked material point',
        initial_marker='Projected original material address on initial visible surface',
        interpretation='Migration of a fixed-material inverse-map address; projection is conditional when original surface compatibility fails',
        not_measured='True frictional slip, confirmed original free-surface contact, and crack-face identity are not established',
        uncertainty='Empirical fit/registration/projection sensitivity envelope, not statistical confidence interval',
        original_ids_only=True, freshly_reseeded_image_points_used=False, physical_truth_validated=False,
        continuity='Choose only among currently qualified maps using shared fixed IDs, material branch, and address continuity; no monotonicity, held coordinate, or unknown-gap interpolation',
        projection_config=asdict(config or SurfaceProjectionConfig()))
    return data
