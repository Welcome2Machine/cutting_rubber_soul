"""Conditional initial-material identity through checked, real-image keyframes.

The graph starts at supplied material IDs in frame zero. Its edges are image
registrations, not physical identity ground truth. Two references share ancestry;
their agreement is a skip-link consistency check, not independent evidence.
Failed observations retain their IDs and both pre-gap references. They never
become templates, and unavailable output coordinates are NaN rather than forecasts.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .material_patch_tracker import (_Bank, _Image, _sample, MaterialPatchConfig,
                                     prepare_verification_image, verify_material_candidates)


@dataclass(frozen=True)
class IdentityChainConfig:
    keyframe_interval: int = 12
    root_check_interval: int = 12
    agreement_px: float = 1.5
    maximum_cycle_budget_px: float = 8.
    cycle_budget_policy: str = 'diagnostic'
    maximum_correction_px: float = 8.
    boundary_mode: str = 'one-sided'
    center_tolerance_px: float = 1.5

    def __post_init__(self):
        for name in ('keyframe_interval', 'root_check_interval'):
            value = getattr(self, name)
            if not np.isfinite(value) or int(value) != value or value < 1:
                raise ValueError(f'{name} must be a finite positive integer')
        for name in ('agreement_px', 'maximum_cycle_budget_px', 'maximum_correction_px'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.boundary_mode != 'one-sided':
            raise ValueError('This explicit-core identity chain requires one-sided boundary mode')
        if self.cycle_budget_policy not in ('diagnostic', 'reject'):
            raise ValueError('Cycle-budget policy must be diagnostic or reject')
        if not np.isfinite(self.center_tolerance_px) or not 0 <= self.center_tolerance_px <= 2:
            raise ValueError('Center tolerance must be between zero and two native pixels')


def _relative_map(prior, reference):
    """Per-ID numerical guard, without imposing a total-deformation J threshold."""
    valid = np.isfinite(prior).all((1, 2)) & np.isfinite(reference).all((1, 2))
    rows = np.flatnonzero(valid)
    relative = np.full(reference.shape, np.nan, np.float32)
    if len(rows):
        stretch = np.linalg.svd(reference[rows].astype(np.float64), compute_uv=False)
        invertible = stretch[:, 1] > stretch[:, 0]*np.finfo(np.float32).eps
        valid[rows[~invertible]] = False
        rows = rows[invertible]
        with np.errstate(over='ignore', invalid='ignore'):
            relative[rows] = prior[rows].astype(np.float64) @ np.linalg.inv(
                reference[rows].astype(np.float64))
        valid &= np.isfinite(relative).all((1, 2))
    return relative, valid


def _raw_image(frame, mask, blade):
    value = np.asarray(frame)
    if value.ndim == 3:
        value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    mask = np.asarray(mask, bool)
    if value.ndim != 2 or mask.shape != value.shape or not np.isfinite(value).all():
        raise ValueError('Finite image and matching material mask required')
    if blade is not None and np.asarray(blade).shape != value.shape:
        raise ValueError('Blade mask must match image')
    # No smoothing here: the common verifier normalizes each real-pixel tile.
    return _Image(value, mask, blade)


class _IdentityBank(_Bank):
    """Reuse the production per-ID pixel atlas, with auditable path metadata."""
    def __init__(self, image, points, cumulative, frame, side, trusted, core):
        super().__init__(image, points, cumulative, frame, side)
        self.trusted = trusted.copy()
        self.budget = np.zeros(len(points), np.float32)
        self.hops = np.zeros(len(points), np.int32)
        self.core_tiles = np.zeros_like(self.blade_tiles)
        self.prepared = None
        self.update_core(core, points, np.ones(len(points), bool))

    def _atlas(self, tiles):
        return tiles.reshape(self.rows, self.columns, self.side, self.side).transpose(
            0, 2, 1, 3).reshape(self.rows*self.side, self.columns*self.side)

    def update_core(self, core, points, valid):
        rows = np.flatnonzero(valid)
        if len(rows):
            xy = np.floor(points[rows])[:, None] + self.offsets
            self.core_tiles[rows] = (0 if core is None else
                _sample(np.asarray(core, np.float32), xy).reshape(-1, self.side, self.side))
        self.core = self._atlas(self.core_tiles)
        if len(rows):
            self.prepared = None

    def prepared_image(self):
        if self.prepared is None:
            self.prepared = prepare_verification_image(self.image.gray, self.image.mask > .5,
                blade=self.image.blade, blade_core=self.core, boundary_mode='one-sided')
        return self.prepared

    def copy_rows(self, source, valid):
        rows = np.flatnonzero(valid)
        if not len(rows):
            return
        for key in ('gray_tiles', 'mask_tiles', 'blade_tiles', 'core_tiles', 'points',
                    'cumulative', 'age', 'atlas_points', 'trusted', 'budget', 'hops'):
            getattr(self, key)[rows] = getattr(source, key)[rows]
        self.image = _Image(self._atlas(self.gray_tiles), self._atlas(self.mask_tiles),
                            self._atlas(self.blade_tiles))
        self.core = self._atlas(self.core_tiles)
        self.prepared = None


class InitialRootedIdentityChain:
    """One fixed set of IDs, strictly increasing frames, bounded reference memory.

    ``proposals`` are positions only, including any saved/neural candidates; they
    cannot promote identity. ``cumulative_affine`` is an optional initialization
    from another tracker, never an observation. Without it, composed accepted
    affine maps initialize matching. All coordinates use native image pixels.
    """
    def __init__(self, initial_frame, material_mask, seed_points, *, point_ids=None,
                 frame_index=0, cutoff_frame=None, blade_mask=None,
                 blade_core_mask=None, config=None, fit_config=None):
        self.config = config or IdentityChainConfig()
        self.fit_config = fit_config or MaterialPatchConfig()
        c = self.config
        if self.fit_config.tile_side < 2*self.fit_config.large_radius+9:
            raise ValueError('Reference tiles must contain both patch support radii with margin')
        points = np.asarray(seed_points, np.float32)
        ids = np.arange(len(points)) if point_ids is None else np.asarray(point_ids)
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
                or not np.isfinite(points).all() or ids.shape != (len(points),)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError('Finite seed positions with immutable unique IDs required')
        if (min(c.keyframe_interval, c.root_check_interval) < 1 or c.agreement_px <= 0
                or c.maximum_cycle_budget_px <= 0 or int(frame_index) != frame_index
                or (cutoff_frame is not None and cutoff_frame < frame_index)):
            raise ValueError('Positive intervals/budgets and ordered frame range required')
        self.point_ids, self.seed_points = ids.copy(), points.copy()
        self.root_frame = self.frame_index = int(frame_index)
        self.cutoff_frame = cutoff_frame
        self.shape = np.asarray(material_mask).shape
        self.last_root_check = self.root_frame
        self.gap_age = np.zeros(len(points), np.int32)
        self.last_supported = np.full(len(points), self.root_frame, np.int64)
        image = _raw_image(initial_frame, material_mask, blade_mask)
        core = blade_mask if blade_core_mask is None else blade_core_mask
        self.options = dict(config=self.fit_config, boundary_mode=c.boundary_mode,
            center_tolerance_px=c.center_tolerance_px,
            maximum_correction_px=c.maximum_correction_px)
        initial = verify_material_candidates(initial_frame, initial_frame, points, points,
            material_mask, material_mask, point_ids=ids, reference_blade=blade_mask,
            current_blade=blade_mask, reference_blade_core=core, current_blade_core=core,
            **self.options)
        self.initial_supported = initial['accepted'].copy()
        self.initial_failure_flags = np.bitwise_or.reduce(
            initial['diagnostics']['failure_flags'], axis=(0, 1))
        self.last_supported[~self.initial_supported] = -1
        cumulative = np.repeat(np.eye(2, dtype=np.float32)[None], len(points), axis=0)
        make = lambda: _IdentityBank(image, points, cumulative, self.root_frame,
            self.fit_config.tile_side, self.initial_supported, core)
        self.root, self.recent, self.keyframe = make(), make(), make()

    def initial_queries(self):
        """Actual initial-frame queries for a neural sequence, without re-labeling."""
        return dict(queries_txy=np.column_stack((np.zeros(len(self.point_ids)), self.seed_points)),
            point_ids=self.point_ids.copy(), rooted_query_supported=self.initial_supported.copy(),
            source_frame_index=self.root_frame)

    def step(self, frame, material_mask, *, frame_index, proposals=None,
             cumulative_affine=None, blade_mask=None, blade_core_mask=None):
        c, n = self.config, len(self.point_ids)
        if (int(frame_index) != frame_index or frame_index <= self.frame_index
                or (self.cutoff_frame is not None and frame_index > self.cutoff_frame)):
            raise ValueError('Strictly increasing frames within the declared cutoff required')
        if np.asarray(material_mask).shape != self.shape:
            raise ValueError('The current image must keep the original pixel coordinate system')
        image = _raw_image(frame, material_mask, blade_mask)
        core = blade_mask if blade_core_mask is None else blade_core_mask
        prepared_current = prepare_verification_image(frame, material_mask,
            blade=blade_mask, blade_core=core, boundary_mode=c.boundary_mode)
        q = np.asarray(self.recent.points if proposals is None else proposals, np.float32)
        if q.ndim == 2:
            q = q[None]
        if q.ndim != 3 or q.shape[1:] != (n, 2) or not len(q):
            raise ValueError('One or more candidate positions in immutable ID order required')
        prior = np.asarray(self.recent.cumulative if cumulative_affine is None
                           else cumulative_affine, np.float32)
        if prior.shape != (n, 2, 2):
            raise ValueError('One cumulative affine initialization per fixed ID required')
        banks = [self.recent, self.keyframe]
        root_due = frame_index-self.last_root_check >= c.root_check_interval
        if root_due:
            banks.append(self.root)
        fits = []
        for bank in banks:
            candidates = q.copy()
            candidates[:, ~bank.trusted] = np.nan
            relative, numerical_valid = _relative_map(prior, bank.cumulative)
            fit = verify_material_candidates(bank.image.gray, frame, bank.atlas_points,
                candidates, bank.image.mask > .5, material_mask, point_ids=self.point_ids,
                initial_affine=relative, reference_blade=bank.image.blade,
                current_blade=blade_mask, reference_blade_core=bank.core,
                current_blade_core=core,
                reference_bounds=np.column_stack((bank.slot+2, bank.slot+bank.side-3)),
                prepared_reference=bank.prepared_image(), prepared_current=prepared_current,
                **self.options)
            fit['accepted'] &= bank.trusted
            fit['numerical_valid'] = numerical_valid
            fits.append(fit)
        if root_due:
            self.last_root_check = int(frame_index)
        accepted = np.stack([f['accepted'] for f in fits])
        ages = np.stack([bank.age for bank in banks])
        scores = np.stack([f['score'] for f in fits])
        positions = np.stack([f['positions'] for f in fits])
        cycles = np.stack([f['cycle_error'] for f in fits])
        anchor = accepted & (ages == self.root_frame)
        disagreement = np.zeros(n, np.float32)
        pair = np.zeros(n, bool)
        for i in range(len(banks)):
            for j in range(i):
                both = accepted[i] & accepted[j] & (ages[i] != ages[j])
                distance = np.linalg.norm(positions[i]-positions[j], axis=1)
                disagreement = np.maximum(disagreement, np.where(both, distance, 0))
                pair |= both & (distance <= c.agreement_px)
        conflict = disagreement > c.agreement_px
        supported = (anchor.any(0) | pair) & ~conflict
        # Root matches take precedence only when other admitted references agree.
        ranked = np.where(accepted, np.where(anchor, scores+10., scores), -np.inf)
        selected = np.argmax(ranked, axis=0)
        row = np.arange(n)
        selected_position = positions[selected, row]
        budget = np.stack([bank.budget for bank in banks]) + cycles
        budget = np.where(accepted, budget, -np.inf)
        accumulated = np.max(budget, axis=0) + disagreement
        direct = anchor.any(0) & supported
        accumulated[direct] = np.min(np.where(anchor, cycles, np.inf), axis=0)[direct]
        over_budget = supported & (accumulated > c.maximum_cycle_budget_px)
        # These correlated cycle/closure residuals accumulate with sample count;
        # they are not an error bound. Keep the former rejection only for ablation.
        if c.cycle_budget_policy == 'reject':
            supported &= ~over_budget
        direct &= supported
        with np.errstate(over='ignore', invalid='ignore'):
            affine = np.stack([fit['relative_affine'] @ bank.cumulative
                               for fit, bank in zip(fits, banks)])[selected, row]
        bad_map = supported & ~np.isfinite(affine).all((1, 2))
        supported &= ~bad_map
        direct &= supported
        hops = np.stack([bank.hops for bank in banks])[selected, row] + 1
        hops[direct] = 1
        recovered = supported & (self.gap_age > 0)
        gap_before = self.gap_age.copy()
        # Only passing current observations can trigger a per-ID bank promotion.
        promote = (supported & self.recent.trusted
                   & (self.recent.age-self.keyframe.age >= c.keyframe_interval))
        self.keyframe.copy_rows(self.recent, promote)
        safe_positions = np.where(supported[:, None], selected_position, self.recent.points)
        safe_affine = np.where(supported[:, None, None], affine, self.recent.cumulative)
        self.recent.update(image, safe_positions, safe_affine, supported,
                           int(frame_index))
        self.recent.update_core(core, safe_positions, supported)
        self.recent.trusted |= supported
        self.recent.budget[supported] = accumulated[supported]
        self.recent.hops[supported] = hops[supported]
        self.gap_age = np.where(supported, 0, self.gap_age+int(frame_index-self.frame_index))
        self.last_supported[supported] = int(frame_index)
        self.frame_index = int(frame_index)
        state = np.full(n, 'unavailable', dtype='<U32')
        state[supported] = 'rooted_conditional'
        state[direct] = 'root_confirmed'
        state[conflict] = 'reference_conflict'
        if c.cycle_budget_policy == 'reject':
            state[over_budget] = 'cycle_budget_exceeded'
        state[bad_map] = 'map_numerical_unavailable'
        # Fixed bank axis: recent, keyframe, immutable root; an unchecked root
        # is -1/False. Flags are unions over all candidates/radii, not mutually
        # exclusive causes or a statement that every candidate failed.
        reference_frames = np.full((3, n), -1, np.int64)
        reference_accepted = np.zeros((3, n), bool)
        reference_failure_flags = np.zeros((3, n), np.uint16)
        reference_attempted = np.zeros((3, n), bool)
        reference_ambiguous = np.zeros((3, n), bool)
        reference_numerical_valid = np.zeros((3, n), bool)
        reference_frames[:len(banks)] = ages
        reference_accepted[:len(banks)] = accepted
        for i, fit in enumerate(fits):
            reference_failure_flags[i] = np.bitwise_or.reduce(
                fit['diagnostics']['failure_flags'], axis=(0, 1))
            reference_attempted[i] = fit['diagnostics']['attempted'].any((0, 1))
            reference_ambiguous[i] = fit['ambiguous']
            reference_numerical_valid[i] = fit['numerical_valid']
        state[~reference_numerical_valid.any(0)] = 'map_numerical_unavailable'
        return dict(frame_index=int(frame_index), point_ids=self.point_ids.copy(),
            positions=np.where(supported[:, None], selected_position, np.nan),
            candidate_positions=q[0].copy(), rooted_supported=supported,
            root_confirmed=direct, rooted_conditional=supported & ~direct,
            cycle_budget_exceeded=over_budget, cycle_budget_policy=c.cycle_budget_policy,
            recovered=recovered, gap_before=gap_before, gap_age=self.gap_age.copy(),
            last_supported_frame=self.last_supported.copy(), state=state,
            reference_frames=reference_frames, reference_accepted=reference_accepted,
            reference_failure_flags=reference_failure_flags,
            reference_attempted=reference_attempted, reference_ambiguous=reference_ambiguous,
            reference_numerical_valid=reference_numerical_valid,
            failure_flag_meanings=fits[0]['failure_flag_meanings'],
            selected_reference_frame=ages[selected, row], reference_disagreement_px=disagreement,
            center_distance_px=np.where(supported,
                np.stack([fit['center_distance_px'] for fit in fits])[selected, row], np.nan),
            reference_center_distance_px=np.where(supported,
                np.stack([fit['reference_center_distance_px'] for fit in fits])[selected, row], np.nan),
            center_extrapolated=supported & np.stack(
                [fit['center_extrapolated'] for fit in fits])[selected, row],
            cumulative_cycle_error_px=np.where(supported, accumulated, np.nan),
            cumulative_affine=np.where(supported[:, None, None], affine, np.nan),
            path_hops=np.where(supported, hops, 0),
            external_affine_prior=cumulative_affine is not None,
            semantics='Image-consistent paths from supplied initial IDs; references share ancestry; not physical identity truth')
