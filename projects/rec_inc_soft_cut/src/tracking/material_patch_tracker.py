"""Experimental multi-reference material-coordinate image registration.

This tracker is independent of the accepted neural pipeline. All original IDs
are retained. ``supported`` means local image checks passed, NOT that material
identity or absence of refraction has been proved. Unsupported coordinates are
explicitly motion forecasts. No point is projected onto the blade or a mask.

Real, unwarped image tiles form an immutable initial bank, an incremental bank,
and a periodically renewed keyframe bank. Each carries a cumulative material
deformation gradient. Registration estimates the relative affine map and
brightness gain/bias; cumulative maps are composed, never averaged. Multiple
support sizes and temporal references provide candidates. The selected one is
checked in reverse before it may update the incremental reference.
"""
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


@dataclass
class MaterialPatchConfig:
    small_radius: int = 10
    large_radius: int = 19
    sample_side: int = 11
    tile_side: int = 65
    iterations: int = 9
    reverse_iterations: int = 6
    keyframe_interval: int = 12
    reference_check_interval: int = 4
    initial_check_interval: int = 24
    min_valid_fraction: float = .24
    min_correlation: float = .80
    max_residual: float = .075
    min_texture: float = .012
    min_photometric_inlier_fraction: float = .65
    max_cycle_error: float = .8
    ambiguity_distance: float = 2.0
    ambiguity_score_gap: float = .018
    max_forecast_step: float = 8.0
    use_lk_prior: bool = True


def _gray(frame):
    image = np.asarray(frame)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    value = image.astype(np.float32)
    if image.dtype == np.uint8 or float(np.nanmax(value)) > 1.5:
        value /= 255.
    if not np.isfinite(value).all():
        raise ValueError('Image contains non-finite values')
    return cv2.GaussianBlur(value, (3, 3), .55)


def masked_gray(frame, valid, minimum_support=.25):
    """Normalized initial blur shared by optional boundary and mesh branches."""
    image = np.asarray(frame)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid = np.asarray(valid, bool)
    if valid.shape != image.shape or not 0 < minimum_support <= 1:
        raise ValueError('Matching material mask and positive kernel support required')
    value = image.astype(np.float32)
    if not np.isfinite(value[valid]).all():
        raise ValueError('Valid material image contains non-finite values')
    if image.dtype == np.uint8 or (valid.any() and float(value[valid].max()) > 1.5):
        value /= 255.
    mass = cv2.GaussianBlur(valid.astype(np.float32), (3, 3), .55)
    total = cv2.GaussianBlur(np.where(valid, value, 0), (3, 3), .55)
    gray = np.divide(total, mass, out=np.zeros_like(total), where=mass > 1.e-8)
    supported = valid & (mass >= minimum_support)
    return np.where(supported, gray, 0), supported


def _sample(image, xy):
    points = np.asarray(xy, np.float32)
    return cv2.remap(image, points[..., 0], points[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _mask_centers(mask, points):
    """Nearest pixel-centre membership, with no clipping of material positions."""
    answer = np.zeros(len(points), bool)
    if mask is None:
        return answer
    p = np.asarray(points)
    finite = np.isfinite(p).all(1)
    selected = np.flatnonzero(finite)
    if not len(selected):
        return answer
    rounded = np.rint(p[selected]).astype(np.int64)
    h, w = mask.shape
    inside = ((rounded[:, 0] >= 0) & (rounded[:, 0] < w)
              & (rounded[:, 1] >= 0) & (rounded[:, 1] < h))
    selected, rounded = selected[inside], rounded[inside]
    answer[selected] = np.asarray(mask, bool)[rounded[:, 1], rounded[:, 0]]
    return answer


def _crosses_opaque_core(core, start, end, eligible):
    """Flag consecutive visible steps crossing persistent opaque blade pixels.

    ``core`` is the intersection of the previous and current blade interiors,
    so pixels swept into the scene by blade motion are not treated as a static
    obstacle. This local test is not applied across an observation gap: material
    may have travelled around the tip while hidden. No half-plane is imposed.
    """
    answer = np.zeros(len(start), bool)
    if core is None or not np.any(core):
        return answer
    selected = np.flatnonzero(eligible & np.isfinite(start).all(1) & np.isfinite(end).all(1))
    if not len(selected):
        return answer
    a, b = np.asarray(start, np.float32)[selected], np.asarray(end, np.float32)[selected]
    h, w = core.shape
    lower = np.clip(np.floor(np.minimum(a, b)).astype(np.int32)-1, [0, 0], [w, h])
    upper = np.clip(np.ceil(np.maximum(a, b)).astype(np.int32)+2, [0, 0], [w, h])
    summed = cv2.integral(np.asarray(core, np.uint8), sdepth=cv2.CV_32S)
    counts = (summed[upper[:, 1], upper[:, 0]]-summed[lower[:, 1], upper[:, 0]]
              -summed[upper[:, 1], lower[:, 0]]+summed[lower[:, 1], lower[:, 0]])
    active = np.flatnonzero(counts)
    if not len(active):
        return answer
    a, b = a[active], b[active]
    # At least one midpoint and at most one native pixel between ray samples.
    samples = max(3, int(np.ceil(np.linalg.norm(b-a, axis=1).max()))+1)
    fractions = np.linspace(0, 1, samples, dtype=np.float32)
    xy = a[:, None] + fractions[None, :, None]*(b-a)[:, None]
    answer[selected[active]] = (_sample(np.asarray(core, np.float32), xy) > .5).any(1)
    return answer


class _Image:
    def __init__(self, gray, mask, blade=None):
        self.gray = np.ascontiguousarray(gray, np.float32)
        self.mask = np.ascontiguousarray(mask, np.float32)
        self.gx = cv2.Sobel(self.gray, cv2.CV_32F, 1, 0, ksize=3) / 8
        self.gy = cv2.Sobel(self.gray, cv2.CV_32F, 0, 1, ksize=3) / 8
        self.blade = (None if blade is None or not np.any(blade)
                      else np.ascontiguousarray(blade, np.float32))
        self.blade_integral = (None if self.blade is None else
            cv2.integral((self.blade > 0).astype(np.uint8), sdepth=cv2.CV_32S))


_RAY_FRACTIONS = np.linspace(.125, .875, 7, dtype=np.float32)
_RAY_FRACTIONS.setflags(write=False)


def _unblocked_support(image, center, samples):
    """Local visibility through the opaque blade, not a global side partition.

    Seven interior ray samples avoid combining material across a tool. Camera
    grids or segmentation holes are NOT treated as physical barriers. The
    discrete test is an approximate visibility check, not contact mechanics.
    Material IDs remain free to move around the tip between successive frames.
    """
    if image.blade is None:
        return np.ones(samples.shape[:2], bool)
    # Ray interiors lie in the endpoint bounding rectangle. A conservative
    # one-pixel halo includes every bilinear interpolation neighbour. If that
    # rectangle contains no nonzero blade pixel, all seven original ray tests
    # are exactly false. Integral queries also work for disjoint atlas tiles.
    lower = np.floor(np.minimum(samples.min(1), center)).astype(np.int32)-1
    upper = np.ceil(np.maximum(samples.max(1), center)).astype(np.int32)+2
    h, w = image.blade.shape
    lower = np.clip(lower, [0, 0], [w, h])
    upper = np.clip(upper, [0, 0], [w, h])
    summed = image.blade_integral
    counts = (summed[upper[:, 1], upper[:, 0]]-summed[lower[:, 1], upper[:, 0]]
              -summed[upper[:, 1], lower[:, 0]]+summed[lower[:, 1], lower[:, 0]])
    ids = np.flatnonzero(counts)
    result = np.ones(samples.shape[:2], bool)
    if not len(ids):
        return result
    centers = center[ids]
    fractions = _RAY_FRACTIONS
    if getattr(image, 'strict_visibility', False):
        # Optional boundary verification resolves even thin opaque cores. The
        # production tracker's original seven-sample approximation is unchanged.
        length = np.linalg.norm(samples[ids]-centers[:, None], axis=2).max()
        fractions = np.linspace(0, 1, max(3, int(np.ceil(length*2))+1), dtype=np.float32)
    rays = centers[:, None, None] + fractions[None, :, None, None]*(
        samples[ids]-centers[:, None])[:, None]
    blocked = _sample(image.blade, rays.reshape(len(ids), -1, 2))
    threshold = .001 if getattr(image, 'strict_visibility', False) else .25
    result[ids] = ~(blocked.reshape(len(ids), len(fractions), -1) > threshold).any(1)
    return result


class _Bank:
    """Per-ID real-pixel tiles packed into an atlas for batched OpenCV remap."""
    def __init__(self, image, points, cumulative, step, tile_side):
        self.n = len(points)
        self.side = tile_side
        self.columns = min(16, self.n)
        self.rows = (self.n + self.columns - 1) // self.columns
        self.slot = np.column_stack((np.arange(self.n) % self.columns,
                                     np.arange(self.n) // self.columns)) * tile_side
        axis = np.arange(tile_side, dtype=np.float32) - tile_side // 2
        yy, xx = np.meshgrid(axis, axis, indexing='ij')
        self.offsets = np.stack((xx.ravel(), yy.ravel()), -1)
        self.gray_tiles = np.zeros((self.rows*self.columns, tile_side, tile_side), np.float32)
        self.mask_tiles = np.zeros_like(self.gray_tiles)
        self.blade_tiles = np.zeros_like(self.gray_tiles)
        self.points = np.array(points, np.float32, copy=True)
        self.cumulative = np.array(cumulative, np.float32, copy=True)
        self.age = np.full(self.n, step, np.int32)
        self.atlas_points = np.zeros_like(self.points)
        self.update(image, points, cumulative, np.ones(self.n, bool), step)

    def update(self, image, points, cumulative, valid, step):
        ids = np.flatnonzero(valid)
        if not len(ids):
            return
        origin = np.floor(points[ids])
        xy = origin[:, None] + self.offsets
        self.gray_tiles[ids] = _sample(image.gray, xy).reshape(-1, self.side, self.side)
        self.mask_tiles[ids] = _sample(image.mask, xy).reshape(-1, self.side, self.side)
        self.blade_tiles[ids] = (0. if image.blade is None else
            _sample(image.blade, xy).reshape(-1, self.side, self.side))
        # Prevent interpolating into adjacent IDs in the packed atlas.
        self.mask_tiles[ids, :2] = 0
        self.mask_tiles[ids, -2:] = 0
        self.mask_tiles[ids, :, :2] = 0
        self.mask_tiles[ids, :, -2:] = 0
        self.atlas_points[ids] = (self.slot[ids] + self.side//2
                                 + points[ids] - origin)
        self.points[ids] = points[ids]
        self.cumulative[ids] = cumulative[ids]
        self.age[ids] = step
        shape = (self.rows, self.columns, self.side, self.side)
        gray = self.gray_tiles.reshape(shape).transpose(0, 2, 1, 3)
        mask = self.mask_tiles.reshape(shape).transpose(0, 2, 1, 3)
        blade = self.blade_tiles.reshape(shape).transpose(0, 2, 1, 3)
        self.image = _Image(gray.reshape(self.rows*self.side, self.columns*self.side),
                            mask.reshape(self.rows*self.side, self.columns*self.side),
                            blade.reshape(self.rows*self.side, self.columns*self.side))


@lru_cache(maxsize=16)
def _cached_offsets(sample_side, radius):
    axis = np.linspace(-radius, radius, sample_side, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    value = np.stack((xx.ravel(), yy.ravel()), -1)
    value.setflags(write=False)
    return value


def _offsets(config, radius):
    return _cached_offsets(config.sample_side, radius)


def _fit(reference, target, p, q0, affine0, config, radius, iterations=None,
         target_bounds=None):
    """Batched registration with optional per-ID atlas rectangle constraints.

    ``target_bounds`` contains xmin,ymin,xmax,ymax in the target atlas. A warp
    may not jump over the zero-mask tile border into another ID's stored tile.
    Real image targets do not need this additional memory-layout constraint.
    """
    offset = _offsets(config, radius)
    uv = offset / radius
    template = _sample(reference.gray, p[:, None] + offset)
    reference_valid = _sample(reference.mask, p[:, None] + offset) > getattr(reference, 'mask_threshold', .90)
    reference_valid &= _unblocked_support(reference, p, p[:, None]+offset)
    q = np.array(q0, np.float32, copy=True)
    affine = np.array(affine0, np.float32, copy=True)
    gain = np.ones(len(p), np.float32)
    bias = np.zeros(len(p), np.float32)
    diagonal = np.diag([1e-5, 1e-5, .001, .001, .001, .001, .003, .0003]).astype(np.float32)

    def target_valid(xy):
        valid = reference_valid & (_sample(target.mask, xy) > getattr(target, 'mask_threshold', .90))
        valid &= _unblocked_support(target, q, xy)
        if target_bounds is not None:
            valid &= ((xy >= target_bounds[:, None, :2])
                      & (xy <= target_bounds[:, None, 2:])).all(2)
        return valid

    for _ in range(iterations or config.iterations):
        xy = q[:, None] + offset @ affine.transpose(0, 2, 1)
        valid = target_valid(xy)
        value = _sample(target.gray, xy)
        dx, dy = _sample(target.gx, xy), _sample(target.gy, xy)
        residual = value - gain[:, None]*template - bias[:, None]
        weight = valid * np.minimum(1., .045/np.maximum(np.abs(residual), 1e-6))
        jac = np.stack((dx, dy, dx*uv[None, :, 0], dx*uv[None, :, 1],
                        dy*uv[None, :, 0], dy*uv[None, :, 1],
                        -template, -np.ones_like(template)), -1)
        jt = jac.transpose(0, 2, 1)
        hessian = jt @ (weight[..., None]*jac) + diagonal
        rhs = -(jt @ (weight*residual)[..., None])
        delta = np.linalg.solve(hessian, rhs)[..., 0]
        q += np.clip(delta[:, :2], -1.8, 1.8)
        affine += np.clip(delta[:, 2:6], -1.6, 1.6).reshape(-1, 2, 2)/radius
        gain = np.clip(gain + np.clip(delta[:, 6], -.2, .2), .25, 4)
        bias = np.clip(bias + np.clip(delta[:, 7], -.15, .15), -.6, .6)
    xy = q[:, None] + offset @ affine.transpose(0, 2, 1)
    valid = target_valid(xy)
    value = _sample(target.gray, xy)
    raw_count = np.maximum(valid.sum(1), 1)
    error = value-gain[:, None]*template-bias[:, None]
    raw_residual = np.sqrt((error**2*valid).sum(1)/raw_count)
    # A small moving glint must not invalidate an otherwise informative patch.
    # Reject only gross photometric outliers, retaining at least 65% of the
    # geometrically visible support. This is not liquid-layer separation and
    # cannot identify coherent refractive motion.
    masked_error = np.where(valid, error, np.nan)
    median = np.zeros(len(p), np.float32)
    scale = np.zeros(len(p), np.float32)
    nonempty = valid.any(1)
    median[nonempty] = np.nanmedian(masked_error[nonempty], axis=1)
    scale[nonempty] = 1.4826*np.nanmedian(
        np.abs(masked_error[nonempty]-median[nonempty, None]), axis=1)
    limit = np.clip(3*scale, .055, .10)
    inlier = valid & (np.abs(error-median[:, None]) <= limit[:, None])
    inlier_fraction = inlier.sum(1)/raw_count
    count = np.maximum(inlier.sum(1), 1)
    rt = (template-(template*inlier).sum(1)[:, None]/count[:, None])*inlier
    vt = (value-(value*inlier).sum(1)[:, None]/count[:, None])*inlier
    correlation = (rt*vt).sum(1)/np.maximum(np.sqrt((rt*rt).sum(1)*(vt*vt).sum(1)), 1e-9)
    texture = np.sqrt((rt*rt).sum(1)/count)
    residual = np.sqrt((error**2*inlier).sum(1)/count)
    fraction = valid.mean(1)
    stretch = np.linalg.svd(affine, compute_uv=False)
    accepted = ((fraction >= config.min_valid_fraction)
                & (texture >= config.min_texture)
                & (correlation >= config.min_correlation)
                & (residual <= config.max_residual)
                & (inlier_fraction >= config.min_photometric_inlier_fraction)
                & (stretch.min(1) > .2) & (stretch.max(1) < 5.)
                & (np.linalg.det(affine) > .08) & np.isfinite(q).all(1))
    if target_bounds is not None:
        accepted &= ((q >= target_bounds[:, :2])
                     & (q <= target_bounds[:, 2:])).all(1)
    # Compare photometric evidence; no score bonus for following neural motion.
    score = (correlation - residual*2 + np.minimum(fraction, .65)*.025
             - (1-inlier_fraction)*.06)
    score[~accepted] = -np.inf
    return dict(positions=q, relative_affine=affine, accepted=accepted,
                correlation=correlation, residual=residual, valid_fraction=fraction,
                texture=texture, gain=gain, bias=bias, score=score,
                minimum_stretch=stretch.min(1), maximum_stretch=stretch.max(1),
                determinant=np.linalg.det(affine),
                raw_residual=raw_residual, photometric_inlier_fraction=inlier_fraction)


_VERIFICATION_FAILURES = {
    1: 'invalid_reference_affine', 2: 'reference_center_geometry',
    4: 'proposal_center_geometry', 8: 'nonfinite_proposal',
    16: 'insufficient_pixel_support', 32: 'insufficient_texture',
    64: 'low_correlation', 128: 'large_residual', 256: 'insufficient_inliers',
    512: 'invalid_fitted_affine', 1024: 'fitted_center_geometry',
    2048: 'excessive_local_correction', 4096: 'reverse_fit',
    8192: 'reverse_center_geometry', 16384: 'cycle_error', 32768: 'ambiguity',
}


def _fit_failure_flags(fit, config):
    """Nonexclusive diagnostic bits, preserving failed photometric evidence."""
    flags = np.zeros(len(fit['positions']), np.uint16)
    checks = ((16, fit['valid_fraction'] >= config.min_valid_fraction),
              (32, fit['texture'] >= config.min_texture),
              (64, fit['correlation'] >= config.min_correlation),
              (128, fit['residual'] <= config.max_residual),
              (256, fit['photometric_inlier_fraction'] >= config.min_photometric_inlier_fraction),
              (512, (fit['minimum_stretch'] > .2) & (fit['maximum_stretch'] < 5.)
                    & (fit['determinant'] > .08) & np.isfinite(fit['positions']).all(1)))
    for bit, passed in checks:
        flags[~passed] |= bit
    return flags


def _center_support(mask, barrier, points, tolerance, distance_map=None):
    """Geometric uncertainty only: never dilate photometric support or move IDs."""
    points = np.asarray(points)
    h, w = mask.shape
    inside = (np.isfinite(points).all(1) & (points >= 0).all(1)
              & (points[:, 0] <= w-1) & (points[:, 1] <= h-1))
    distance = np.full(len(points), np.inf, np.float32)
    if mask.any() and inside.any():
        outside = (cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2,
                                        cv2.DIST_MASK_PRECISE)
                   if distance_map is None else distance_map)
        distance[inside] = _sample(outside, points[inside, None])[:, 0]
    blocked = _mask_centers(barrier, points)
    return inside & (distance <= tolerance) & ~blocked, distance


@dataclass(frozen=True)
class _PreparedVerificationImage:
    image: _Image
    geometry: tuple
    boundary_aware: bool
    boundary_mode: str | None


def prepare_verification_image(frame, mask, *, blade=None, blade_core=None,
                               boundary_aware=True, boundary_mode=None):
    """Immutable pixel/geometry snapshot reusable across reference comparisons.

    A supplied snapshot is authoritative; callers must replace it after image,
    mask or core changes. No array shares writable storage with the inputs.
    """
    one_sided = boundary_mode == 'one-sided'
    if (boundary_mode not in (None, 'one-sided') or (one_sided and not boundary_aware)
            or (blade_core is not None and not one_sided)):
        raise ValueError('Consistent boundary mode and blade-core geometry required')
    mask = np.array(mask, bool, copy=True)
    if mask.shape != np.asarray(frame).shape[:2]:
        raise ValueError('Material mask shape must match its image')
    if blade is not None:
        blade = np.array(blade, bool, copy=True)
        if blade.shape != mask.shape:
            raise ValueError('Blade mask shape must match its image')
        mask &= ~blade
    barrier = blade
    if blade_core is not None:
        barrier = np.array(blade_core, bool, copy=True)
        if barrier.shape != mask.shape:
            raise ValueError('Blade core shape must match its image')
        mask &= ~barrier
    gray, valid = masked_gray(frame, mask) if boundary_aware else (_gray(frame), mask)
    if boundary_aware:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    image = _Image(gray, valid, barrier)
    if one_sided:
        image.mask_threshold, image.strict_visibility = .999, True
    distance = (cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE) if one_sided else None)
    for value in [*vars(image).values(), mask, barrier, distance]:
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return _PreparedVerificationImage(image, (mask, barrier, distance),
                                      bool(boundary_aware), boundary_mode)


def verify_material_candidates(reference_frame, current_frame, reference_points, proposals,
                               reference_mask, current_mask, *, reference_blade=None,
                               current_blade=None, initial_affine=None, point_ids=None,
                               config=None, boundary_aware=True, maximum_correction_px=8.,
                               boundary_mode=None, center_tolerance_px=1.5,
                               reference_blade_core=None, current_blade_core=None,
                               reference_bounds=None, prepared_reference=None,
                               prepared_current=None):
    """Image-verify proposals against ONE explicit material reference.

    Proposals have shape (K,N,2), or (N,2); rows are immutable IDs. Two support
    sizes, gain/bias affine registration and independently refitted reverse
    checks share the production fit kernel. Distinct competitive matches remain
    ambiguous. Acceptance is image consistency, not certified material identity;
    the caller must compare initial and recent references independently.
    ``boundary_mode='one-sided'`` separates geometric center uncertainty from
    pixel support. Centers may lie within a declared narrow material-edge band;
    the existing eroded, material-only pixels still supply texture and gradients.
    Supplied full blade masks always exclude photometric samples. Centers and
    support rays exclude the full blade unless a trusted opaque core is supplied
    explicitly. This is conditional one-sided extrapolation, not visible proof
    of a surface point or permission to match across the tool. Default behavior
    is unchanged. Diagnostics retain failed fits for every proposal and radius.
    Optional ``reference_bounds[N,4]`` constrain reverse samples to each ID's
    real-pixel atlas tile; reference and current image shapes may differ.
    Prepared snapshots optionally replace repeated pixel/geometry processing;
    their mode and shape must agree and callers invalidate them after changes.
    No forecast, point deletion, tool projection or neural confidence promotion.
    """
    c = config or MaterialPatchConfig()
    if boundary_mode not in (None, 'one-sided'):
        raise ValueError("boundary_mode must be None or 'one-sided'")
    one_sided = boundary_mode == 'one-sided'
    if one_sided and (not boundary_aware or not np.isfinite(center_tolerance_px)
                      or not 0 <= center_tolerance_px <= 2.):
        raise ValueError('One-sided support requires masked pixels and center tolerance between 0 and 2 pixels')
    if not one_sided and (reference_blade_core is not None or current_blade_core is not None):
        raise ValueError('Explicit blade-core center geometry requires one-sided mode')
    p = np.asarray(reference_points, np.float32)
    q = np.asarray(proposals, np.float32)
    if q.ndim == 2:
        q = q[None]
    ids = np.arange(len(p)) if point_ids is None else np.asarray(point_ids)
    if (p.ndim != 2 or p.shape[1] != 2 or q.ndim != 3 or q.shape[1:] != p.shape
            or not len(q) or not np.isfinite(p).all() or ids.shape != (len(p),)
            or len(np.unique(ids)) != len(ids)
            or not np.isfinite(maximum_correction_px) or maximum_correction_px <= 0):
        raise ValueError('Finite reference positions and unique corresponding material IDs required')
    A = (np.repeat(np.eye(2, dtype=np.float32)[None], len(p), axis=0)
         if initial_affine is None else np.asarray(initial_affine, np.float32))
    if A.shape != (len(p), 2, 2):
        raise ValueError('One initial affine map per material identity required')
    if reference_bounds is not None:
        reference_bounds = np.asarray(reference_bounds, np.float32)
        if (reference_bounds.shape != (len(p), 4) or not np.isfinite(reference_bounds).all()
                or not (reference_bounds[:, :2] <= reference_bounds[:, 2:]).all()):
            raise ValueError('Reference atlas bounds must be finite N x 4 ordered rectangles')
    images, geometry = [], []
    for frame, mask, blade, core, prepared in [
            (reference_frame, reference_mask, reference_blade, reference_blade_core, prepared_reference),
            (current_frame, current_mask, current_blade, current_blade_core, prepared_current)]:
        if prepared is None:
            prepared = prepare_verification_image(frame, mask, blade=blade, blade_core=core,
                boundary_aware=boundary_aware, boundary_mode=boundary_mode)
        elif (not isinstance(prepared, _PreparedVerificationImage)
                or prepared.boundary_aware != boundary_aware or prepared.boundary_mode != boundary_mode
                or prepared.image.gray.shape != np.asarray(frame).shape[:2]
                or prepared.image.gray.shape != np.asarray(mask).shape):
            raise ValueError('Prepared verification snapshot must match image shape and boundary mode')
        images.append(prepared.image)
        geometry.append(prepared.geometry)
    ref, cur = images
    def centers(index, points):
        if one_sided:
            mask, barrier, distance = geometry[index]
            return _center_support(mask, barrier, points, center_tolerance_px, distance)
        valid = _mask_centers(images[index].mask, points)
        return valid, np.where(valid, 0., np.inf)

    radii = (c.small_radius, c.large_radius)
    shape = (len(q), len(radii), len(p))
    diagnostic = {key: np.full(shape, np.nan, np.float32) for key in (
        'correlation', 'residual', 'raw_residual', 'valid_fraction', 'texture',
        'photometric_inlier_fraction', 'minimum_stretch', 'maximum_stretch', 'determinant',
        'reverse_correlation', 'reverse_residual', 'reverse_valid_fraction',
        'cycle_error', 'correction_px', 'proposal_distance_px', 'fitted_distance_px')}
    diagnostic.update(failure_flags=np.zeros(shape, np.uint16),
                      reverse_failure_flags=np.zeros(shape, np.uint16),
                      attempted=np.zeros(shape, bool), accepted=np.zeros(shape, bool))
    result = dict(positions=q[0].copy(), accepted=np.zeros(len(p), bool),
        ambiguous=np.zeros(len(p), bool), point_ids=ids.copy(),
        selected_proposal=np.full(len(p), -1, int), relative_affine=A.copy(),
        cycle_error=np.full(len(p), np.nan), correlation=np.full(len(p), np.nan),
        residual=np.full(len(p), np.nan), valid_fraction=np.full(len(p), np.nan),
        score=np.full(len(p), -np.inf), boundary_aware=bool(boundary_aware),
        boundary_mode=boundary_mode or ('eroded' if boundary_aware else 'legacy'),
        center_tolerance_px=float(center_tolerance_px) if one_sided else 0.,
        center_extrapolated=np.zeros(len(p), bool), diagnostics=diagnostic,
        failure_flag_meanings=_VERIFICATION_FAILURES.copy(), support_radii=np.array(radii),
        observation_definition='Independent image and reverse checks of a supplied reference; not physical identity ground truth')
    finite_A = np.isfinite(A).all((1, 2))
    safe_A = np.where(finite_A[:, None, None], A, np.eye(2))
    affine_ok = finite_A & (np.linalg.det(safe_A) > .08)
    reference_ok, reference_distance = centers(0, p)
    if reference_bounds is not None:
        reference_ok &= ((p >= reference_bounds[:, :2]) & (p <= reference_bounds[:, 2:])).all(1)
    usable = affine_ok & reference_ok
    result['reference_center_distance_px'] = reference_distance
    candidates = []
    for k, proposal in enumerate(q):
        proposal_ok, proposal_distance = centers(1, proposal)
        finite_q = np.isfinite(proposal).all(1)
        flags = ((~affine_ok).astype(np.uint16) | (~reference_ok).astype(np.uint16)*2
                 | (~proposal_ok).astype(np.uint16)*4 | (~finite_q).astype(np.uint16)*8)
        diagnostic['failure_flags'][k] = flags
        diagnostic['proposal_distance_px'][k] = proposal_distance
        use = np.flatnonzero(usable & finite_q & proposal_ok)
        if not len(use):
            continue
        for r, radius in enumerate(radii):
            ix = (k, r, use)
            diagnostic['attempted'][ix] = True
            fit = _fit(ref, cur, p[use], proposal[use], A[use], c, radius)
            fitted_ok, fitted_distance = centers(1, fit['positions'])
            correction = np.linalg.norm(fit['positions']-proposal[use], axis=1)
            flags = _fit_failure_flags(fit, c)
            flags[~fitted_ok] |= 1024
            flags[correction > maximum_correction_px] |= 2048
            for key in diagnostic:
                if key in fit:
                    diagnostic[key][ix] = fit[key]
            diagnostic['fitted_distance_px'][ix] = fitted_distance
            diagnostic['correction_px'][ix] = correction
            valid = fit['accepted'] & fitted_ok & (correction <= maximum_correction_px)
            valid_rows = np.flatnonzero(valid)
            cycle = np.full(len(use), np.inf)
            if len(valid_rows):
                reverse = _fit(cur, ref, fit['positions'][valid_rows], p[use[valid_rows]],
                    np.linalg.inv(fit['relative_affine'][valid_rows]), c, radius,
                    iterations=c.reverse_iterations, target_bounds=(None if reference_bounds is None
                        else reference_bounds[use[valid_rows]]))
                cycle[valid_rows] = np.linalg.norm(reverse['positions']-p[use[valid_rows]], axis=1)
                reverse_ok, _ = centers(0, reverse['positions'])
                flags[valid_rows[~reverse['accepted']]] |= 4096
                if one_sided:
                    flags[valid_rows[~reverse_ok]] |= 8192
                flags[valid_rows[cycle[valid_rows] > c.max_cycle_error]] |= 16384
                valid[valid_rows] &= reverse['accepted'] & (cycle[valid_rows] <= c.max_cycle_error)
                if one_sided:
                    valid[valid_rows] &= reverse_ok
                for key in ('correlation', 'residual', 'valid_fraction'):
                    diagnostic['reverse_'+key][k, r, use[valid_rows]] = reverse[key]
                diagnostic['reverse_failure_flags'][k, r, use[valid_rows]] = _fit_failure_flags(reverse, c)
            diagnostic['failure_flags'][ix] = flags
            diagnostic['cycle_error'][ix] = cycle
            diagnostic['accepted'][ix] = valid
            candidates.append((use, fit, valid, cycle, k, r))
            better = valid & (fit['score'] > result['score'][use])
            chosen = use[better]
            for key in ('positions', 'relative_affine', 'correlation', 'residual', 'valid_fraction', 'score'):
                result[key][chosen] = fit[key][better]
            result['cycle_error'][chosen] = cycle[better]
            result['selected_proposal'][chosen] = k
            result['accepted'][chosen] = True
    for use, fit, valid, cycle, k, r in candidates:
        competitive = valid & (fit['score'] >= result['score'][use]-c.ambiguity_score_gap)
        separated = np.linalg.norm(fit['positions']-result['positions'][use], axis=1) > c.ambiguity_distance
        result['ambiguous'][use] |= competitive & separated
    result['accepted'] &= ~result['ambiguous']
    diagnostic['failure_flags'][:, :, result['ambiguous']] |= 32768
    diagnostic['accepted'][:, :, result['ambiguous']] = False
    _, result['center_distance_px'] = centers(1, result['positions'])
    if one_sided:
        result['center_extrapolated'] = (result['accepted']
            & (result['center_distance_px'] > 1.e-6))
    return result


class MaterialPatchTracker:
    """Incremental material tracker; coordinates are pixels (x, y).

    ``material_mask`` is boolean material support. Excluded pixels may reduce
    evidence but do not erase IDs. ``initial_positions`` is an optional neural
    proposal tested only as another starting point, never blindly copied.
    ``blade_core_mask`` optionally supplies a trusted opaque blade interior,
    e.g. registered geometry confirmed by segmentation. Otherwise the semantic
    blade is eroded by two pixels; its uncertain boundary is not a hard core.
    ``blade_occluded`` labels an opaque projection even when a latent forecast
    remains finite. It does not claim the material itself has ceased to exist.
    All timing is in input sample indices; no physical fps is assumed.
    """
    def __init__(self, frame, material_mask, points, config=None, blade_mask=None,
                 blade_core_mask=None):
        self.config = config or MaterialPatchConfig()
        c = self.config
        if c.tile_side < 2*c.large_radius+9 or c.tile_side % 2 != 1:
            raise ValueError('tile_side must be odd and leave a margin around large_radius')
        if min(c.keyframe_interval, c.reference_check_interval, c.initial_check_interval) < 1:
            raise ValueError('Reference intervals must be positive')
        self.positions = np.asarray(points, np.float32).copy()
        if self.positions.ndim != 2 or self.positions.shape[1] != 2 or not len(points):
            raise ValueError('points must be a nonempty N x 2 array')
        if not np.isfinite(self.positions).all():
            raise ValueError('Initial point coordinates must be finite')
        self.ids = np.arange(len(points), dtype=np.int32)
        self.cumulative = np.repeat(np.eye(2, dtype=np.float32)[None], len(points), 0)
        self.velocity = np.zeros_like(self.positions)
        self.gap = np.zeros(len(points), np.int32)
        self.step_index = 0
        self.previous = self._prepare(frame, material_mask, blade_mask, blade_core_mask)
        self.shape = self.previous.gray.shape
        self.recent = _Bank(self.previous, self.positions, self.cumulative, 0, c.tile_side)
        self.keyframe = _Bank(self.previous, self.positions, self.cumulative, 0, c.tile_side)
        self.initial = _Bank(self.previous, self.positions, self.cumulative, 0, c.tile_side)

    @staticmethod
    def _prepare(frame, material_mask, blade_mask, blade_core_mask=None):
        gray = _gray(frame)
        mask = np.asarray(material_mask, bool).copy()
        if mask.shape != gray.shape:
            raise ValueError('Mask shape must match the image')
        blade = np.zeros_like(mask)
        if blade_mask is not None:
            blade = np.asarray(blade_mask, bool).copy()
            if blade.shape != mask.shape:
                raise ValueError('Blade mask shape must match the image')
        if blade_core_mask is not None:
            # Caller-supplied interior must come from trusted registered blade
            # geometry. It may fill a semantic hole; it is not a material mask.
            core = np.asarray(blade_core_mask, bool).copy()
            if core.shape != mask.shape:
                raise ValueError('Blade core mask shape must match the image')
            blade |= core
        else:
            core = cv2.erode(blade.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        mask &= ~blade
        image = _Image(gray, mask, blade)
        image.blade_core = core if core.any() else None
        image.blade_boundary = (cv2.dilate(blade.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0) & ~core
        return image

    def _prior(self, image):
        c = self.config
        velocity = self.velocity * np.power(.9, np.minimum(self.gap, 12))[:, None]
        prediction = self.positions + np.clip(velocity, -c.max_forecast_step, c.max_forecast_step)
        if not c.use_lk_prior:
            return prediction
        old = np.clip(self.previous.gray*255, 0, 255).astype(np.uint8)
        new = np.clip(image.gray*255, 0, 255).astype(np.uint8)
        q, status, _ = cv2.calcOpticalFlowPyrLK(
            old, new, self.positions[:, None], prediction[:, None].astype(np.float32),
            winSize=(25, 25), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        if q is None:
            return prediction
        reverse, back_status, _ = cv2.calcOpticalFlowPyrLK(
            new, old, q, None, winSize=(25, 25), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
        good = (status[:, 0] > 0) & (back_status[:, 0] > 0)
        good &= np.linalg.norm(reverse[:, 0]-self.positions, axis=1) < 1.2
        good &= self.gap == 0
        prediction[good] = q[good, 0]
        return prediction

    def step(self, frame, material_mask, blade_mask=None, initial_positions=None,
             blade_core_mask=None):
        c = self.config
        image = self._prepare(frame, material_mask, blade_mask, blade_core_mask)
        if image.gray.shape != self.shape:
            raise ValueError('Frame dimensions changed')
        self.step_index += 1
        prior = self._prior(image)
        n = len(self.positions)
        candidates = []
        rejected_blade = np.zeros(n, bool)
        crossed_blade = np.zeros(n, bool)
        persistent_core = (None if image.blade_core is None or self.previous.blade_core is None
                           else image.blade_core & self.previous.blade_core)

        def candidate(bank, bank_id, radius, ids, start):
            ids = np.asarray(ids, np.int32)
            if not len(ids):
                return
            relative = self.cumulative[ids] @ np.linalg.pinv(bank.cumulative[ids])
            fit = _fit(bank.image, image, bank.atlas_points[ids], start[ids],
                       relative, c, radius)
            # Reject individual opaque candidates before competition, so an
            # admissible alternative remains eligible. The old late-only check
            # could reject the winner and unnecessarily forecast the whole ID.
            opaque = _mask_centers(image.blade_core, fit['positions'])
            covered = (np.zeros(len(ids), bool) if image.blade is None else
                _sample(image.blade, fit['positions'][:, None])[:, 0] > .9)
            crossing = _crosses_opaque_core(persistent_core, self.positions[ids],
                fit['positions'], fit['accepted'] & (self.gap[ids] == 0))
            rejected_blade[ids] |= (opaque | covered) & fit['accepted']
            crossed_blade[ids] |= crossing
            fit['accepted'] &= ~(opaque | covered | crossing)
            fit['score'][~fit['accepted']] = -np.inf
            fit.update(ids=ids, bank=bank, bank_id=bank_id, radius=radius)
            candidates.append(fit)

        all_ids = self.ids
        candidate(self.recent, 0, c.small_radius, all_ids, prior)
        first = candidates[0]
        weak = ~first['accepted'] | (first['correlation'] < .92) | (self.gap > 0)
        candidate(self.recent, 0, c.large_radius, all_ids[weak], prior)
        if self.step_index % c.reference_check_interval == 0:
            temporal_ids = all_ids
        else:
            temporal_ids = all_ids[weak]
        candidate(self.keyframe, 1, c.small_radius, temporal_ids, prior)
        if self.step_index % c.initial_check_interval == 0:
            candidate(self.initial, 2, c.large_radius, all_ids, prior)
        if initial_positions is not None:
            external = np.asarray(initial_positions, np.float32)
            if external.shape != self.positions.shape:
                raise ValueError('initial_positions must have shape N x 2')
            distinct = np.isfinite(external).all(1) & (np.linalg.norm(external-prior, axis=1) > 1.)
            # Independent starts are compared even when the local match passes:
            # otherwise repeated texture could silently trap the local solver.
            candidate(self.recent, 3, c.large_radius, all_ids[distinct], external)

        best_score = np.full(n, -np.inf)
        second_score = np.full(n, -np.inf)
        best_index = np.full(n, -1, np.int32)
        best_local = np.zeros(n, np.int32)
        second_position = prior.copy()
        best_position = prior.copy()
        for index, fit in enumerate(candidates):
            ids = fit['ids']
            good = fit['score'] > best_score[ids]
            alternate = ~good & (fit['score'] > second_score[ids])
            second_score[ids[good]] = best_score[ids[good]]
            second_position[ids[good]] = best_position[ids[good]]
            second_score[ids[alternate]] = fit['score'][alternate]
            second_position[ids[alternate]] = fit['positions'][alternate]
            best_score[ids[good]] = fit['score'][good]
            best_index[ids[good]] = index
            best_local[ids[good]] = np.flatnonzero(good)
            best_position[ids[good]] = fit['positions'][good]

        disagreement = np.linalg.norm(best_position-second_position, axis=1)
        disagreement[~np.isfinite(second_score)] = np.nan
        score_gap = np.full(n, np.inf)
        paired = np.isfinite(best_score) & np.isfinite(second_score)
        score_gap[paired] = best_score[paired]-second_score[paired]
        ambiguous = ((disagreement > c.ambiguity_distance)
                     & (score_gap < c.ambiguity_score_gap))
        supported = (best_index >= 0) & ~ambiguous
        cycle = np.full(n, np.nan, np.float32)
        residual = np.full(n, np.nan, np.float32)
        raw_residual = np.full(n, np.nan, np.float32)
        inlier_fraction = np.zeros(n, np.float32)
        correlation = np.full(n, np.nan, np.float32)
        fraction = np.zeros(n, np.float32)
        selected_ref = np.full(n, -1, np.int8)
        selected_age = np.full(n, -1, np.int32)
        radius = np.zeros(n, np.float32)
        cumulative = self.cumulative.copy()
        for index, fit in enumerate(candidates):
            ids = np.flatnonzero(best_index == index)
            if not len(ids):
                continue
            local = best_local[ids]
            bank = fit['bank']
            relative = fit['relative_affine'][local]
            reverse = _fit(image, bank.image, best_position[ids], bank.atlas_points[ids],
                           np.linalg.pinv(relative), c, fit['radius'], c.reverse_iterations,
                           target_bounds=np.column_stack((bank.slot[ids]+2,
                                                          bank.slot[ids]+bank.side-3)))
            cycle[ids] = np.linalg.norm(reverse['positions']-bank.atlas_points[ids], axis=1)
            supported[ids] &= reverse['accepted'] & (cycle[ids] <= c.max_cycle_error)
            residual[ids] = fit['residual'][local]
            raw_residual[ids] = fit['raw_residual'][local]
            inlier_fraction[ids] = fit['photometric_inlier_fraction'][local]
            correlation[ids] = fit['correlation'][local]
            fraction[ids] = fit['valid_fraction'][local]
            selected_ref[ids] = fit['bank_id']
            selected_age[ids] = self.step_index-bank.age[ids]
            radius[ids] = fit['radius']
            cumulative[ids] = relative @ bank.cumulative[ids]

        if image.blade is not None:
            # A surface patch can contain usable material while its centre is
            # currently hidden by the blade. Keep that ID but label the centre
            # unobserved; do not project it to the nearest mask boundary.
            covered = _sample(image.blade, best_position[:, None])[:, 0] > .9
            supported &= ~covered & ~_mask_centers(image.blade_core, best_position)

        # For unsupported points do not promote any image match to a measurement.
        # Preserve IDs and finite forecast positions, never snap to material/tool.
        forecast = self.positions + np.clip(self.velocity, -c.max_forecast_step,
                                            c.max_forecast_step)*np.power(.9, np.minimum(self.gap, 12))[:, None]
        position = np.where(supported[:, None], best_position, forecast).astype(np.float32)
        cumulative[~supported] = self.cumulative[~supported]
        delta = position-self.positions
        self.velocity[supported] = .65*delta[supported] + .35*self.velocity[supported]
        self.gap = np.where(supported, 0, self.gap+1).astype(np.int32)
        self.positions, self.cumulative = position, cumulative.astype(np.float32)
        # Only image-supported and reverse-consistent samples become references.
        self.recent.update(image, position, self.cumulative, supported, self.step_index)
        if self.step_index % c.keyframe_interval == 0:
            stable = supported & (correlation >= .90)
            stable &= ~np.isfinite(disagreement) | (disagreement <= c.ambiguity_distance)
            self.keyframe.update(image, position, self.cumulative, stable, self.step_index)
        self.previous = image
        # An occluded projection is retained as a latent prediction. It is not
        # projected to a contact edge, labelled an observed surface point, or
        # removed from the fixed cohort. Renderers can respect opaque visibility
        # while reporting hidden IDs in the reference-material panel/counts.
        blade_occluded = _mask_centers(image.blade_core, position)
        return dict(positions=position.copy(), point_ids=self.ids.copy(),
                    supported=supported, inferred=~supported, residual=residual,
                    raw_residual=raw_residual, photometric_inlier_fraction=inlier_fraction,
                    correlation=correlation, cycle_error=cycle,
                    affine=self.cumulative.copy(), cumulative_affine=self.cumulative.copy(),
                    valid_fraction=fraction, reference_disagreement=disagreement,
                    gap_age=self.gap.copy(), selected_reference=selected_ref,
                    reference_age=selected_age, support_radius=radius,
                    ambiguous=ambiguous, blade_occluded=blade_occluded,
                    blade_boundary_uncertain=_mask_centers(image.blade_boundary, position),
                    candidate_rejected_by_blade=rejected_blade,
                    candidate_crossed_blade=crossed_blade)
