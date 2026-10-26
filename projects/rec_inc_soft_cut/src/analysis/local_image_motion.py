"""Short-time, freshly seeded image motion near a blade tip.

These samples describe current visible texture, not persistent original material
identities. They can complement a failed long trajectory when estimating a
local motion trend; they cannot establish which original surface point is in
contact, nor distinguish rubber texture from coherently moving lubricant.

Pyramidal LK supplies candidates. A material-only, gain/bias-invariant patch
check and a reverse check validate them. Difficult candidates may use a local
affine patch registration. Opaque blade pixels never supply registration
residuals, candidates inside the blade and consecutive paths across its
persistent interior are rejected. No spatial smoothness is imposed across it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LocalImageMotionConfig:
    radius_px: float = 110.
    max_points: int = 320
    min_distance_px: float = 5.
    corner_quality: float = .008
    stratify_blade_bands: bool = True
    vertical_bin_px: float = 0.
    lk_window: int = 21
    pyramid_levels: int = 3
    patch_radius: int = 7
    min_patch_fraction: float = .32
    min_patch_std: float = .035
    min_correlation: float = .72
    max_cycle_error_px: float = 1.0
    max_displacement_px: float = 24.
    blade_core_margin_px: int = 2
    affine_rescue: bool = True
    max_affine_rescues: int = 60
    affine_iterations: int = 7


def _gray(image):
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError('Images must be gray or BGR')
    value = image.astype(np.float32)
    if image.dtype == np.uint8 or value.max(initial=0) > 1.5:
        value /= 255.
    if not np.isfinite(value).all():
        raise ValueError('Images must be finite')
    return value


def _normalize(gray):
    # Smooth illumination is removed without global histogram changes. The
    # noise floor prevents untextured/shadow regions being amplified freely.
    low = cv2.GaussianBlur(gray, (0, 0), 5.)
    residual = gray-low
    scale = np.sqrt(cv2.GaussianBlur(residual*residual, (0, 0), 5.) + .015**2)
    return np.clip(.5 + .18*residual/scale, 0., 1.).astype(np.float32)


def _sample(image, xy, nearest=False):
    xy = np.asarray(xy, np.float32)
    return cv2.remap(np.asarray(image), xy[..., 0], xy[..., 1],
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _inside(shape, xy):
    h, w = shape
    return (np.isfinite(xy).all(-1) & (xy[..., 0] >= 0) &
            (xy[..., 0] <= w-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h-1))


def _centers(mask, xy):
    if not len(xy):
        return np.zeros(0, bool)
    return _inside(mask.shape, xy) & (_sample(mask.astype(np.uint8), xy[:, None], True)[:, 0] > 0)


def _path_crosses(core, start, end):
    if not len(start) or not core.any():
        return np.zeros(len(start), bool)
    distance = np.linalg.norm(end-start, axis=1)
    samples = min(128, max(3, int(np.ceil(np.nanmax(distance, initial=0)))+1))
    xy = start[:, None] + np.linspace(0, 1, samples)[None, :, None]*(end-start)[:, None]
    return (_sample(core.astype(np.uint8), xy, True) > 0).any(1)


def _visible_support(core, centers, xy):
    answer = np.ones(xy.shape[:-1], bool)
    if not core.any() or not len(centers):
        return answer
    # This excludes a patch reaching around/through the blade to its other side.
    for fraction in (.25, .5, .75):
        ray = centers[:, None] + fraction*(xy-centers[:, None])
        answer &= _sample(core.astype(np.uint8), ray, True) == 0
    return answer


def _offsets(radius):
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    return np.stack((xx.ravel(), yy.ravel()), -1).astype(np.float32)


def _select_seeds(gray, allowed, blade, tip, config):
    """Reserve seeds for both visible blade flanks before filling the near field."""
    def detect(mask, quota):
        if quota <= 0 or not mask.any():
            return np.empty((0, 2), np.float32)
        found = cv2.goodFeaturesToTrack(gray, maxCorners=quota,
            qualityLevel=config.corner_quality, minDistance=config.min_distance_px,
            mask=mask.astype(np.uint8)*255, blockSize=5, useHarrisDetector=False)
        return np.empty((0, 2), np.float32) if found is None else found[:, 0]
    if not config.stratify_blade_bands or not blade.any():
        return detect(allowed, config.max_points)
    distance = cv2.distanceTransform((~blade).astype(np.uint8), cv2.DIST_L2, 5)
    yy, xx = np.indices(gray.shape)
    remaining = allowed.astype(np.uint8)
    chunks = []
    if config.vertical_bin_px > 0:
        # Equal quotas by blade-height section keep upper sparse-track regions
        # represented even when the near-tip image has stronger corners.
        bins = np.floor((tip[1]-yy)/config.vertical_bin_px).astype(int)
        zones = []
        for section in np.unique(bins[allowed]):
            for side in (xx < tip[0], xx >= tip[0]):
                for low, high in ((0., 12.), (12., 40.)):
                    zone = allowed & (bins == section) & side & (distance > low) & (distance <= high)
                    if zone.any():
                        zones.append(zone)
        quota = max(1, config.max_points//max(1, len(zones)))
        for zone in zones:
            seeds = detect(remaining.astype(bool) & zone, min(quota, config.max_points-sum(map(len, chunks))))
            chunks.append(seeds)
            for point in seeds:
                cv2.circle(remaining, tuple(np.rint(point).astype(int)),
                           int(np.ceil(config.min_distance_px)), 0, -1)
        chunks.append(detect(remaining.astype(bool), config.max_points-sum(map(len, chunks))))
        return np.concatenate(chunks) if chunks else np.empty((0, 2), np.float32)
    quota = max(1, int(config.max_points*.18))
    for side in (xx < tip[0], xx >= tip[0]):
        for low, high in ((0., 12.), (12., 40.)):
            seeds = detect(remaining.astype(bool) & side & (distance > low) & (distance <= high), quota)
            chunks.append(seeds)
            for point in seeds:
                cv2.circle(remaining, tuple(np.rint(point).astype(int)),
                           int(np.ceil(config.min_distance_px)), 0, -1)
    used = sum(map(len, chunks))
    chunks.append(detect(remaining.astype(bool), config.max_points-used))
    return np.concatenate(chunks)


def _patch_quality(a, b, mask0, mask1, core0, core1, p, q, offsets, affine=None):
    xy0 = p[:, None] + offsets
    warped = np.broadcast_to(offsets, (len(p), *offsets.shape))
    if affine is not None:
        warped = np.einsum('nij,kj->nki', affine, offsets)
    xy1 = q[:, None] + warped
    values0, values1 = _sample(a, xy0), _sample(b, xy1)
    usable = ((_sample(mask0.astype(np.uint8), xy0, True) > 0) &
              (_sample(mask1.astype(np.uint8), xy1, True) > 0) &
              _visible_support(core0, p, xy0) & _visible_support(core1, q, xy1))
    count = usable.sum(1)
    weight = usable/np.maximum(count[:, None], 1)
    x = values0 - (values0*weight).sum(1)[:, None]
    y = values1 - (values1*weight).sum(1)[:, None]
    sx, sy = np.sqrt((x*x*weight).sum(1)), np.sqrt((y*y*weight).sum(1))
    correlation = (x*y*weight).sum(1)/np.maximum(sx*sy, 1e-8)
    return correlation, usable.mean(1), np.minimum(sx, sy)


def _fit_affine(a, b, mask0, mask1, core0, core1, p, q, offsets, config, initial_affine=None):
    """Robust material-only affine fit, with brightness gain and offset."""
    source_xy = p[None] + offsets
    source = _sample(a, source_xy[None])[0]
    source_ok = ((_sample(mask0.astype(np.uint8), source_xy[None], True)[0] > 0) &
                 _visible_support(core0, p[None], source_xy[None])[0])
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)/8
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)/8
    center = np.array(q, np.float32, copy=True)
    affine = np.eye(2, dtype=np.float32) if initial_affine is None else np.array(initial_affine, np.float32)
    gain, bias = 1., 0.
    radius = float(config.patch_radius)
    uv = offsets/radius
    for _ in range(config.affine_iterations):
        target_xy = center + offsets@affine.T
        ok = (source_ok & (_sample(mask1.astype(np.uint8), target_xy[None], True)[0] > 0) &
              _visible_support(core1, center[None], target_xy[None])[0])
        if ok.sum() < max(24, len(offsets)*config.min_patch_fraction):
            return center, affine, False
        target = _sample(b, target_xy[None])[0]
        dx, dy = _sample(gx, target_xy[None])[0], _sample(gy, target_xy[None])[0]
        residual = target-gain*source-bias
        jac = np.column_stack((dx, dy, dx*uv[:, 0], dx*uv[:, 1],
                               dy*uv[:, 0], dy*uv[:, 1], -source, -np.ones(len(source))))
        r, j = residual[ok], jac[ok]
        robust_scale = max(.02, 1.4826*np.median(abs(r-np.median(r))))
        weight = np.minimum(1., 1.5*robust_scale/np.maximum(abs(r), 1e-6))
        jw = j*weight[:, None]
        hessian = j.T@jw + np.diag([1e-4, 1e-4, .002, .002, .002, .002, .001, .001])
        try:
            step = np.linalg.solve(hessian, -jw.T@r)
        except np.linalg.LinAlgError:
            return center, affine, False
        # Cap one iteration; this is an optimization safeguard, never a valid
        # measurement limit or a projection onto a material/blade boundary.
        scale = max(1., np.linalg.norm(step[:2])/2., np.max(abs(step[2:6]))/2.)
        step /= scale
        center += step[:2]
        affine += step[2:6].reshape(2, 2)/radius
        gain += step[6]
        bias += step[7]
        singular = np.linalg.svd(affine, compute_uv=False)
        if singular.min() < .5 or singular.max() > 2. or np.linalg.det(affine) <= 0:
            return center, affine, False
        if np.linalg.norm(step[:6]) < .015:
            break
    return center, affine, True


def estimate_local_image_motion(frame0, frame1, labels0, labels1, tip0_xy, tip1_xy,
                                *, dt_s=None, config=None, seed_xy=None, seed_region_mask=None):
    """Estimate independent current-texture motion between two nearby frames.

    Labels: 1 material, 2 blade; other labels never seed/support a measurement.
    Returned arrays keep invalid seeds for diagnostics. Validity means local
    image checks passed, not independently proven physical material identity.
    Positive y is image-down. ``relative_*`` subtracts blade translation.
    ``dt_s`` must use acquisition time, not slowed visualization playback time.
    """
    cfg = config or LocalImageMotionConfig()
    if cfg.radius_px <= 0 or cfg.max_points < 1 or cfg.patch_radius < 2 or cfg.vertical_bin_px < 0:
        raise ValueError('Invalid motion configuration')
    if dt_s is not None and (not np.isfinite(dt_s) or dt_s <= 0):
        raise ValueError('dt_s must be positive physical elapsed time')
    gray0, gray1 = _gray(frame0), _gray(frame1)
    labels0, labels1 = np.asarray(labels0), np.asarray(labels1)
    tip0, tip1 = np.asarray(tip0_xy, float), np.asarray(tip1_xy, float)
    if gray0.shape != gray1.shape or labels0.shape != gray0.shape or labels1.shape != gray0.shape:
        raise ValueError('Both images and semantic masks must have equal size')
    if tip0.shape != (2,) or tip1.shape != (2,) or not np.isfinite([tip0, tip1]).all():
        raise ValueError('Both blade tips must be finite two-dimensional coordinates')
    # Restrict work to a union crop while preserving all public coordinates.
    pad = cfg.radius_px + cfg.max_displacement_px + cfg.lk_window
    h, w = gray0.shape
    lower = np.maximum(np.floor(np.minimum(tip0, tip1)-pad).astype(int), 0)
    upper = np.minimum(np.ceil(np.maximum(tip0, tip1)+pad+1).astype(int), [w, h])
    if seed_region_mask is not None:
        seed_region_mask = np.asarray(seed_region_mask, bool)
        if seed_region_mask.shape != gray0.shape:
            raise ValueError('seed_region_mask must match the source image')
        ys, xs = np.nonzero(seed_region_mask & (labels0 == 1))
        if len(xs):
            margin = cfg.max_displacement_px+cfg.lk_window
            lower = np.minimum(lower, np.maximum(np.floor([xs.min()-margin, ys.min()-margin]), 0).astype(int))
            upper = np.maximum(upper, np.minimum(np.ceil([xs.max()+margin+1, ys.max()+margin+1]), [w, h]).astype(int))
    if (upper <= lower).any():
        raise ValueError('Tip neighborhood is outside the image')
    sl = np.s_[lower[1]:upper[1], lower[0]:upper[0]]
    a, b = _normalize(gray0[sl]), _normalize(gray1[sl])
    mask0, mask1 = labels0[sl] == 1, labels1[sl] == 1
    blade0, blade1 = labels0[sl] == 2, labels1[sl] == 2
    kernel = np.ones((2*cfg.blade_core_margin_px+1,)*2, np.uint8)
    core0 = cv2.erode(blade0.astype(np.uint8), kernel).astype(bool)
    core1 = cv2.erode(blade1.astype(np.uint8), kernel).astype(bool)
    if seed_xy is None:
        yy, xx = np.indices(a.shape)
        allowed = mask0 & ((xx+lower[0]-tip0[0])**2+(yy+lower[1]-tip0[1])**2 <= cfg.radius_px**2)
        if seed_region_mask is not None:
            allowed = mask0 & seed_region_mask[sl]
        # Feature selection uses real contrast, before local normalization.
        p = _select_seeds(gray0[sl], allowed, blade0, tip0-lower, cfg)
    else:
        p = np.asarray(seed_xy, np.float32)-lower.astype(np.float32)
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError('seed_xy must be finite N,2 coordinates')
    n = len(p)
    q = np.array(p, copy=True)
    status = np.zeros(n, bool)
    fb = np.full(n, np.inf, np.float32)
    method = np.zeros(n, np.uint8)  # 0 LK, 1 affine rescue
    affine = np.repeat(np.eye(2, dtype=np.float32)[None], n, axis=0)
    offsets = _offsets(cfg.patch_radius)
    if n:
        a8, b8 = np.uint8(a*255), np.uint8(b*255)
        args = dict(winSize=(cfg.lk_window,)*2, maxLevel=cfg.pyramid_levels,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, .01))
        tracked, forward_status, _ = cv2.calcOpticalFlowPyrLK(a8, b8, p[:, None], None, **args)
        q = tracked[:, 0]
        back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(b8, a8, q[:, None], None, **args)
        fb = np.linalg.norm(back[:, 0]-p, axis=1)
        status = (forward_status[:, 0] > 0) & (reverse_status[:, 0] > 0) & np.isfinite(q).all(1)
        correlation, fraction, texture = _patch_quality(a, b, mask0, mask1, core0, core1, p, q, offsets)
        if cfg.affine_rescue:
            eligible = (status & (texture >= cfg.min_patch_std) &
                        (fraction >= cfg.min_patch_fraction) &
                        (np.linalg.norm(q-p, axis=1) <= cfg.max_displacement_px) &
                        _centers(mask0, p) & _centers(mask1, q) &
                        ((correlation < .90) | (fb > .45)))
            ids = np.flatnonzero(eligible)
            # Prefer potentially useful candidates over grossly unrelated ones.
            ids = ids[np.argsort(correlation[ids])[::-1]][:cfg.max_affine_rescues]
            for i in ids:
                candidate, transform, ok = _fit_affine(a, b, mask0, mask1, core0, core1,
                                                       p[i], q[i], offsets, cfg)
                if not ok:
                    continue
                reverse, _, reverse_ok = _fit_affine(b, a, mask1, mask0, core1, core0,
                      candidate, p[i], offsets, cfg, np.linalg.inv(transform))
                if not reverse_ok:
                    continue
                cc, ff, tt = _patch_quality(a, b, mask0, mask1, core0, core1,
                           p[i:i+1], candidate[None], offsets, transform[None])
                cycle = np.linalg.norm(reverse-p[i])
                improvement = cc[0]-.05*cycle > correlation[i]-.05*min(fb[i], 10.)+.005
                eligible = (cc[0] >= cfg.min_correlation and ff[0] >= cfg.min_patch_fraction and
                    tt[0] >= cfg.min_patch_std and cycle <= cfg.max_cycle_error_px and
                    _centers(mask1, candidate[None])[0] and
                    not _path_crosses(core0 & core1, p[i:i+1], candidate[None])[0] and
                    np.linalg.norm(candidate-p[i]) <= cfg.max_displacement_px)
                if improvement and eligible:
                    q[i], affine[i], fb[i], method[i] = candidate, transform, cycle, 1
            correlation, fraction, texture = _patch_quality(a, b, mask0, mask1, core0, core1, p, q, offsets, affine)
    else:
        correlation, fraction, texture = (np.empty(0, np.float32) for _ in range(3))
    crossing = _path_crosses(core0 & core1, p, q)
    on_material = _centers(mask0, p) & _centers(mask1, q)
    displacement = q-p
    valid = (status & on_material & ~crossing &
             (np.linalg.norm(displacement, axis=1) <= cfg.max_displacement_px) &
             (fb <= cfg.max_cycle_error_px) & (correlation >= cfg.min_correlation) &
             (fraction >= cfg.min_patch_fraction) & (texture >= cfg.min_patch_std))
    confidence = (np.clip((correlation-.4)/.6, 0, 1)*np.exp(-.5*(fb/cfg.max_cycle_error_px)**2)*
                  np.clip(fraction/.7, 0, 1)*np.clip(texture/.1, 0, 1))
    confidence[~valid] = 0
    source, current = p+lower, q+lower
    branch = np.where(source[:, 0] < tip0[0], 1, 2).astype(np.int8)
    # Distinct below-tip region is retained, never averaged into both flanks.
    branch[source[:, 1] > tip0[1]+4] = 3
    relative = displacement-(tip1-tip0)
    return dict(source_xy=source.astype(np.float32), current_xy=current.astype(np.float32),
        displacement_xy=displacement.astype(np.float32), relative_displacement_xy=relative.astype(np.float32),
        velocity_xy=None if dt_s is None else (displacement/dt_s).astype(np.float32),
        relative_velocity_xy=None if dt_s is None else (relative/dt_s).astype(np.float32),
        valid=valid, confidence=confidence.astype(np.float32), fb_error_px=fb.astype(np.float32),
        masked_correlation=correlation.astype(np.float32), support_fraction=fraction.astype(np.float32),
        texture=texture.astype(np.float32), source_branch=branch, blade_crossing=crossing,
        on_material=on_material, affine=affine, method=method,
        identity_scope='fresh visible texture in this frame pair; not original material IDs')
