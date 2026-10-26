"""Image geometry of a material outline, independent of material identity.

Contour stations are geometric samples, NEVER persistent material IDs. The
closed outline includes frame-clipped edges; the open path excludes them.
States: 0 frame-clipped, 1 segmentation-only, 2 image-supported, 3 blade-adjacent.
Contact state: 0 no qualified contact, 1 conditional geometry, 2 image-supported.
The nearest geometric point is returned even when contact_state is zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ContourGeometryConfig:
    sample_step_px: int = 3
    normal_search_px: int = 7
    image_sigma_px: float = 1.1
    minimum_gradient: float = 5.
    minimum_contrast: float = 8.
    segmentation_weight: float = .035
    smoothness_weight: float = .22
    temporal_weight: float = .025
    temporal_radius_px: float = 65.
    temporal_maximum_motion_px: float = 25.
    crop_margin_px: float = 2.
    blade_band_px: float = 2.5
    contact_tolerance_px: float = 6.
    minimum_component_pixels: int = 64
    minimum_path_points: int = 8


def _sample(image, xy):
    xy = np.asarray(xy)
    original_shape = xy.shape[:-1]
    flat = xy.reshape(-1, 2).astype(np.float32)
    result = cv2.remap(image.astype(np.float32), flat[:, 0:1], flat[:, 1:2],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return result.reshape(original_shape)


def _blade_at(blade, xy):
    xy = np.asarray(xy)
    q = np.rint(xy).astype(int)
    inside = (q[..., 0] >= 0) & (q[..., 0] < blade.shape[1]) & (q[..., 1] >= 0) & (q[..., 1] < blade.shape[0])
    result = np.ones(q.shape[:-1], bool)
    result[inside] = blade[q[..., 1][inside], q[..., 0][inside]]
    return result


def _clear(blade, a, b):
    length = float(np.linalg.norm(np.asarray(b)-a))
    q = np.asarray(a) + np.linspace(0., 1., max(2, int(np.ceil(length*3))+1))[:, None]*(np.asarray(b)-a)
    return not _blade_at(blade, q).any()


def _crop(xy, shape, margin):
    h, w = shape
    return ((xy[:, 0] <= margin) | (xy[:, 0] >= w-1-margin) |
            (xy[:, 1] <= margin) | (xy[:, 1] >= h-1-margin))


def _open_path(closed, states, tip, tangent, minimum):
    eligible = states != 0
    xy, kinds = closed, states
    if (~eligible).any():
        cut = int(np.flatnonzero(~eligible)[0])
        xy, kinds, eligible = (np.roll(x, -cut, axis=0) for x in (xy, kinds, eligible))
    else:
        # A closed specimen has no canonical seam. Put it opposite the tip.
        cut = int(np.linalg.norm(xy-tip, axis=1).argmax())
        xy, kinds, eligible = (np.roll(x, -cut, axis=0) for x in (xy, kinds, eligible))
    change = np.diff(np.r_[False, eligible, False].astype(int))
    pieces = [(xy[a:b], kinds[a:b]) for a, b in
              zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)) if b-a >= minimum]
    if not pieces:
        return np.empty((0, 2)), np.empty(0, np.int8)
    path, state = min(pieces, key=lambda p: (float(np.linalg.norm(p[0]-tip, axis=1).min()), -len(p[0])))
    if np.dot(path[-1]-path[0], tangent) < 0:
        path, state = path[::-1], state[::-1]
    return path.copy(), state.copy()


def _project_tip(path, tip):
    delta = np.diff(path, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    alpha = np.clip(np.sum((tip-path[:-1])*delta, axis=1)/np.maximum(lengths**2, 1.e-12), 0., 1.)
    candidates = path[:-1]+alpha[:, None]*delta
    gap = np.linalg.norm(candidates-tip, axis=1)
    k = int(np.argmin(gap))
    return candidates[k], float(gap[k]), arc-(arc[k]+alpha[k]*lengths[k]), k


class ContourGeometryTracker:
    def __init__(self, config=None):
        self.config = config or ContourGeometryConfig()
        self.previous_closed = None

    def _prediction(self, coarse, previous_points, current_points, supported):
        cfg = self.config
        prediction = np.full_like(coarse, np.nan)
        if self.previous_closed is None or previous_points is None or current_points is None:
            return prediction
        p, q = np.asarray(previous_points, float), np.asarray(current_points, float)
        if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 2:
            raise ValueError('Paired flow arrays must have equal N,2 shape and preserve material IDs')
        use = np.isfinite(p).all(1) & np.isfinite(q).all(1)
        if supported is not None:
            support = np.asarray(supported, bool)
            if support.shape != (len(p),):
                raise ValueError('common_supported must match paired flow IDs')
            use &= support
        motion = q-p
        use &= np.linalg.norm(motion, axis=1) <= cfg.temporal_maximum_motion_px
        if use.sum() < 4:
            return prediction
        tree = cKDTree(p[use])
        distances, indices = tree.query(self.previous_closed, k=min(8, use.sum()))
        if distances.ndim == 1:
            distances, indices = distances[:, None], indices[:, None]
        selected_motion = motion[use][indices]
        selected_motion[distances > cfg.temporal_radius_px] = np.nan
        good = np.isfinite(selected_motion).all(2).sum(1) >= 3
        advected = self.previous_closed[good] + np.nanmedian(selected_motion[good], axis=1)
        if len(advected):
            distance, nearest = cKDTree(advected).query(coarse)
            valid = distance <= cfg.normal_search_px*2
            prediction[valid] = advected[nearest[valid]]
        return prediction

    def estimate(self, frame_bgr, labels, tip_xy, tangent, previous_points=None,
                 current_points=None, common_supported=None, geometry_conditional=False):
        cfg = self.config
        labels = np.asarray(labels)
        image = np.asarray(frame_bgr)
        tip, tangent = np.asarray(tip_xy, float), np.asarray(tangent, float)
        if labels.ndim != 2 or image.shape[:2] != labels.shape or tip.shape != (2,) or not np.isfinite(tip).all():
            raise ValueError('Image, labels and finite registered tip must share one image coordinate system')
        if tangent.shape != (2,) or not np.isfinite(tangent).all() or np.linalg.norm(tangent) < 1.e-8:
            raise ValueError('A finite, nonzero orientation tangent is required')
        empty = dict(closed_xy=np.empty((0, 2)), closed_state=np.empty(0, np.int8),
                     closed_edge_valid=np.empty(0, bool), path_edge_valid=np.empty(0, bool),
                     raw_closed_xy=np.empty((0, 2)), path_xy=np.empty((0, 2)), path_arc=np.empty(0),
                     path_state=np.empty(0, np.int8), raw_path_xy=np.empty((0, 2)), raw_path_arc=np.empty(0),
                     contact_xy=np.full(2, np.nan), tip_gap_px=np.nan, contact_state=0,
                     raw_tip_gap_px=np.nan, diagnostics=dict(reason='material_component_unavailable'))
        count, components, stats, _ = cv2.connectedComponentsWithStats((labels == 1).astype(np.uint8))
        if count < 2:
            self.previous_closed = None
            return empty
        main = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[main, cv2.CC_STAT_AREA] < cfg.minimum_component_pixels:
            self.previous_closed = None
            return empty
        material = (components == main).astype(np.uint8)
        contours, _ = cv2.findContours(material, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        raw = max(contours, key=len)[:, 0].astype(float)
        blade = labels == 2
        # Anchor the optimization seam on a cropped edge, or far from contact.
        cropped = _crop(raw, labels.shape, cfg.crop_margin_px)
        seam = int(np.flatnonzero(cropped)[0]) if cropped.any() else int(np.linalg.norm(raw-tip, axis=1).argmax())
        raw = np.roll(raw, -seam, axis=0)
        selected = np.arange(0, len(raw), cfg.sample_step_px)
        coarse = raw[selected]
        cropped = _crop(coarse, labels.shape, cfg.crop_margin_px)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), cfg.image_sigma_px)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
        direction = np.roll(coarse, -1, axis=0)-np.roll(coarse, 1, axis=0)
        normal = np.column_stack((direction[:, 1], -direction[:, 0]))
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1.e-8)
        flip = _sample(material, coarse+normal*3) > _sample(material, coarse-normal*3)
        normal[flip] *= -1.
        offsets = np.arange(-cfg.normal_search_px, cfg.normal_search_px+1, dtype=float)
        candidates = coarse[:, None]+normal[:, None]*offsets[None, :, None]
        gradient = abs(_sample(gx, candidates)*normal[:, None, 0]+_sample(gy, candidates)*normal[:, None, 1])
        contrast = abs(_sample(gray, candidates-normal[:, None]*3)-_sample(gray, candidates+normal[:, None]*3))
        # Both gradient AND cross-boundary contrast must be present. Texture
        # inside the material is further discouraged by the narrow mask prior.
        # Soft compression retains the gradient peak; hard clipping creates a
        # flat reward plateau and pulls high-contrast edges back toward a
        # biased segmentation boundary.
        evidence = np.log1p(gradient/cfg.minimum_gradient)*np.log1p(contrast/cfg.minimum_contrast)
        cost = cfg.segmentation_weight*offsets[None]**2 - evidence
        prediction = self._prediction(coarse, previous_points, current_points, common_supported)
        predicted = np.isfinite(prediction).all(1)
        shift = np.sum((prediction-coarse)*normal, axis=1)
        cost[predicted] += cfg.temporal_weight*(offsets[None]-shift[predicted, None])**2
        h, w = labels.shape
        allowed = ((candidates[..., 0] >= 0) & (candidates[..., 0] <= w-1) &
                   (candidates[..., 1] >= 0) & (candidates[..., 1] <= h-1) & ~_blade_at(blade, candidates))
        # Motion and regularity may choose between image-supported candidates;
        # neither is permission to move a boundary with absent image evidence.
        strong = (gradient >= cfg.minimum_gradient) & (contrast >= cfg.minimum_contrast)
        strong &= evidence >= evidence[:, cfg.normal_search_px, None]-1.e-8
        strong[:, cfg.normal_search_px] = True
        allowed &= strong
        allowed[cropped] = False
        allowed[cropped, cfg.normal_search_px] = True
        cost[~allowed] = np.inf
        # A fixed seam ensures closed-DP endpoint consistency. The registered
        # tip never appears as a fitting target or forced point on the curve.
        cost[0] = np.inf
        cost[0, cfg.normal_search_px] = 0.
        transition = cfg.smoothness_weight*(offsets[:, None]-offsets[None, :])**2
        history = np.zeros((len(coarse), len(offsets)), np.int16)
        cumulative = cost[0].copy()
        apex = int(np.linalg.norm(coarse-tip, axis=1).argmin())
        split_apex = np.linalg.norm(coarse[apex]-tip) <= 16.
        for i in range(1, len(coarse)):
            # The two sides may have a real corner at the tip. Do not impose
            # the same normal correction on opposing faces across that corner.
            local_transition = 0. if split_apex and i in (apex, apex+1) else transition
            choices = cumulative[:, None]+local_transition
            if np.ndim(local_transition) == 0:
                choices = np.broadcast_to(choices, transition.shape)
            history[i] = np.argmin(choices, axis=0)
            cumulative = cost[i]+np.min(choices, axis=0)
        route = np.zeros(len(coarse), int)
        route[-1] = int(np.argmin(cumulative+transition[:, cfg.normal_search_px]))
        for i in range(len(coarse)-1, 0, -1):
            route[i-1] = history[i, route[i]]
        refined = candidates[np.arange(len(coarse)), route].copy()
        # Any proposed chord cutting through the blade restores its endpoints
        # to the original material contour. No geometric bridge crosses steel.
        restored = np.zeros(len(refined), bool)
        for _ in range(3):
            bad = np.array([not _clear(blade, refined[i], refined[(i+1)%len(refined)]) for i in range(len(refined))])
            if not bad.any():
                break
            restore = bad | np.roll(bad, 1)
            refined[restore] = coarse[restore]
            restored |= restore
        distance_blade = cv2.distanceTransform((~blade).astype(np.uint8), cv2.DIST_L2, 5) if blade.any() else np.full(labels.shape, 1.e6, np.float32)
        image_supported = ((gradient[np.arange(len(coarse)), route] >= cfg.minimum_gradient) &
                           (contrast[np.arange(len(coarse)), route] >= cfg.minimum_contrast) & ~restored)
        state = np.where(image_supported, 2, 1).astype(np.int8)
        state[_sample(distance_blade, refined) <= cfg.blade_band_px] = 3
        state[cropped] = 0
        if geometry_conditional:
            state[state == 2] = 1
        # Preserve detailed raster corners wherever coarsening alone would
        # create an illegal chord. These points remain segmentation-only.
        pieces, kinds = [], []
        for i in range(len(refined)):
            j = (i+1)%len(refined)
            if _clear(blade, refined[i], refined[j]):
                pieces.append(refined[i:i+1]); kinds.append(state[i:i+1])
            else:
                end = selected[j] if j else len(raw)
                part = raw[selected[i]:end].copy()
                part_state = np.ones(len(part), np.int8)
                part_state[_sample(distance_blade, part) <= cfg.blade_band_px] = 3
                part_state[_crop(part, labels.shape, cfg.crop_margin_px)] = 0
                pieces.append(part); kinds.append(part_state)
        closed, state = np.concatenate(pieces), np.concatenate(kinds)
        # Resolve raster diagonal corner ambiguity with a one-pixel material
        # route instead of accepting a line that samples a blade pixel.
        safe_xy, safe_state = [], []
        for i, a in enumerate(closed):
            b = closed[(i+1)%len(closed)]
            safe_xy.append(a); safe_state.append(state[i])
            if not _clear(blade, a, b):
                corners = (np.array([a[0], b[1]]), np.array([b[0], a[1]]))
                legal = [p for p in corners if _clear(blade, a, p) and _clear(blade, p, b)]
                if legal:
                    safe_xy.append(legal[0])
                    safe_state.append(0 if _crop(legal[0][None], labels.shape, cfg.crop_margin_px)[0] else 3)
        closed, state = np.asarray(safe_xy), np.asarray(safe_state, np.int8)
        # Raster corner repair can insert the same vertex twice. Keep the
        # geometric curve unchanged while giving arc-length mappings a strict
        # ordering; a zero-length edge is not a separate surface station.
        keep = np.r_[True, np.linalg.norm(np.diff(closed, axis=0), axis=1) > 1.e-8]
        closed, state = closed[keep], state[keep]
        if len(closed) > 1 and np.linalg.norm(closed[-1]-closed[0]) <= 1.e-8:
            closed, state = closed[:-1], state[:-1]
        path, path_state = _open_path(closed, state, tip, tangent, cfg.minimum_path_points)
        raw_state = np.where(_crop(raw, labels.shape, cfg.crop_margin_px), 0, 1).astype(np.int8)
        raw_path, _ = _open_path(raw, raw_state, tip, tangent, cfg.minimum_path_points)
        result = dict(empty, closed_xy=closed, closed_state=state, raw_closed_xy=raw,
                      path_xy=path, path_state=path_state, raw_path_xy=raw_path)
        if len(raw_path) >= 2:
            _, result['raw_tip_gap_px'], result['raw_path_arc'], _ = _project_tip(raw_path, tip)
        if len(path) >= 2:
            contact, gap, arc, index = _project_tip(path, tip)
            qualified = gap <= cfg.contact_tolerance_px
            observed = qualified and np.all(path_state[index:index+2] == 2) and not geometry_conditional
            result.update(contact_xy=contact, tip_gap_px=gap, path_arc=arc,
                          contact_state=2 if observed else (1 if qualified else 0))
        closed_edges = np.array([_clear(blade, a, b) for a, b in zip(closed, np.roll(closed, -1, axis=0))])
        path_edges = np.array([_clear(blade, a, b) for a, b in zip(path[:-1], path[1:])], bool)
        result.update(closed_edge_valid=closed_edges, path_edge_valid=path_edges)
        crossings = int((~closed_edges).sum())
        if len(path) >= 2 and not path_edges[index]:
            result['contact_state'] = 0
        result['diagnostics'] = dict(reason='image_and_segmentation_outline', component_pixels=int(stats[main, cv2.CC_STAT_AREA]),
            component_count=int(count-1), contour_stations=len(closed), image_supported_fraction=float(np.mean(state == 2)),
            crop_fraction=float(np.mean(state == 0)), blade_adjacent_fraction=float(np.mean(state == 3)),
            temporal_predicted_stations=int(predicted.sum()), restored_stations=int(restored.sum()),
            median_normal_correction_px=float(np.median(np.linalg.norm(refined-coarse, axis=1))),
            blade_crossing_edges=int(crossings), geometry_conditional=bool(geometry_conditional),
            contact_is_material_identity=False, closed_outline_has_implicit_last_to_first_edge=True)
        visible = ~cropped
        normal_gradient = abs(_sample(gx, refined)*normal[:, 0]+_sample(gy, refined)*normal[:, 1])
        boundary_contrast = abs(_sample(gray, refined-normal*3)-_sample(gray, refined+normal*3))
        correction = np.linalg.norm(refined-coarse, axis=1)
        for key, values in (('raw_normal_gradient', gradient[:, cfg.normal_search_px]),
                            ('refined_normal_gradient', normal_gradient),
                            ('raw_boundary_contrast', contrast[:, cfg.normal_search_px]),
                            ('refined_boundary_contrast', boundary_contrast)):
            result['diagnostics'][key+'_median'] = float(np.median(values[visible])) if visible.any() else None
            result['diagnostics'][key+'_mean'] = float(np.mean(values[visible])) if visible.any() else None
        result['diagnostics']['visible_correction_p95_px'] = float(np.quantile(correction[visible], .95)) if visible.any() else None
        result['diagnostics']['visible_correction_max_px'] = float(np.max(correction[visible])) if visible.any() else None
        result['diagnostics']['image_corrected_stations'] = int(np.sum(correction > 1.e-6))
        self.previous_closed = closed.copy()
        return result
