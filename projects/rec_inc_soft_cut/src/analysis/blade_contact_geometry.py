"""Image-space blade exclusion and contact proximity, without moving tracks.

Semantic proximity is a contact candidate, not independently observed contact.
Routing branches prevent fits straight across the opaque blade; they are not
labels of separate cracked bodies. All distances below are native pixels.
"""
from __future__ import annotations

import cv2
import numpy as np


def sample_pixels(image, points, fill=0):
    image = np.asarray(image)
    points = np.asarray(points, float)
    if image.ndim != 2 or points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError('Expected a scalar pixel image and points ending in x,y')
    finite = np.isfinite(points).all(axis=-1)
    rounded = np.rint(np.where(np.isfinite(points), points, 0.))
    h, w = image.shape[:2]
    valid = finite & (rounded[..., 0] >= 0) & (rounded[..., 0] < w) & (rounded[..., 1] >= 0) & (rounded[..., 1] < h)
    # Retained forecasts can be arbitrarily far outside the image. Check the
    # pixel bounds in floating point before converting safe indices to int.
    xy = np.where(valid[..., None], rounded, 0.).astype(np.intp)
    result = np.full(points.shape[:-1], fill, dtype=image.dtype)
    result[valid] = image[xy[..., 1][valid], xy[..., 0][valid]]
    return result, valid


def blade_contact_geometry(labels, points, tip_xy, *, tip_supported=True,
                           core_margin_px=2, contact_gap_px=4., near_tip_px=90.):
    labels = np.asarray(labels, np.uint8)
    points, tip = np.asarray(points, float), np.asarray(tip_xy, float)
    if labels.ndim != 2 or points.ndim != 2 or points.shape[1] != 2 or tip.shape != (2,):
        raise ValueError('Expected label image, N,2 points and a 2D blade tip')
    if core_margin_px < 0 or contact_gap_px <= 0 or near_tip_px <= 0:
        raise ValueError('Invalid geometry tolerances')
    blade = labels == 2
    radius = int(core_margin_px)
    core = cv2.erode(blade.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    on_blade, inside = sample_pixels(blade, points)
    in_core, _ = sample_pixels(core, points)
    material, _ = sample_pixels(labels == 1, points)
    count = len(points)
    signed_gap = np.full(count, np.nan)
    normal = np.full((count, 2), np.nan)
    tangent = np.full((count, 2), np.nan)
    nearest_xy = np.full((count, 2), np.nan)
    if blade.any():
        outside_distance, closest = cv2.distanceTransformWithLabels((~blade).astype(np.uint8),
            cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        iy, ix = np.nonzero(blade)
        matched, _ = sample_pixels(closest, points)
        has_match = inside & (matched > 0) & (matched <= len(ix))
        nearest_xy[has_match] = np.column_stack((ix, iy))[matched[has_match]-1]
        delta = points - nearest_xy
        distance = np.linalg.norm(delta, axis=1)
        signed_gap[has_match] = distance[has_match]
        if on_blade.any():
            inside_distance = cv2.distanceTransform(blade.astype(np.uint8), cv2.DIST_L2, 5)
            sampled, _ = sample_pixels(inside_distance, points)
            signed_gap[on_blade] = -sampled[on_blade]
        usable_normal = has_match & ~on_blade & (distance > 1e-8)
        normal[usable_normal] = delta[usable_normal]/distance[usable_normal, None]
        tangent[usable_normal] = np.column_stack((normal[usable_normal, 1], -normal[usable_normal, 0]))
        flip = (tangent[:, 1] < 0) | ((abs(tangent[:, 1]) < 1e-6) & (tangent[:, 0] < 0))
        tangent[flip] *= -1
    branch = np.zeros(count, np.int8)
    candidate = np.zeros(count, bool)
    geometry_valid = bool(tip_supported and np.isfinite(tip).all())
    if geometry_valid:
        eligible = inside & material & ~in_core
        above = points[:, 1] <= tip[1] + 4.
        branch[eligible & above & (points[:, 0] < tip[0])] = 1
        branch[eligible & above & (points[:, 0] >= tip[0])] = 2
        branch[eligible & ~above] = 3
        near = np.linalg.norm(points-tip, axis=1) <= near_tip_px
        candidate = eligible & near & (signed_gap >= 0) & (signed_gap <= contact_gap_px)
    return dict(on_blade=on_blade, in_opaque_core=in_core, in_image=inside,
        material_pixel=material, signed_blade_gap_px=signed_gap, nearest_blade_xy=nearest_xy,
        outward_normal_xy=normal, down_or_right_tangent_xy=tangent,
        routing_branch=branch, contact_proximity_candidate=candidate,
        geometry_supported=geometry_valid)
