"""Conservative, temporary analysis-only repair of a missing blade-tip mask.

Registered blade translation aligns two originally observed neighbor masks.
Only their local intersection can add blade pixels. Material masks are never
expanded, original input labels are never modified, and estimates never become
evidence for further repairs.
"""
from __future__ import annotations

from collections import OrderedDict
import cv2
import numpy as np


def _tip_metrics(labels, tip, radius):
    blade = np.asarray(labels) == 2
    h, w = blade.shape
    x, y = np.rint(tip).astype(int)
    if not (0 <= x < w and 0 <= y < h):
        return float('inf'), 0
    if blade.any():
        gap = float(cv2.distanceTransform((~blade).astype(np.uint8), cv2.DIST_L2, 5)[y, x])
    else:
        gap = float('inf')
    yy, xx = np.ogrid[:h, :w]
    local = (xx-tip[0])**2+(yy-tip[1])**2 <= radius**2
    return gap, int(np.count_nonzero(blade & local))


class LocalBladeMaskRepair:
    """Cached label reader returning ``(temporary_labels, mask_estimated)``.

    Array row i and reader index i refer to the same source frame. ``tips_xy``
    must contain independently registered blade tips. The caller must treat all
    motion estimates using a repaired mask as conditional geometry evidence.
    """
    def __init__(self, reader, tips_xy, tip_supported=None, *, max_neighbor_frames=8,
                 tip_radius_px=90., missing_tip_distance_px=8.,
                 reliable_tip_distance_px=6., minimum_local_blade_pixels=64,
                 maximum_tip_shift_px=32., cache_size=24):
        self.reader = reader
        self.tips = np.asarray(tips_xy, float)
        if self.tips.ndim != 2 or self.tips.shape[1] != 2:
            raise ValueError('tips_xy must be T,2 and indexed as the source label reader')
        self.supported = np.isfinite(self.tips).all(axis=1)
        if tip_supported is not None:
            valid = np.asarray(tip_supported, bool)
            if valid.shape != (len(self.tips),):
                raise ValueError('tip_supported must match tips_xy')
            self.supported &= valid
        if (max_neighbor_frames < 1 or tip_radius_px <= 0 or missing_tip_distance_px <= 0
                or reliable_tip_distance_px <= 0 or reliable_tip_distance_px > missing_tip_distance_px
                or minimum_local_blade_pixels < 1 or maximum_tip_shift_px <= 0 or cache_size < 2):
            raise ValueError('Invalid local blade-mask repair settings')
        self.max_neighbor_frames = int(max_neighbor_frames)
        self.radius = float(tip_radius_px)
        self.missing_gap = float(missing_tip_distance_px)
        self.reliable_gap = float(reliable_tip_distance_px)
        self.minimum_pixels = int(minimum_local_blade_pixels)
        self.maximum_tip_shift = float(maximum_tip_shift_px)
        self.cache_size = int(cache_size)
        self._original = OrderedDict()
        self._metrics = {}
        self.diagnostics = {}

    def _read_original(self, frame):
        if frame in self._original:
            self._original.move_to_end(frame)
            return self._original[frame]
        labels = np.asarray(self.reader(int(frame)))
        if labels.ndim != 2:
            raise ValueError('Reader must return a scalar semantic label image')
        self._original[frame] = labels
        while len(self._original) > self.cache_size:
            self._original.popitem(last=False)
        return labels

    def _frame_metrics(self, frame):
        if frame not in self._metrics:
            self._metrics[frame] = _tip_metrics(self._read_original(frame), self.tips[frame], self.radius)
        return self._metrics[frame]

    def _neighbor(self, frame, direction):
        for step in range(1, self.max_neighbor_frames+1):
            candidate = frame + direction*step
            if not 0 <= candidate < len(self.tips):
                break
            if not self.supported[candidate]:
                continue
            if np.linalg.norm(self.tips[candidate]-self.tips[frame]) > self.maximum_tip_shift:
                continue
            gap, count = self._frame_metrics(candidate)
            if gap <= self.reliable_gap and count >= self.minimum_pixels:
                return candidate
        return None

    def read(self, frame):
        frame = int(frame)
        if not 0 <= frame < len(self.tips):
            raise IndexError('Frame outside blade geometry array')
        original = self._read_original(frame)
        diagnostic = dict(frame=frame, mask_estimated=False, reason='original_tip_mask_available',
            added_blade_pixels=0, material_pixels_removed=0, material_pixels_added=0,
            tip_radius_px=self.radius, conditional_geometry=False)
        self.diagnostics[frame] = diagnostic
        if not self.supported[frame]:
            diagnostic['reason'] = 'registered_tip_unavailable'
            return original, False
        gap, count = self._frame_metrics(frame)
        diagnostic.update(original_tip_distance_px=gap if np.isfinite(gap) else None,
                          original_local_blade_pixels=count)
        if gap <= self.missing_gap and count >= self.minimum_pixels:
            return original, False
        before, after = self._neighbor(frame, -1), self._neighbor(frame, 1)
        diagnostic.update(previous_source_frame=before, next_source_frame=after)
        if before is None or after is None:
            diagnostic['reason'] = 'two_reliable_original_neighbors_unavailable'
            return original, False
        h, w = original.shape
        aligned = []
        for neighbor in (before, after):
            labels = self._read_original(neighbor)
            if labels.shape != original.shape:
                raise ValueError('Label frame sizes must remain constant')
            delta = self.tips[frame]-self.tips[neighbor]
            matrix = np.array([[1., 0., delta[0]], [0., 1., delta[1]]], np.float32)
            aligned.append(cv2.warpAffine((labels == 2).astype(np.uint8), matrix, (w, h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool))
        yy, xx = np.ogrid[:h, :w]
        tip = self.tips[frame]
        local = (xx-tip[0])**2+(yy-tip[1])**2 <= self.radius**2
        consensus = aligned[0] & aligned[1] & local
        consensus_labels = np.zeros_like(original)
        consensus_labels[consensus] = 2
        consensus_gap, consensus_count = _tip_metrics(consensus_labels, tip, self.radius)
        diagnostic.update(consensus_tip_distance_px=consensus_gap if np.isfinite(consensus_gap) else None,
                          consensus_local_blade_pixels=consensus_count)
        if consensus_gap > self.reliable_gap or consensus_count < self.minimum_pixels:
            diagnostic['reason'] = 'aligned_neighbor_intersection_insufficient'
            return original, False
        addition = consensus & (original != 2)
        if not addition.any():
            diagnostic['reason'] = 'no_consensus_pixels_to_add'
            return original, False
        repaired = original.copy()
        repaired[addition] = 2
        diagnostic.update(mask_estimated=True, reason='two_original_neighbors_rigid_local_intersection',
            added_blade_pixels=int(addition.sum()),
            material_pixels_removed=int(np.count_nonzero(addition & (original == 1))),
            conditional_geometry=True)
        return repaired, True

    __call__ = read
