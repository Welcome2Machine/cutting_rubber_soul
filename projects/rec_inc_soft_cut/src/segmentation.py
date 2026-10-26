"""Existing temporal blade/material segmentation, without training entry points.

The checkpoint supplies the network settings. Source-coordinate preprocessing,
bounded frame/feature caches and conservative mask cleanup retain the deployed
numerical behavior. Labels are background=0, material=1, blade=2, ignored=255.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional

from src.tracking.fixed_grid import FixedGridProcessor, grid_0915_config

LABEL_BACKGROUND, LABEL_MATERIAL, LABEL_BLADE, LABEL_IGNORE = 0, 1, 2, 255


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ConvGRUCell(nn.Module):
    """A spatial GRU cell whose hidden state is a feature map."""

    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, value: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                value.shape[0], self.hidden_channels, value.shape[2], value.shape[3],
                device=value.device, dtype=value.dtype,
            )
        joined = torch.cat((value, hidden), dim=1)
        update, reset = self.gates(joined).chunk(2, dim=1)
        update, reset = torch.sigmoid(update), torch.sigmoid(reset)
        candidate = torch.tanh(self.candidate(torch.cat((value, reset * hidden), dim=1)))
        return (1.0 - update) * hidden + update * candidate


class TemporalSegmenter(nn.Module):
    """Bidirectional ConvGRU U-Net returning the centre frame semantic logits."""

    def __init__(
        self,
        sequence_length: int = 7,
        center_index: int | None = None,
        input_channels: int = 3,
        num_classes: int = 3,
    ):
        super().__init__()
        if sequence_length < 3 or sequence_length % 2 != 1:
            raise ValueError("sequence_length must be an odd number >= 3")
        self.sequence_length = sequence_length
        self.center_index = sequence_length // 2 if center_index is None else center_index
        if not 0 <= self.center_index < sequence_length:
            raise ValueError("center_index outside sequence")
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)

        # Module names and shapes are part of the deployed checkpoint contract.
        self.stem = ConvBlock(self.input_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 96))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(96, 128))
        self.forward_gru = ConvGRUCell(128, 128)
        self.backward_gru = ConvGRUCell(128, 128)
        self.up2 = ConvBlock(128 + 96, 96)
        self.up1 = ConvBlock(96 + 64, 64)
        self.up0 = ConvBlock(64 + 32, 32)
        self.output = nn.Conv2d(32, self.num_classes, 1)

    @staticmethod
    def _up(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def _encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Encode flattened ``[B*T,C,H,W]`` frames and keep skip features."""
        stem = self.stem(frames)
        down1 = self.down1(stem)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        return stem, down1, down2, down3

    def _decode(self, bottleneck: torch.Tensor, stem: torch.Tensor, down1: torch.Tensor, down2: torch.Tensor) -> torch.Tensor:
        up2 = self.up2(torch.cat((self._up(bottleneck, down2), down2), dim=1))
        up1 = self.up1(torch.cat((self._up(up2, down1), down1), dim=1))
        up0 = self.up0(torch.cat((self._up(up1, stem), stem), dim=1))
        return self.output(up0)

    @staticmethod
    def _run_direction(cell: ConvGRUCell, encoded: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        # encoded: [B,T,C,H,W]
        indices = range(encoded.shape[1] - 1, -1, -1) if reverse else range(encoded.shape[1])
        states: list[torch.Tensor | None] = [None] * encoded.shape[1]
        hidden = None
        for index in indices:
            hidden = cell(encoded[:, index], hidden)
            states[index] = hidden
        return torch.stack([state for state in states if state is not None], dim=1)

    def _temporal_features(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a sequence once and return fused bottleneck plus skip maps."""
        if value.ndim != 5:
            raise ValueError(f"expected [B,T,C,H,W], got {tuple(value.shape)}")
        batch, steps, channels, height, width = value.shape
        if steps != self.sequence_length or channels != self.input_channels:
            raise ValueError(
                f"expected T={self.sequence_length}, C={self.input_channels}, got T={steps}, C={channels}"
            )
        stem, down1, down2, down3 = self._encode(value.reshape(batch * steps, channels, height, width))
        feature_height, feature_width = down3.shape[-2:]
        encoded = down3.reshape(batch, steps, 128, feature_height, feature_width)
        forward = self._run_direction(self.forward_gru, encoded, reverse=False)
        backward = self._run_direction(self.backward_gru, encoded, reverse=True)
        fused = 0.5 * (forward + backward)
        stem = stem.reshape(batch, steps, 32, stem.shape[-2], stem.shape[-1])
        down1 = down1.reshape(batch, steps, 64, down1.shape[-2], down1.shape[-1])
        down2 = down2.reshape(batch, steps, 96, down2.shape[-2], down2.shape[-1])
        return fused, stem, down1, down2

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Only the centre frame is decoded during inference.
        fused, stem, down1, down2 = self._temporal_features(value)
        index = self.center_index
        return self._decode(fused[:, index], stem[:, index], down1[:, index], down2[:, index])


def redact_overlay_region(frame: np.ndarray, preprocessing: dict[str, Any] | None = None) -> np.ndarray:
    """Remove fixed camera overlay text before a frame reaches the network.

    The acquisition metadata is white text on a black strip in the upper-right
    corner.  It is useful for OCR, but it is not a physical object and must not
    become a segmentation cue.  Fractions make the same rule work on the
    reduced annotation videos and the original-resolution videos.
    """
    cfg = (preprocessing or {}).get("overlay_redaction", {})
    if not bool(cfg.get("enabled", False)):
        return frame
    height, width = frame.shape[:2]
    x0 = int(np.clip(float(cfg.get("x_start_fraction", 0.84)) * width, 0, width))
    y0 = int(np.clip(float(cfg.get("y_start_fraction", 0.0)) * height, 0, height))
    x1 = int(np.clip(float(cfg.get("x_end_fraction", 1.0)) * width, x0, width))
    y1 = int(np.clip(float(cfg.get("y_end_fraction", 0.18)) * height, y0, height))
    result = frame.copy()
    fill = str(cfg.get("fill", "black")).lower()
    if fill == "median" and x0 > 0 and y1 > y0:
        sample = result[y0:y1, max(0, x0 - max(4, width // 40)):x0]
        value = np.median(sample.reshape(-1, result.shape[2]), axis=0).astype(np.uint8) if sample.size else np.zeros(3, dtype=np.uint8)
        result[y0:y1, x0:x1] = value
    else:
        result[y0:y1, x0:x1] = 0
    return result


def remove_fixed_grid(frame: np.ndarray, preprocessing: dict[str, Any] | None = None) -> np.ndarray:
    """Inpaint the fixed acquisition grid without altering object geometry."""
    cfg = (preprocessing or {}).get("grid_suppression", {})
    if not bool(cfg.get("enabled", False)):
        return frame
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    width_px = max(1, int(round(float(cfg.get("line_width_fraction", 0.0025)) * min(height, width))))
    for fraction in cfg.get("vertical_fractions", [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]):
        x = int(round(float(fraction) * (width - 1)))
        cv2.line(mask, (x, 0), (x, height - 1), 255, width_px * 2 + 1)
    for fraction in cfg.get("horizontal_fractions", [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]):
        y = int(round(float(fraction) * (height - 1)))
        cv2.line(mask, (0, y), (width - 1, y), 255, width_px * 2 + 1)
    # Fast fixed-line replacement.  Full Navier-Stokes inpainting is needlessly
    # expensive for a static one-pixel acquisition grid and made training
    # appear stalled.  Averaging opposite neighbours preserves local texture
    # while removing the artificial line.
    # OpenCV's native box filter avoids allocating four full-size NumPy
    # shifted arrays for every frame.
    kernel_side = max(3, int(width_px) * 2 + 1)
    estimate = cv2.blur(frame, (kernel_side, kernel_side))
    result = frame.copy()
    result[mask > 0] = estimate[mask > 0]
    return result


def frame_features(frame: np.ndarray) -> np.ndarray:
    """RGB plus luminance and green-excess cues robust to material glints."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    luminance = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[..., None]
    green_excess = np.clip(rgb[..., 1] - 0.5 * (rgb[..., 0] + rgb[..., 2]) + 0.5, 0.0, 1.0)[..., None]
    return np.concatenate((rgb, luminance, green_excess), axis=2).transpose(2, 0, 1).astype(np.float32)


def build_model(model_config: dict[str, Any]) -> TemporalSegmenter:
    return TemporalSegmenter(
        sequence_length=int(model_config["sequence_length"]),
        input_channels=int(model_config.get("input_channels", 5)),
        num_classes=int(model_config.get("num_classes", 3)),
    )


def conservative_postprocess(
    mask: np.ndarray,
    max_material_hole_area: int = 700,
    min_blade_component_area: int = 36,
    min_isolated_material_area: int = 2500,
    min_material_hole_aspect_ratio: float = 3.0,
) -> np.ndarray:
    """Apply topology-safe cleanup without inventing a geometric template.

    The material is a macroscopic workpiece extending to the image boundary.
    Small material islands floating entirely in the background are therefore
    treated as high-light false positives.  Border-connected material is
    untouched, as are large interior components that may be a real split
    surface.  The blade is only pruned by area; no shaft or tip is fabricated.
    Narrow, elongated background components are preserved because a
    post-withdrawal separation gap is not a lubricant bubble.
    """
    result = mask.copy().astype(np.uint8)
    blade = result == LABEL_BLADE
    material = result == LABEL_MATERIAL
    inverse = (~material).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    height, width = material.shape
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x, y, component_width, component_height = (int(stats[label, index]) for index in range(4))
        touches_border = x == 0 or y == 0 or x + component_width >= width or y + component_height >= height
        component = labels == label
        aspect_ratio = max(component_width, component_height) / max(1.0, min(component_width, component_height))
        elongated_gap = aspect_ratio >= float(min_material_hole_aspect_ratio)
        if not touches_border and area <= max_material_hole_area and not elongated_gap and not np.any(blade & component):
            result[component] = LABEL_MATERIAL
    blade = result == LABEL_BLADE
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blade.astype(np.uint8), connectivity=8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_blade_component_area:
            result[labels == label] = LABEL_BACKGROUND
    # Remove isolated material speckles, but retain every border-connected
    # component and any large interior region for manual review.
    material = (result == LABEL_MATERIAL).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(material, connectivity=8)
    height, width = result.shape
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x, y, component_width, component_height = (int(stats[label, index]) for index in range(4))
        touches_border = x == 0 or y == 0 or x + component_width >= width or y + component_height >= height
        if not touches_border and area < int(min_isolated_material_area):
            result[labels == label] = LABEL_BACKGROUND
    return result


def temporal_blade_cleanup(
    mask: np.ndarray,
    previous_mask: np.ndarray | None = None,
    min_component_area: int = 36,
    max_gap_px: int = 18,
) -> np.ndarray:
    """Retain the dominant blade component, ranked by area and elongation.

    The deployed algorithm does not consult ``previous_mask`` or ``max_gap_px``;
    these arguments remain compatible with saved preparation configurations.
    No missing tip, shaft or other blade pixels are fabricated.
    """
    result = mask.astype(np.uint8, copy=True)
    blade = (result == LABEL_BLADE).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blade, connectivity=8)
    if count <= 1:
        return result
    components = [
        (int(stats[index, cv2.CC_STAT_AREA]), index)
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_component_area)
    ]
    if not components:
        result[result == LABEL_BLADE] = LABEL_BACKGROUND
        return result
    # Prefer the elongated, shaft-like component when several fragments have
    # similar area.  A compact bright ring is a common false positive in the
    # lubricant/grid region and should not become the tracked object.
    def tool_score(item: tuple[int, int]) -> float:
        area, index = item
        width = max(1, int(stats[index, cv2.CC_STAT_WIDTH]))
        height = max(1, int(stats[index, cv2.CC_STAT_HEIGHT]))
        elongation = max(width, height) / min(width, height)
        return float(area) * (1.0 + 0.55 * min(elongation, 5.0))

    components.sort(key=tool_score, reverse=True)
    keep = {components[0][1]}
    # The component gate is intentionally local and dilation-free.  Large
    # 37x37 dilations on every 640x480 frame dominated inference time and did
    # not improve the rejection of compact bright islands.  Temporal support is
    # represented by the stable component ranking across consecutive windows;
    # the previous mask remains an API-compatible argument for future tracking.
    accepted = np.zeros_like(blade)
    for index in keep:
        accepted |= (labels == index).astype(np.uint8)
    result[(blade > 0) & (accepted == 0)] = LABEL_BACKGROUND
    return result


def semantic_overlay(frame: np.ndarray, mask: np.ndarray, confidence: float, frame_index: int, annotate: bool = True) -> np.ndarray:
    """Truthful review overlay: masks and their observed contours only."""
    result = frame.copy()
    colors = {LABEL_MATERIAL: (55, 210, 65), LABEL_BLADE: (255, 220, 0)}
    color = np.zeros_like(frame)
    visible = np.zeros(mask.shape, dtype=bool)
    for label, bgr in colors.items():
        region = mask == label
        color[region] = bgr
        visible |= region
    if np.any(visible):
        cv2.copyTo(cv2.addWeighted(result, 0.72, color, 0.28, 0), visible.astype(np.uint8), result)
    contour_support = 1 - cv2.dilate((mask == 255).astype(np.uint8), np.ones((5, 5), np.uint8))
    for label, bgr in colors.items():
        contours, _ = cv2.findContours((mask == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outlined = result.copy()
        cv2.drawContours(outlined, contours, -1, bgr, 2, cv2.LINE_AA)
        cv2.copyTo(outlined, contour_support, result)
    if annotate:
        cv2.rectangle(result, (0, 0), (470, 32), (0, 0, 0), -1)
        cv2.putText(result, f"full-mask temporal segmentation  f={frame_index}  confidence={confidence:.3f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return result


@lru_cache(maxsize=1)
def _processor_0915():
    return FixedGridProcessor(grid_0915_config())


def segmentation_frame_features(frame, input_size, preprocessing=None):
    """Keep legacy checkpoint behaviour unless source coordinates are explicit.

    source_0915 is only for the verified 1024x768 acquisition grid. Removal
    precedes top cropping and resize; resized fractional grid removal is then
    bypassed. Redaction retains its existing resized-ROI coordinate convention.
    """
    config = preprocessing or {}
    crop = int(config.get('source_crop_top_px', 0))
    if not 0 <= crop < frame.shape[0]:
        raise ValueError('invalid source_crop_top_px')
    height, width = map(int, input_size)
    if min(height, width) < 1:
        raise ValueError('input_size must be positive')
    mode = config.get('grid_coordinate_space', 'legacy_resized')
    if mode not in ('legacy_resized', 'source_0915'):
        raise ValueError(f'Unknown grid_coordinate_space: {mode}')
    if mode == 'source_0915' and config.get('grid_suppression', {}).get('enabled', False):
        frame = _processor_0915()(frame)
    resized = cv2.resize(frame[crop:], (width, height), interpolation=cv2.INTER_AREA)
    resized = redact_overlay_region(resized, config)
    if mode == 'legacy_resized':
        resized = remove_fixed_grid(resized, config)
    return frame_features(resized)


class SourceFrameCache:
    """Sequential sliding windows with a bounded full-resolution cache.

    Seeking to every neighbour frame is disastrous for some H.264 long-GOP
    files.  Frames are therefore decoded once in display order; the cache is
    only used for the short backwards portion of the next temporal window.
    """

    def __init__(self, path: Path, cache_size: int = 48):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 12.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.frame_count < 1 or self.width < 1 or self.height < 1:
            raise RuntimeError(f"video has invalid metadata: {path}")
        self.cache_size = max(16, int(cache_size))
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.next_decode_index = 0

    def prefetch_through(self, index: int) -> None:
        """Decode source frames monotonically through ``index`` (inclusive)."""
        index = int(np.clip(index, 0, self.frame_count - 1))
        while self.next_decode_index <= index:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot decode frame {self.next_decode_index}: {self.path}")
            self.cache[self.next_decode_index] = frame
            self.next_decode_index += 1
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

    def read(self, index: int) -> np.ndarray:
        index = int(np.clip(index, 0, self.frame_count - 1))
        if index >= self.next_decode_index:
            self.prefetch_through(index)
        if index in self.cache:
            frame = self.cache.pop(index)
            self.cache[index] = frame
            return frame.copy()
        raise RuntimeError(
            f"temporal cache evicted frame {index}; increase cache size "
            f"(next decoded={self.next_decode_index}, retained={list(self.cache)[:1]}..{list(self.cache)[-1:]})"
        )

    def close(self) -> None:
        self.cap.release()
        self.cache.clear()


class SegmentationFeatureCache:
    """Reuse deterministic per-frame input features in overlapping windows."""

    def __init__(self, reader, input_size, preprocessing, enabled=True):
        self.reader = reader
        self.input_size = input_size
        self.preprocessing = preprocessing
        self.enabled = enabled
        self.cache = OrderedDict()

    def read(self, index):
        index = int(np.clip(index, 0, self.reader.frame_count - 1))
        if self.enabled and index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value
        value = segmentation_frame_features(self.reader.read(index), self.input_size, self.preprocessing)
        if self.enabled:
            value.setflags(write=False)
            self.cache[index] = value
            while len(self.cache) > self.reader.cache_size:
                self.cache.popitem(last=False)
        return value


def _topology(binary: np.ndarray) -> tuple[int, int]:
    components = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)[0] - 1
    n, _, stats, _ = cv2.connectedComponentsWithStats((~binary).astype(np.uint8), 8)
    h, w = binary.shape
    holes = sum(not (s[0] == 0 or s[1] == 0 or s[0]+s[2] == w or s[1]+s[3] == h) for s in stats[1:])
    return int(components), int(holes)


def refine_material_boundary(frame: np.ndarray, labels: np.ndarray, band_radius: int = 6,
                             protected_radius: int = 8, iterations: int = 3):
    """Return (semantic labels, diagnostics), without modifying either input.

    Labels are background=0, material=1, blade=2 and ignored=255. Both hard
    blade/ignore pixels and their guard band remain fixed. Interior background
    holes remain fixed. A changed number of components or holes rejects the
    entire candidate. New graph-cut speckle components without a fixed class
    seed are projected back onto a connected outer edge before this check.
    Original point trajectories and strain are never touched.
    """
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected uint8 BGR image')
    if labels.dtype != np.uint8 or labels.ndim != 2 or frame.shape[:2] != labels.shape:
        raise ValueError('Frame/uint8 semantic label shape mismatch')
    if not np.isin(labels, [0, 1, 2, 255]).all():
        raise ValueError('Unsupported semantic class')
    if band_radius < 1 or protected_radius < 0 or iterations < 1:
        raise ValueError('Invalid refinement parameter')
    material = labels == 1
    info = dict(changed_pixels=0, candidate_changed_pixels=0, allowed_pixels=0,
                band_radius_px=int(band_radius), protected_radius_px=int(protected_radius),
                topology_guard=True, accepted=False)
    if not material.any() or material.all():
        info['reason'] = 'no_material_boundary'
        return labels.copy(), info
    n, lab, stats, _ = cv2.connectedComponentsWithStats((labels == 0).astype(np.uint8), 8)
    h, w = labels.shape
    exterior_ids = [i for i in range(1, n) if stats[i, 0] == 0 or stats[i, 1] == 0 or stats[i, 0]+stats[i, 2] >= w or stats[i, 1]+stats[i, 3] >= h]
    exterior = np.isin(lab, exterior_ids)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band_radius+1, 2*band_radius+1))
    inner = cv2.erode(material.astype(np.uint8), kernel).astype(bool)
    outer = cv2.dilate(material.astype(np.uint8), kernel).astype(bool)
    allowed = (outer & ~inner) & cv2.dilate(exterior.astype(np.uint8), kernel).astype(bool)
    protected = cv2.dilate(((labels == 2) | (labels == 255)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*protected_radius+1, 2*protected_radius+1))).astype(bool)
    protected |= cv2.dilate(((labels == 0) & ~exterior).astype(np.uint8), kernel).astype(bool)
    allowed &= ~protected
    info['allowed_pixels'] = int(allowed.sum())
    trimap = np.where(material, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)
    trimap[allowed & material] = cv2.GC_PR_FGD
    trimap[allowed & ~material] = cv2.GC_PR_BGD
    if not np.any(trimap == cv2.GC_PR_FGD) or not np.any(trimap == cv2.GC_PR_BGD):
        info['reason'] = 'no_unprotected_two_sided_band'
        return labels.copy(), info
    cv2.grabCut(frame, trimap, None, np.zeros((1, 65)), np.zeros((1, 65)), iterations, cv2.GC_INIT_WITH_MASK)
    result = labels.copy()
    result[allowed] = np.isin(trimap[allowed], [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    info['unprojected_changed_pixels'] = int(np.count_nonzero(result != labels))
    # Hard class seeds distinguish the real exterior/core from newly invented
    # holes/islands. This projection cannot modify any protected or interior
    # gap pixel, and the final global topology check remains mandatory.
    projected = 0
    for value in (0, 1):
        region = result == value
        _, components = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
        anchored_ids = np.unique(components[(labels == value) & ~allowed])
        unanchored = region & allowed & ~np.isin(components, anchored_ids)
        projected += int(unanchored.sum())
        result[unanchored] = 1 - value
    info['projected_speckle_pixels'] = projected
    info['candidate_changed_pixels'] = int(np.count_nonzero(result != labels))
    info['topology_before'] = _topology(material)
    info['topology_candidate'] = _topology(result == 1)
    if info['topology_before'] != info['topology_candidate']:
        info['reason'] = 'rejected_component_or_hole_change'
        return labels.copy(), info
    info['accepted'] = True
    info['changed_pixels'] = info['candidate_changed_pixels']
    info['reason'] = 'accepted_outer_boundary_only'
    assert np.array_equal(result[~allowed], labels[~allowed])
    return result, info
