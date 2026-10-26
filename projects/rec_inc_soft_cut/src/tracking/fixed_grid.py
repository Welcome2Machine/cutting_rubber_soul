"""Remove a confirmed, burned-in acquisition grid before cropping or resizing.

Only explicitly configured full-frame rows/columns are interpolated. This
cannot reconstruct texture hidden by the overlay, and is not a displacement
filter: neither point coordinates nor pixels outside those bands are changed.
No grid is assumed for an arbitrary video. The 0915 preset is based on the
one-pixel, intensity-85 lines in original 1024 x 768 AVI frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class FixedGridConfig:
    """Explicit grid geometry in original, uncropped pixel coordinates."""

    width: int
    height: int
    vertical_lines: tuple[int, ...] = ()
    horizontal_lines: tuple[int, ...] = ()
    half_width_px: int = 0

    def __post_init__(self) -> None:
        for name in ("width", "height", "half_width_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.width < 3 or self.height < 3 or self.half_width_px < 0:
            raise ValueError("frame dimensions must be >= 3 and half_width_px >= 0")
        for name, limit in (("vertical_lines", self.width), ("horizontal_lines", self.height)):
            values = tuple(getattr(self, name))
            if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in values):
                raise ValueError(f"{name} must contain integer pixel coordinates")
            values = tuple(sorted(set(int(v) for v in values)))
            if any(v - self.half_width_px < 1 or v + self.half_width_px >= limit - 1 for v in values):
                raise ValueError("each grid band needs an unmasked pixel on both sides")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "vertical_lines": list(self.vertical_lines),
            "horizontal_lines": list(self.horizontal_lines),
            "half_width_px": self.half_width_px,
        }


def grid_0915_config() -> FixedGridConfig:
    """Measured grid in raw 0915 AVIs; not a normalized-coordinate guess.

    Do not apply this to an already cropped/resized image. Lossy compressed
    inputs can additionally have ringing outside the original one-pixel grid;
    prefer the original AVI for quantitative tracking when available.
    """
    return FixedGridConfig(
        width=1024,
        height=768,
        vertical_lines=(170, 341, 511, 682, 853),
        horizontal_lines=(127, 255, 383, 511, 639),
    )


def _bands(centers: tuple[int, ...], radius: int) -> tuple[tuple[int, int], ...]:
    """Inclusive intervals, merging touching bands before interpolation."""
    result: list[tuple[int, int]] = []
    for center in centers:
        lo, hi = center - radius, center + radius
        if result and lo <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((lo, hi))
    return tuple(result)


class FixedGridProcessor:
    """Callable, deterministic, CPU-only interpolation with immutable config.

    Horizontal strips are filled from their two neighboring clean rows, then
    vertical strips from their clean neighboring columns. At crossings this
    is bilinear interpolation of the four unmasked corners, without leaking
    overlay intensities back into the result. uint8 rounding is at most half
    an intensity level per pass. All unmasked pixels remain bit-identical.
    """

    ALGORITHM = "fixed-grid-opposite-neighbors-v1"

    def __init__(self, config: FixedGridConfig):
        if not isinstance(config, FixedGridConfig):
            raise TypeError("config must be a FixedGridConfig")
        self._config = config
        self._rows = _bands(config.horizontal_lines, config.half_width_px)
        self._columns = _bands(config.vertical_lines, config.half_width_px)

    @property
    def config(self) -> FixedGridConfig:
        return self._config

    def metadata(self) -> dict:
        """Fresh JSON-serializable metadata, suitable for run/cache manifests."""
        return {"algorithm": self.ALGORITHM, "coordinate_space": "original_full_frame", **self.config.to_dict()}

    def signature(self) -> str:
        """Stable fingerprint; caches must distinguish identity from this."""
        payload = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def mask(self) -> np.ndarray:
        """Fresh boolean mask identifying interpolated (not measured) pixels."""
        mask = np.zeros((self.config.height, self.config.width), dtype=bool)
        for lo, hi in self._rows:
            mask[lo:hi + 1] = True
        for lo, hi in self._columns:
            mask[:, lo:hi + 1] = True
        return mask

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.shape[:2] != (self.config.height, self.config.width):
            raise ValueError("fixed grid must be processed at the configured full-frame size, before crop/resize")
        if image.dtype != np.uint8 or (image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3)):
            raise ValueError("frame must be uint8 grayscale or BGR")
        result = image.copy()
        for lo, hi in self._rows:
            weight = np.arange(1, hi - lo + 2, dtype=np.float32) / (hi - lo + 2)
            weight = weight.reshape((-1,) + (1,) * (image.ndim - 1))
            top = result[lo - 1:lo].astype(np.float32)
            bottom = result[hi + 1:hi + 2].astype(np.float32)
            result[lo:hi + 1] = np.rint(top * (1 - weight) + bottom * weight).astype(np.uint8)
        for lo, hi in self._columns:
            weight = np.arange(1, hi - lo + 2, dtype=np.float32) / (hi - lo + 2)
            weight = weight.reshape((1, -1) + (1,) * (image.ndim - 2))
            left = result[:, lo - 1:lo].astype(np.float32)
            right = result[:, hi + 1:hi + 2].astype(np.float32)
            result[:, lo:hi + 1] = np.rint(left * (1 - weight) + right * weight).astype(np.uint8)
        return result
