"""Read saved video observations without changing clocks, frames or coordinates.

The preferred format is the fixed material-ID displacement package. The older
exchange and MMT/OBT CSVs are inspected as summaries only: their playback clocks
and model contact estimates must not be silently promoted to measurements.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


_POINT_MASKS = (
    "measurement_supported", "displacement_endpoint_supported",
    "all_intermediate_measurement_supported", "tracker_supported",
    "tracker_inferred", "tracker_blade_occluded", "tracker_ambiguous",
)
_OPTIONAL_MASKS = ("measurement_motion_step_valid", "tracker_motion_step_valid")
_POINT_FIELDS = (
    "seed_points_source_px", "contact_positions_source_px",
    "contact_image_xright_yup_mm",
)
_VECTOR_FIELDS = (
    "positions_source_px", "image_xright_yup_mm", "displacement_from_contact_px",
    "displacement_from_contact_xright_yup_mm",
)
_FRAME_FIELDS = (
    "avi_source_frames", "clip_frames", "capture_time_from_contact_s",
    "indentation_estimate_mm", "blade_reference_state",
)
_DIAGNOSTIC_FIELDS = (
    "tracker_correlation", "tracker_cycle_error", "tracker_reference_disagreement",
    "tracker_position_kind",
)


def _metadata(folder: Path) -> dict:
    path = folder / "metadata.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata.json must contain an object: {path}")
    return value


def _shape(data: dict, name: str, expected: tuple[int, ...]) -> None:
    if name not in data:
        raise ValueError(f"Missing fixed-ID field: {name}")
    if data[name].shape != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {data[name].shape}")


def _mask(data: dict, name: str, shape: tuple[int, int]) -> np.ndarray:
    _shape(data, name, shape)
    value = data[name]
    if value.dtype.kind not in "buif" or not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name}: mask must contain only boolean or 0/1 values")
    return value.astype(bool, copy=False)


def _finite_vectors(data: dict, name: str) -> np.ndarray:
    try:
        return np.isfinite(data[name]).all(axis=-1)
    except TypeError as error:
        raise ValueError(f"{name}: expected numeric coordinates") from error


def load_fixed_id(path: Path) -> dict:
    """Load the fixed-ID NPZ, its metadata, and conservative validity masks.

    All native arrays are returned under their original names and are unchanged.
    ``endpoint_valid`` applies the native endpoint support plus inferred,
    ambiguous and blade-hidden exclusions at both endpoints. A bad contact
    reference excludes that ID at every frame. ``continuous_valid`` applies the
    native all-intermediate support and the entire preceding history of those
    exclusions, as specified by video_fe_comparison.md section 5.2. Motion-edge
    flags remain available as native diagnostics and do not change these masks.

    Invalid observations may contain NaN. Nonfinite displacements declared valid
    after the endpoint quality checks are rejected as inconsistent input. Depth
    fluctuations and the stored frame order are preserved, never sorted.
    """
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "material_displacements.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    for name in ("source_frames", "point_ids"):
        if name not in data or data[name].ndim != 1 or data[name].size == 0:
            raise ValueError(f"{name}: expected a nonempty one-dimensional ID array")
        if data[name].dtype.kind not in "iu" or np.any(data[name] < 0):
            raise ValueError(f"{name}: expected nonnegative integer IDs")
        if len(np.unique(data[name])) != len(data[name]):
            raise ValueError(f"{name}: IDs must be unique")
    frames, points = len(data["source_frames"]), len(data["point_ids"])
    for name, shape in (
        ("indentation_estimate_mm", (frames,)),
        ("capture_time_from_contact_s", (frames,)),
        ("contact_image_xright_yup_mm", (points, 2)),
        ("displacement_from_contact_xright_yup_mm", (frames, points, 2)),
    ):
        _shape(data, name, shape)
    for names, shape in (
        (_POINT_FIELDS, (points, 2)), (_VECTOR_FIELDS, (frames, points, 2)),
        (_FRAME_FIELDS, (frames,)), (_DIAGNOSTIC_FIELDS, (frames, points)),
        (("blade_tip_source_px",), (frames, 2)),
        (("image_origin_source_px",), (2,)),
    ):
        for name in names:
            if name in data:
                _shape(data, name, shape)
    masks = {name: _mask(data, name, (frames, points)) for name in _POINT_MASKS}
    for name in _OPTIONAL_MASKS:
        if name in data:
            masks[name] = _mask(data, name, (frames, points))
    if "scale_um_per_px" in data:
        _shape(data, "scale_um_per_px", ())
        if not math.isfinite(float(data["scale_um_per_px"])) or float(data["scale_um_per_px"]) <= 0:
            raise ValueError("scale_um_per_px must be finite and positive")
    for name in ("indentation_estimate_mm", "capture_time_from_contact_s"):
        if data[name].dtype.kind not in "iuf" or not np.isfinite(data[name]).all():
            raise ValueError(f"{name}: expected finite numeric samples")
    bad = (masks["tracker_inferred"] | masks["tracker_ambiguous"]
           | masks["tracker_blade_occluded"])
    reference_good = ~bad[0] & _finite_vectors(data, "contact_image_xright_yup_mm")
    if "contact_positions_source_px" in data:
        reference_good &= _finite_vectors(data, "contact_positions_source_px")
    endpoint = masks["displacement_endpoint_supported"] & ~bad & reference_good[None, :]
    continuous = (masks["all_intermediate_measurement_supported"]
                  & ~np.logical_or.accumulate(bad, axis=0) & reference_good[None, :])
    # A nonfinite saved position is unusable, including the first/contact frame.
    for name in ("positions_source_px", "image_xright_yup_mm"):
        if name in data:
            finite = _finite_vectors(data, name)
            endpoint &= finite & finite[0][None, :]
            continuous &= finite & finite[0][None, :]
    for name in ("displacement_from_contact_xright_yup_mm", "displacement_from_contact_px"):
        if name in data and np.any((endpoint | continuous) & ~_finite_vectors(data, name)):
            raise ValueError(f"{name}: nonfinite displacement at a qualified endpoint")
    data.update(metadata=_metadata(path.parent), source_path=str(path),
                endpoint_valid=endpoint, continuous_valid=continuous,
                frame_count=frames, point_count=points,
                validity_policy="video_fe_comparison.md section 5.2; finite coordinates and displacements required; motion-edge diagnostics do not alter masks")
    return data


def _range(values) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return [float(finite.min()), float(finite.max())] if finite.size else None


def _clock_limits(metadata: dict) -> dict:
    source = metadata.get("source", {})
    clock = metadata.get("time", {})
    return {
        "capture_clock_verified": bool(clock.get("physical_clock_verified", source.get("physical_capture_time_verified", False))),
        "playback_is_physical_time": False,
        "force_synchronization_established": False,
        "capture_time_basis": clock.get("mapping", source.get("capture_time_estimate_formula", "not supplied")),
        "native_clock_metadata": clock or source.get("capture_to_playback_time_mapping", {}),
        "note": "Capture-time estimates and MP4 playback time are separate clocks; no force synchronization is inferred.",
    }


def inspect_video(path: Path) -> dict:
    """Return a JSON-safe summary; use fixed IDs first when given a folder."""
    path = Path(path).expanduser().resolve()
    fixed = path / "material_displacements.npz" if path.is_dir() else path
    if fixed.name == "material_displacements.npz" and fixed.is_file():
        data = load_fixed_id(fixed)
        meta = data["metadata"]
        return {
            "kind": "fixed_id", "label": meta.get("case_id", fixed.parent.name),
            "case_id": meta.get("case_id", fixed.parent.name), "path": str(fixed.parent),
            "source_path": data["source_path"], "frame_count": data["frame_count"],
            "point_count": data["point_count"],
            "source_frame_range": [int(data["source_frames"].min()), int(data["source_frames"].max())],
            "depth_range_mm": _range(data["indentation_estimate_mm"]),
            "capture_time_range_s": _range(data["capture_time_from_contact_s"]),
            "endpoint_valid_count": int(data["endpoint_valid"].sum()),
            "continuous_valid_count": int(data["continuous_valid"].sum()),
            "endpoint_valid_fraction": float(data["endpoint_valid"].mean()),
            "continuous_valid_fraction": float(data["continuous_valid"].mean()),
            "endpoint_valid_last_frame": int(data["endpoint_valid"][-1].sum()),
            "continuous_valid_last_frame": int(data["continuous_valid"][-1].sum()),
            "nominal_angle_deg": meta.get("requested_nominal_angle_deg"),
            "clock": _clock_limits(meta), "scale_status": meta.get("scale_status", "unknown"),
            "contact_zero_method": meta.get("contact_zero_method", "unknown"),
            "coordinates": meta.get("coordinates", {}),
            "limitations": [
                "Depth and millimetre coordinates use conditional image calibration, not laboratory registration.",
                "Endpoint support does not certify identity across unsupported intervals; use continuous_valid for uninterrupted observations.",
                "The analysis cutoff is not a fracture detection; intact material is not guaranteed.",
            ],
            "metadata": meta,
        }
    if path.is_dir():
        if (path / "frames.csv").is_file():
            return inspect_exchange_frames(path)
        if (path / "comparison.csv").is_file():
            return inspect_mmt_csv(path)
    elif path.name == "frames.csv":
        return inspect_exchange_frames(path)
    elif path.name == "comparison.csv" or path.suffix.lower() == ".csv":
        return inspect_mmt_csv(path)
    raise ValueError(f"No supported video package at {path}")


def _rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _number(text) -> float | None:
    try:
        result = float(text)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def inspect_exchange_frames(path: Path) -> dict:
    """Read only the small legacy frames table, never the material tracks."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "frames.csv"
    metadata = _metadata(path.parent)
    columns, frame_count, valid_count = [], 0, 0
    ranges = {name: [] for name in ("frame_id", "time_s", "capture_time_estimate_s", "depth_video_estimate_mm", "depth_mm")}
    for row in _rows(path):
        if not columns:
            columns = list(row)
            if not {"frame_id", "time_s", "depth_video_estimate_mm"}.issubset(row):
                raise ValueError("Not a legacy exchange frames.csv schema")
        frame_count += 1
        valid_count += row.get("valid_frame", "") in ("1", "True", "true")
        for field in ranges:
            value = _number(row.get(field))
            if value is not None:
                ranges[field].append(value)
    if not frame_count:
        raise ValueError("Exchange frames table is empty")
    return {
        "kind": "legacy_exchange", "label": metadata.get("case_id", path.parent.name),
        "case_id": metadata.get("case_id", path.parent.name), "source_path": str(path),
        "frame_count": frame_count, "valid_frame_count": valid_count, "columns": columns,
        "source_frame_range": _range(ranges["frame_id"]),
        "playback_time_range_s": _range(ranges["time_s"]),
        "capture_time_estimate_range_s": _range(ranges["capture_time_estimate_s"]),
        "depth_range_mm": _range(ranges["depth_video_estimate_mm"]),
        "laboratory_depth_range_mm": _range(ranges["depth_mm"]),
        "clock": _clock_limits(metadata), "metadata": metadata,
        "limitations": ["time_s is MP4 playback time, not a physical acquisition clock.",
                        "depth_video_estimate_mm is conditional image depth; empty depth_mm remains unknown.",
                        "This summary does not read or qualify the per-point tracks."],
    }


def inspect_mmt_csv(path: Path) -> dict:
    """Summarize MMT/OBT v1/v2 curves without conflating them with observations."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "comparison.csv"
    columns, methods, values, depths, times = [], [], {}, [], []
    row_count = 0
    for row in _rows(path):
        if not columns:
            columns = list(row)
            methods = [name[:-len("_offset_um")] for name in columns
                       if name.endswith("_offset_um") and name.startswith(("MMT-", "OBT-"))]
            if not methods or not {"source_frame", "depth_mm", "physical_time_s"}.issubset(row):
                raise ValueError("Not an MMT/OBT comparison.csv schema")
            values = {method: [] for method in methods}
        row_count += 1
        for field, target in (("depth_mm", depths), ("physical_time_s", times)):
            value = _number(row.get(field))
            if value is not None:
                target.append(value)
        for method in methods:
            value = _number(row.get(method + "_offset_um"))
            if value is not None:
                values[method].append(value)
    if not row_count:
        raise ValueError("MMT/OBT comparison table is empty")
    return {
        "kind": "contact_model_curves", "label": path.parent.name, "source_path": str(path),
        "frame_count": row_count, "columns": columns, "methods": methods,
        "depth_range_mm": _range(depths), "estimated_capture_time_range_s": _range(times),
        "offset_ranges_um": {method: _range(value) for method, value in values.items()},
        "clock": {"capture_clock_verified": False, "playback_is_physical_time": False,
                  "force_synchronization_established": False,
                  "note": "physical_time_s retains the upstream acquisition-rate assumption; the CSV alone does not verify it."},
        "limitations": ["Offset is model-estimated original-surface address migration, not independently measured interface slip.",
                        "Model sensitivity is not a calibrated measurement confidence interval.",
                        "The first zero is imposed by the contact reference."],
    }


def discover_video_packages(video_root: Path) -> list[dict]:
    """Find fixed-ID packages with bounded traversal; prefer the declared 1 mm set.

    Only three directory levels under results are inspected, avoiding traversal
    through the many thousands of masks, images and model files.
    """
    root = Path(video_root).expanduser().resolve()
    results = root if root.name == "results" else root / "results"
    preferred = results / "20261024_n_group_first_1mm"
    candidates = []
    if preferred.is_dir():
        candidates.extend(sorted(preferred.glob("*/material_displacements.npz")))
    if results.is_dir():
        for pattern in ("*/material_displacements.npz", "*/*/material_displacements.npz", "*/*/*/material_displacements.npz"):
            candidates.extend(sorted(results.glob(pattern)))
    found, seen = [], set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        metadata = _metadata(path.parent)
        case_id = str(metadata.get("case_id", path.parent.name))
        angle = metadata.get("requested_nominal_angle_deg")
        suffix = f" · {angle}°" if angle is not None else ""
        found.append({"label": f"{case_id}{suffix} · {path.parent.parent.name}", "path": str(path.parent)})
    return found
