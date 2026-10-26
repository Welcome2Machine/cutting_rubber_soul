"""Focused regression checks for video evidence and quality masks."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from softcut_solution.video import (
    discover_video_packages, inspect_exchange_frames, inspect_mmt_csv,
    inspect_video, load_fixed_id,
)


def make_package(folder: Path, **overrides):
    folder.mkdir(parents=True, exist_ok=True)
    shape = (4, 3)
    fields = {
        "source_frames": np.array([100, 101, 102, 103]),
        "point_ids": np.array([11, 12, 13]),
        "indentation_estimate_mm": np.array([0., .3, .29, .5]),
        "capture_time_from_contact_s": np.array([0., .015, .025, .04]),
        "contact_image_xright_yup_mm": np.zeros((3, 2)),
        "displacement_from_contact_xright_yup_mm": np.zeros((4, 3, 2)),
        "measurement_supported": np.ones(shape, bool),
        "displacement_endpoint_supported": np.ones(shape, bool),
        "all_intermediate_measurement_supported": np.ones(shape, bool),
        "measurement_motion_step_valid": np.ones(shape, bool),
        "tracker_supported": np.ones(shape, bool),
        "tracker_inferred": np.zeros(shape, bool),
        "tracker_ambiguous": np.zeros(shape, bool),
        "tracker_blade_occluded": np.zeros(shape, bool),
        "tracker_motion_step_valid": np.ones(shape, bool),
        "native_extra": np.array([42]),
    }
    fields.update(overrides)
    for name in [key for key, value in fields.items() if value is None]:
        del fields[name]
    np.savez(folder / "material_displacements.npz", **fields)
    (folder / "metadata.json").write_text(json.dumps({
        "case_id": "test33", "requested_nominal_angle_deg": 33,
        "video_fps": 12, "time": {"physical_clock_verified": False,
        "playback_is_physical_time": False, "mapping": "AVI index / header fps"},
        "scale_status": "conditional stroke calibration",
    }), encoding="utf-8")
    return folder


def test_preserves_native_order_and_nonmonotone_depth(tmp_path):
    path = make_package(tmp_path / "case", source_frames=np.array([100, 103, 101, 102]))
    data = load_fixed_id(path)
    assert data["source_frames"].tolist() == [100, 103, 101, 102]
    assert data["indentation_estimate_mm"].tolist() == [0., .3, .29, .5]
    assert data["native_extra"].tolist() == [42]
    assert data["source_path"] == str((path / "material_displacements.npz").resolve())
    assert data["endpoint_valid"].all()
    assert data["continuous_valid"].all()


@pytest.mark.parametrize("flag", ["tracker_inferred", "tracker_ambiguous", "tracker_blade_occluded"])
def test_invalid_flags_exclude_current_and_reference(tmp_path, flag):
    bad = np.zeros((4, 3), bool)
    bad[0, 0] = True  # bad contact reference excludes this ID forever
    bad[1, 1] = True  # later reacquisition is endpoint-only
    result = load_fixed_id(make_package(tmp_path, **{flag: bad}))
    assert not result["endpoint_valid"][:, 0].any()
    assert result["endpoint_valid"][:, 1].tolist() == [True, False, True, True]
    assert result["continuous_valid"][:, 1].tolist() == [True, False, False, False]
    assert result[flag][0, 0]  # original flag remains unchanged


def test_nonfinite_reference_is_excluded_and_invalid_nan_is_allowed(tmp_path):
    reference = np.zeros((3, 2))
    reference[0] = np.nan
    displacements = np.zeros((4, 3, 2))
    displacements[:, 0] = np.nan
    result = load_fixed_id(make_package(tmp_path,
        contact_image_xright_yup_mm=reference,
        displacement_from_contact_xright_yup_mm=displacements))
    assert not result["endpoint_valid"][:, 0].any()
    assert result["endpoint_valid"][:, 1:].all()


def test_valid_nonfinite_displacement_is_rejected(tmp_path):
    values = np.zeros((4, 3, 2))
    values[2, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite displacement"):
        load_fixed_id(make_package(tmp_path, displacement_from_contact_xright_yup_mm=values))


def test_motion_edges_are_diagnostics_not_implicit_extra_filters(tmp_path):
    edges = np.ones((4, 3), bool)
    edges[0] = False  # edge before clip must not invalidate observations
    edges[2, 1] = False
    data = load_fixed_id(make_package(tmp_path, measurement_motion_step_valid=edges))
    assert data["continuous_valid"][0].all()
    assert data["continuous_valid"].all()
    assert data["endpoint_valid"].all()
    assert not data["measurement_motion_step_valid"][2, 1]


def test_masks_follow_documented_contract_without_extra_tracker_support_gate(tmp_path):
    raw_support = np.ones((4, 3), bool)
    raw_support[2, 1] = False
    continuous = np.ones((4, 3), bool)
    continuous[2:, 2] = False
    data = load_fixed_id(make_package(tmp_path, tracker_supported=raw_support,
        all_intermediate_measurement_supported=continuous))
    assert data["endpoint_valid"].all()
    assert data["continuous_valid"][:, 2].tolist() == [True, True, False, False]


@pytest.mark.parametrize("overrides,match", [
    ({"point_ids": np.array([1, 1, 2])}, "unique"),
    ({"source_frames": np.array([1, 2, 2, 3])}, "unique"),
    ({"measurement_supported": np.ones((4, 2), bool)}, "shape"),
    ({"tracker_inferred": None}, "Missing"),
    ({"tracker_ambiguous": np.full((4, 3), np.nan)}, "mask"),
])
def test_malformed_schema_rejected(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        load_fixed_id(make_package(tmp_path, **overrides))


def test_inspection_keeps_capture_estimate_separate_from_playback(tmp_path):
    report = inspect_video(make_package(tmp_path))
    assert report["capture_time_range_s"] == [0., .04]
    assert report["clock"]["capture_clock_verified"] is False
    assert report["clock"]["playback_is_physical_time"] is False
    assert report["clock"]["force_synchronization_established"] is False
    assert report["scale_status"] == "conditional stroke calibration"
    json.dumps(report, allow_nan=False)


def test_legacy_clocks_and_unknown_lab_depth_are_distinct(tmp_path):
    path = tmp_path / "frames.csv"
    path.write_text("frame_id,time_s,capture_time_estimate_s,depth_video_estimate_mm,depth_mm,valid_frame\n"
                    "0,0,0,0,,1\n1,0.083333,0.015,0.02,,1\n", encoding="utf-8")
    report = inspect_exchange_frames(path)
    assert report["playback_time_range_s"] == [0., .083333]
    assert report["capture_time_estimate_range_s"] == [0., .015]
    assert report["laboratory_depth_range_mm"] is None
    assert report["clock"]["capture_clock_verified"] is False
    json.dumps(report, allow_nan=False)


def test_mmt_v1_and_v2_versions_remain_distinct(tmp_path):
    path = tmp_path / "comparison.csv"
    path.write_text("source_frame,depth_mm,physical_time_s,MMT-v1_offset_um,OBT-v2_offset_um\n"
                    "137,0,1.715,0,0\n139,0.01,1.74,12,14\n", encoding="utf-8")
    report = inspect_mmt_csv(path)
    assert report["methods"] == ["MMT-v1", "OBT-v2"]
    assert report["offset_ranges_um"]["MMT-v1"] == [0., 12.]
    assert not report["clock"]["capture_clock_verified"]


def test_discovery_prefers_declared_set_without_duplicates(tmp_path):
    preferred = make_package(tmp_path / "results" / "20261024_n_group_first_1mm" / "case")
    make_package(tmp_path / "results" / "another" / "case")
    result = discover_video_packages(tmp_path)
    assert len(result) == 2
    assert result[0]["path"] == str(preferred.resolve())
