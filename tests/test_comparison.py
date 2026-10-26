"""Known observation operators, not synthetic evidence of physical accuracy."""
import csv
import gzip
import hashlib
import json

import numpy as np
import pytest

from softcut_solution import comparison as comp


X = np.array([[0., 0.], [2., 0.], [0., 2.], [2., 2.]])
TRI = np.array([[0, 1, 2], [1, 3, 2]])
POINTS = np.array([[.2, .3], [1.2, .3], [3., 3.]])


def manufactured_fe(tmp_path, qualified=(True, True, True), exported=(0, 1, 2)):
    """Accepted reference is translated, deliberately different from mesh rest."""
    reference = X + [3., -2.]
    frames = []
    for index in exported:
        d = index * .5
        x = reference + np.column_stack((.1 * d * X[:, 0], -.2 * d * X[:, 1]))
        frames.append(dict(index=index, depth_mm=d, positions=x.tolist(), blade=[[3, -2-d], [4, 0-d], [2, 0-d]]))
    result = dict(schema_version="softcut-result/1", metadata=dict(name="Manufactured affine fixture", angle_deg=33,
                  qualification=dict(qualified_states=3)), series=[dict(depth_mm=i*.5, qualified=q,
                  qualification="audited_prefix" if q else "unqualified_masked") for i, q in enumerate(qualified)],
                  mesh=dict(reference_positions=X.tolist(), triangles=TRI.tolist()), frames=frames)
    path = tmp_path / "fe" / "result.bundle.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result))
    return path


def manufactured_video(tmp_path, depths=(0, .25, .5, .48, 1., 1.2)):
    depth = np.asarray(depths, float)
    displacement = np.stack([np.column_stack((.1*d*POINTS[:, 0], -.2*d*POINTS[:, 1])) for d in depth])
    metadata = dict(case_id="fixture", requested_nominal_angle_deg=33, scale_um_per_px=1000.,
                    coordinates=dict(image_origin_source_px=[0, 0]), scale_status="manufactured")
    data = dict(metadata=metadata, point_ids=np.arange(3), contact_image_xright_yup_mm=POINTS.copy(),
                displacement_from_contact_xright_yup_mm=displacement, indentation_estimate_mm=depth,
                source_frames=np.arange(len(depth)) + 100, avi_source_frames=np.arange(len(depth)) * 2,
                capture_time_from_contact_s=np.arange(len(depth))*.1,
                endpoint_valid=np.ones((len(depth), 3), bool), continuous_valid=np.ones((len(depth), 3), bool),
                blade_tip_source_px=np.zeros((len(depth), 2)))
    folder = tmp_path / "video"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps(metadata))
    np.savez(folder / "material_displacements.npz", fixture=np.arange(3))
    return folder, data


def run_fixture(tmp_path, monkeypatch, *, qualified=(True, True, True), options=None, modifier=None):
    fe = manufactured_fe(tmp_path, qualified)
    video_path, video = manufactured_video(tmp_path)
    if modifier:
        modifier(video)
    monkeypatch.setattr(comp, "_load_video", lambda path: video)
    result = comp.compare(fe, video_path, tmp_path / "comparison", dict(registration="tip_translation", max_depth_mm=2., **(options or {})))
    with open(result["observables_path"], encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return result, rows


def test_known_affine_displacement_uses_accepted_reference_fixed_barycentrics(tmp_path, monkeypatch):
    result, rows = run_fixture(tmp_path, monkeypatch)
    assert result["validation"]["mapped_points"] == 2
    assert result["validation"]["selected_cohort_points"] == 2
    for row in rows[:-1]:
        assert float(row["vector_rmse_mm"]) < 1e-14
        assert int(row["valid_point_count"]) == 2
    with open(tmp_path / "comparison/mapping.csv", encoding="utf-8-sig") as stream:
        mapping = list(csv.DictReader(stream))
    assert mapping[2]["mapped"] == "0"
    assert mapping[2]["triangle_index"] == ""
    assert mapping[2]["b0"] == ""
    assert "not independent laboratory calibration" in (tmp_path / "comparison/index.html").read_text(encoding="utf8")


def test_does_not_sort_nonmonotone_video_depth_or_extrapolate(tmp_path, monkeypatch):
    result, rows = run_fixture(tmp_path, monkeypatch)
    assert [float(r["depth_mm"]) for r in rows] == [0, .25, .5, .48, 1., 1.2]
    assert [int(r["source_frame_id"]) for r in rows] == list(range(100, 106))
    assert rows[-1]["predicted_mean_y_mm"] == ""
    assert rows[-1]["valid_point_count"] == "0"
    assert result["validation"]["video_depth_nonmonotone_steps"] == 1


def test_first_failed_audit_blocks_later_qualified_frames(tmp_path, monkeypatch):
    result, rows = run_fixture(tmp_path, monkeypatch, qualified=(True, False, True))
    assert rows[0]["valid_point_count"] == "2"
    assert all(r["valid_point_count"] == "0" for r in rows[1:])
    manifest = json.loads((tmp_path / "comparison/manifest.json").read_text())
    assert manifest["qualification"]["qualified_prefix_states"] == 1


def test_continuous_cohort_is_fixed_across_the_entire_overlap(tmp_path, monkeypatch):
    def unsupported(data):
        data["continuous_valid"][2:, 1] = False
        data["endpoint_valid"][2, 1] = False
    result, rows = run_fixture(tmp_path, monkeypatch, modifier=unsupported)
    assert result["validation"]["selected_cohort_points"] == 1
    assert all(r["valid_point_count"] == "1" for r in rows[:-1])
    with gzip.open(tmp_path / "comparison/trajectories.csv.gz", "rt", encoding="utf-8-sig") as stream:
        tracks = list(csv.DictReader(stream))
    assert len(tracks) == 18
    assert all(r["observed_y_mm"] == "" for r in tracks if r["used"] == "0")


def test_endpoint_support_does_not_interpolate_through_missing_point(tmp_path, monkeypatch):
    def unsupported(data):
        data["endpoint_valid"][2, 1] = False
    result, rows = run_fixture(tmp_path, monkeypatch, modifier=unsupported, options={"support_mode": "endpoint"})
    assert rows[1]["valid_point_count"] == "2"
    assert rows[2]["valid_point_count"] == "1"
    assert rows[3]["valid_point_count"] == "2"
    assert any("can change" in w for w in result["validation"]["limitations"])


def test_angle_mismatch_rejected_before_writing(tmp_path, monkeypatch):
    fe = manufactured_fe(tmp_path)
    path, data = manufactured_video(tmp_path)
    data["metadata"]["requested_nominal_angle_deg"] = 57
    monkeypatch.setattr(comp, "_load_video", lambda p: data)
    with pytest.raises(ValueError, match="angle"):
        comp.compare(fe, path, tmp_path / "report", {"registration": "tip_translation"})
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("options,error", [
    ({"max_depth_mm": 3}, "max_depth"),
    ({"depth_offset_mm": .1}, "reference depth"),
    ({"fe_reference_index": 7}, "reference index"),
    ({"intact_status": "intact"}, "intactness"),
    ({"registration": "explicit"}, "translation_y"),
])
def test_invalid_physical_or_reference_options_fail(tmp_path, monkeypatch, options, error):
    fe = manufactured_fe(tmp_path)
    path, data = manufactured_video(tmp_path)
    monkeypatch.setattr(comp, "_load_video", lambda p: data)
    args = {"registration": "tip_translation", **options}
    with pytest.raises(ValueError, match=error):
        comp.compare(fe, path, tmp_path / "report", args)


def test_missing_scale_is_not_guessed_for_tip_registration(tmp_path, monkeypatch):
    fe = manufactured_fe(tmp_path)
    path, data = manufactured_video(tmp_path)
    del data["metadata"]["scale_um_per_px"]
    monkeypatch.setattr(comp, "_load_video", lambda p: data)
    with pytest.raises(ValueError, match="scale_um_per_px"):
        comp.compare(fe, path, tmp_path / "report", {"registration": "tip_translation"})


def test_contradictory_pixel_origin_is_rejected(tmp_path, monkeypatch):
    fe = manufactured_fe(tmp_path)
    path, data = manufactured_video(tmp_path)
    data["image_origin_source_px"] = np.array([1., 0.])
    monkeypatch.setattr(comp, "_load_video", lambda p: data)
    with pytest.raises(ValueError, match="image origin"):
        comp.compare(fe, path, tmp_path / "report", {"registration": "tip_translation"})


def test_inspect_fe_reports_only_continuous_qualified_frames(tmp_path):
    path = manufactured_fe(tmp_path, qualified=(True, False, True))
    data = comp.inspect_fe(path)
    assert data["qualified_states"] == 1
    assert data["saved_frame_indices"] == [0]


def test_direct_barycentric_mapping_reconstructs_points_and_keeps_outside_missing():
    indices, weights = comp._map_points(X, TRI, POINTS)
    assert list(indices >= 0) == [True, True, False]
    actual = np.einsum("ni,nij->nj", weights[:2], X[TRI[indices[:2]]])
    np.testing.assert_allclose(actual, POINTS[:2], atol=1e-14)


def test_direct_renamed_npz_input_is_fingerprinted(tmp_path, monkeypatch):
    fe = manufactured_fe(tmp_path)
    folder, data = manufactured_video(tmp_path)
    source = folder / "renamed.npz"
    (folder / "material_displacements.npz").rename(source)
    data["source_path"] = str(source)
    monkeypatch.setattr(comp, "_load_video", lambda p: data)
    result = comp.compare(fe, source, tmp_path / "report", {"registration": "tip_translation"})
    manifest = json.loads((tmp_path / "report/manifest.json").read_text())
    assert manifest["sources_sha256"][str(source.resolve())] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not any(key.endswith("material_displacements.npz") for key in manifest["sources_sha256"])
    assert result["validation"]["source_files_unchanged"]


def test_modification_to_renamed_input_during_read_is_rejected(tmp_path, monkeypatch):
    fe = manufactured_fe(tmp_path)
    folder, data = manufactured_video(tmp_path)
    source = folder / "renamed.npz"
    (folder / "material_displacements.npz").rename(source)
    data["source_path"] = str(source)
    def changed_input(path):
        with source.open("ab") as stream:
            stream.write(b"changed")
        return data
    monkeypatch.setattr(comp, "_load_video", changed_input)
    with pytest.raises(ValueError, match="Source files changed"):
        comp.compare(fe, source, tmp_path / "report", {"registration": "tip_translation"})
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("depths,maximum,first_reason", [
    ((0, .25, .6, .4), .5, "outside_requested_window"),
    ((0, .25, 1.2, .4), 2., "outside_qualified_saved_frame_range"),
])
def test_first_window_exit_permanently_cuts_off_later_frames(tmp_path, monkeypatch, depths, maximum, first_reason):
    fe = manufactured_fe(tmp_path)
    path, data = manufactured_video(tmp_path, depths=depths)
    # A future unsupported point must not retroactively shrink the pre-cutoff cohort.
    data["continuous_valid"][3, 1] = False
    monkeypatch.setattr(comp, "_load_video", lambda p: data)
    result = comp.compare(fe, path, tmp_path / "report", {"registration": "tip_translation", "max_depth_mm": maximum})
    with open(tmp_path / "report/frames.csv", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["status"] for r in rows] == ["sampled_saved_states", "sampled_saved_states", first_reason, "after_cutoff"]
    assert [r["used_point_count"] for r in rows] == ["2", "2", "0", "0"]
    assert result["validation"]["selected_cohort_points"] == 2
