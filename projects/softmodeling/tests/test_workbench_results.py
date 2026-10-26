"""Reusable saved-state export regressions, with explicitly manufactured fields.

The tiny fixture is not a physical loading solution or experimental evidence.
Tests check reconstruction, provenance masking, field range and IO semantics.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from workbench.results import export_case, saved_state_fields
from workbench import results as postprocess


MATERIAL = dict(mu_eq=[.14], alpha_eq=[2.], mu_ov=[1.24], alpha_ov=[2.],
                poisson=.49, relaxation_s=1.7, viscous_flow="isochoric_maxwell",
                volumetric="quadratic_j")
X = np.array([[0., 0.], [1., 0.], [0., 1.]])
TRI = np.array([[0, 1, 2]])


def manufactured_case(path):
    (path / "states").mkdir(parents=True)
    steps, rows = [], []
    for i in range(3):
        F = np.diag([1 + .1*i, 1 / (1 + .1*i)])
        x = X @ F.T
        metric = np.eye(3)[None]
        fields, energy, areas = saved_state_fields(X, x, TRI, metric, MATERIAL)
        blade = np.array([[.4, 2-i*.1], [.5, 1-i*.1], [.6, 2-i*.1]])
        state_path = path / "states" / f"{i:04}.npz"
        np.savez(state_path, reference_xy=np.vstack([X, blade]), positions=np.vstack([x, blade]),
                 triangles=TRI, body_nodes=3, depth_mm=i*.1, material_metric=metric)
        step = dict(depth_mm=i*.1, reaction_fz_n=float(4*i), reaction_x_n=float(-3*i),
                    residual_n=1e-6, min_j=float(fields["J"].min()), material={
                        "stored_energy_n_mm": float(20 * (areas @ energy)), "time_s": i*.5,
                        "dt_s": .5, "equilibrium_energy_n_mm": .1*i,
                        "overstress_energy_n_mm": .3*i, "viscous_dissipation_increment_n_mm": .01*i},
                    tip_contact={"status": "ambiguous", "reference_offset_mm": .02})
        steps.append(step)
        rows.append(dict(depth_mm=step["depth_mm"], reaction_fz_n=step["reaction_fz_n"],
                         residual_n=step["residual_n"], min_j=step["min_j"], passed=True,
                         checkpoint_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest()))
    report = dict(completed=True, target_mm=.2, signature=dict(material=MATERIAL,
                  config=dict(width_mm=20., inclination_deg=57., friction_coefficient=.35)),
                  settings=dict(tip_radius=.02),
                  source_sha256={"retired/source.py": "historical-not-current"}, steps=steps,
                  fixture_kind="manufactured affine fields, not a solved loading history")
    (path / "result.json").write_text(json.dumps(report), encoding="utf8")
    (path / "audit.json").write_text(json.dumps(dict(passed=True, rows=rows)), encoding="utf8")
    return path


def test_affine_isochoric_fields_have_known_stress_and_strain():
    x = X @ np.diag([2., .5])
    fields, energy, areas = saved_state_fields(X, x, TRI, np.eye(3)[None], MATERIAL)
    # Three stretches (2, .5, 1), J=1, total G=1.38; spectral values are explicit.
    assert fields["tensile_MPa"][0] == pytest.approx(1.38*(4-5.25/3))
    assert fields["green_max"][0] == pytest.approx(1.5)
    assert fields["green_min"][0] == pytest.approx(-.375)
    assert fields["J"][0] == pytest.approx(1.)
    assert energy[0] == pytest.approx(1.38/2*(5.25-3))
    assert areas[0] == pytest.approx(.5)


def test_rigid_rotation_has_no_stress_or_strain_and_keeps_translation():
    theta = .73
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x = X @ R.T + [1.2, -.8]
    fields, energy, _ = saved_state_fields(X, x, TRI, np.eye(3)[None], MATERIAL)
    for key in ("tensile_MPa", "green_max", "green_min"):
        np.testing.assert_allclose(fields[key], 0, atol=1e-13)
    np.testing.assert_allclose(energy, 0, atol=1e-13)
    np.testing.assert_allclose(fields["displacement_mm"], np.linalg.norm(x-X, axis=1))


def test_committed_metric_is_used_instead_of_elastic_reset():
    metric = np.diag([4., .25, 1.])[None]
    fields, energy, _ = saved_state_fields(X, X, TRI, metric, MATERIAL)
    assert fields["tensile_MPa"][0] == pytest.approx(1.24*(4-5.25/3))
    assert energy[0] == pytest.approx(1.24/2*(5.25-3))


@pytest.mark.parametrize("invalid", [dict(alpha_ov=[1.]), dict(viscous_flow="paper2023")])
def test_unsupported_material_fails_closed(invalid):
    with pytest.raises(ValueError, match="alpha=2"):
        saved_state_fields(X, X, TRI, np.eye(3)[None], MATERIAL | invalid)


def test_export_signed_components_fixed_range_and_historical_source(tmp_path):
    source = manufactured_case(tmp_path / "manufactured")
    bundle = export_case(source, tmp_path / "view", max_frames=2)
    assert bundle["schema_version"] == "softcut-result/1"
    assert bundle["series"][1]["fy_N"] == -3.
    assert bundle["series"][1]["fz_N"] == 4.
    assert bundle["series"][1]["fyz_N"] == 5.
    assert bundle["metadata"]["name"] == "57° · G∞/Gv=0.14/1.24 MPa · μ=0.35 · r=0.02 mm"
    assert bundle["series"][1]["dt_s"] == .5
    assert bundle["series"][1]["equilibrium_energy_Nmm"] == .1
    assert bundle["series"][1]["overstress_energy_Nmm"] == .3
    assert bundle["series"][1]["viscous_dissipation_increment_Nmm"] == .01
    assert bundle["series"][1]["contact_reference_offset_mm"] is None
    assert bundle["metadata"]["qualification"]["complete_path_qualified"]
    assert [f["index"] for f in bundle["frames"]] == [0, 2]
    assert len(bundle["frames"][0]["blade"]) == 3
    for key, info in bundle["field_info"].items():
        values = np.concatenate([f["fields"][key] for f in bundle["frames"]])
        np.testing.assert_equal(info["range"], [values.min(), values.max()])
    assert (tmp_path / "view" / "curves.csv").is_file()
    assert json.loads((tmp_path / "view" / "result.bundle.json").read_text(encoding="utf8")) == bundle


def test_audit_hole_stops_qualified_history_even_if_later_passes(tmp_path):
    source = manufactured_case(tmp_path / "manufactured")
    audit = json.loads((source / "audit.json").read_text())
    audit["rows"][1]["passed"] = False
    alternative = tmp_path / "partial_audit.json"
    alternative.write_text(json.dumps(audit))
    bundle = export_case(source, tmp_path / "view", audit_path=alternative)
    assert len(bundle["frames"]) == 1
    assert bundle["series"][2]["fz_N"] is None
    assert bundle["series"][2]["raw_fz_N"] == 8
    assert not bundle["metadata"]["qualification"]["complete_path_qualified"]


def test_no_audit_is_not_promoted_to_physical_completion(tmp_path):
    source = manufactured_case(tmp_path / "manufactured")
    (source / "audit.json").unlink()
    bundle = export_case(source, tmp_path / "view")
    assert not bundle["frames"]
    assert bundle["series"][-1]["fz_N"] is None
    assert bundle["series"][-1]["raw_fz_N"] == 8


def test_failed_history_chain_cannot_be_overridden_by_passing_rows(tmp_path):
    source = manufactured_case(tmp_path / "manufactured")
    path = source / "audit.json"
    audit = json.loads(path.read_text())
    audit.update(passed=False, history_chain_checks_passed=False)
    path.write_text(json.dumps(audit))
    bundle = export_case(source, tmp_path / "view")
    assert not bundle["frames"]
    assert bundle["metadata"]["qualification"]["qualified_states"] == 0


def test_selected_checkpoint_change_and_audit_record_mismatch_rejected(tmp_path):
    source = manufactured_case(tmp_path / "manufactured")
    state = source / "states" / "0002.npz"
    with state.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        export_case(source, tmp_path / "view", max_frames=2)
    audit_path = source / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["rows"][0]["reaction_fz_n"] = 9
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="Audit/result mismatch"):
        export_case(source, tmp_path / "view")


def test_missing_ffmpeg_is_explicit_and_does_not_erase_export(tmp_path, monkeypatch):
    source = manufactured_case(tmp_path / "manufactured")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    bundle = export_case(source, tmp_path / "view", movie=True)
    assert bundle["metadata"]["movie"]["status"] == "failed"
    assert "FFmpeg" in bundle["metadata"]["movie"]["reason"]
    assert (tmp_path / "view" / "result.bundle.json").is_file()


def test_failed_movie_removes_own_partial_file_and_keeps_previous_complete_movie(tmp_path, monkeypatch):
    source = manufactured_case(tmp_path / "manufactured")
    output = tmp_path / "view"
    output.mkdir()
    (output / "indentation.mp4").write_bytes(b"previous complete export")
    def interrupted_movie(bundle, destination, **kwargs):
        destination.write_bytes(b"partial")
        raise RuntimeError("fixture encoder interruption")
    monkeypatch.setattr(postprocess, "render_movie", interrupted_movie)
    bundle = export_case(source, output, movie=True)
    assert bundle["metadata"]["movie"]["status"] == "failed"
    assert not (output / "indentation.partial.mp4").exists()
    assert (output / "indentation.mp4").read_bytes() == b"previous complete export"


def test_historical_unrecorded_tip_radius_is_not_replaced_with_current_default(tmp_path):
    source = manufactured_case(tmp_path / "manufactured")
    path = source / "result.json"
    report = json.loads(path.read_text())
    del report["settings"]["tip_radius"]
    path.write_text(json.dumps(report))
    bundle = export_case(source, tmp_path / "view")
    assert bundle["metadata"]["name"].endswith("r=? mm")


def test_postprocess_matches_an_actual_saved_state_when_available():
    """Optional local regression; the repository does not ship binary results."""
    case = Path(__file__).resolve().parents[1] / "outputs/baseline/angle57"
    if not (case / "states/0050.npz").is_file():
        pytest.skip("Local accepted baseline data not distributed with code")
    report = json.loads((case / "result.json").read_text(encoding="utf8"))
    with np.load(case / "states/0050.npz", allow_pickle=False) as state:
        n = int(state["body_nodes"])
        fields, energy, area = saved_state_fields(state["reference_xy"][:n], state["positions"][:n],
            state["triangles"], state["material_metric"], report["signature"]["material"])
    record = report["steps"][50]
    assert fields["J"].min() == pytest.approx(record["min_j"], abs=1e-10)
    assert area @ energy * report["signature"]["config"]["width_mm"] == pytest.approx(
        record["material"]["stored_energy_n_mm"], rel=1e-8, abs=1e-9)
