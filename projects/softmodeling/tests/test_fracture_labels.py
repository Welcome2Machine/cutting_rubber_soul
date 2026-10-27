"""Evidence and identity gates for derived fracture-event comparisons."""
import copy
import json
from pathlib import Path

import pytest

from tools.fracture_labels import SCHEMA, compare_event, load_labels, validate_labels


def observed(**changes):
    row = dict(id="observed_1", trial_id="trial_a2", kind="verified_observation",
               event_kind="physical_onset", depth_interval_mm=[1.2, 1.3],
               depth_reference="registered_first_contact",
               evidence=dict(source="review.json", method="independent image review",
                             verified=True, interval_verified=True), split="validation")
    row.update(changes)
    return row


def predicted(**changes):
    row = dict(trial_id="trial_a2", event_kind="local_criterion_crossing",
               target_event_kind="physical_onset", status="crossed",
               depth_interval_mm=[1.25, 1.26], depth_reference="registered_first_contact")
    row.update(changes)
    return row


def catalogue(row):
    return dict(schema=SCHEMA, labels=[row])


def test_example_preserves_unknowns_priors_and_wrong_replicate():
    path = Path(__file__).resolve().parents[1] / "configs/fracture_labels.example.json"
    data = load_labels(path)
    assert len(data["labels"]) == 7
    assert all(row["split"] == "unused" for row in data["labels"])
    assert not any(row["kind"] == "verified_observation" for row in data["labels"])
    a1 = next(row for row in data["labels"] if row["id"] == "historical_0deg_a1_approximate")
    assert a1["trial_id"] != "1008n0a2"
    assert a1["depth_interval_mm"] is None


def test_verified_observation_requires_interval_and_interval_evidence():
    row = observed()
    row["evidence"].pop("interval_verified")
    with pytest.raises(ValueError, match="verified evidence and interval"):
        validate_labels(catalogue(row))
    with pytest.raises(ValueError, match="needs a depth interval"):
        validate_labels(catalogue(observed(depth_interval_mm=None)))


@pytest.mark.parametrize("interval", [[1.3, 1.2], [0, float("nan")], [True, 1], [1], "1.2"])
def test_bad_intervals_rejected(interval):
    with pytest.raises(ValueError):
        validate_labels(catalogue(observed(depth_interval_mm=interval)))


@pytest.mark.parametrize("kind", ["group_cutoff", "approximate_prior", "unknown"])
def test_priors_and_cutoffs_cannot_enter_calibration(kind):
    row = observed(kind=kind, split="calibration")
    row["evidence"]["verified"] = False
    with pytest.raises(ValueError, match="must remain unused"):
        validate_labels(catalogue(row))


def test_trial_and_depth_reference_are_never_inferred():
    assert compare_event(predicted(trial_id="trial_a1"), observed())["reason"] == "trial_id_mismatch"
    assert compare_event(predicted(depth_reference="video_motion_onset"), observed())["reason"] == "depth_reference_mismatch"


def test_overlap_and_signed_gap_preserve_intervals_without_midpoints():
    result = compare_event(predicted(), observed())
    assert result["comparable"] and result["intervals_overlap"]
    assert result["signed_interval_gap_mm"] == 0
    assert result["event_mapping"]["explicit"]
    assert result["physical_onset_validated"] is False
    assert "midpoint_error_mm" not in result
    assert compare_event(predicted(depth_interval_mm=[1.4, 1.41]), observed())["signed_interval_gap_mm"] == pytest.approx(.1)
    assert compare_event(predicted(depth_interval_mm=[1.0, 1.1]), observed())["signed_interval_gap_mm"] == pytest.approx(-.1)


def test_visible_opening_is_not_physical_initiation():
    row = observed(event_kind="visible_opening")
    assert compare_event(predicted(), row)["reason"] == "event_definition_mismatch"
    event = predicted(event_kind="visible_opening")
    event.pop("target_event_kind")
    assert compare_event(event, row)["comparison_kind"] == "visible_opening"


def test_force_proxy_is_kept_separate_and_cannot_be_remapped_to_fracture():
    row = observed(kind="proxy_force_change", event_kind="force_trend_change")
    row["evidence"] = dict(source="force.csv", method="specified hinge regression", verified=False)
    event = predicted(event_kind="force_trend_change")
    assert compare_event(event, observed())["reason"] == "unsupported_event_mapping"
    event.pop("target_event_kind")
    result = compare_event(event, row)
    assert result["comparison_kind"] == "proxy_force_change"
    assert result["physical_onset_validated"] is False
    row["calibration_eligible"] = True
    with pytest.raises(ValueError, match="Only a verified physical-onset"):
        validate_labels(catalogue(row))


@pytest.mark.parametrize("status", ["parameters_missing", "numerical_censored", "not_reached_in_qualified_window", "already_exceeded_at_first_state"])
def test_no_event_and_censored_history_never_become_zero_error(status):
    result = compare_event(predicted(status=status, depth_interval_mm=None), observed())
    assert not result["comparable"]
    assert result["prediction_status"] == status
    assert "signed_interval_gap_mm" not in result


def test_explicit_event_mapping_required():
    event = predicted()
    event.pop("target_event_kind")
    assert compare_event(event, observed())["reason"] == "event_definition_mismatch"


@pytest.mark.parametrize("raw, canonical", [
    ("already_above_at_first_qualified_state", "already_exceeded_at_first_state"),
    ("not_triggered_in_qualified_range", "not_reached_in_qualified_window"),
    ("censored_by_qualification", "numerical_censored"),
    ("missing_threshold", "parameters_missing"),
])
def test_saved_state_onset_status_aliases_preserve_censoring(raw, canonical):
    interval = [None, 0.0] if raw.startswith("already_") else None
    result = compare_event(predicted(status=raw, depth_interval_mm=interval), observed())
    assert result["prediction_status"] == canonical
    assert result["prediction_status_raw"] == raw
    assert result["prediction_depth_interval_mm"] == interval
    assert result["reason"] == "prediction_not_bracketed_or_censored"
    assert not result["comparable"]


def test_same_trial_cannot_leak_across_calibration_and_validation():
    data = catalogue(observed(split="calibration"))
    data["labels"].append(observed(id="second_label", split="validation"))
    with pytest.raises(ValueError, match="both calibration and validation"):
        validate_labels(data)


def test_duplicate_ids_rejected_and_input_never_changed(tmp_path):
    data = catalogue(observed())
    original = copy.deepcopy(data)
    result = validate_labels(data)
    result["labels"][0]["evidence"]["source"] = "different"
    assert data == original
    path = tmp_path / "derived.json"
    text = json.dumps(data)
    path.write_text(text, encoding="utf-8")
    assert load_labels(path) == data
    assert path.read_text(encoding="utf-8") == text
    data["labels"].append(copy.deepcopy(data["labels"][0]))
    with pytest.raises(ValueError, match="Duplicate observation"):
        validate_labels(data)
