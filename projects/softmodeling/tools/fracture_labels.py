"""Validate derived fracture observations and compare explicitly registered events.

No force change, historical estimate, or analysis cutoff is silently promoted to
physical crack initiation. This module reads JSON only; it does not modify raw
experimental inputs, select a threshold, register clocks, or launch a solver.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


SCHEMA = "softcut-fracture-labels/1"
KINDS = {"verified_observation", "proxy_force_change", "approximate_prior",
         "group_cutoff", "unknown"}
SPLITS = {"calibration", "validation", "unused"}
OBSERVED_EVENTS = {"physical_onset", "visible_opening"}
PREDICTION_STATUSES = {"crossed", "already_exceeded_at_first_state",
                       "not_reached_in_qualified_window", "numerical_censored",
                       "parameters_missing", "insufficient_state_fields"}
STATUS_ALIASES = {
    "already_above_at_first_qualified_state": "already_exceeded_at_first_state",
    "not_triggered_in_qualified_range": "not_reached_in_qualified_window",
    "censored_by_qualification": "numerical_censored",
    "missing_threshold": "parameters_missing",
}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _interval(value, name, *, required=False):
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [lower_mm, upper_mm] or null")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in value):
        raise ValueError(f"{name} must contain two finite numbers")
    low, high = map(float, value)
    if low > high:
        raise ValueError(f"{name} is reversed")
    return [low, high]


def _validate_label(label):
    if not isinstance(label, dict):
        raise ValueError("Each label must be an object")
    for key in ("id", "trial_id", "event_kind", "depth_reference"):
        _text(label.get(key), key)
    kind, split = label.get("kind"), label.get("split")
    if kind not in KINDS:
        raise ValueError(f"Unknown label kind: {kind!r}")
    if split not in SPLITS:
        raise ValueError(f"Unknown label split: {split!r}")
    evidence = label.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be an object")
    _text(evidence.get("source"), "evidence.source")
    _text(evidence.get("method"), "evidence.method")
    if not isinstance(evidence.get("verified"), bool):
        raise ValueError("evidence.verified must be an explicit boolean")
    interval = _interval(label.get("depth_interval_mm"), "depth_interval_mm")
    if kind == "verified_observation":
        if label["event_kind"] not in OBSERVED_EVENTS:
            raise ValueError("verified_observation needs physical_onset or visible_opening")
        if not evidence["verified"] or evidence.get("interval_verified") is not True:
            raise ValueError("A verified observation needs verified evidence and interval")
        if interval is None:
            raise ValueError("A verified observation needs a depth interval")
    elif kind == "proxy_force_change":
        if label["event_kind"] != "force_trend_change":
            raise ValueError("A force proxy must keep event_kind=force_trend_change")
        if interval is None:
            raise ValueError("A force proxy needs its detected depth interval")
    else:
        if split != "unused":
            raise ValueError("Unknowns, approximate priors and group cutoffs must remain unused")
        if evidence["verified"]:
            raise ValueError("A prior/cutoff/unknown cannot claim verified event evidence")
    if kind == "unknown" and interval is not None:
        raise ValueError("An unknown event cannot contain an invented interval")
    if kind == "group_cutoff" and label["event_kind"] != "analysis_cutoff":
        raise ValueError("A group cutoff must keep event_kind=analysis_cutoff")
    if "depth_estimate_mm" in label and label["depth_estimate_mm"] is not None:
        _interval([label["depth_estimate_mm"]] * 2, "depth_estimate_mm")
    if "calibration_eligible" in label:
        expected = kind == "verified_observation" and label["event_kind"] == "physical_onset"
        if not isinstance(label["calibration_eligible"], bool):
            raise ValueError("calibration_eligible must be boolean")
        if label["calibration_eligible"] and not expected:
            raise ValueError("Only a verified physical-onset interval is calibration eligible")
    return label


def validate_labels(document):
    """Return a deep copy of a valid catalogue; raise ValueError otherwise.

    Verification is an explicit provenance declaration, not an automated review
    of source images. Validation cannot establish that the declaration is true.
    Labels marked unused remain ineligible for quantitative evaluation.
    """
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"Expected schema={SCHEMA}")
    labels = document.get("labels")
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    identifiers = set()
    used_splits = {}
    for label in labels:
        _validate_label(label)
        if label["id"] in identifiers:
            raise ValueError(f"Duplicate observation id: {label['id']}")
        identifiers.add(label["id"])
        if label["split"] != "unused":
            previous = used_splits.setdefault(label["trial_id"], label["split"])
            if previous != label["split"]:
                raise ValueError(f"Trial {label['trial_id']} appears in both calibration and validation")
    return copy.deepcopy(document)


def load_labels(path):
    """Read and validate a UTF-8 JSON catalogue without modifying it."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        return validate_labels(json.load(stream))


def compare_event(prediction, label):
    """Compare one model event with one observation, preserving event semantics.

    prediction fields: trial_id, event_kind, status, depth_interval_mm and
    depth_reference. For local_criterion_crossing or damage_onset, the caller may
    explicitly provide target_event_kind='physical_onset'. That mapping is
    reported; it never upgrades an observation's evidence. A force-trend model
    event can only be compared with a force-trend proxy. No trial alias, depth
    offset, midpoint replacement or extrapolation is inferred here.

    All structured non-comparability results retain source identifiers. Invalid
    schemas raise ValueError. Only crossed events have a two-state bracket;
    already-exceeded initial states and truncated histories remain censored.
    """
    _validate_label(label)
    if not isinstance(prediction, dict):
        raise ValueError("prediction must be an object")
    for key in ("trial_id", "event_kind", "depth_reference"):
        _text(prediction.get(key), f"prediction.{key}")
    raw_status = prediction.get("status")
    status = STATUS_ALIASES.get(raw_status, raw_status)
    if status not in PREDICTION_STATUSES:
        raise ValueError(f"Unknown prediction status: {status!r}")
    raw_interval = prediction.get("depth_interval_mm")
    # An initially exceeded criterion has an unknown lower bound, not a zero-
    # width event at the first stored (already preloaded) state.
    if (status == "already_exceeded_at_first_state" and isinstance(raw_interval, (list, tuple))
            and len(raw_interval) == 2 and raw_interval[0] is None):
        upper = _interval([raw_interval[1]] * 2, "prediction.upper_depth_bound_mm")[1]
        interval = [None, upper]
    else:
        interval = _interval(raw_interval, "prediction.depth_interval_mm", required=status == "crossed")
    observed = label.get("depth_interval_mm")
    result = dict(comparable=False, comparison_kind=None, reason=None,
                  prediction_trial_id=prediction["trial_id"], label_id=label["id"],
                  observation_trial_id=label["trial_id"], label_kind=label["kind"],
                  split=label["split"], prediction_status=status, prediction_status_raw=raw_status,
                  prediction_event_kind=prediction["event_kind"],
                  observation_event_kind=label["event_kind"],
                  prediction_depth_interval_mm=interval,
                  observation_depth_interval_mm=copy.deepcopy(observed))
    def reject(reason):
        result["reason"] = reason
        return result
    if prediction["trial_id"] != label["trial_id"]:
        return reject("trial_id_mismatch")
    if label["split"] == "unused":
        return reject("label_marked_unused")
    if label["kind"] not in {"verified_observation", "proxy_force_change"}:
        return reject("no_verified_observation_or_defined_proxy")
    if prediction["depth_reference"] != label["depth_reference"]:
        return reject("depth_reference_mismatch")
    target = prediction.get("target_event_kind", prediction["event_kind"])
    _text(target, "prediction.target_event_kind")
    mapped = target != prediction["event_kind"]
    if mapped and not (prediction["event_kind"] in {"local_criterion_crossing", "damage_onset"}
                       and target == "physical_onset"):
        return reject("unsupported_event_mapping")
    result["event_mapping"] = dict(model=prediction["event_kind"], target=target,
                                    explicit=mapped)
    if target != label["event_kind"]:
        return reject("event_definition_mismatch")
    if status != "crossed":
        return reject("prediction_not_bracketed_or_censored")
    low, high = interval
    obs_low, obs_high = _interval(observed, "label.depth_interval_mm", required=True)
    gap = low - obs_high if low > obs_high else high - obs_low if high < obs_low else 0.0
    result.update(comparable=True, reason=None,
                  comparison_kind=("proxy_force_change" if label["kind"] == "proxy_force_change"
                                   else label["event_kind"]),
                  intervals_overlap=bool(low <= obs_high and high >= obs_low),
                  signed_interval_gap_mm=gap,
                  physical_onset_validated=False,
                  interpretation=("Interval agreement alone does not validate a fracture model; "
                                  "positive gap means the prediction is later/deeper."))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Derived fracture-label catalogue JSON")
    args = parser.parse_args(argv)
    document = load_labels(args.path)
    summary = dict(schema=document["schema"], labels=len(document["labels"]),
                   physical_onset_labels=sum(x["kind"] == "verified_observation"
                                             and x["event_kind"] == "physical_onset"
                                             and x["split"] != "unused"
                                             for x in document["labels"]),
                   proxy_labels=sum(x["kind"] == "proxy_force_change"
                                    and x["split"] != "unused" for x in document["labels"]),
                   scope="Schema validation only; source evidence is not automatically certified")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
