"""Compare an intact IPC parameter study using reported accepted states only.

This postprocessor never launches a solve, interpolates observations, or silently
mixes incompatible settings. Missing cases and failed prefixes remain visible.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


DEPTH_TOL = 1e-10
PHYSICS_SOURCES = (
    "mechanics/finite_strain.py", "prototype/ipc_reference.py",
    "prototype/material.py", "prototype/model_config.py", "prototype/initiation.py",
)
COMMON_KEYS = (
    "case", "target_mm", "mesh_settings", "simplifications", "dhat_mm",
    "loading_increment_mm", "barrier_stiffness_mpa", "force_tolerance_n",
    "ipc_version", "ipc_binary_sha256", "hessian_mode",
)
FIELDS = (
    "name", "source", "source_result_sha256", "status", "reported_status", "included", "exclusion_reason",
    "angle_deg", "friction_coefficient", "depth_mm", "reaction_fz_n", "residual_n",
    "min_j", "minimum_distance_mm", "friction_work_n_mm", "contact_status",
    "reference_offset_mm", "gap_mm", "contact_source",
)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _queue_overlay(plan):
    """Read one queue snapshot, without inspecting or polling any process."""
    if not plan.get("queue_state"):
        return {}, None
    path = Path(plan["queue_state"]).resolve()
    provenance = dict(path=str(path), status="not_created")
    if not path.exists():
        return {}, provenance
    try:
        data = path.read_bytes()
        provenance["sha256"] = _hash(data)
        state = json.loads(data)
        result = {}
        for group in ("jobs", "observed_runs"):
            for run in state.get(group, []):
                key = os.path.normcase(str(Path(run["output"]).resolve()))
                result[key] = dict(source_group=group, **{field: run[field] for field in
                    ("status", "pid", "exit_code", "mechanics_completed", "error", "last_checked", "finished")
                    if field in run})
        provenance["status"] = "loaded"
        return result, provenance
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        provenance.update(status="unreadable", error=f"{type(exc).__name__}: {exc}")
        return {}, provenance


def _apply_queue_status(item, observation):
    item.setdefault("reported_status", item["status"])
    item["queue_observation"] = observation
    status = str(observation.get("status") or "")
    terminal_failure = (status in ("failed", "timed_out", "cancelled", "killed", "interrupted")
                        or status.startswith("external_"))
    item["queue_terminal_failure"] = terminal_failure
    if terminal_failure or (status == "running" and item["status"] == "pending"):
        item["status"] = status
    # queue status=completed never promotes a missing, partial or invalid report.


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _same(left, right):
    if _finite(left) and _finite(right):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            label = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(f"{label}: missing from {'reference' if key not in left else 'case'}")
            else:
                result.extend(_differences(left[key], right[key], label))
        return result
    return [] if _same(left, right) else [f"{prefix}: reference={left!r}, case={right!r}"]


def _load_control(report):
    control = report.get("load_control")
    if control is None:
        return {"mode": "fixed"}
    if not isinstance(control, dict) or control.get("mode") not in ("fixed", "adaptive_bisection"):
        raise ValueError("invalid load-control mode")
    if control["mode"] == "fixed":
        return {"mode": "fixed"}
    increment = report.get("loading_increment_mm")
    if not _finite(increment) or increment <= 0:
        raise ValueError("invalid loading increment")
    minimum, maximum, output = (control.get(key) for key in
        ("minimum_increment_mm", "maximum_increment_mm", "output_grid_increment_mm"))
    window, fraction = control.get("stagnation_window"), control.get("stagnation_improvement_fraction")
    if (not all(_finite(value) for value in (minimum, maximum, output))
            or not 0 < minimum <= maximum
            or abs(maximum-increment) > DEPTH_TOL or abs(output-increment) > DEPTH_TOL
            or control.get("retry_predictor") != "affine"
            or not isinstance(window, int) or isinstance(window, bool) or window < 1
            or not _finite(fraction) or not 0 < fraction < 1):
        raise ValueError("invalid adaptive load-control settings")
    return dict(control)


def _output_grid(report):
    increment, target = report["loading_increment_mm"], report["target_mm"]
    if not all(_finite(value) and value > 0 for value in (increment, target)):
        raise ValueError("invalid target or loading increment")
    return [min(index*increment, target) for index in range(math.ceil(target/increment-1e-12)+1)]


def _signature(report):
    """Hash only code on the intact mechanical path, not queue/plot helpers."""
    missing = [key for key in COMMON_KEYS if key not in report]
    for key in ("config", "source_sha256", "simplifications", "mesh_settings"):
        if not isinstance(report.get(key), dict):
            missing.append(key)
    sources = report.get("source_sha256") or {}
    missing += [f"source_sha256.{key}" for key in PHYSICS_SOURCES if not sources.get(key)]
    if missing:
        raise ValueError("missing comparison metadata: " + ", ".join(missing))
    result = {key: report[key] for key in COMMON_KEYS}
    # The host description records the expected mu=0 Newton / mu>0 PTC choice;
    # that descriptive string is not a changed material/contact discretization.
    result["simplifications"] = {key: value for key, value in report["simplifications"].items()
                                 if key != "host"}
    result["config"] = {key: value for key, value in report["config"].items()
                        if key not in ("inclination_deg", "friction_coefficient")}
    result["config"].setdefault("ogden_alpha", 2.)
    alpha = result["config"]["ogden_alpha"]
    if not _finite(alpha) or alpha <= 0:
        raise ValueError("invalid Ogden exponent")
    result["contact_formulation"] = report.get("contact_formulation", "improved-max")
    if result["contact_formulation"] not in ("improved-max", "offset-geometric"):
        raise ValueError("invalid contact formulation")
    result["source_sha256"] = {key: sources[key] for key in PHYSICS_SOURCES}
    result["predictor"] = report.get("predictor", "affine")
    result["linear_solver"] = report.get("linear_solver", "direct")
    if (result["predictor"] not in ("affine", "secant")
            or result["linear_solver"] not in ("direct", "reused-lu")):
        raise ValueError("invalid predictor or linear solver")
    result["predictor_settings"] = report.get("predictor_settings")
    if result["predictor"] == "secant" and not isinstance(result["predictor_settings"], dict):
        raise ValueError("secant predictor settings are missing")
    result["load_control"] = _load_control(report)
    json.dumps(result, allow_nan=False)
    return result


def _prefix(report):
    """Retain only the contiguous reported-accepted prefix; do not skip bad rows."""
    increment, target, tolerance = (report[key] for key in
                                    ("loading_increment_mm", "target_mm", "force_tolerance_n"))
    if (not all(_finite(value) and value > 0 for value in (increment, target, tolerance))
            or tolerance > 1e-4):
        raise ValueError("invalid target, increment or force tolerance")
    adaptive = _load_control(report)["mode"] == "adaptive_bisection"
    output_grid = _output_grid(report)
    steps = report.get("steps")
    if not isinstance(steps, list):
        raise ValueError("steps is not a list")
    accepted, warnings, next_output = [], [], 0
    for index, step in enumerate(steps):
        if (not isinstance(step, dict)
                or not all(_finite(step.get(key)) for key in
                           ("depth_mm", "reaction_fz_n", "residual_n", "min_j",
                            "minimum_distance_mm", "friction_work_n_mm"))
                or not -DEPTH_TOL <= step["depth_mm"] <= target+DEPTH_TOL
                or (not accepted and abs(step["depth_mm"]) > DEPTH_TOL)
                or (accepted and not DEPTH_TOL < step["depth_mm"]-accepted[-1]["depth_mm"] <= increment+DEPTH_TOL)
                or next_output >= len(output_grid)
                or step["depth_mm"] > output_grid[next_output]+DEPTH_TOL
                or (not adaptive and abs(step["depth_mm"]-output_grid[next_output]) > DEPTH_TOL)
                or step["residual_n"] < 0 or step["residual_n"] > tolerance
                or step["min_j"] <= 0 or step["minimum_distance_mm"] <= 0
                or step["friction_work_n_mm"] < -1e-12):
            warnings.append(f"invalid accepted history at step {index}; this and all later rows omitted")
            break
        accepted.append(step)
        if abs(step["depth_mm"]-output_grid[next_output]) <= DEPTH_TOL:
            next_output += 1
    return accepted, warnings


def _trace(case, report, digest, warnings):
    if not case.get("trace"):
        return {}
    path = Path(case["trace"]) / "contact_trace.json"
    try:
        data = path.read_bytes()
        trace = json.loads(data)
        if trace.get("status") != "completed_postprocessing" or trace.get("source_result_sha256") != digest:
            raise ValueError("trace status or source_result_sha256 mismatch")
        steps = trace["steps"]
        source_steps = report["steps"]
        if len(steps) != len(source_steps) or any(
                not _finite(item.get("depth_mm"))
                or abs(item["depth_mm"] - source["depth_mm"]) > 1e-10
                for item, source in zip(steps, source_steps)):
            raise ValueError("trace depths do not match source report")
        return dict(steps=steps, path=str(path.resolve()), sha256=_hash(data))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        warnings.append(f"offline contact trace ignored: {exc}")
        return {}


def _contact(step, fallback, depth, dhat):
    source = "online" if isinstance(step.get("tip_contact"), dict) else "offline"
    item = step.get("tip_contact") if source == "online" else fallback
    if not isinstance(item, dict):
        return dict(contact_status="not_recorded", contact_source="", reference_offset_mm=None, gap_mm=None)
    status = item.get("status", "invalid_observation")
    valid_depth = _finite(item.get("depth_mm")) and abs(item["depth_mm"] - depth) <= 1e-10
    gap = item.get("gap_mm")
    gap = gap if _finite(gap) and gap >= 0 else None
    offset = item.get("reference_offset_mm")
    if not valid_depth or status not in ("unique", "no_tip_contact", "ambiguous", "error", "observer_error"):
        status = "invalid_observation"
    if status == "unique" and (not _finite(offset) or gap is None or gap >= dhat):
        status = "invalid_observation"
    return dict(contact_status=status, contact_source=source,
                reference_offset_mm=offset if status == "unique" else None, gap_mm=gap)


def _exact_comparisons(cases, rows):
    """Compare actual shared states; all unpaired substeps stay in the long table."""
    pairs = []
    for index, left in enumerate(cases):
        if not left["included"]:
            continue
        for right in cases[index+1:]:
            if not right["included"]:
                continue
            sweeps = []
            if _same(left["angle_deg"], right["angle_deg"]):
                sweeps.append("friction")
            if _same(left["friction_coefficient"], right["friction_coefficient"]):
                sweeps.append("angle")
            if not sweeps:
                continue
            a = [row for row in rows if row["name"] == left["name"] and _finite(row.get("depth_mm"))]
            b = [row for row in rows if row["name"] == right["name"] and _finite(row.get("depth_mm"))]
            ia, ib, points, unmatched_a, unmatched_b = 0, 0, [], [], []
            while ia < len(a) and ib < len(b):
                x, y = a[ia], b[ib]
                if abs(x["depth_mm"]-y["depth_mm"]) <= DEPTH_TOL:
                    both_unique = x["contact_status"] == y["contact_status"] == "unique"
                    points.append(dict(depth_mm=x["depth_mm"], right_depth_mm=y["depth_mm"],
                        left_fz_n=x["reaction_fz_n"], right_fz_n=y["reaction_fz_n"],
                        fz_difference_n=y["reaction_fz_n"]-x["reaction_fz_n"],
                        left_contact_status=x["contact_status"], right_contact_status=y["contact_status"],
                        left_reference_offset_mm=x["reference_offset_mm"],
                        right_reference_offset_mm=y["reference_offset_mm"],
                        reference_offset_difference_mm=(y["reference_offset_mm"]-x["reference_offset_mm"]
                                                        if both_unique else None)))
                    ia += 1; ib += 1
                elif x["depth_mm"] < y["depth_mm"]:
                    unmatched_a.append(x["depth_mm"]); ia += 1
                else:
                    unmatched_b.append(y["depth_mm"]); ib += 1
            unmatched_a += [row["depth_mm"] for row in a[ia:]]
            unmatched_b += [row["depth_mm"] for row in b[ib:]]
            pairs.append(dict(left=left["name"], right=right["name"], sweeps=sweeps,
                              common_depths_mm=[point["depth_mm"] for point in points],
                              unmatched_left_depths_mm=unmatched_a, unmatched_right_depths_mm=unmatched_b,
                              points=points))
    return dict(method="exact common accepted depth; no interpolation/extrapolation",
                depth_roundoff_mm=DEPTH_TOL, difference_direction="right minus left", pairs=pairs)


def collect_study(plan):
    """Return a provenance/status summary and a CSV-ready long table."""
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("plan.cases must be a nonempty list")
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("case names must be unique")
    queue, queue_provenance = _queue_overlay(plan)
    reference = _signature(plan["expected"]) if plan.get("expected") else None
    reference_name = "plan.expected" if reference else None
    summaries, rows = [], []
    for case in cases:
        path = Path(case["output"]).resolve() / "result.json"
        item = dict(name=case["name"], source=str(path), angle_deg=case["angle"],
                    friction_coefficient=case["friction"], status="pending", included=False,
                    exclusion_reasons=[], warnings=[], accepted_states=0, last_depth_mm=None,
                    numerical_complete=False, contact_observations_complete=False, source_result_sha256=None)
        summaries.append(item)
        base = {key: item.get(key) for key in FIELDS if key in item}
        if not path.exists():
            rows.append(dict(base, contact_status="not_recorded"))
            continue
        try:
            data = path.read_bytes()
            report = json.loads(data)
            if not isinstance(report, dict):
                raise ValueError("result report must be a JSON object")
            item.update(source_result_sha256=_hash(data), status=report.get("status", "unknown"),
                        reported_status=report.get("status", "unknown"))
            signature = _signature(report)
            if reference is None:
                reference, reference_name = signature, case["name"]
            item["exclusion_reasons"].extend(_differences(reference, signature))
            if report["case"] != "oblique" or report["simplifications"].get("fracture") is not False:
                item["exclusion_reasons"].append("not an intact oblique IPC model")
            for key, expected in (("inclination_deg", case["angle"]), ("friction_coefficient", case["friction"])):
                if not _same(report["config"].get(key), expected):
                    item["exclusion_reasons"].append(f"plan/config mismatch: {key}")
            steps, warnings = _prefix(report)
            item["warnings"].extend(warnings)
            item["reported_states"] = len(report["steps"])
            item["accepted_states"] = len(steps)
            item["last_depth_mm"] = steps[-1]["depth_mm"] if steps else None
            output_grid = _output_grid(report)
            missing_outputs = [depth for depth in output_grid
                               if not any(abs(step["depth_mm"]-depth) <= DEPTH_TOL for step in steps)]
            item["loading_grid"] = dict(mode=signature["load_control"]["mode"],
                required_output_depths_mm=output_grid, missing_output_depths_mm=missing_outputs,
                accepted_substep_depths_mm=[step["depth_mm"] for step in steps
                    if not any(abs(step["depth_mm"]-depth) <= DEPTH_TOL for depth in output_grid)])
            item["numerical_complete"] = (report.get("status") == "completed"
                                          and report.get("completed") is True and bool(steps)
                                          and abs(steps[-1]["depth_mm"] - report["target_mm"]) <= DEPTH_TOL
                                          and not missing_outputs and not warnings
                                          and len(steps) == len(report["steps"])
                                          and ("last_depth_mm" not in report or
                                               (_finite(report["last_depth_mm"]) and
                                                abs(report["last_depth_mm"]-report["target_mm"]) <= DEPTH_TOL)))
            if (report.get("completed") or report.get("status") == "completed") and not item["numerical_complete"]:
                item["status"] = "inconsistent_completion"
            if warnings:
                item["status"] = "invalid_history"
            item["included"] = not item["exclusion_reasons"]
            trace = _trace(case, report, item["source_result_sha256"], item["warnings"])
            item["trace"] = {key: trace[key] for key in ("path", "sha256") if key in trace}
            item["physics_signature"] = signature
            item["solver_metadata"] = {key: report.get(key) for key in
                                       ("friction_solver", "friction_solver_settings", "line_search",
                                        "predictor", "predictor_settings", "linear_solver", "load_control")}
            item["solver_metadata"]["host"] = report["simplifications"].get("host")
            base = {key: item.get(key) for key in FIELDS if key in item}
            base["exclusion_reason"] = "; ".join(item["exclusion_reasons"])
            current = []
            for index, step in enumerate(steps):
                fallback = trace.get("steps", [])[index].get("tip_contact") if trace else None
                row = dict(base, **{key: step[key] for key in ("depth_mm", "reaction_fz_n", "residual_n", "min_j",
                                                             "minimum_distance_mm", "friction_work_n_mm")})
                row.update(_contact(step, fallback, step["depth_mm"], report["dhat_mm"]))
                current.append(row)
            item["contact_status_counts"] = dict(Counter(row["contact_status"] for row in current))
            item["contact_observations_complete"] = bool(current) and all(
                row["contact_status"] in ("unique", "no_tip_contact", "ambiguous") for row in current)
            rows.extend(current or [dict(base, contact_status="not_recorded")])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            item["status"] = "unreadable_or_invalid"
            item["exclusion_reasons"].append(f"{type(exc).__name__}: {exc}")
            base = {key: item.get(key) for key in FIELDS if key in item}
            rows.append(dict(base, status=item["status"], included=False,
                             exclusion_reason="; ".join(item["exclusion_reasons"]), contact_status="not_recorded"))
    for case, item in zip(cases, summaries):
        key = os.path.normcase(str(Path(case["output"]).resolve()))
        _apply_queue_status(item, queue.get(key, {}))
        for row in rows:
            if row["name"] == item["name"]:
                row.update(status=item["status"], reported_status=item["reported_status"])
    mechanics_complete = all(item["numerical_complete"] and item["included"] for item in summaries)
    observations_complete = all(item["contact_observations_complete"] and item["included"] for item in summaries)
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(), study=plan.get("study", {}),
                   complete=mechanics_complete and observations_complete
                            and not any(item["queue_terminal_failure"] for item in summaries),
                   all_mechanics_completed=mechanics_complete,
                   contact_observations_complete=observations_complete,
                   queue_state=queue_provenance,
                   reference_case=reference_name, cases=summaries,
                   exact_depth_comparisons=_exact_comparisons(summaries, rows),
                   semantics=dict(reference_offset_mm="signed original-surface material coordinate minus initial drop_s; not accumulated tangential slip",
                                  reaction_fz_n="reported blade vertical reaction in N", depth_mm="accepted solver indentation depth in mm"),
                   limitations="Reported accepted diagnostics checked; checkpoint mechanics not re-solved. No interpolation, smoothing, or extrapolation. No fracture.")
    return summary, rows


def _curve_rows(rows, name, field):
    selected = [row for row in rows if row["name"] == name and row.get("included") and _finite(row.get("depth_mm"))]
    return ([row["depth_mm"] for row in selected],
            [row[field] if _finite(row.get(field)) else float("nan") for row in selected])


def _plot(summary, rows, output, field, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    study = summary["study"]
    if sweep == "friction":
        angle = study.get("friction_angle", 60)
        cases = sorted((item for item in summary["cases"] if _same(item["angle_deg"], angle)),
                       key=lambda item: item["friction_coefficient"])
        title = f"{angle:g} deg: friction coefficient sensitivity"
    else:
        mu = study.get("angle_friction", .35)
        cases = sorted((item for item in summary["cases"] if _same(item["friction_coefficient"], mu)),
                       key=lambda item: item["angle_deg"])
        title = f"Angle comparison at friction coefficient {mu:g}"
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#333333", "#946B00")
    markers = ("o", "s", "^", "D", "v", "P")
    styles = ("-", "--", "-.", ":", "-")
    notes = []
    with plt.rc_context({"font.size": 10, "axes.grid": True, "grid.alpha": .25, "savefig.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.subplots_adjust(left=.10, right=.97, bottom=.25, top=.90)
        for index, case in enumerate(cases):
            label = (f"mu={case['friction_coefficient']:g}" if sweep == "friction" else f"{case['angle_deg']:g} deg")
            end = f"{case['last_depth_mm']:g} mm" if case["last_depth_mm"] is not None else "no states"
            state = case["status"] + ("; excluded" if case["exclusion_reasons"] else "")
            x, y = _curve_rows(rows, case["name"], field)
            if x and any(math.isfinite(value) for value in y):
                color = colors[index % len(colors)]
                ax.plot(x, y, color=color, linestyle=styles[index % len(styles)],
                        marker=markers[index % len(markers)], markersize=3, markevery=max(1, len(x)//12),
                        linewidth=1.5, label=f"{label} | {state}, {end}")
                if math.isfinite(y[-1]):
                    ax.plot(x[-1], y[-1], marker=markers[index % len(markers)], markersize=7,
                            color=color, markerfacecolor=color if case["numerical_complete"] else "white")
            else:
                notes.append(f"{label}: {state}, {end}" + ("; no defined tip contact" if x else ""))
        if ax.lines:
            ax.legend(loc="best", fontsize=9)
        else:
            ax.text(.5, .5, "No comparable accepted observations yet", transform=ax.transAxes, ha="center")
        ax.set_title(title)
        ax.set_xlabel("Indentation depth (mm)")
        ax.set_ylabel("Reference material contact offset $\\Delta s_0$ (mm)" if field == "reference_offset_mm" else "Vertical reaction $F_z$ (N)")
        ax.axhline(0, color=".6", linewidth=.7)
        ax.set_xlim(left=0)
        caption = "Intact model | accepted states only | raw curves, no smoothing or extrapolation"
        if field == "reference_offset_mm":
            caption += "\nGaps: no/ambiguous/unrecorded tip contact. Offset is not accumulated slip."
        caption += "\nOpen endpoint: incomplete case; filled endpoint: reported target reached."
        if notes:
            caption += "\n" + " ; ".join(notes[:3])
            if len(notes) > 3:
                caption += "\n" + " ; ".join(notes[3:])
        fig.text(.10, .025, caption, fontsize=8.5, va="bottom")
        fig.savefig(output, dpi=160, metadata={"Description": caption})
        plt.close(fig)


def render_study(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    sweeps = plan.get("study", {}).get("sweeps", ["friction", "angle"])
    if (not isinstance(sweeps, list) or not sweeps or len(set(sweeps)) != len(sweeps)
            or any(value not in ("friction", "angle") for value in sweeps)):
        raise ValueError("study.sweeps must be a nonempty unique list of friction and/or angle")
    summary, rows = collect_study(plan)
    output.mkdir(parents=True, exist_ok=True)
    table = output / "comparison.csv"
    temporary = table.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(table)
    figures = [(name, field, sweep) for name, field, sweep in
               (("tip_offset_by_friction.png", "reference_offset_mm", "friction"),
                ("fz_by_friction.png", "reaction_fz_n", "friction"),
                ("fz_by_angle.png", "reaction_fz_n", "angle")) if sweep in sweeps]
    for name, field, sweep in figures:
        _plot(summary, rows, output / name, field, sweep)
    summary.update(plan=str(plan_path), plan_sha256=_hash(plan_bytes),
                   postprocessor_sha256=_hash(Path(__file__).read_bytes()),
                   products={path.name: dict(sha256=_hash(path.read_bytes()))
                             for path in [table] + [output / name for name, _, _ in figures]})
    path = output / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render_study(args.plan, args.output)
    print(json.dumps(dict(complete=result["complete"], cases=len(result["cases"]), output=str(args.output))))


if __name__ == "__main__":
    main()
