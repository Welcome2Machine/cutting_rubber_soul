"""Export and query material coordinates of accepted intact IPC tip contact.

No native contact library or new equilibrium solve is needed for offline export.
The query returns accepted depths (or brackets), never inventing a contact point.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_support import write_json
from .tip_contact import TipContactTracker


def write_trace_csv(path, steps):
    """Atomically rewrite the small accepted-state table; blank means undefined."""
    path = Path(path)
    fields = ["depth_mm", "status", "reference_offset_mm", "absolute_reference_offset_mm",
              "reference_s_mm", "gap_mm", "edge_alpha", "edge_nodes", "reference_point_mm",
              "current_point_mm", "knife_tip_mm", "nose_status", "nose_offset_intervals_mm"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            item = dict(step.get("tip_contact") or {}, depth_mm=step["depth_mm"])
            row = {key: item.get(key) for key in fields}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, separators=(",", ":"), allow_nan=False)
            writer.writerow(row)
    temporary.replace(path)


def query_depth(steps, depth):
    depths = np.asarray([step["depth_mm"] for step in steps], dtype=float)
    if (not np.isfinite(depth) or len(depths) == 0 or not np.isfinite(depths).all()
            or np.any(np.diff(depths) <= 0)):
        raise ValueError("Depths must be finite and strictly increasing")
    exact = np.flatnonzero(np.isclose(depths, depth, atol=1e-10, rtol=0))
    if len(exact):
        indices, kind = [int(exact[0])], "exact_accepted_depth"
    elif depth < depths[0] or depth > depths[-1]:
        raise ValueError(f"Depth outside accepted range [{depths[0]}, {depths[-1]}] mm")
    else:
        upper = int(np.searchsorted(depths, depth))
        indices, kind = [upper - 1, upper], "bracketing_accepted_depths"
    return dict(requested_depth_mm=float(depth), kind=kind, interpolated=False,
                samples=[dict(depth_mm=steps[i]["depth_mm"],
                              tip_contact=steps[i].get("tip_contact")) for i in indices])


def validate_loading_schedule(report, *, require_complete=False):
    """Validate recorded accepted depths, including adaptive substeps.

    A partial adaptive history may end between output anchors. Every anchor
    already crossed must be a saved accepted state; nothing is interpolated.
    The report's original target is used even when a caller plans an extension.
    """
    tolerance = 1e-10
    depths = np.asarray([step["depth_mm"] for step in report["steps"]], dtype=float)
    increment = float(report.get("loading_increment_mm", report.get("settings", {}).get("increment", np.nan)))
    target = float(report["target_mm"])
    if (not np.isfinite(increment) or increment <= 0 or not np.isfinite(target) or target <= 0
            or len(depths) == 0 or not np.isfinite(depths).all()
            or abs(depths[0]) > tolerance or np.any(np.diff(depths) <= 0)
            or depths[-1] > target + tolerance):
        raise ValueError("Invalid accepted loading schedule")
    if (require_complete or report.get("completed") is True) and abs(depths[-1] - target) > tolerance:
        raise ValueError("History does not span zero to target")
    expected = np.r_[0., np.arange(increment, target - 1e-12, increment), target]
    control = report.get("load_control")
    if control is None:
        if len(depths) > len(expected) or not np.allclose(depths, expected[:len(depths)],
                                                         atol=tolerance, rtol=0):
            raise ValueError("Accepted depths do not match the fixed loading schedule")
    else:
        if not isinstance(control, dict) or control.get("mode") != "adaptive_bisection":
            raise ValueError("Unknown loading control")
        minimum = float(control["minimum_increment_mm"])
        maximum = float(control["maximum_increment_mm"])
        output_grid = float(control["output_grid_increment_mm"])
        if (not np.isfinite([minimum, maximum, output_grid]).all() or not 0 < minimum <= maximum
                or abs(maximum - increment) > tolerance or abs(output_grid - increment) > tolerance):
            raise ValueError("Invalid adaptive loading control")
        if np.any(np.diff(depths) > maximum + tolerance):
            raise ValueError("Accepted substep exceeds the maximum increment")
        for anchor in expected[expected <= depths[-1] + tolerance]:
            if not np.isclose(depths, anchor, atol=tolerance, rtol=0).any():
                raise ValueError(f"Missing accepted output-grid depth {anchor:g} mm")
        # Do not impose minimum on accepted intervals: a target/anchor remainder
        # can be smaller. The solver controls how many failed trials are retried.
    return depths


def export_trace(source, output):
    """Postprocess a completed intact history, preserving evidence and provenance.

    Checks saved acceptance diagnostics, topology, displacement history and J.
    This is not an independent force-equilibrium recomputation or validation.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    report_path = source / "result.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    signature = report.get("signature", {})
    maxwell = signature.get("material", {}).get("viscous_flow") == "isochoric_maxwell"
    intact = (maxwell or (report.get("case") == "oblique"
                         and report.get("simplifications", {}).get("fracture") is False))
    if (report.get("status") != "completed" or report.get("completed") is not True or not intact):
        raise ValueError("Offline export requires a completed intact oblique history")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Export must be separate from the original run")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export directory must be empty")
    config = signature.get("config", report.get("config"))
    dhat = signature.get("dhat", report.get("dhat_mm"))
    nose_facets = report.get("simplifications", {}).get("nose_facets", 64 if maxwell else None)
    if config is None or dhat is None or nose_facets is None:
        raise ValueError("Missing intact geometry metadata")
    steps = report["steps"]
    validate_loading_schedule(report, require_complete=True)
    tracker, prior, reference, triangles = None, None, None, None
    hashes, observed = {}, []
    for i, step in enumerate(steps):
        if (not np.isfinite(step["residual_n"]) or not np.isfinite(step["min_j"])
                or step["residual_n"] > report["force_tolerance_n"] or step["min_j"] <= 0):
            raise ValueError(f"State {i} has unaccepted reported diagnostics")
        relative = Path("states") / f"{i:04d}.npz"
        state_path = source / relative
        hashes[relative.as_posix()] = hashlib.sha256(state_path.read_bytes()).hexdigest()
        with np.load(state_path, allow_pickle=False) as state:
            x, previous = state["positions"], state["previous_positions"]
            n = int(state["body_nodes"])
            if tracker is None:
                reference, triangles = state["reference_xy"].copy(), state["triangles"].copy()
                tracker = TipContactTracker(reference, triangles, n, config, dhat, nose_facets)
                body_nodes = n
                prior = reference
                zero_knife = x[n:].copy()
                start = -2 * dhat / max(np.cos(np.deg2rad(config["inclination_deg"])), .1)
                np.testing.assert_allclose(zero_knife, reference[n:] + [0., start], atol=1e-9, rtol=0)
                dm = np.stack((reference[triangles[:, 1]] - reference[triangles[:, 0]],
                               reference[triangles[:, 2]] - reference[triangles[:, 0]]), axis=2)
                reference_det = np.linalg.det(dm)
                if np.any(reference_det == 0):
                    raise ValueError("Degenerate reference triangle")
            np.testing.assert_array_equal(state["reference_xy"], reference)
            np.testing.assert_array_equal(state["triangles"], triangles)
            np.testing.assert_array_equal(previous, prior)
            if n != body_nodes or x.shape != reference.shape or not np.isfinite(x).all():
                raise ValueError(f"State {i} has invalid geometry")
            np.testing.assert_allclose(float(state["depth_mm"]), step["depth_mm"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(x[n:], zero_knife - [0., step["depth_mm"]], atol=1e-9, rtol=0)
            ds = np.stack((x[triangles[:, 1]] - x[triangles[:, 0]],
                           x[triangles[:, 2]] - x[triangles[:, 0]]), axis=2)
            minimum_j = float(np.min(np.linalg.det(ds) / reference_det))
            if minimum_j <= 0:
                raise ValueError(f"State {i} contains an inverted body element")
            np.testing.assert_allclose(minimum_j, step["min_j"], atol=1e-9, rtol=1e-7)
            observed.append(dict(depth_mm=step["depth_mm"], tip_contact=tracker.observe(x, step["depth_mm"])))
            prior = x.copy()
    output.mkdir(parents=True, exist_ok=True)
    result = dict(status="completed_postprocessing", mechanical_solve_performed=False,
                  source_directory=str(source), source_result_sha256=hashlib.sha256(report_bytes).hexdigest(),
                  source_checkpoints_sha256=hashes, source_solver_sha256=report.get("source_sha256"),
                  postprocessor_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (Path(__file__), Path(__file__).with_name("tip_contact.py"),
                                                  Path(__file__).resolve().parents[1] / "tools/visualize_ipc.py")},
                  checks="saved acceptance diagnostics, topology, slip history, knife motion, positive J; no force re-solve",
                  tip_contact_tracking=tracker.metadata, steps=observed)
    write_trace_csv(output / "contact_trace.csv", observed)
    result["visualization"] = dict(status="running")
    write_json(output / "contact_trace.json", result)
    try:
        from tools.visualize_ipc import plot_tip_contact
        plot_tip_contact(observed, output / "tip_contact_offset.png")
        result["visualization"]["status"] = "completed"
    except Exception as exc:
        result["visualization"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_json(output / "contact_trace.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    query = sub.add_parser("query")
    query.add_argument("--source", type=Path, required=True)
    query.add_argument("--depth", type=float, required=True)
    args = parser.parse_args()
    if args.action == "export":
        result = export_trace(args.source, args.output)
        print(json.dumps(dict(status=result["status"], accepted_states=len(result["steps"]), output=str(args.output))))
    else:
        path = args.source / "contact_trace.json"
        if not path.exists():
            path = args.source / "result.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("tip_contact_tracking"):
            parser.error("Contact was not tracked; export this run first")
        print(json.dumps(query_depth(report["steps"], args.depth), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
