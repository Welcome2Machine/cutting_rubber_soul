"""Synthetic report integrity checks; these fixtures are not mechanical solves."""

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.compare_ipc_study import PHYSICS_SOURCES, collect_study, render_study, _curve_rows


def report(angle=60, friction=.35, status="completed"):
    result = dict(case="oblique", target_mm=.3, loading_increment_mm=.1,
                  mesh_settings=dict(local_spacing_mm=.05, surface_spacing_mm=.005),
                  simplifications=dict(fracture=False, tip_radius_mm=.02, nose_facets=64,
                                       host="frictional harness"),
                  config=dict(inclination_deg=angle, friction_coefficient=friction,
                              shear_modulus_n_mm2=.66, height_mm=5),
                  source_sha256={key: "fixture-hash" for key in PHYSICS_SOURCES},
                  dhat_mm=.0025, barrier_stiffness_mpa=6.6, force_tolerance_n=1e-4,
                  ipc_version="fixture", ipc_binary_sha256="fixture-binary", hessian_mode="regularized",
                  status=status, completed=status == "completed", steps=[])
    for index in range(4):
        depth = index * .1
        contact = dict(depth_mm=depth, status="unique", reference_offset_mm=index*.01, gap_mm=.001)
        result["steps"].append(dict(depth_mm=depth, reaction_fz_n=index*10,
                                    residual_n=1e-5, min_j=.9, minimum_distance_mm=.001,
                                    friction_work_n_mm=0., tip_contact=contact))
    return result


def adaptive_report(friction=.35, extra_depth=.05, status="completed"):
    result = report(friction=friction, status=status)
    result.update(predictor="secant", predictor_settings={"history": "last two accepted states"},
                  linear_solver="direct", load_control=dict(mode="adaptive_bisection",
                    minimum_increment_mm=.003125, maximum_increment_mm=.1,
                    output_grid_increment_mm=.1, retry_predictor="affine",
                    stagnation_window=40, stagnation_improvement_fraction=.1))
    step = copy.deepcopy(result["steps"][0])
    step.update(depth_mm=extra_depth, reaction_fz_n=100*extra_depth)
    step["tip_contact"].update(depth_mm=extra_depth, reference_offset_mm=extra_depth*.1)
    result["steps"].append(step)
    result["steps"].sort(key=lambda item: item["depth_mm"])
    return result


class IPCComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = dict(study=dict(angle_friction=.35, friction_angle=60), cases=[])

    def tearDown(self):
        self.temp.cleanup()

    def add(self, name, data=None, angle=60, friction=.35):
        folder = self.root / name
        case = dict(name=name, output=str(folder), angle=angle, friction=friction)
        if data is not None:
            folder.mkdir()
            (folder / "result.json").write_text(json.dumps(data), encoding="utf-8")
        self.plan["cases"].append(case)
        return case

    def test_pending_case_is_named_and_does_not_count_as_complete(self):
        self.add("completed", report())
        self.add("pending", angle=0)
        summary, rows = collect_study(self.plan)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["cases"][1]["status"], "pending")
        self.assertTrue(summary["cases"][0]["numerical_complete"])
        self.assertIsNone(rows[-1].get("depth_mm"))
        self.assertEqual(rows[-1]["name"], "pending")

    def test_failed_prefix_and_contact_gaps_are_preserved(self):
        data = report(status="failed")
        data["steps"] = data["steps"][:3]
        data["steps"][1]["tip_contact"].update(status="ambiguous", reference_offset_mm=.7)
        self.add("partial", data)
        summary, rows = collect_study(self.plan)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["cases"][0]["last_depth_mm"], .2)
        self.assertIsNone(rows[1]["reference_offset_mm"])
        depths, offsets = _curve_rows(rows, "partial", "reference_offset_mm")
        self.assertEqual(depths, [0., .1, .2])
        self.assertTrue(math.isnan(offsets[1]))
        self.assertEqual(offsets[0], 0.)

    def test_adaptive_substeps_are_preserved_and_only_exact_shared_depths_compared(self):
        self.add("mu35", adaptive_report())
        other = adaptive_report(friction=.2, extra_depth=.15)
        other["steps"][3]["tip_contact"].update(status="no_tip_contact", reference_offset_mm=None)
        self.add("mu20", other, friction=.2)
        summary, rows = collect_study(self.plan)
        self.assertTrue(summary["complete"])
        self.assertEqual([case["accepted_states"] for case in summary["cases"]], [5, 5])
        self.assertEqual(summary["cases"][0]["loading_grid"]["accepted_substep_depths_mm"], [.05])
        self.assertEqual(_curve_rows(rows, "mu20", "reaction_fz_n")[0], [0., .1, .15, .2, .30000000000000004])
        pair = summary["exact_depth_comparisons"]["pairs"][0]
        self.assertEqual(pair["common_depths_mm"], [0., .1, .2, .30000000000000004])
        self.assertEqual(pair["unmatched_left_depths_mm"], [.05])
        self.assertEqual(pair["unmatched_right_depths_mm"], [.15])
        self.assertEqual(len(pair["points"]), 4)
        self.assertIsNone(pair["points"][2]["reference_offset_difference_mm"])
        self.assertEqual(pair["points"][2]["fz_difference_n"], 0.)

    def test_adaptive_failed_substep_prefix_does_not_claim_completion(self):
        data = adaptive_report(status="failed")
        data["steps"] = data["steps"][:2]
        self.add("partial", data)
        summary, rows = collect_study(self.plan)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["cases"][0]["last_depth_mm"], .05)
        self.assertFalse(summary["cases"][0]["warnings"])
        self.assertEqual(len(rows), 2)

    def test_missing_anchor_and_bad_substep_diagnostics_stop_prefix(self):
        for key, value in (("depth_mm", .15), ("residual_n", .001), ("min_j", 0.),
                           ("minimum_distance_mm", 0.), ("friction_work_n_mm", -1e-5)):
            with self.subTest(key=key):
                self.plan["cases"] = []
                data = adaptive_report(); data["steps"][1][key] = value
                self.add("bad_"+key, data)
                summary, rows = collect_study(self.plan)
                self.assertFalse(summary["cases"][0]["numerical_complete"])
                self.assertEqual(len(rows), 1)
                self.assertEqual(summary["cases"][0]["status"], "invalid_history")
        self.plan["cases"] = []
        data = adaptive_report(); data["steps"][2]["depth_mm"] = .125
        self.add("skipped_anchor", data)
        summary, rows = collect_study(self.plan)
        self.assertEqual([row["depth_mm"] for row in rows], [0., .05])
        self.assertFalse(summary["all_mechanics_completed"])

    def test_required_gap_work_and_reported_terminal_depth_are_checked(self):
        for key in ("minimum_distance_mm", "friction_work_n_mm"):
            self.plan["cases"] = []
            data = report(); data["steps"][1].pop(key)
            self.add("missing_"+key, data)
            summary, rows = collect_study(self.plan)
            self.assertEqual(len(rows), 1)
            self.assertFalse(summary["all_mechanics_completed"])
        self.plan["cases"] = []
        data = report(); data["last_depth_mm"] = .2
        self.add("wrong_terminal", data)
        summary, _ = collect_study(self.plan)
        self.assertEqual(summary["cases"][0]["status"], "inconsistent_completion")

    def test_solver_signature_defaults_and_adaptive_settings_are_enforced(self):
        self.plan["expected"] = report()
        explicit = report(); explicit.update(predictor="affine", linear_solver="direct", load_control={"mode": "fixed"})
        self.add("explicit_defaults", explicit)
        summary, _ = collect_study(self.plan)
        self.assertTrue(summary["complete"])
        self.plan["cases"] = []
        self.plan["expected"] = adaptive_report()
        for key, value in (("linear_solver", "reused-lu"), ("predictor", "affine"),
                           ("load_control", dict(adaptive_report()["load_control"], minimum_increment_mm=.00625))):
            altered = adaptive_report(); altered[key] = value
            self.add("changed_"+key, altered)
        summary, _ = collect_study(self.plan)
        self.assertTrue(all(not item["included"] for item in summary["cases"]))
        self.assertTrue(all(item["numerical_complete"] for item in summary["cases"]))

    def test_malformed_adaptive_control_is_excluded(self):
        for key, value in (("mode", "unknown"), ("minimum_increment_mm", 0.),
                           ("maximum_increment_mm", .2), ("output_grid_increment_mm", .2),
                           ("stagnation_improvement_fraction", float("nan"))):
            with self.subTest(key=key):
                self.plan["cases"] = []
                data = adaptive_report(); data["load_control"][key] = value
                self.add("control_"+key, data)
                summary, _ = collect_study(self.plan)
                self.assertEqual(summary["cases"][0]["status"], "unreadable_or_invalid")
                self.assertFalse(summary["complete"])

    def test_adaptive_frictionless_algorithm_branch_is_expected(self):
        self.plan["expected"] = adaptive_report()
        data = adaptive_report(friction=0.)
        data["simplifications"]["host"] = "frictionless Newton"
        data.update(friction_solver=None, friction_solver_settings=None, line_search="armijo")
        self.add("mu0", data, friction=0.)
        summary, _ = collect_study(self.plan)
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["cases"][0]["included"])

    def test_setting_mismatch_excludes_curve_but_preserves_table_evidence(self):
        self.add("reference", report())
        altered = report(angle=30)
        altered["mesh_settings"]["surface_spacing_mm"] = .1
        self.add("coarse", altered, angle=30)
        summary, rows = collect_study(self.plan)
        self.assertFalse(summary["cases"][1]["included"])
        self.assertIn("surface_spacing_mm", summary["cases"][1]["exclusion_reasons"][0])
        self.assertEqual(_curve_rows(rows, "coarse", "reaction_fz_n"), ([], []))
        self.assertEqual(len([row for row in rows if row["name"] == "coarse"]), 4)

    def test_physics_source_difference_is_not_silently_waived(self):
        self.plan["expected"] = report()
        altered = report()
        altered["source_sha256"]["prototype/ipc_reference.py"] = "different"
        self.add("different", altered)
        summary, _ = collect_study(self.plan)
        self.assertFalse(summary["cases"][0]["included"])
        self.assertIn("ipc_reference.py", summary["cases"][0]["exclusion_reasons"][0])

    def test_material_and_contact_formulation_are_part_of_physics_signature(self):
        self.plan['expected'] = report()
        legacy_equivalent = report()
        legacy_equivalent['config']['ogden_alpha'] = 2.
        legacy_equivalent['contact_formulation'] = 'improved-max'
        self.add('explicit_defaults', legacy_equivalent)
        changed_alpha = report(); changed_alpha['config']['ogden_alpha'] = 1.
        self.add('changed_alpha', changed_alpha)
        changed_contact = report(); changed_contact['contact_formulation'] = 'offset-geometric'
        self.add('changed_contact', changed_contact)
        summary, _ = collect_study(self.plan)
        self.assertEqual([case['included'] for case in summary['cases']], [True, False, False])

    def test_expected_frictionless_host_difference_is_allowed(self):
        self.plan["expected"] = report()
        data = report(friction=0)
        data["simplifications"]["host"] = "frictionless Newton harness"
        data["line_search"] = "energy"
        self.add("frictionless", data, friction=0)
        summary, _ = collect_study(self.plan)
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["cases"][0]["included"])
        self.assertEqual(summary["cases"][0]["solver_metadata"]["line_search"], "energy")

    def test_invalid_acceptance_stops_prefix_instead_of_skipping_bad_row(self):
        data = report()
        data["steps"][2]["residual_n"] = .5
        self.add("bad", data)
        summary, rows = collect_study(self.plan)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["cases"][0]["status"], "invalid_history")
        self.assertFalse(summary["cases"][0]["numerical_complete"])

    def test_contact_error_and_gap_outside_support_do_not_erase_force(self):
        data = report()
        data["steps"][1]["tip_contact"].update(status="error", reference_offset_mm=None)
        data["steps"][2]["tip_contact"]["gap_mm"] = data["dhat_mm"]
        self.add("bad_contact", data)
        summary, rows = collect_study(self.plan)
        self.assertTrue(summary["all_mechanics_completed"])
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["contact_observations_complete"])
        self.assertEqual(rows[1]["contact_status"], "error")
        self.assertEqual(rows[2]["contact_status"], "invalid_observation")
        self.assertIsNone(rows[2]["reference_offset_mm"])
        self.assertEqual(len(_curve_rows(rows, "bad_contact", "reaction_fz_n")[0]), 4)

    def test_missing_accepted_depth_stops_prefix(self):
        data = report()
        del data["steps"][1]
        self.add("gap", data)
        summary, rows = collect_study(self.plan)
        self.assertEqual(len(rows), 1)
        self.assertIn("step 1", summary["cases"][0]["warnings"][0])

    def test_online_observation_precedes_hash_verified_offline_trace(self):
        data = report()
        del data["steps"][1]["tip_contact"]
        case = self.add("source", data)
        trace_dir = self.root / "trace"
        trace_dir.mkdir()
        trace_steps = copy.deepcopy(report()["steps"])
        for step in trace_steps:
            step["tip_contact"]["reference_offset_mm"] = .5
        trace = dict(status="completed_postprocessing", source_result_sha256=hashlib.sha256(
            (Path(case["output"]) / "result.json").read_bytes()).hexdigest(), steps=trace_steps)
        trace_path = trace_dir / "contact_trace.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        case["trace"] = str(trace_dir)
        summary, rows = collect_study(self.plan)
        self.assertEqual(rows[0]["reference_offset_mm"], 0.)
        self.assertEqual(rows[1]["reference_offset_mm"], .5)
        self.assertEqual(rows[1]["contact_source"], "offline")
        self.assertIn("sha256", summary["cases"][0]["trace"])
        trace["source_result_sha256"] = "wrong"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        summary, rows = collect_study(self.plan)
        self.assertEqual(rows[1]["contact_status"], "not_recorded")
        self.assertTrue(summary["cases"][0]["warnings"])

    def test_offline_depth_mismatch_rejected_while_force_remains_available(self):
        data = report()
        case = self.add("source", data)
        trace_dir = self.root / "trace"
        trace_dir.mkdir()
        trace = dict(status="completed_postprocessing", source_result_sha256=hashlib.sha256(
            (Path(case["output"]) / "result.json").read_bytes()).hexdigest(), steps=[])
        (trace_dir / "contact_trace.json").write_text(json.dumps(trace), encoding="utf-8")
        case["trace"] = str(trace_dir)
        summary, rows = collect_study(self.plan)
        self.assertTrue(summary["cases"][0]["included"])
        self.assertEqual(len(_curve_rows(rows, "source", "reaction_fz_n")[0]), 4)
        self.assertIn("depths", summary["cases"][0]["warnings"][0])

    def test_render_records_products_and_blank_missing_values(self):
        data = report()
        data["steps"][1]["tip_contact"].update(status="no_tip_contact", reference_offset_mm=None)
        self.add("synthetic", data)
        self.add("pending", angle=0)
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        result = render_study(plan_path, self.root / "comparison")
        self.assertEqual(len(result["products"]), 4)
        for name, metadata in result["products"].items():
            path = self.root / "comparison" / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["sha256"])
        with (self.root / "comparison/comparison.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[1]["reference_offset_mm"], "")
        self.assertEqual(rows[-1]["depth_mm"], "")
        self.assertEqual(rows[-1]["name"], "pending")

    def test_friction_only_sweep_omits_unrequested_angle_figure(self):
        self.plan["study"]["sweeps"] = ["friction"]
        self.add("mu35", adaptive_report())
        plan_path = self.root/"plan.json"
        plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        summary = render_study(plan_path, self.root/"comparison")
        self.assertEqual(set(summary["products"]), {"comparison.csv", "tip_offset_by_friction.png", "fz_by_friction.png"})
        self.assertFalse((self.root/"comparison/fz_by_angle.png").exists())

    def test_dead_external_status_overrides_stale_running_report(self):
        data = report(status="running")
        data["steps"] = data["steps"][:2]
        case = self.add("external", data)
        queue_path = self.root / "queue.json"
        queue_path.write_text(json.dumps(dict(jobs=[], observed_runs=[dict(
            output=case["output"], status="external_process_missing", pid=123,
            mechanics_completed=False)])), encoding="utf-8")
        self.plan["queue_state"] = str(queue_path)
        summary, rows = collect_study(self.plan)
        self.assertEqual(summary["cases"][0]["status"], "external_process_missing")
        self.assertEqual(summary["cases"][0]["reported_status"], "running")
        self.assertEqual(summary["queue_state"]["sha256"], hashlib.sha256(queue_path.read_bytes()).hexdigest())
        self.assertEqual(rows[0]["status"], "external_process_missing")
        self.assertEqual(len(_curve_rows(rows, "external", "reaction_fz_n")[0]), 2)
        self.assertFalse(summary["complete"])

    def test_native_failure_prefix_and_missing_report_use_queue_outcome(self):
        data = report(status="running")
        data["steps"] = data["steps"][:2]
        case = self.add("native_crash", data)
        missing = self.add("launch_failure", angle=30)
        queue_path = self.root / "queue.json"
        queue_path.write_text(json.dumps(dict(jobs=[
            dict(output=case["output"], status="failed", exit_code=-1),
            dict(output=missing["output"], status="failed", error="launch failure")],
            observed_runs=[])), encoding="utf-8")
        self.plan["queue_state"] = str(queue_path)
        summary, rows = collect_study(self.plan)
        self.assertEqual([item["status"] for item in summary["cases"]], ["failed", "failed"])
        self.assertEqual([item["reported_status"] for item in summary["cases"]], ["running", "pending"])
        self.assertEqual(len(_curve_rows(rows, "native_crash", "reaction_fz_n")[0]), 2)
        self.assertEqual(rows[-1]["status"], "failed")
        self.assertIsNone(rows[-1].get("depth_mm"))

    def test_queue_success_cannot_promote_partial_report_and_missing_queue_is_allowed(self):
        data = report(status="running")
        data["steps"] = data["steps"][:2]
        case = self.add("partial", data)
        queue_path = self.root / "queue.json"
        self.plan["queue_state"] = str(queue_path)
        summary, _ = collect_study(self.plan)
        self.assertEqual(summary["queue_state"]["status"], "not_created")
        queue_path.write_text(json.dumps(dict(jobs=[dict(output=case["output"], status="completed",
                                                       mechanics_completed=True)])), encoding="utf-8")
        summary, _ = collect_study(self.plan)
        self.assertEqual(summary["cases"][0]["status"], "running")
        self.assertFalse(summary["all_mechanics_completed"])


if __name__ == "__main__":
    unittest.main()
