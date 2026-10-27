"""Fracture job boundaries: no implicit solve, copied inputs, explicit budgets."""
from pathlib import Path
import sys

import pytest

from softcut_solution import api
from softcut_solution.__main__ import main


@pytest.fixture
def fracture_workspace(tmp_path):
    for project in api.projects(tmp_path).values():
        project.mkdir(parents=True)
    api.write_json(tmp_path / "settings.local.json", {
        "simulation_python": sys.executable, "video_python": sys.executable,
    })
    sim = api.projects(tmp_path)["simulation"]
    for name in ("angle 0", "angle33"):
        case = sim / "outputs" / name
        (case / "states").mkdir(parents=True)
        api.write_json(case / "result.json", {"name": name})
        api.write_json(case / "audit.json", {"qualified": True})
    api.write_json(tmp_path / "config with spaces.json", {"calibration": "uncalibrated_demo"})
    return tmp_path


@pytest.fixture
def job_calls(monkeypatch):
    calls = []
    def run(root, kind, args, **kwargs):
        calls.append({"root": root, "kind": kind, "args": args, **kwargs})
        output = Path(args[args.index("--output") + 1])
        assert not output.exists()
        output.mkdir()
        (output / "index.html").write_text("<title>Fixture report</title>", encoding="utf-8")
        return {"status": "completed"}
    monkeypatch.setattr(api, "_run_job", run)
    return calls


def test_onset_uses_multiple_saved_cases_and_leaves_threshold_unset(fracture_workspace, job_calls):
    root = fracture_workspace
    cases = sorted((api.projects(root)["simulation"] / "outputs").iterdir())
    result = api.fracture_onset(root, cases, length_mm=.2, seconds=240)
    call = job_calls[0]
    args = call["args"]
    assert args[:2] == ["-m", "tools.fracture_onset"]
    assert args.count("--case") == 2
    assert "--threshold-mpa" not in args
    assert args[args.index("--length-mm") + 1] == .2
    assert args[args.index("--seconds") + 1] == 240
    assert call["kind"] == "simulation" and call["timeout"] == 300
    assert call["inputs"]["postprocessing_only"] is True
    assert call["inputs"]["threshold_mpa"] is None
    assert call["inputs"]["cases"][0]["result_sha256"] == api.digest(cases[0] / "result.json")
    assert Path(result["report_path"]).is_relative_to(root / "runs")
    assert api.read_json(Path(result["run_dir"]) / "artifact.json")["report_path"] == result["report_path"]
    assert all(sorted(p.name for p in case.iterdir()) == ["audit.json", "result.json", "states"] for case in cases)


def test_fracture_calculation_uses_workspace_config_and_fresh_result(fracture_workspace, job_calls):
    root = fracture_workspace
    config = root / "config with spaces.json"
    before = config.read_bytes()
    result = api.fracture_run(root, config.name, seconds=301)
    call = job_calls[0]
    assert call["args"][:4] == ["-m", "tools.run_cohesive_cut", "--config", config]
    assert call["timeout"] == 361
    assert call["inputs"]["explicit_execution"] is True
    assert call["inputs"]["config_sha256"] == api.digest(config)
    assert config.read_bytes() == before
    assert Path(result["output_dir"]) == Path(result["run_dir"]) / "result"
    assert "experimental calibration" in result["qualification_note"]


@pytest.mark.parametrize("seconds", [True, 0, 3601, 1.5, "300", float("inf")])
def test_invalid_budget_starts_no_job(fracture_workspace, job_calls, seconds):
    with pytest.raises(ValueError, match="seconds"):
        api.fracture_run(fracture_workspace, "config with spaces.json", seconds=seconds)
    assert not job_calls
    assert not (fracture_workspace / "runs").exists()


@pytest.mark.parametrize("kwargs", [
    {"length_mm": 0}, {"length_mm": float("nan")}, {"length_mm": True},
    {"threshold_mpa": -1}, {"threshold_mpa": float("inf")},
])
def test_invalid_indicator_parameters_start_no_job(fracture_workspace, job_calls, kwargs):
    case = api.projects(fracture_workspace)["simulation"] / "outputs/angle33"
    with pytest.raises(ValueError, match="positive finite"):
        api.fracture_onset(fracture_workspace, [case], **kwargs)
    assert not job_calls
    assert not (fracture_workspace / "runs").exists()


def test_outside_paths_and_missing_states_are_rejected_before_job(fracture_workspace, job_calls):
    root = fracture_workspace
    with pytest.raises(ValueError, match="inside"):
        api.fracture_run(root, root.parent / "original.json")
    with pytest.raises(ValueError, match="inside"):
        api.fracture_onset(root, [root / "original/case"])
    case = api.projects(root)["simulation"] / "outputs/angle33"
    (case / "states").rmdir()
    with pytest.raises(ValueError, match="saved states"):
        api.fracture_onset(root, [case])
    assert not job_calls
    assert not (root / "runs").exists()


def test_success_without_report_is_not_presented_as_complete(fracture_workspace, monkeypatch):
    monkeypatch.setattr(api, "_run_job", lambda *a, **kw: {"status": "completed"})
    with pytest.raises(RuntimeError, match="no report"):
        api.fracture_run(fracture_workspace, "config with spaces.json")


def test_inventory_discovers_nested_fracture_report_without_starting_solver(fracture_workspace, monkeypatch):
    report = fracture_workspace / "runs/fracture_run/result/index.html"
    report.parent.mkdir(parents=True)
    report.write_text("report", encoding="utf-8")
    monkeypatch.setattr(api, "_run_job", lambda *a, **kw: pytest.fail("inventory started a job"))
    result = api.inventory(fracture_workspace)
    assert str(report) in [item["path"] for item in result["reports"]]


def test_audited_cohesive_history_is_not_offered_as_an_intact_case(fracture_workspace):
    sim = api.projects(fracture_workspace)["simulation"]
    coupled = sim / "outputs/cohesive_case"
    (coupled / "states").mkdir(parents=True)
    api.write_json(coupled / "result.json", {"schema": "softcut-cohesive-result/1", "completed": True})
    api.write_json(coupled / "audit.json", {"qualified": True})
    paths = [item["path"] for item in api.inventory(fracture_workspace)["simulation_cases"]]
    assert str(coupled) not in paths
    assert str(sim / "outputs/angle33") in paths


def test_cli_preserves_repeated_cases_and_explicit_threshold(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(api, "fracture_onset", lambda root, cases, **kw:
                        calls.append((root, cases, kw)) or {"status": "fixture"})
    assert main(["fracture-onset", "--case", "a path", "--case", "another", "--length-mm", ".1",
                 "--threshold-mpa", ".4", "--seconds", "90"]) == 0
    assert calls[0][1] == [Path("a path"), Path("another")]
    assert calls[0][2] == {"length_mm": .1, "threshold_mpa": .4, "seconds": 90}
    assert "fixture" in capsys.readouterr().out


def test_cli_calculation_requires_config_and_carries_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "fracture_run", lambda root, config, **kw:
                        calls.append((config, kw)) or {"status": "fixture"})
    assert main(["fracture-run", "--config", "demo.json", "--seconds", "60"]) == 0
    assert calls == [(Path("demo.json"), {"seconds": 60})]


def test_desktop_fracture_callbacks_capture_values_before_worker(fracture_workspace):
    from types import SimpleNamespace
    from softcut_solution.gui import Workbench
    calls, captured = [], []
    class Backend:
        def fracture_run(self, root, config, **kwargs):
            calls.append(("run", config, kwargs))
            return {}
        def fracture_onset(self, root, cases, **kwargs):
            calls.append(("onset", cases, kwargs))
            return {}
    # Exercise event handlers without creating another Tcl interpreter.
    app = Workbench.__new__(Workbench)
    app.workspace_root = fracture_workspace
    app.backend = Backend()
    case = api.projects(fracture_workspace)["simulation"] / "outputs/angle33"
    app.asset_choices = {"simulation_cases": {"angle33": str(case)}}
    values = {"config": str(fracture_workspace / "config with spaces.json"),
              "seconds": "123", "case": "angle33", "length": ".15", "threshold": ""}
    for attribute, key in (("fracture_config", "config"), ("fracture_seconds", "seconds"),
                           ("fracture_case_choice", "case"), ("fracture_length", "length"),
                           ("fracture_threshold", "threshold")):
        setattr(app, attribute, SimpleNamespace(get=lambda k=key: values[k]))
    app._task = lambda label, function, on_success=None: captured.append(function)
    app._run_fracture()
    app._diagnose_fracture()
    assert calls == []
    assert len(captured) == 2
    values["seconds"] = "999"
    values["threshold"] = "99"
    for function in captured:
        function()
    assert calls[0][0] == "run" and calls[0][2]["seconds"] == 123
    assert calls[1][0] == "onset" and calls[1][2]["threshold_mpa"] is None
