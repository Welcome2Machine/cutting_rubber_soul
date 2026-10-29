"""Orchestration contracts: copied paths, immutable inputs and auditable jobs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from softcut_solution import api


@pytest.fixture
def workspace(tmp_path):
    for project in api.projects(tmp_path).values():
        project.mkdir(parents=True)
    api.write_json(tmp_path / "settings.local.json", {
        "simulation_python": sys.executable, "video_python": sys.executable,
        "simulation_port": 8876, "video_timeout_s": 3600,
    })
    return tmp_path


def manifest_fixture(root, cases):
    path = root / "historical manifest.json"
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    return path


def test_relocated_manifest_never_rewrites_historical_file_or_uses_its_outputs(workspace):
    original = {
        "name": "1014n33a1", "contact_analysis_stop_mm": 0.998,
        "tracks_root": "D:/original/video/results/tracks",
        "measurements_root": "D:/original/video/results/measurements",
        "review_root": "D:/original/video/results/review",
        "video": "D:/original/video/data/1014n33a1.mp4",
        "source": "E:/acquisition/1014n33a1.avi",
        "stroke_status": "user_confirmed", "microscopy_offset_um": 123,
    }
    manifest = manifest_fixture(workspace, [original])
    before = manifest.read_bytes()
    run = api._new_run(workspace, "test")
    target, output = api.prepare_video_manifest(workspace, manifest, run)
    result = api.read_json(target)
    case = result["cases"][0]
    assert manifest.read_bytes() == before
    assert result["integration_source_sha256"] == api.digest(manifest)
    assert target == run / "video_manifest.json"
    assert output.is_relative_to(api.projects(workspace)["video"] / "results/integration")
    for field, child in (("tracks_root", "tracks"), ("measurements_root", "measurements"), ("review_root", "review")):
        assert Path(case[field]) == output / child
        assert "original" not in case[field]
    assert "video" not in case
    assert case["source"] == original["source"]  # provenance, not a batch import command
    assert case["contact_analysis_stop_mm"] == 0.998
    assert case["microscopy_offset_um"] == 123
    assert not output.exists()  # preparing a recipe starts no processing


@pytest.mark.parametrize("names", [
    ["1014n33a1", "1014n33a1"], ["../outside"], ["a/b"], ["a\\b"],
    ["D:drive"], [""], ["case.mp4"],
])
def test_unsafe_or_duplicate_manifest_cases_rejected_without_partial_recipe(workspace, names):
    path = manifest_fixture(workspace, [{"name": name} for name in names])
    before = path.read_bytes()
    run = api._new_run(workspace, "test")
    with pytest.raises(ValueError, match="unique|safe"):
        api.prepare_video_manifest(workspace, path, run)
    assert path.read_bytes() == before
    assert not (run / "video_manifest.json").exists()


@pytest.mark.parametrize("cases", [[], None, {}, "1014n33a1"])
def test_manifest_requires_nonempty_case_list(workspace, cases):
    path = manifest_fixture(workspace, cases)
    with pytest.raises(ValueError, match="non-empty"):
        api.prepare_video_manifest(workspace, path, api._new_run(workspace, "test"))


def test_path_boundary_resolves_parent_segments_and_rejects_sibling_prefix(workspace):
    parent = workspace / "outputs"
    assert api.within(parent / "case/../other", parent) == parent / "other"
    for outside in (parent / "../escape", workspace / "outputs_backup/case"):
        with pytest.raises(ValueError, match="inside"):
            api.within(outside, parent)


def test_fe_export_reads_input_and_targets_fresh_copied_output(workspace, monkeypatch):
    sim = api.projects(workspace)["simulation"]
    case = sim / "outputs/accepted case"
    case.mkdir(parents=True)
    (case / "result.json").write_text('{"accepted":true}', encoding="utf-8")
    (case / "audit.json").write_text('{"qualification":"accepted"}', encoding="utf-8")
    before = {file.name: file.read_bytes() for file in case.iterdir()}
    called = {}
    def run(root, kind, args, **kwargs):
        called.update(root=root, kind=kind, args=args, **kwargs)
        return {"status": "completed"}
    monkeypatch.setattr(api, "_run_job", run)
    result = api.export_fe(workspace, case, frames=7)
    assert called["kind"] == "simulation"
    assert called["args"][called["args"].index("--export") + 1] == case
    target = Path(called["args"][called["args"].index("--output") + 1])
    assert target.is_relative_to(sim / "outputs/integration")
    assert target != case
    assert result["bundle_path"] == str(target / "result.bundle.json")
    assert called["inputs"] == {"result_sha256": api.digest(case / "result.json"),
                                "audit_sha256": api.digest(case / "audit.json")}
    assert before == {file.name: file.read_bytes() for file in case.iterdir()}


def test_fe_export_rejects_outside_case_before_starting_job(workspace, monkeypatch):
    outside = workspace / "outside"
    outside.mkdir()
    for name in ("result.json", "audit.json"):
        (outside / name).write_text("{}")
    monkeypatch.setattr(api, "_run_job", lambda *a, **kw: pytest.fail("Started outside export"))
    with pytest.raises(ValueError, match="inside"):
        api.export_fe(workspace, outside)
    assert not (workspace / "runs").exists()


@pytest.mark.parametrize("frames", [True, 1, 202, 2.5, "21"])
def test_fe_export_rejects_invalid_frame_requests_before_run(workspace, frames):
    case = api.projects(workspace)["simulation"] / "outputs/case"
    case.mkdir(parents=True)
    for name in ("result.json", "audit.json"):
        (case / name).write_text("{}")
    with pytest.raises(ValueError, match="frames"):
        api.export_fe(workspace, case, frames)
    assert not (workspace / "runs").exists()


@pytest.mark.parametrize("patch", [
    {"simulation_port": True}, {"simulation_port": 1023}, {"simulation_port": 65536},
    {"simulation_port": 1234.5}, {"simulation_port": None}, {"simulation_port": []},
    {"simulation_port": float("inf")}, {"video_timeout_s": 0},
    {"video_timeout_s": 86401}, {"video_timeout_s": float("nan")},
    {"video_python": "missing-python.exe"}, {"unknown_option": 42},
])
def test_invalid_settings_rejected_without_corrupting_saved_values(workspace, patch):
    path = workspace / "settings.local.json"
    before = path.read_bytes()
    with pytest.raises(ValueError):
        api.save_settings(workspace, patch)
    assert path.read_bytes() == before


def test_valid_settings_update_preserves_interpreters_and_converts_integer_text(workspace):
    updated = api.save_settings(workspace, {"simulation_port": "8877", "video_timeout_s": "240"})
    assert updated["simulation_port"] == 8877
    assert updated["video_timeout_s"] == 240
    assert updated["video_python"] == os.path.abspath(sys.executable)
    assert api.read_json(workspace / "settings.local.json") == updated


@pytest.mark.skipif(os.name == "nt", reason="POSIX venv symlink regression")
def test_job_preserves_selected_venv_instead_of_resolving_system_python(workspace):
    import venv
    environment = workspace / "isolated environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    executable = environment / "bin/python"
    assert executable.is_symlink()
    api.save_settings(workspace, {"simulation_python": str(executable)})
    job = api._run_job(workspace, "simulation", ["-c", "import sys; print(sys.prefix)"])
    assert job["command"][0] == str(executable)
    assert job["output"].strip() == str(environment)


def test_batch_preflight_uses_relocated_recipe_and_never_requests_execution(workspace, monkeypatch):
    path = manifest_fixture(workspace, [{"name": "1014n33a1", "tracks_root": "D:/old"}])
    called = {}
    def run(root, kind, args, **kwargs):
        called.update(kind=kind, args=args, **kwargs)
        return {"status": "completed"}
    monkeypatch.setattr(api, "_run_job", run)
    api.video_batch(workspace, path)
    assert "--check-only" in called["args"]
    assert called["inputs"]["check_only"] is True
    generated = Path(called["args"][called["args"].index("--manifest") + 1])
    assert generated != path
    assert "D:/old" not in generated.read_text(encoding="utf-8")


@pytest.mark.parametrize("case", ["../x", "x/y", "x\\y", "x;do", ""])
def test_video_preflight_rejects_unsafe_case_before_spawn(workspace, monkeypatch, case):
    monkeypatch.setattr(api, "_run_job", lambda *a, **kw: pytest.fail("Unsafe spawn"))
    with pytest.raises(ValueError):
        api.video_preflight(workspace, case)


@pytest.mark.parametrize("returncode", [0, 7])
def test_job_records_completion_or_failure_and_isolates_import_paths(workspace, monkeypatch, returncode):
    monkeypatch.setenv("PYTHONPATH", "D:/original/project")
    started = {}
    class Process:
        pid = 4321
        def __init__(self, command, **kwargs):
            started.update(command=command, **kwargs)
            kwargs["stdout"].write("fixture output\n")
        def wait(self, timeout):
            return returncode
    monkeypatch.setattr(api.subprocess, "Popen", Process)
    run = api._new_run(workspace, "job_test")
    if returncode:
        with pytest.raises(RuntimeError, match="Exit 7"):
            api._run_job(workspace, "video", ["softcut.py", "check"], run=run)
    else:
        assert api._run_job(workspace, "video", ["softcut.py", "check"], run=run)["status"] == "completed"
    record = api.read_json(run / "job.json")
    assert record["status"] == ("failed" if returncode else "completed")
    assert record["returncode"] == returncode
    assert record["finished_utc"]
    assert started["cwd"] == api.projects(workspace)["video"]
    assert "PYTHONPATH" not in started["env"]
    assert started["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "-B" in started["command"]
    assert not any("shell" == key and value for key, value in started.items())


def test_job_timeout_records_failure_and_stops_child(workspace, monkeypatch):
    stopped, processes = [], []
    class Process:
        pid = 5432
        calls = 0
        def __init__(self, *args, **kwargs):
            processes.append(self)
        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return -9
        def kill(self):
            stopped.append("kill")
    monkeypatch.setattr(api.subprocess, "Popen", Process)
    monkeypatch.setattr(api.subprocess, "run", lambda command, **kwargs: stopped.append(command))
    run = api._new_run(workspace, "timeout_test")
    with pytest.raises(TimeoutError, match="exceeded"):
        api._run_job(workspace, "video", ["softcut.py", "check"], timeout=1, run=run)
    assert stopped
    record = api.read_json(run / "job.json")
    assert record["status"] == "failed"
    assert "exceeded" in record["error"]
    assert record["finished_utc"]
