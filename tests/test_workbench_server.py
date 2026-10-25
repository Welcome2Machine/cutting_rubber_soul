"""Loopback workbench HTTP/configuration regressions; never launch FE loading."""
import copy
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest

from workbench import server as app
from workbench.config import catalog


@pytest.fixture
def service(tmp_path, monkeypatch):
    static = tmp_path / "static"
    static.mkdir()
    for filename, content in {"index.html": "<!doctype html><title>fixture</title>",
                              "app.js": "'use strict';", "app.css": "body{color:black}"}.items():
        (static / filename).write_text(content, encoding="utf8")
    monkeypatch.setattr(app, "STATIC", static)
    jobs = app.Jobs(tmp_path)
    httpd = app.make_server(0, jobs=jobs)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    yield httpd, jobs
    jobs.close()
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)
    assert not thread.is_alive()


def request(httpd, path, *, method="GET", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=3)
    try:
        payload = json.dumps(body).encode("utf8") if body is not None else None
        connection.request(method, path, body=payload, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        content = json.loads(data) if "application/json" in response.getheader("Content-Type", "") else data
        return response.status, content, dict(response.getheaders())
    finally:
        connection.close()


def session_headers(httpd):
    code, data, _ = request(httpd, "/api/catalog")
    assert code == 200
    return {"X-Workbench-Token": data["session_token"],
            "Origin": f"http://127.0.0.1:{httpd.server_port}", "Content-Type": "application/json"}


def test_loopback_catalog_and_validation_are_read_only(service):
    httpd, jobs = service
    assert httpd.server_address[0] == "127.0.0.1"
    code, data, headers = request(httpd, "/api/catalog")
    assert code == 200 and len(data["session_token"]) >= 32
    assert headers["Cache-Control"] == "no-store"
    code, result, _ = request(httpd, "/api/validate", method="POST",
        body={"config": data["defaults"]}, headers=session_headers(httpd))
    assert code == 200
    assert result["config"]["simulation"]["speed_mm_s"] == .2
    assert result["config"]["simulation"]["depth_mm"] == 2
    assert result["summary"]["force_tolerance_N"] == 1e-4
    assert jobs.active is None
    assert not list((jobs.base / "runs").glob("*"))


def test_missing_token_cross_origin_and_untrusted_host_are_rejected(service):
    httpd, _ = service
    body = {"config": catalog()["defaults"]}
    assert request(httpd, "/api/validate", method="POST", body=body)[0] == 403
    headers = session_headers(httpd)
    headers["Origin"] = "https://untrusted.example"
    assert request(httpd, "/api/validate", method="POST", body=body, headers=headers)[0] == 403
    headers = session_headers(httpd)
    headers["X-Workbench-Token"] = "wrong"
    assert request(httpd, "/api/runs", method="POST", body=body, headers=headers)[0] == 403
    assert request(httpd, "/api/catalog", headers={"Host": "untrusted.example"})[0] == 403


def test_invalid_config_never_spawns_or_creates_job(service, monkeypatch):
    httpd, jobs = service
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid workbench configuration attempted to spawn a process")
    monkeypatch.setattr(app.subprocess, "Popen", forbidden)
    config = catalog()["defaults"]
    config["simulation"]["depth_mm"] = 3
    status, error, _ = request(httpd, "/api/runs", method="POST", body={"config": config},
                               headers=session_headers(httpd))
    assert status == 400 and error["error"]
    assert jobs.active is None
    assert not list((jobs.base / "runs").glob("*"))
    assert request(httpd, "/api/validate", method="POST", body=[], headers=session_headers(httpd))[0] == 400


def test_static_allowlist_and_confined_result_files(service):
    httpd, jobs = service
    (jobs.root / "private.json").write_text('{"private":true}')
    (jobs.outputs / "script.py").write_text("not public")
    (jobs.outputs / "plain.csv").write_text("0123456789", encoding="utf8")
    assert request(httpd, "/")[0] == 200
    assert request(httpd, "/app.js")[0] == 200
    assert request(httpd, "/server.py")[0] == 404
    assert request(httpd, "/.git/config")[0] == 404
    assert request(httpd, "/files/script.py")[0] == 403
    assert request(httpd, "/files/%2e%2e/private.json")[0] == 400
    assert request(httpd, "/files/%2e%2e%5cprivate.json")[0] in (400, 404)
    code, data, headers = request(httpd, "/files/plain.csv", headers={"Range": "bytes=3-7"})
    assert code == 206 and data == b"34567"
    assert headers["Content-Range"] == "bytes 3-7/10"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert request(httpd, "/files/plain.csv", headers={"Range": "bytes=40-"})[0] == 416


@contextmanager
def directory_link(link, target):
    """Use a real filesystem link; Windows junction fallback needs no elevation."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        if os.name != "nt":
            pytest.skip("Filesystem does not support test directory symlinks")
        quoted = lambda value: "'" + str(value).replace("'", "''") + "'"
        command = ("$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "
                   + quoted(link) + " -Target " + quoted(target) + " | Out-Null")
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                capture_output=True, timeout=5)
        if result.returncode:
            pytest.skip("Cannot create local junction fixture")
    try:
        yield
    finally:
        # Remove only the directory link, never recurse through its target.
        if os.name == "nt":
            os.rmdir(link)
        else:
            link.unlink()


def test_junction_or_symlink_cannot_escape_output_root(service):
    httpd, jobs = service
    outside = jobs.root / "outside"
    outside.mkdir()
    (outside / "private.json").write_text('{"private":true}')
    link = jobs.outputs / "linked"
    with directory_link(link, outside):
        with pytest.raises(ValueError, match="目录"):
            app.confined(link / "private.json", jobs.outputs)
        assert request(httpd, "/files/linked/private.json")[0] == 400
    assert (outside / "private.json").is_file()


def test_result_listing_and_source_scope(service):
    httpd, jobs = service
    case = jobs.outputs / "saved_case"
    case.mkdir()
    (case / "result.json").write_text(json.dumps(dict(status="paused", steps=[{"depth_mm": .25}],
        signature={"config": {"inclination_deg": 33}})), encoding="utf8")
    empty = jobs.outputs / "empty"
    empty.mkdir()
    (empty / "result.json").write_text('{"steps":[]}')
    code, data, _ = request(httpd, "/api/results")
    assert code == 200 and len(data["results"]) == 1
    assert data["results"][0]["path"] == "outputs/saved_case"
    assert data["results"][0]["depth_mm"] == .25
    assert jobs.source("outputs/saved_case") == case.resolve()
    with pytest.raises(ValueError):
        jobs.source("../unrelated")
    assert request(httpd, "/api/runs/missing")[0] == 404


def test_launch_records_normalized_request_and_enforces_one_job(tmp_path, monkeypatch):
    jobs = app.Jobs(tmp_path)
    targets = []
    class RecordedThread:
        def __init__(self, *, target, args=(), daemon=False):
            targets.append((target, args))
        def start(self):
            pass  # Explicitly do not execute solver worker.
    monkeypatch.setattr(app.threading, "Thread", RecordedThread)
    monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="fixture-git-head"))
    config = catalog()["defaults"]
    config["simulation"]["depth_mm"] = .5
    original = copy.deepcopy(config)
    job = jobs.launch(config)
    assert job["status"] == "prepared" and not job["qualified"]
    assert job["config"] == original and config == original
    folder = jobs.base / "runs" / job["id"]
    assert json.loads((folder / "config.json").read_text(encoding="utf8")) == original
    assert job["git_head"] == "fixture-git-head"
    assert len(targets) == 1 and targets[0][0] == jobs._run
    with pytest.raises(ValueError, match="已有"):
        jobs.launch(config)
    assert jobs.stop(job["id"])["status"] == "stop_requested"
    assert jobs.stop_event.is_set()
    jobs.active = None
    jobs.close()


def test_restart_marks_stale_record_without_resuming_or_killing(tmp_path, monkeypatch):
    folder = tmp_path / "outputs/workbench/runs/stale"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps({"id": "stale", "status": "running", "pid": 12345}))
    def forbidden(*args, **kwargs):
        pytest.fail("Opening old job records must not spawn or kill a solver")
    monkeypatch.setattr(app.subprocess, "Popen", forbidden)
    jobs = app.Jobs(tmp_path)
    assert jobs.get("stale")["status"] == "interrupted"
    assert jobs.active is None and jobs.process is None


def test_git_metadata_is_optional_for_source_archive(tmp_path, monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError('git')
    monkeypatch.setattr(app.subprocess, 'run', missing_git)
    assert app.git_revision(tmp_path) is None
