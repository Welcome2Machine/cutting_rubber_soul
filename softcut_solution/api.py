"""Local orchestration. Subprojects run in separate Python processes/environments."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
SETTING_KEYS = {"simulation_python", "video_python", "simulation_port", "video_timeout_s"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def settings(root=ROOT):
    root = Path(root).resolve()
    value = {"simulation_python": sys.executable,
             "video_python": str(root / "projects/rec_inc_soft_cut/.venv" /
                                 ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
             "simulation_port": 8876, "video_timeout_s": 3600}
    local = root / "settings.local.json"
    if local.is_file():
        saved = read_json(local)
        if not isinstance(saved, dict) or set(saved) - SETTING_KEYS:
            raise ValueError("settings.local.json contains unknown or invalid settings")
        value.update(saved)
    return _validate_settings(value)


def _validate_settings(value, require_files=False):
    current = dict(value)
    for key in ("simulation_python", "video_python"):
        if not isinstance(current[key], str) or not current[key].strip():
            raise ValueError(f"{key} must be a non-empty interpreter path")
        # On POSIX the venv executable is normally a symlink to system Python.
        # Resolving it changes sys.prefix and silently drops the chosen packages.
        # Normalize the spelling, but execute the configured venv entry point.
        p = Path(os.path.abspath(Path(current[key]).expanduser()))
        if require_files and not p.is_file():
            raise ValueError(f"Python interpreter does not exist: {p}")
        current[key] = str(p)
    for key, low, high in (("simulation_port", 1024, 65535), ("video_timeout_s", 1, 86400)):
        item = current[key]
        try:
            valid = not isinstance(item, bool) and math.isfinite(float(item)) and int(item) == float(item)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"{key} must be a finite integer")
        current[key] = int(item)
        if not low <= current[key] <= high:
            raise ValueError(f"{key} must be in [{low}, {high}]")
    return current


def save_settings(root, value):
    current = settings(root)
    if not isinstance(value, dict):
        raise ValueError("settings must be an object")
    unknown = set(value) - SETTING_KEYS
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    current.update(value)
    current = _validate_settings(current, require_files=True)
    write_json(Path(root) / "settings.local.json", current)
    return current


def projects(root=ROOT):
    root = Path(root).resolve()
    return {"simulation": root / "projects/softmodeling", "video": root / "projects/rec_inc_soft_cut"}


def within(path, parent):
    p, base = Path(path).resolve(), Path(parent).resolve()
    if not p.is_relative_to(base):
        raise ValueError(f"Path must be inside {base}: {p}")
    return p


def inventory(root=ROOT):
    from .video import discover_video_packages
    root = Path(root).resolve()
    p = projects(root)
    sim = p["simulation"]
    bundles = sorted((sim / "outputs").rglob("result.bundle.json"))
    cases = []
    for result in (sim / "outputs").rglob("result.json"):
        if not (result.parent / "audit.json").is_file() or not (result.parent / "states").is_dir():
            continue
        try:
            metadata = read_json(result)
        except (OSError, ValueError):
            continue
        if not isinstance(metadata, dict) or "softcut-cohesive-result/1" in (
                metadata.get("schema"), metadata.get("schema_version")):
            continue
        cases.append(result.parent)
    reports = sorted((root / "runs").rglob("index.html"), reverse=True)
    known = [sim / "outputs/video_fe_1mm_20261024/index.html",
             p["video"] / "results/20261024_n_group_first_1mm/index.html",
             p["video"] / "results/20261025_surface_identity/validated/index.html"]
    reports += [f for f in known if f.is_file()]
    def entries(paths):
        return [{"label": str(f.relative_to(root)), "path": str(f)} for f in paths]
    return {"simulation_bundles": entries(bundles), "simulation_cases": entries(sorted(cases)),
            "video_packages": discover_video_packages(p["video"]), "reports": entries(reports),
            "projects": {k: str(v) for k, v in p.items()}, "settings": settings(root)}


def inspect(kind, path):
    if kind == "simulation":
        from .comparison import inspect_fe
        return inspect_fe(Path(path))
    if kind == "video":
        from .video import inspect_video
        return inspect_video(Path(path))
    raise ValueError("kind must be simulation or video")


def _python(root, kind):
    executable = Path(settings(root)[kind + "_python"])
    if not executable.is_file():
        raise FileNotFoundError(f"Configure {kind}_python in settings.local.json: {executable}")
    return str(executable)


def _new_run(root, name):
    folder = Path(root) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + name + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    return folder


def _hidden():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _environment():
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return env


def code_identity(root):
    root = Path(root).resolve()
    head = None
    if (root / ".git").exists():
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                    text=True, timeout=10, **_hidden())
            head = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {"git_head": head,
            "integration_sha256": {p.name: digest(p) for p in (root / "softcut_solution").glob("*.py")}}


def _run_job(root, kind, args, *, timeout=180, run=None, inputs=None):
    root = Path(root).resolve()
    run = run or _new_run(root, kind)
    command = [_python(root, kind), "-X", "utf8", "-B", *map(str, args)]
    cwd = projects(root)[kind]
    record = {"schema_version": "softcut-job/1", "command": command, "cwd": str(cwd),
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "timeout_s": timeout, "inputs": inputs or {}, "log_path": str(run / "process.log")}
    record["code"] = code_identity(root)
    write_json(run / "job.json", record)
    env = _environment()
    try:
        with (run / "process.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, **_hidden())
            record["pid"] = process.pid
            write_json(run / "job.json", record)
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, **_hidden())
                else:
                    process.kill()
                process.wait(timeout=10)
                raise TimeoutError(f"Job exceeded {timeout}s; see {run / 'process.log'}")
        record.update(returncode=returncode, status="completed" if returncode == 0 else "failed")
        if returncode:
            tail = (run / "process.log").read_text(encoding="utf-8", errors="replace")[-3500:]
            raise RuntimeError(f"Exit {returncode}: {tail}\nLog: {run / 'process.log'}")
        return {**record, "run_dir": str(run), "output": (run / "process.log").read_text(encoding="utf-8", errors="replace")[-12000:]}
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run / "job.json", record)


def export_fe(root, case, frames=21):
    root = Path(root).resolve()
    sim = projects(root)["simulation"]
    case = within(case, sim / "outputs")
    if not (case / "result.json").is_file() or not (case / "audit.json").is_file():
        raise ValueError("Export requires an accepted result.json and audit.json")
    if isinstance(frames, bool) or not isinstance(frames, int) or not 2 <= frames <= 201:
        raise ValueError("frames must be an integer from 2 to 201")
    run = _new_run(root, "fe_export")
    out = sim / "outputs/integration" / run.name
    result = _run_job(root, "simulation", ["-m", "tools.workbench", "--export", case,
                      "--output", out, "--frames", frames], timeout=600, run=run,
                      inputs={"result_sha256": digest(case / "result.json"), "audit_sha256": digest(case / "audit.json")})
    result["bundle_path"] = str(out / "result.bundle.json")
    write_json(run / "artifact.json", {"bundle_path": result["bundle_path"]})
    return result


def _fracture_budget(seconds):
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 3600:
        raise ValueError("seconds must be an integer from 1 to 3600")
    return seconds


def _positive_finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _fracture_source(root, path, parent):
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    return within(path, parent)


def _fracture_result(run, result):
    report = run / "result/index.html"
    if not report.is_file():
        raise RuntimeError(f"Fracture process produced no report: {report}; see {run / 'process.log'}")
    artifact = {"report_path": str(report), "output_dir": str(report.parent),
                "qualification_note": "Execution status is separate from numerical qualification and experimental calibration."}
    write_json(run / "artifact.json", artifact)
    return {**result, **artifact, "run_dir": str(run)}


def fracture_onset(root, cases, *, length_mm=.15, threshold_mpa=None, seconds=300):
    """Diagnose saved intact FE states; never load a new FE increment."""
    root = Path(root).resolve()
    sim = projects(root)["simulation"]
    seconds = _fracture_budget(seconds)
    length_mm = _positive_finite(length_mm, "length_mm")
    if threshold_mpa is not None:
        threshold_mpa = _positive_finite(threshold_mpa, "threshold_mpa")
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("cases must be a non-empty list of saved FE cases")
    paths = []
    for case in cases:
        path = _fracture_source(root, case, sim / "outputs")
        if not all((path / name).is_file() for name in ("result.json", "audit.json")):
            raise ValueError(f"Onset diagnosis requires result.json and audit.json: {path}")
        if not (path / "states").is_dir():
            raise ValueError(f"Onset diagnosis requires saved states: {path}")
        if path not in paths:
            paths.append(path)
    run = _new_run(root, "fracture_onset")
    args = ["-m", "tools.fracture_onset"]
    for case in paths:
        args.extend(["--case", case])
    args.extend(["--output", run / "result", "--length-mm", length_mm, "--seconds", seconds])
    if threshold_mpa is not None:
        args.extend(["--threshold-mpa", threshold_mpa])
    inputs = {"cases": [{"path": str(case), "result_sha256": digest(case / "result.json"),
                          "audit_sha256": digest(case / "audit.json")} for case in paths],
              "length_mm": length_mm, "threshold_mpa": threshold_mpa, "postprocessing_only": True}
    source = sim / "tools/fracture_onset.py"
    if source.is_file():
        inputs["entrypoint_sha256"] = digest(source)
    result = _run_job(root, "simulation", args, timeout=seconds + 60, run=run, inputs=inputs)
    return _fracture_result(run, result)


def fracture_run(root, config, *, seconds=300):
    """Explicitly start a budgeted cohesive-cut calculation in a new output directory."""
    root = Path(root).resolve()
    seconds = _fracture_budget(seconds)
    config = _fracture_source(root, config, root)
    if not config.is_file() or config.suffix.lower() != ".json":
        raise ValueError("Fracture calculation requires an existing workspace JSON configuration")
    if not isinstance(read_json(config), dict):
        raise ValueError("Fracture configuration must be a JSON object")
    run = _new_run(root, "fracture_run")
    inputs = {"config_path": str(config), "config_sha256": digest(config), "explicit_execution": True}
    source = projects(root)["simulation"] / "tools/run_cohesive_cut.py"
    if source.is_file():
        inputs["entrypoint_sha256"] = digest(source)
    result = _run_job(root, "simulation", ["-m", "tools.run_cohesive_cut", "--config", config,
                      "--output", run / "result", "--seconds", seconds],
                      timeout=seconds + 60, run=run, inputs=inputs)
    return _fracture_result(run, result)


def video_preflight(root, case_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise ValueError("case_id may contain letters, numbers, underscore and hyphen")
    return _run_job(root, "video", ["softcut.py", "track", "--names", case_id, "--check-only"])


def prepare_video_manifest(root, manifest, run):
    """Generate fresh copied-project output roots; never overwrite historical manifests."""
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    data = read_json(manifest)
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must contain a non-empty cases list")
    out = projects(root)["video"] / "results/integration" / run.name
    names = set()
    clean = []
    for original in cases:
        if not isinstance(original, dict):
            raise ValueError("Every manifest case must be an object")
        case = dict(original)
        name = case.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name.lower() in names:
            raise ValueError("Case names must be unique and safe path components")
        if name.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            raise ValueError("Case names must be safe path components, not Windows device names")
        names.add(name.lower())
        for key, child in (("tracks_root", "tracks"), ("measurements_root", "measurements"), ("review_root", "review")):
            case[key] = str(out / child)
        # The batch pipeline resolves its MP4 from copied ROOT/data, not historical plan.json.
        case.pop("video", None)
        clean.append(case)
    fresh = {"cases": clean, "integration_source": str(manifest), "integration_source_sha256": digest(manifest)}
    target = run / "video_manifest.json"
    write_json(target, fresh)
    return target, out


def video_batch(root, manifest, check_only=True):
    run = _new_run(root, "video_preflight" if check_only else "video_batch")
    target, out = prepare_video_manifest(root, manifest, run)
    args = ["softcut.py", "batch", "--manifest", target, "--out", out]
    if check_only:
        args.append("--check-only")
    return _run_job(root, "video", args, run=run, timeout=180 if check_only else settings(root)["video_timeout_s"],
                    inputs={"manifest_sha256": digest(target), "check_only": check_only})


def compare(root, fe_bundle, video_package, options):
    from .comparison import compare as perform_comparison
    run = _new_run(root, "comparison")
    job = {"schema_version": "softcut-job/1", "code": code_identity(root),
           "options": options, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
           "inputs": {"fe": str(fe_bundle), "video": str(video_package)}}
    try:
        result = perform_comparison(Path(fe_bundle), Path(video_package), run, options)
        job.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(run / "job.json", job)
        return {**result, "run_dir": str(run), "report_path": str(run / "index.html")}
    except Exception as exc:
        write_json(run / "failure.json", {"status": "failed", "error": str(exc), "options": options})
        job.update(status="failed", error=str(exc), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(run / "job.json", job)
        raise


def start_fe_workbench(root=ROOT):
    """Start only the copied FE server on a free port; never attach by schema alone."""
    root = Path(root).resolve()
    port = settings(root)["simulation_port"]
    lock = root / "runs/fe_server.local.json"
    def owned_ready(state):
        try:
            with urllib.request.urlopen(state["url"] + "api/catalog", timeout=1) as response:
                catalog = json.load(response)
            return catalog.get("session_token") == state.get("session_token") and state.get("root") == str(root)
        except (OSError, ValueError, KeyError):
            return False
    if lock.is_file():
        state = read_json(lock)
        if owned_ready(state):
            webbrowser.open(state["url"])
            return state
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    folder = _new_run(root, "fe_server")
    cmd = [_python(root, "simulation"), "-X", "utf8", "-B", "-m", "tools.workbench", "--port", str(port)]
    with (folder / "server.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(cmd, cwd=projects(root)["simulation"], env=_environment(), stdout=log, stderr=subprocess.STDOUT, **_hidden())
    for _ in range(60):
        if child.poll() is not None:
            raise RuntimeError(f"FE workbench exited; see {folder / 'server.log'}")
        try:
            with urllib.request.urlopen(url + "api/catalog", timeout=.5) as response:
                catalog = json.load(response)
            state = {"url": url, "pid": child.pid, "root": str(root), "session_token": catalog["session_token"],
                     "command": cmd, "log_path": str(folder / "server.log")}
            write_json(lock, state)
            webbrowser.open(url)
            return state
        except (OSError, ValueError, KeyError):
            time.sleep(.1)
    child.terminate()
    raise RuntimeError(f"FE workbench did not become ready; see {folder / 'server.log'}")


def doctor(root=ROOT):
    root = Path(root).resolve()
    checks = {}
    probe = "import importlib.util,json,sys; print(json.dumps({'python':sys.version,'packages':{x:importlib.util.find_spec(x) is not None for x in %s}}))"
    for kind, libs in (("simulation", ["numpy", "scipy", "matplotlib", "psutil", "threadpoolctl", "ipctk"]),
                       ("video", ["numpy", "scipy", "cv2", "torch", "pandas"])):
        try:
            p = subprocess.run([_python(root, kind), "-B", "-c", probe % repr(libs)], cwd=projects(root)[kind],
                               capture_output=True, text=True, encoding="utf-8", env=_environment(), timeout=20, **_hidden())
            checks[kind] = json.loads(p.stdout) if p.returncode == 0 else {"error": p.stderr[-1500:]}
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            checks[kind] = {"error": str(exc)}
    checks["simulation"]["local_ipc_runtime"] = (projects(root)["simulation"] / ".runtime/ipc-python").is_dir()
    checks["video"]["segmentation_weights"] = (projects(root)["video"] / "models/segmentation/clean_roi_20260909/checkpoint_best.pt").is_file()
    checks["root"] = str(root)
    checks["note"] = "Import availability only; numerical qualification belongs to each result audit. No FE or tracking started."
    return checks
