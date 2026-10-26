"""Reject data, credentials, runtime artifacts and oversized blobs from Git publication."""
import argparse
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".onnx", ".pdf", ".mp4", ".mkv", ".avi", ".mov", ".png", ".jpg", ".jpeg", ".gif", ".csv", ".tsv", ".xlsx", ".xls", ".zip", ".7z", ".bundle", ".patch", ".exe", ".dll", ".pyd", ".pem", ".key", ".pyc"}
BLOCKED_PARTS = {".venv", ".runtime", "__pycache__", ".pytest_cache", ".codex", "node_modules"}
BLOCKED_PREFIXES = ("runs/", "logs/", "projects/softmodeling/outputs/", "projects/rec_inc_soft_cut/results/", "projects/rec_inc_soft_cut/data/", "projects/rec_inc_soft_cut/models/", "projects/rec_inc_soft_cut/environment/local_maintenance/")


def reason(path, size):
    p = PurePosixPath(path)
    if any(part in BLOCKED_PARTS for part in p.parts) or path.startswith(BLOCKED_PREFIXES):
        return "local data/runtime directory"
    if p.name == "settings.local.json" or p.name == ".env" or p.name.startswith(".env."):
        return "local settings or credentials"
    if p.suffix.lower() in BLOCKED_SUFFIXES:
        return "data/binary/provenance artifact"
    if size > 2 * 1024 * 1024:
        return "blob exceeds 2 MiB; review data/code split"
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--ref", help="Inspect every blob reachable from a commit (e.g. HEAD)")
    mode.add_argument("--pre-push", action="store_true", help="Read the actual pushed local refs from Git hook stdin")
    args = p.parse_args(argv)
    violations, count = [], 0
    if args.staged:
        raw = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=ROOT)
        records = []
        for item in raw.split(b"\0"):
            if item:
                name = item.decode("utf-8")
                size = int(subprocess.check_output(["git", "cat-file", "-s", ":" + name], cwd=ROOT))
                records.append((name, size))
    else:
        # Walk historical trees as well as HEAD so deleted data cannot hide in history.
        records, seen = [], set()
        refs = [args.ref]
        if args.pre_push:
            refs = []
            for line in sys.stdin:
                fields = line.split()
                if len(fields) != 4:
                    raise ValueError("Invalid Git pre-push input")
                if set(fields[1]) != {"0"}:
                    refs.append(fields[1])
        commits = sorted({commit for ref in refs for commit in subprocess.check_output(
            ["git", "rev-list", ref], cwd=ROOT).decode().splitlines()})
        for commit in commits:
            entries = subprocess.check_output(["git", "ls-tree", "-rlz", commit], cwd=ROOT).split(b"\0")
            for entry in entries:
                if not entry:
                    continue
                info, name_raw = entry.split(b"\t", 1)
                fields = info.split()
                if fields[1] != b"blob":
                    continue
                name = name_raw.decode("utf-8")
                key = (fields[2], name)
                if key not in seen:
                    records.append((name, int(fields[3])))
                    seen.add(key)
    for name, size in records:
        count += 1
        problem = reason(name, size)
        if problem:
            violations.append(f"{name}: {problem} ({size} bytes)")
    print("\n".join(violations) if violations else f"Publication check passed: {count} blobs")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
