"""Reject data, credentials, runtime artifacts and oversized blobs from Git publication."""
import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".onnx", ".pdf", ".mp4", ".mkv", ".avi", ".mov", ".png", ".jpg", ".jpeg", ".gif", ".csv", ".tsv", ".xlsx", ".xls", ".zip", ".7z", ".bundle", ".patch", ".exe", ".dll", ".pyd", ".pem", ".key", ".pyc"}
BLOCKED_PARTS = {".venv", ".runtime", "__pycache__", ".pytest_cache", ".codex", "node_modules"}
BLOCKED_SUFFIXES.update({".pptx", ".ppt"})
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


def tree_records(ref):
    """Yield path, size and object identity from one complete Git tree."""
    entries = subprocess.check_output(["git", "ls-tree", "-rlz", ref], cwd=ROOT).split(b"\0")
    for entry in entries:
        if not entry:
            continue
        info, name_raw = entry.split(b"\t", 1)
        fields = info.split()
        if fields[1] == b"blob":
            yield name_raw.decode("utf-8"), int(fields[3]), fields[2].decode("ascii")


def history_records(ref, exclude=None):
    """Visit all historical trees, including files added and later deleted."""
    command = ["git", "rev-list", ref]
    if exclude is not None:
        command.extend(["--not", exclude])
    commits = subprocess.check_output(command, cwd=ROOT).decode().splitlines()
    seen = set()
    for commit in commits:
        for name, size, blob in tree_records(commit):
            if (name, blob) not in seen:
                seen.add((name, blob))
                yield name, size, blob


def pre_push_records(lines):
    """Check clean tips and new history against each actual destination head.

    A published ancestor can contain historical debt. It is not new publication,
    but it must be removed from the current tree. Only an exact path/blob pair
    already reachable from the remote head is excluded from the new-history
    check. A new destination has no trusted baseline and receives a full audit.
    """
    for line in lines:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("Invalid Git pre-push input")
        _, local_sha, _, remote_sha = fields
        if any(re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", sha) is None
               for sha in (local_sha, remote_sha)):
            raise ValueError("Invalid object identity in Git pre-push input")
        if set(local_sha) == {"0"}:
            continue  # Deleting a destination publishes no new objects.
        subprocess.check_output(["git", "rev-parse", "--verify", local_sha + "^{commit}"],
                                cwd=ROOT, stderr=subprocess.STDOUT)
        if set(remote_sha) == {"0"}:
            yield from history_records(local_sha)
            continue
        try:
            subprocess.check_output(["git", "rev-parse", "--verify", remote_sha + "^{commit}"],
                                    cwd=ROOT, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise ValueError("Remote head is unavailable locally; fetch and review it before pushing") from exc
        ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", remote_sha, local_sha],
                                  cwd=ROOT, capture_output=True, check=False)
        if ancestry.returncode:
            raise ValueError("Remote head is not a verified ancestor; fetch and review, do not force push")
        yield from tree_records(local_sha)
        published = {(name, blob) for name, _, blob in history_records(remote_sha)}
        for name, size, blob in history_records(local_sha, exclude=remote_sha):
            if (name, blob) not in published:
                yield name, size, blob


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--ref", help="Inspect every blob reachable from a commit (e.g. HEAD)")
    mode.add_argument("--pre-push", action="store_true",
                      help="Check clean tips and unpublished history using the actual remote heads from Git hook stdin")
    args = p.parse_args(argv)
    violations, count = [], 0
    if args.staged:
        raw = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=ROOT)
        records = []
        for item in raw.split(b"\0"):
            if item:
                name = item.decode("utf-8")
                size = int(subprocess.check_output(["git", "cat-file", "-s", ":" + name], cwd=ROOT))
                records.append((name, size, "index"))
    elif args.ref is not None:
        records = history_records(args.ref)
    else:
        records = pre_push_records(sys.stdin)
    seen = set()
    try:
        for name, size, blob in records:
            if (name, blob) in seen:
                continue
            seen.add((name, blob))
            count += 1
            problem = reason(name, size)
            if problem:
                violations.append(f"{name}: {problem} ({size} bytes)")
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"Publication check could not verify history: {exc}")
        return 1
    print("\n".join(violations) if violations else f"Publication check passed: {count} blobs")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
