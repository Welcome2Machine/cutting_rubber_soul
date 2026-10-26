"""Verify copied assets by size and authored text by SHA256; original projects are read-only."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "simulation": (Path(r"D:\Doctoral Research\AcademicForge\softmodeling"), ROOT / "projects/softmodeling"),
    "video": (Path(r"D:\Doctoral Research\Softcutting\0910imaging\rec_inc_soft_cut"), ROOT / "projects/rec_inc_soft_cut"),
}
EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".venv", ".codex"}
ASSETS = {"outputs", "results", "data", "models", ".runtime", "local_maintenance"}
TEXT = {".py", ".md", ".json", ".txt", ".toml", ".ini", ".ps1", ".js", ".css", ".html", ".sh", ".cmd", ".yml", ".yaml"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    summary = {"schema_version": "softcut-import/1", "verified_utc": datetime.now(timezone.utc).isoformat(),
               "method": "All non-excluded source files: existence and byte size; authored text outside asset trees: SHA256 equality. Binary assets are not claimed content-hashed.",
               "excluded_directories": sorted(EXCLUDED), "projects": {}}
    failed = False
    for name, (source, target) in SOURCES.items():
        count, byte_count, mismatches, code = 0, 0, [], {}
        for base, dirs, files in os.walk(source):
            dirs[:] = [d for d in dirs if d not in EXCLUDED and not (Path(base) / d).is_symlink()]
            for item in files:
                src = Path(base) / item
                rel = src.relative_to(source)
                dst = target / rel
                count += 1
                byte_count += src.stat().st_size
                if not dst.is_file() or src.stat().st_size != dst.stat().st_size:
                    mismatches.append({"path": rel.as_posix(), "reason": "missing or size differs"})
                elif not any(x in ASSETS for x in rel.parts) and (src.suffix.lower() in TEXT or src.name in {".gitignore", ".gitattributes"}):
                    original, copied = sha(src), sha(dst)
                    code[rel.as_posix()] = copied
                    if original != copied:
                        mismatches.append({"path": rel.as_posix(), "reason": "SHA256 differs"})
        def git(*args):
            return subprocess.check_output(["git", "-C", str(source), *args], text=True, encoding="utf-8").strip()
        bundle, patch = ROOT / f"provenance/{name}-original.bundle", ROOT / f"provenance/{name}-working-tree.patch"
        summary["projects"][name] = {"source": str(source), "destination": target.relative_to(ROOT).as_posix(),
            "head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "working_tree_status": git("status", "--porcelain=v1"), "files": count, "bytes": byte_count,
            "authored_text_sha256": code, "mismatches": mismatches,
            "history_bundle_sha256": sha(bundle), "working_tree_patch_sha256": sha(patch)}
        failed |= bool(mismatches)
        print(f"{name}: {count} files, {byte_count} bytes, {len(code)} text hashes, {len(mismatches)} mismatches")
    summary["passed"] = not failed
    (ROOT / "provenance/import_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
