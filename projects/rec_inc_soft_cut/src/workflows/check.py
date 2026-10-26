"""Read-only checks of the preserved calculation chain; never run inference.

Run from the repository root: python softcut.py check
"""
from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from src.io import ROOT


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href", "poster") and value:
                parsed = urlsplit(value)
                if not parsed.scheme and not parsed.netloc and parsed.path:
                    self.links.append(unquote(parsed.path))


def validate(root=ROOT):
    root = Path(root).resolve()
    results = root / "results"
    config = json.loads((root / "configs/retained_results.json").read_text(encoding="utf-8"))
    errors = []
    counts = dict(html_links=0, npz_archives=0, cases=0, presentations=0)

    def require(path):
        if not path.exists():
            errors.append(f"Missing: {path.relative_to(root)}")
            return False
        return True

    def npz(path):
        if require(path):
            try:
                # Read the archive directory, without allocating full trajectory arrays.
                with ZipFile(path) as archive:
                    if not archive.namelist() or any(not x.endswith(".npy") for x in archive.namelist()):
                        raise ValueError("Not an array archive")
                counts["npz_archives"] += 1
            except Exception as exc:
                errors.append(f"Invalid NPZ: {path.relative_to(root)}: {exc}")

    for name in config["results"]:
        require(results / name)
    for name in config["entry_points"]:
        path = results / name
        if require(path) and path.suffix == ".html":
            parser = LocalLinks()
            parser.feed(path.read_text(encoding="utf-8"))
            for link in parser.links:
                counts["html_links"] += 1
                require((path.parent / link).resolve())

    for group in config.get('validation_groups', []):
        manifest = results/group['manifest']
        if not require(manifest):
            continue
        cases = json.loads(manifest.read_text(encoding='utf-8'))['cases']
        counts[group['counter']] = len(cases)
        if len(cases) != group['expected_cases']:
            errors.append(f"Expected {group['expected_cases']} cases in {group['manifest']}, got {len(cases)}")
        for case in cases:
            name = case[group['case_key']]
            for pattern in group['npz']:
                npz(results/pattern.format(name=name))
            for pattern in group['files']:
                require(results/pattern.format(name=name))
    return dict(ok=not errors, read_only=True, counts=counts, errors=errors,
                numerical_accuracy_certified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = validate(args.root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
