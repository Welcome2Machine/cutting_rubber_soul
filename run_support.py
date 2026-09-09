"""Shared atomic outputs and repository-relative source provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(payload, encoding='utf-8')
    temporary.replace(path)


def save_npz(path, **values):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def source_hashes(packages=('prototype', 'mechanics'), *, snapshot=None):
    sources = [Path(__file__).resolve()]
    for package in packages:
        sources.extend((ROOT / package).glob('*.py'))
    result = {}
    for source in sorted(set(sources)):
        relative = source.relative_to(ROOT)
        result[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        if snapshot is not None:
            target = Path(snapshot) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return result
