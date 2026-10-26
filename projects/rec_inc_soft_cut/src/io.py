"""Shared local file IO and process constants; no inference or network side effects."""
from pathlib import Path
import hashlib
import gzip
import json
import subprocess
import os
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HIDDEN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def atomic_write_json(path, value, *, allow_nan=True, compact=False):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=None if compact else 2,
                         separators=(',', ':') if compact else None, allow_nan=allow_nan)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each writer owns its temporary file; the previous JSON stays readable
    # until replacement succeeds. Never truncate the destination as a fallback.
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temp = Path(stream.name)
            stream.write(payload)
        for attempt in range(12):
            try:
                os.replace(temp, path)
                return
            except OSError as error:
                retryable = isinstance(error, PermissionError) or getattr(error, 'winerror', None) in (5, 32, 33)
                if not retryable or attempt == 11:
                    raise
                time.sleep(min(.05 * 2 ** attempt, .5))
    finally:
        if temp is not None:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                # Preserve the replacement error if even cleanup is locked.
                pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value, *, compact=False):
    atomic_write_json(path, clean(value), allow_nan=False, compact=compact)


def gzip_file(path):
    """Replace a file by gzip only after a complete byte-for-byte hash check.

    Leaves the original untouched on any compression/verification failure.
    Callers own metadata/link migration; this never searches or deletes folders.
    """
    path = Path(path)
    target = path.with_name(path.name + '.gz')
    if target.exists() or path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular source and no existing gzip destination')
    before = path.stat()
    original = hashlib.sha256()
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=target.name+'.',
                                         suffix='.tmp', delete=False) as output:
            temp = Path(output.name)
            with path.open('rb') as source, gzip.GzipFile(filename='', mode='wb',
                    fileobj=output, compresslevel=1, mtime=0) as compressed:
                for block in iter(lambda: source.read(2**20), b''):
                    original.update(block)
                    compressed.write(block)
        restored = hashlib.sha256()
        with gzip.open(temp, 'rb') as stream:
            for block in iter(lambda: stream.read(2**20), b''):
                restored.update(block)
        after = path.stat()
        if (restored.digest() != original.digest() or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns or sha(path) != original.hexdigest()):
            raise ValueError('Gzip verification failed or source changed during compression')
        if temp.stat().st_size >= before.st_size:
            return dict(path=str(path), compressed=False, bytes=before.st_size)
        # rename (not replace) refuses a concurrently created destination on Windows.
        temp.rename(target)
        path.unlink()
        return dict(path=str(target), original_path=str(path), compressed=True,
                    original_bytes=before.st_size, bytes=target.stat().st_size,
                    original_sha256=original.hexdigest(), sha256=sha(target),
                    decompressed_sha256=restored.hexdigest())
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def slice_ragged(arrays, prefix, keep, fields):
    offsets = arrays[prefix + 'offsets']
    spans = [slice(int(offsets[i]), int(offsets[i+1])) for i in np.flatnonzero(keep)]
    result = {prefix + 'offsets': np.r_[0, np.cumsum([s.stop-s.start for s in spans])]}
    result.update({key: np.concatenate([arrays[key][s] for s in spans]) for key in fields})
    return result
