"""Inspect output storage; losslessly archive/restore inactive NPZ histories.

Run only when the selected sources have no active solver or postprocessor.
Original NPZ bytes, filenames and audit SHA256s survive archival. Reports stay
in place; no numerical array is rounded, repacked or removed from a history.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

from run_support import ROOT, background_child_options, write_json

SCHEMA = 'softcut-output-archive/1'
MARKER = 'storage.archive.json'


def confined(path, root):
    root = Path(root).resolve()
    path = Path(path).resolve()
    if root not in path.parents:
        raise ValueError('Expected a child of outputs: '+str(path))
    return path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hash_stream(stream)


def hash_stream(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''):
        value.update(block)
    return value.hexdigest()


def inventory(root):
    """Logical bytes and unique file identities; hard links count only once."""
    root = Path(root).resolve()
    sizes = defaultdict(int)
    files = 0
    identities = set()
    unique = 0
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = confined(Path(directory)/name, root)
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino)
            sizes[path.relative_to(root).parts[0]] += stat.st_size
            files += 1
            if key not in identities:
                identities.add(key)
                unique += stat.st_size
    return dict(files=files, logical_bytes=sum(sizes.values()),
                unique_file_bytes=unique, groups=dict(sorted(sizes.items())))


def assert_idle(outputs):
    """Do not prune while a solver or workbench field export is active."""
    import psutil
    modules = ('tools.run_material_candidate', 'tools.run_material_queue',
               'tools.visualize_ipc', 'tools.qualify_intact_baseline')
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.pid == os.getpid():
            continue
        command = ' '.join(process.info['cmdline'] or [])
        if any(module in command for module in modules):
            try:
                if Path(process.cwd()).resolve() == outputs.parent:
                    raise RuntimeError(f'Active result consumer PID {process.pid}; stop it before storage maintenance')
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
    for path in (outputs/'workbench').glob('*/**/job.json'):
        job = json.loads(path.read_text(encoding='utf-8'))
        if job.get('status') in ('prepared', 'running', 'postprocessing'):
            raise RuntimeError('Active/stale workbench job must be resolved before pruning: '+str(path))


@contextmanager
def tar_stream(path, mode):
    """Streaming codecs avoid an uncompressed multi-GB temporary tar file."""
    path = Path(path)
    if path.name.endswith('.tar.xz'):
        options = dict(filters=[dict(id=lzma.FILTER_LZMA2, preset=1, dict_size=16*1024*1024)]) if mode == 'w' else {}
        with lzma.open(path, mode+'b', **options) as stream:
            with tarfile.open(fileobj=stream, mode=mode+'|', format=tarfile.PAX_FORMAT) as archive:
                yield archive
        return
    if not path.name.endswith('.tar.zst'):
        raise ValueError('Archive must end in .tar.zst or .tar.xz')
    executable = shutil.which('zstd')
    if not executable:
        raise ValueError('zstd CLI is required for .tar.zst; .tar.xz uses the Python standard library')
    with tempfile.TemporaryFile() as errors:
        if mode == 'w':
            output = path.open('wb')
            command = [executable, '-q', '-3', '--long=27', '-T2', '-c']
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output,
                                       stderr=errors, **background_child_options())
            stream = process.stdin
        else:
            output = None
            process = subprocess.Popen([executable, '-q', '-d', '-c', str(path)],
                                       stdout=subprocess.PIPE, stderr=errors,
                                       **background_child_options())
            stream = process.stdout
        try:
            with tarfile.open(fileobj=stream, mode=mode+'|', format=tarfile.PAX_FORMAT) as archive:
                yield archive
            if mode == 'r':
                # Drain padding/footer so decoder checksum failures are visible.
                while stream.read(1024*1024):
                    pass
            stream.close()
            code = process.wait()
            if code:
                errors.seek(0)
                raise OSError('zstd failed: '+errors.read().decode('utf-8', errors='replace'))
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            stream.close()
            if output:
                output.close()


def _manifest(path, outputs):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unsupported output archive')
    seen = set()
    for row in manifest['files']:
        relative = PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in row['path'] or ':' in row['path']:
            raise ValueError('Unsafe archive destination')
        confined(outputs.joinpath(*relative.parts), outputs)
        if row['path'] in seen or len(row['sha256']) != 64:
            raise ValueError('Duplicate/invalid archive record')
        seen.add(row['path'])
    return manifest


def verify(manifest_path, *, outputs):
    outputs = Path(outputs).resolve()
    manifest_path = confined(manifest_path, outputs)
    manifest = _manifest(manifest_path, outputs)
    path = confined(outputs/manifest['archive'], outputs)
    if digest(path) != manifest['archive_sha256']:
        raise ValueError('Archive fingerprint changed')
    expected = {row['sha256']: row['bytes'] for row in manifest['files']}
    seen = set()
    with tar_stream(path, 'r') as archive:
        for member in archive:
            key = member.name.removeprefix('objects/').removesuffix('.npz')
            if (not member.isfile() or member.name != f'objects/{key}.npz'
                    or key not in expected or key in seen or member.size != expected[key]):
                raise ValueError('Unexpected archive member')
            with archive.extractfile(member) as stream:
                if hash_stream(stream) != key:
                    raise ValueError('Archived NPZ fingerprint changed')
            seen.add(key)
    if seen != set(expected):
        raise ValueError('Missing archive objects')
    return manifest


def archive_states(sources, destination, *, outputs, prune=False):
    outputs = Path(outputs).resolve()
    if prune:
        assert_idle(outputs)
    destination = confined(destination, outputs)
    manifest_path = destination.with_name(destination.name+'.manifest.json')
    if destination.exists() or manifest_path.exists():
        raise FileExistsError('Choose a new archive name')
    sources = [confined(source, outputs) for source in sources]
    protected = [outputs/'baseline', outputs/'paper2023']
    if any(source == p or p in source.parents or source in p.parents for source in sources for p in protected):
        raise ValueError('Complete baselines and paper reproduction stay online')
    if any(not source.is_dir() for source in sources):
        raise ValueError('Archive source must be an existing directory')
    paths = sorted({confined(path, outputs) for source in sources for path in source.rglob('*.npz')})
    if not paths:
        raise ValueError('No NPZ states to archive')
    if any(path.is_symlink() for source in sources for path in source.rglob('*.npz')):
        raise ValueError('Refusing linked archive inputs')
    rows = [dict(path=path.relative_to(outputs).as_posix(), bytes=path.stat().st_size,
                 sha256=digest(path)) for path in paths]
    destination.parent.mkdir(parents=True, exist_ok=True)
    unique = {}
    print(f'ARCHIVE files={len(rows)} input_bytes={sum(r["bytes"] for r in rows)}', flush=True)
    # The final suffix remains visible to the codec selector; partial is not advertised.
    temporary = destination.with_name('.partial-'+destination.name)
    try:
        with tar_stream(temporary, 'w') as archive:
            for i, row in enumerate(rows):
                if row['sha256'] not in unique:
                    path = outputs/row['path']
                    info = tarfile.TarInfo(f'objects/{row["sha256"]}.npz')
                    info.size = row['bytes']
                    with path.open('rb') as stream:
                        archive.addfile(info, stream)
                    unique[row['sha256']] = True
                if (i+1) % 100 == 0:
                    print(f'ARCHIVE written={i+1}/{len(rows)} objects={len(unique)}', flush=True)
        temporary.replace(destination)
        manifest = dict(schema=SCHEMA, archive=destination.relative_to(outputs).as_posix(),
                        archive_sha256=digest(destination), files=rows,
                        source_directories=[p.relative_to(outputs).as_posix() for p in sources],
                        unique_objects=len(unique), original_bytes=sum(r['bytes'] for r in rows),
                        archive_bytes=destination.stat().st_size, pruned=False,
                        representation='Original NPZ bytes; content-addressed objects; no array conversion')
        write_json(manifest_path, manifest)
        verify(manifest_path, outputs=outputs)
        # Check EVERY source before removing ANY source. A changing input aborts pruning.
        if any(not (outputs/r['path']).is_file() or digest(outputs/r['path']) != r['sha256'] for r in rows):
            raise ValueError('Source changed; no original files removed')
        if prune:
            assert_idle(outputs)
            for source in sources:
                write_json(source/MARKER, dict(schema=SCHEMA, manifest=manifest_path.relative_to(outputs).as_posix(),
                    note='NPZ histories archived losslessly; restore before field export/audit/resumption'))
            for row in rows:
                confined(outputs/row['path'], outputs).unlink()
            manifest['pruned'] = True
            write_json(manifest_path, manifest)
        print(f'ARCHIVE verified={len(rows)} bytes={destination.stat().st_size} pruned={prune}', flush=True)
        return manifest_path
    finally:
        temporary.unlink(missing_ok=True)


def restore(manifest_path, *, outputs, prefix=None):
    outputs = Path(outputs).resolve()
    manifest = verify(manifest_path, outputs=outputs)
    prefix = confined(prefix, outputs) if prefix is not None else None
    selected = [row for row in manifest['files'] if prefix is None or prefix == outputs/row['path'] or prefix in (outputs/row['path']).parents]
    if not selected:
        raise ValueError('No matching archived files')
    wanted = defaultdict(list)
    for row in selected:
        target = confined(outputs/row['path'], outputs)
        if target.exists():
            if digest(target) != row['sha256']:
                raise ValueError('Refusing to overwrite a different file: '+str(target))
        else:
            wanted[row['sha256']].append(target)
    with tar_stream(outputs/manifest['archive'], 'r') as archive:
        for member in archive:
            key = member.name.removeprefix('objects/').removesuffix('.npz')
            targets = wanted.get(key, [])
            if not targets:
                continue
            first = targets[0]
            first.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=first.parent, suffix='.restore', delete=False) as temporary:
                temporary_path = Path(temporary.name)
                with archive.extractfile(member) as stream:
                    shutil.copyfileobj(stream, temporary, 1024*1024)
            try:
                if digest(temporary_path) != key:
                    raise ValueError('Restore checksum mismatch')
                # Every destination was checked above; never overwrite another history.
                for target in targets:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise FileExistsError('Destination appeared during restoration')
                    # Accepted snapshots are immutable; shared bytes remain protected
                    # by atomic checkpoint replacement if a future runner writes a path.
                    os.link(temporary_path, target)
            finally:
                temporary_path.unlink(missing_ok=True)
    if any(digest(outputs/row['path']) != row['sha256'] for row in selected):
        raise ValueError('Incomplete restoration')
    print(f'RESTORE verified={len(selected)}', flush=True)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('inventory')
    archive = commands.add_parser('archive')
    archive.add_argument('--source', type=Path, action='append', required=True)
    archive.add_argument('--output', type=Path, required=True)
    archive.add_argument('--prune', action='store_true', help='Remove originals only after full archive/SHA verification')
    for name in ('verify', 'restore'):
        command = commands.add_parser(name)
        command.add_argument('--manifest', type=Path, required=True)
        if name == 'restore':
            command.add_argument('--prefix', type=Path)
    args = parser.parse_args()
    outputs = ROOT/'outputs'
    if args.command == 'inventory':
        print(json.dumps(inventory(outputs), indent=2))
    elif args.command == 'archive':
        archive_states(args.source, args.output, outputs=outputs, prune=args.prune)
    elif args.command == 'restore':
        restore(args.manifest, outputs=outputs, prefix=args.prefix)
    else:
        value = verify(args.manifest, outputs=outputs)
        print(f'VERIFIED files={len(value["files"])} objects={value["unique_objects"]}')


if __name__ == '__main__':
    main()
