"""Lossless cold-history storage; no solver, relaxed gate or precision change."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from tools import manage_outputs as storage


def fixture_case(outputs):
    folder = outputs/'retired'/'angle57'
    states = folder/'states'
    states.mkdir(parents=True)
    np.savez_compressed(states/'0000.npz', positions=np.arange(12, dtype=np.float64).reshape(6, 2),
                        material_metric=np.tile(np.eye(3), (3, 1, 1)),
                        previous_positions=np.zeros((6, 2)), depth_mm=np.array(0.))
    shutil.copyfile(states/'0000.npz', states/'0001.npz')
    originals = {p: p.read_bytes() for p in states.iterdir()}
    report = folder/'result.json'
    report.write_text(json.dumps({'steps': [dict(depth_mm=0), dict(depth_mm=.01)]}))
    audit = folder/'audit.json'
    audit.write_text(json.dumps({'rows': [dict(checkpoint_sha256=hashlib.sha256(raw).hexdigest()) for raw in originals.values()]}))
    return folder, originals, {report: report.read_bytes(), audit: audit.read_bytes()}


@pytest.mark.parametrize('extension', ['.tar.xz', '.tar.zst'])
def test_archive_restores_exact_audited_bytes_and_deduplicates(tmp_path, extension):
    if extension == '.tar.zst' and not shutil.which('zstd'):
        pytest.skip('Optional zstd CLI not installed; standard-library xz path is always covered')
    outputs = tmp_path/'outputs'
    case, originals, reports = fixture_case(outputs)
    manifest_path = storage.archive_states([case], outputs/'archives'/('retired'+extension),
                                          outputs=outputs, prune=True)
    manifest = storage.verify(manifest_path, outputs=outputs)
    assert manifest['pruned'] and manifest['unique_objects'] == 1
    assert all(not p.exists() for p in originals)
    assert all(p.read_bytes() == raw for p, raw in reports.items())
    assert storage.restore(manifest_path, outputs=outputs, prefix=case) == 2
    for p, raw in originals.items():
        assert p.read_bytes() == raw
        with np.load(p, allow_pickle=False) as state:
            np.testing.assert_array_equal(state['material_metric'], np.tile(np.eye(3), (3, 1, 1)))
    assert storage.inventory(outputs)['unique_file_bytes'] < storage.inventory(outputs)['logical_bytes']


def test_changed_source_never_gets_pruned(tmp_path, monkeypatch):
    outputs = tmp_path/'outputs'
    case, originals, _ = fixture_case(outputs)
    real_verify = storage.verify
    def changed_after_verification(*args, **kwargs):
        value = real_verify(*args, **kwargs)
        next(iter(originals)).write_bytes(b'new source from another process')
        return value
    monkeypatch.setattr(storage, 'verify', changed_after_verification)
    with pytest.raises(ValueError, match='Source changed'):
        storage.archive_states([case], outputs/'archives/retired.tar.xz', outputs=outputs, prune=True)
    assert all(p.exists() for p in originals)


def test_restoring_one_case_keeps_other_histories_cold(tmp_path):
    outputs=tmp_path/'outputs'
    case,originals,_=fixture_case(outputs)
    other=outputs/'retired/angle33/states/0000.npz'
    other.parent.mkdir(parents=True)
    shutil.copyfile(next(iter(originals)),other)
    manifest=storage.archive_states([outputs/'retired'],outputs/'archives/retired.tar.xz',
                                    outputs=outputs,prune=True)
    assert storage.restore(manifest,outputs=outputs,prefix=case)==2
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    assert not other.exists()


def test_modified_archive_and_conflicting_restoration_are_rejected(tmp_path):
    outputs = tmp_path/'outputs'
    case, originals, _ = fixture_case(outputs)
    manifest_path = storage.archive_states([case], outputs/'archives/retired.tar.xz', outputs=outputs)
    first = next(iter(originals))
    first.write_bytes(b'another material history')
    with pytest.raises(ValueError, match='overwrite'):
        storage.restore(manifest_path, outputs=outputs)
    assert first.read_bytes() == b'another material history'
    archive = outputs/'archives/retired.tar.xz'
    with archive.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='fingerprint'):
        storage.verify(manifest_path, outputs=outputs)


def test_protected_paths_and_manifest_escape_are_rejected(tmp_path):
    outputs = tmp_path/'outputs'
    case, originals, _ = fixture_case(outputs)
    with pytest.raises(ValueError, match='baselines'):
        storage.archive_states([outputs/'baseline/angle57'], outputs/'archives/base.tar.xz', outputs=outputs)
    with pytest.raises(ValueError, match='child of outputs'):
        storage.archive_states([tmp_path/'external'], outputs/'archives/external.tar.xz', outputs=outputs)
    manifest_path = storage.archive_states([case], outputs/'archives/retired.tar.xz', outputs=outputs)
    value = json.loads(manifest_path.read_text(encoding='utf-8'))
    value['files'][0]['path'] = '../outside.npz'
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Unsafe'):
        storage.restore(manifest_path, outputs=outputs)
    assert all(p.read_bytes() == raw for p, raw in originals.items())
