"""Regression checks for transient Windows JSON replacement failures."""
import errno
import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.io import atomic_write_json, gzip_file, write_json


def test_transient_lock_recovers_and_uses_unique_temp(tmp_path):
    target = tmp_path / 'heartbeat.json'
    target.write_text('{"old": true}', encoding='utf-8')
    unrelated = tmp_path / 'heartbeat.tmp'
    unrelated.write_text('another writer', encoding='utf-8')
    replace = os.replace
    attempts = []

    def locked_then_success(source, destination):
        attempts.append(Path(source))
        assert json.loads(target.read_text(encoding='utf-8')) == {'old': True}
        assert Path(source).name != unrelated.name
        if len(attempts) < 3:
            raise PermissionError(errno.EACCES, 'simulated sharing violation')
        replace(source, destination)

    with patch('src.io.os.replace', side_effect=locked_then_success), \
            patch('src.io.time.sleep') as sleep:
        atomic_write_json(target, {'stage': '计算', 'updated': 123})
    assert sleep.call_count == 2
    assert json.loads(target.read_text(encoding='utf-8')) == {'stage': '计算', 'updated': 123}
    assert unrelated.read_text(encoding='utf-8') == 'another writer'
    assert list(tmp_path.glob('heartbeat.json.*.tmp')) == []


def test_permanent_denial_is_bounded_and_preserves_previous_json(tmp_path):
    target = tmp_path / 'status.json'
    target.write_text('{"valid": true}', encoding='utf-8')
    with patch('src.io.os.replace', side_effect=PermissionError('locked')) as replace, \
            patch('src.io.time.sleep') as sleep, pytest.raises(PermissionError):
        atomic_write_json(target, {'valid': False})
    assert replace.call_count == 12
    assert sleep.call_count == 11
    assert json.loads(target.read_text(encoding='utf-8')) == {'valid': True}
    assert list(tmp_path.glob('*.tmp')) == []


def test_other_io_failure_is_not_retried(tmp_path):
    with patch('src.io.os.replace', side_effect=OSError(errno.ENOSPC, 'full')) as replace, \
            patch('src.io.time.sleep') as sleep, pytest.raises(OSError):
        atomic_write_json(tmp_path / 'status.json', {})
    assert replace.call_count == 1
    sleep.assert_not_called()
    assert list(tmp_path.glob('*.tmp')) == []


def test_invalid_payload_does_not_touch_destination(tmp_path):
    target = tmp_path / 'status.json'
    atomic_write_json(target, {'valid': True})
    with pytest.raises(ValueError):
        atomic_write_json(target, {'invalid': float('nan')}, allow_nan=False)
    assert json.loads(target.read_text(encoding='utf-8')) == {'valid': True}
    assert list(tmp_path.glob('*.tmp')) == []


def test_compact_json_preserves_values_and_reduces_size(tmp_path):
    payload = {'name': '材料', 'diagnostics': [{'values': list(range(100))}] * 30}
    pretty, compact = tmp_path/'pretty.json', tmp_path/'compact.json'
    write_json(pretty, payload)
    write_json(compact, payload, compact=True)
    assert json.loads(compact.read_text(encoding='utf-8')) == payload
    assert compact.stat().st_size < pretty.stat().st_size / 2


def test_verified_gzip_preserves_every_byte_and_refuses_existing_target(tmp_path):
    path = tmp_path/'tracks.csv'
    payload = b'case,point,x,y\r\n' + '材料,123,0.123456789,-3.45\r\n'.encode('utf-8') * 1000
    path.write_bytes(payload)
    report = gzip_file(path)
    assert report['compressed'] and not path.exists()
    assert report['original_sha256'] == report['decompressed_sha256']
    assert gzip.open(report['path'], 'rb').read() == payload
    path.write_bytes(b'another writer')
    with pytest.raises(ValueError):
        gzip_file(path)
    assert path.read_bytes() == b'another writer'


def test_gzip_failure_preserves_source_and_cleans_temporary_file(tmp_path):
    path = tmp_path/'tracks.csv'
    path.write_bytes(b'point,x,y\n' * 100)
    with patch('src.io.gzip.open', side_effect=OSError('bad compressed stream')):
        with pytest.raises(OSError):
            gzip_file(path)
    assert path.read_bytes() == b'point,x,y\n' * 100
    assert sorted(p.name for p in tmp_path.iterdir()) == ['tracks.csv']


def test_unhelpful_compression_retains_original(tmp_path):
    path = tmp_path/'tiny.csv'
    path.write_bytes(b'x')
    assert not gzip_file(path)['compressed']
    assert path.read_bytes() == b'x'
    assert list(tmp_path.glob('*.gz')) == []
