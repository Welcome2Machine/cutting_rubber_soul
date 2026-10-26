"""Keep worker inputs reproducible outside historical results directories."""
import sys
from pathlib import Path

import pytest

from src.workflows import motion as runner


def test_custom_sources_reach_workers(tmp_path, monkeypatch):
    source, measurements, out = [tmp_path / name for name in ('tracks', 'measurements', 'review')]
    monkeypatch.setattr(runner, 'SOURCE', source)
    monkeypatch.setattr(runner, 'MEASUREMENTS', measurements)
    monkeypatch.setattr(runner, 'index', lambda args: None)
    calls = []

    def worker(cmd, **kwargs):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(runner.subprocess, 'call', worker)
    monkeypatch.setattr(sys, 'argv', ['runner', '--source', str(source), '--measurements', str(measurements),
        '--out', str(out), '--names', '0930l30a1', '--stage', 'measure'])
    runner.main()
    assert len(calls) == 1
    cmd = calls[0]
    assert Path(cmd[cmd.index('--source') + 1]) == source.resolve()
    assert Path(cmd[cmd.index('--measurements') + 1]) == measurements.resolve()
    assert cmd[cmd.index('--scope') + 1] == 'interface'


@pytest.mark.parametrize('collision', ['tracks', 'measurements'])
def test_output_cannot_replace_input(tmp_path, monkeypatch, collision):
    monkeypatch.setattr(runner, 'SOURCE', tmp_path / 'tracks')
    monkeypatch.setattr(runner, 'MEASUREMENTS', tmp_path / 'measurements')
    monkeypatch.setattr(sys, 'argv', ['runner', '--source', str(tmp_path / 'tracks'),
        '--measurements', str(tmp_path / 'measurements'), '--out', str(tmp_path / collision)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not (tmp_path / collision).exists()
