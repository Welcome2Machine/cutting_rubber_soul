"""The sole root CLI forwards commands without loading unrelated workflows."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import softcut


EXPECTED = {
    'track': 'src.workflows.tracking', 'batch': 'src.workflows.batch',
    'import': 'src.workflows.video_import', 'measure': 'src.workflows.measurement',
    'motion': 'src.workflows.motion', 'slip': 'src.workflows.sliding',
    'gallery': 'src.workflows.gallery',
    'optimize': 'src.workflows.optimize',
    'exchange-export': 'src.exchange.export', 'exchange-check': 'src.exchange.validate',
    'check': 'src.workflows.check',
}


@pytest.mark.parametrize('command,module', EXPECTED.items())
def test_dispatch_imports_only_requested_workflow_and_preserves_arguments(monkeypatch, command, module):
    entry = Mock(return_value=7)
    loader = Mock(return_value=SimpleNamespace(main=entry))
    monkeypatch.setattr(softcut.importlib, 'import_module', loader)
    arguments = [command, '--names', '0930l30a1', '--worker']
    assert softcut.main(arguments) == 7
    assert arguments == [command, '--names', '0930l30a1', '--worker']
    loader.assert_called_once_with(module)
    entry.assert_called_once_with(arguments[1:])


@pytest.mark.parametrize('arguments', [[], ['--help'], ['-h']])
def test_help_is_read_only_and_lists_all_supported_commands(monkeypatch, capsys, arguments):
    loader = Mock(side_effect=AssertionError('Help imported a workflow'))
    monkeypatch.setattr(softcut.importlib, 'import_module', loader)
    assert softcut.main(arguments) == 0
    output = capsys.readouterr().out
    assert all(command in output for command in EXPECTED)
    loader.assert_not_called()


def test_unknown_command_exits_with_error_without_processing(monkeypatch, capsys):
    loader = Mock(side_effect=AssertionError('Unknown command started processing'))
    monkeypatch.setattr(softcut.importlib, 'import_module', loader)
    assert softcut.main(['cotracker']) == 2
    assert 'Unknown command' in capsys.readouterr().err
    loader.assert_not_called()


def test_system_arguments_and_successful_none_return(monkeypatch):
    entry = Mock(return_value=None)
    monkeypatch.setattr(softcut.importlib, 'import_module', Mock(return_value=SimpleNamespace(main=entry)))
    monkeypatch.setattr(sys, 'argv', ['softcut.py', 'track', '--help'])
    assert softcut.main() == 0
    entry.assert_called_once_with(['--help'])


def test_worker_failure_is_not_reported_as_success(monkeypatch):
    entry = Mock(side_effect=RuntimeError('failed tracking'))
    monkeypatch.setattr(softcut.importlib, 'import_module', Mock(return_value=SimpleNamespace(main=entry)))
    with pytest.raises(RuntimeError, match='failed tracking'):
        softcut.main(['track', '--worker'])
