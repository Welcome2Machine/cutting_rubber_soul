"""Generic cutoffs, worker isolation, and the paired scientific output contract."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.io import read_json, write_json
from src.workflows import sliding, motion
from tests.test_dual_contact import rigid_scene


def flat_scene():
    data, geometry, origin, recovered = rigid_scene()
    data['blade_observation_qualified'] = np.ones(len(data['frames']), bool)
    recovered['valid'] = np.ones_like(data['support'])
    query, initial, paths = data.pop('query'), data.pop('initial'), data.pop('paths')
    data.update({'query__'+key: value for key, value in query.items()})
    data.update({'initial__'+key: value for key, value in initial.items()})
    data['path__offsets'] = np.r_[0, np.cumsum([len(path['xy']) for path in paths])]
    data['path__xy'] = np.concatenate([path['xy'] for path in paths])
    data['path__arc'] = np.concatenate([path['arc'] for path in paths])
    data['path__gap'] = np.array([path['tip_gap'] for path in paths])
    data['path__projection'] = np.array([path['tip_projection'] for path in paths])
    return data, geometry, origin, recovered


def test_manifest_cutoffs_and_paths_remain_explicit(tmp_path):
    manifest = tmp_path/'input.json'
    write_json(manifest, dict(cases=[dict(name='new60a1', tracks_root='tracks',
        measurements_root='measurements', cutoff=dict(depth_um=825))]))
    case = sliding.load_cases(manifest)[0]
    assert case['cutoff']['depth_mm'] == .825
    assert Path(case['tracks_root']) == tmp_path/'tracks'
    assert len(case['missing_prerequisites']) == 11
    assert sliding.load_cases(manifest, stop_map={'new60a1': 1.2})[0]['cutoff']['depth_um'] == 1200
    with pytest.raises(ValueError, match='Unknown case'):
        sliding.load_cases(manifest, names=['absent'])
    with pytest.raises(ValueError, match='not guessed'):
        sliding.load_cases(names=['0930l60a1'], source=tmp_path/'tracks', measurements=tmp_path/'measurements')
    with pytest.raises(ValueError, match='single directory'):
        sliding.load_cases(names=['../escape'], source=tmp_path, measurements=tmp_path, stop_mm=1.)


def test_inconsistent_depth_units_and_unknown_overrides_are_rejected(tmp_path):
    manifest = tmp_path/'input.json'
    write_json(manifest, dict(cases=[dict(name='case', tracks_root='tracks', measurements_root='measures',
        cutoff=dict(depth_mm=1., depth_um=2000))]))
    with pytest.raises(ValueError, match='inconsistent cutoff'):
        sliding.load_cases(manifest)
    with pytest.raises(ValueError, match='unknown cases'):
        sliding.load_cases(manifest, stop_map={'different': 1.})


def test_worker_preserves_selected_recipe_and_portable_paths(tmp_path):
    args = SimpleNamespace(estimator='recovery', render=True, overwrite=False, audit_contact_zero=True)
    command = sliding.worker_command(args, 'case', tmp_path/'directory with spaces')
    assert command[2:4] == ['-m', 'src.workflows.sliding']
    assert command[command.index('--manifest')+1] == str(tmp_path/'directory with spaces'/'manifest.json')
    assert command[command.index('--estimator')+1] == 'recovery'
    assert '--audit-contact-zero' in command and '--render' in command and '--overwrite' not in command


def test_contact_zero_loader_is_restored_on_failure():
    original = motion.data
    with pytest.raises(RuntimeError, match='synthetic failure'):
        with sliding.candidate_contact_zero(dict(measurements_root='unused'), {}):
            assert motion.data is not original
            raise RuntimeError('synthetic failure')
    assert motion.data is original


@pytest.mark.parametrize('estimator', ['bounded', 'recovery'])
def test_workflow_retains_both_zero_slip_rigid_solutions_without_mutating_inputs(estimator):
    inputs = flat_scene()
    snapshot = copy.deepcopy(inputs)
    results, reports = sliding.solve_prepared(*inputs, estimator=estimator)
    assert set(results) == {'mmt', 'obt'}
    for key, method in zip(('mmt', 'obt'), sliding.methods_for(estimator)):
        np.testing.assert_allclose(results[key]['offset_um'], 0., atol=1e-7)
        assert results[key]['method_identifier'].item() == method.identifier
    for before, after in zip(snapshot, inputs):
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])


def test_calculation_exports_explicit_endpoints_and_keeps_recipes_distinct(tmp_path, monkeypatch):
    case = dict(name='synthetic', tracks_root=str(tmp_path/'tracks'), measurements_root=str(tmp_path/'measures'),
                missing_prerequisites=[], cutoff=dict(depth_mm=.05, depth_um=50.))
    prepared = flat_scene()
    metadata = dict(provenance={}, cutoff=case['cutoff'])
    guard = dict(source=case['tracks_root']+'/synthetic')
    monkeypatch.setattr(sliding.contact_inputs, 'prepare_guarded', lambda *args: (*prepared, metadata, guard))
    report = sliding.calculate(case, tmp_path/'output')
    folder = tmp_path/'output'/'synthetic'
    assert report['configuration']['estimator'] == 'bounded'
    assert [row['method'] for row in report['methods']] == ['MMT-v2', 'OBT-v2']
    assert (folder/'MMT-v2.npz').exists() and (folder/'OBT-v2.npz').exists()
    assert (folder/'comparison.png').exists() and (folder/'comparison.csv').exists()
    sliding.write_index(tmp_path/'output', [case], {'synthetic': dict(stage='complete')})
    summary = read_json(tmp_path/'output'/'summary.json')
    assert {row['method'] for row in summary} == {'mmt', 'obt'}
    assert all(row['last_estimate_depth_um'] < row['cutoff_um'] for row in summary)
    with pytest.raises(RuntimeError, match='different recipe'):
        sliding.calculate(case, tmp_path/'output', estimator='recovery')


def test_report_only_keeps_solver_and_manifest_unchanged_and_marks_gap_lines(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    case = dict(name='saved', tracks_root='tracks', measurements_root='measures', cutoff=dict(depth_mm=.4, depth_um=400.))
    folder = tmp_path/'saved'
    folder.mkdir()
    write_json(tmp_path/'manifest.json', dict(cases=[case], estimator='recovery', audit_contact_zero=True))
    methods = sliding.RECOVERY_METHODS
    write_json(folder/'complete.json', dict(methods=[dict(method=m.identifier) for m in methods],
        configuration=dict(estimator='recovery')))
    source_files = [tmp_path/'manifest.json', folder/'complete.json']
    for method in methods:
        path = folder/(method.identifier+'.npz')
        arrays = dict(depth_mm=np.arange(4)*.1, offset_um=np.array([0., np.nan, 20., np.nan]),
            sigma_um=np.ones(4), frames=np.arange(4), times_s=np.arange(4)*.025, um_per_px=np.array(1.),
            method_identifier=np.array(method.identifier), transport_state=np.array([5, 0, 2, 0]),
            transport_offset_available=np.array([True, False, True, False]),
            transport_absolute_used=np.array([False, False, True, False]), transport_velocity_used=np.zeros(4, bool))
        np.savez_compressed(path, **arrays)
        source_files.append(path)
    before = {path: path.read_bytes() for path in source_files}
    captured, original = [], plt.subplots
    def subplots(*args, **kwargs):
        figure, axes = original(*args, **kwargs)
        captured.extend(axes)
        return figure, axes
    monkeypatch.setattr(plt, 'subplots', subplots)
    monkeypatch.setattr(sliding, 'calculate', lambda *a, **kw: pytest.fail('Reporting must not rerun estimation'))
    sliding.main(['--report-only', '--out', str(tmp_path)])
    assert all(path.read_bytes() == value for path, value in before.items())
    report = np.load(folder/(methods[0].identifier+'.report.npz'))
    np.testing.assert_allclose(report['offset_completed_um'], [0., 10., 20., np.nan], equal_nan=True)
    np.testing.assert_array_equal(report['estimate_kind'], [1, 2, 1, 0])
    assert np.isnan(report['completed_sigma_um'][1])
    assert all(any(line.get_label() == 'Conditional interpolation' and line.get_linestyle() == '--'
        for line in ax.lines) for ax in captured)


def test_batch_failure_returns_nonzero_to_its_caller(tmp_path, monkeypatch):
    case = dict(name='case', tracks_root=str(tmp_path/'tracks'), measurements_root=str(tmp_path/'measures'),
        cutoff=dict(depth_mm=1., depth_um=1000.), missing_prerequisites=[])
    monkeypatch.setattr(sliding, 'load_cases', lambda *args, **kwargs: [case])
    monkeypatch.setattr(sliding.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=7))
    assert sliding.main(['--names', 'case', '--stop-mm', '1', '--out', str(tmp_path/'out')]) == 1
    assert read_json(tmp_path/'out'/'status.json')['cases']['case']['stage'] == 'failed'
