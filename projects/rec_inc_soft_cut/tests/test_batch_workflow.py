"""Batch contracts: reuse, failure isolation, bounded work and honest status."""
import json
import hashlib
from dataclasses import asdict
import importlib
from pathlib import Path
from threading import Lock
import time
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest
import numpy as np

from src.workflows import batch as runner


def test_saved_track_verification_checks_real_hashes_and_rejects_changed_masks(monkeypatch, tmp_path):
    from src.tracking.material_patch_tracker import MaterialPatchConfig
    name = '1014fixture'
    case = make_case(tmp_path, name)
    folder = case['tracks_root']/name
    (folder/'tracking').mkdir(parents=True)
    masks = folder/'segmentation'/name/'semantic_masks'
    masks.mkdir(parents=True)
    source = tmp_path/'imported'
    source.mkdir()
    monkeypatch.setattr(runner, 'video_directory', lambda _: source)
    (source/f'{name}.json').write_text(json.dumps(dict(output_sha256='source-identity')))
    (folder/'plan.json').write_text(json.dumps(dict(source_sha256='source-identity',
        target_depth_mm=3., contact_to_reversal_mm=3., query_scope='full_material',
        full_material_coarse_pitch_px=24, reversal_frame=3)))
    mapping, ids = np.arange(4), np.array([13, 97], np.int64)
    seed = np.array([[10., 20.], [30., 40.]], np.float32)
    np.savez(folder/'queries.npz', seed_points=seed, point_ids=ids)
    np.savez(folder/'tracking/tracks.npz', positions=np.repeat(seed[None], 4, axis=0),
             point_ids=ids, source_frame_indices=mapping)
    normalized = folder/'normalized_lossless.mkv'
    normalized.write_bytes(b'fixture normalization bytes; verification does not decode this stream')
    for index in mapping:
        (masks/f'f{index:06d}.png').write_bytes(bytes([index]))
    specification = dict(video_sha256=runner.sha(normalized),
        seed_sha256=hashlib.sha256(seed.tobytes()).hexdigest(),
        ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        semantic_masks_sha256=runner.mask_fingerprint(masks, mapping))
    (folder/'tracking/complete.json').write_text(json.dumps(dict(
        method='material_independent', neural_proposals=False, config=asdict(MaterialPatchConfig()),
        code_sha256=runner.sha(runner.ROOT/'src/tracking/material_patch_tracker.py'),
        specification=specification)))
    verified = runner.verify_saved_tracks(case)
    assert verified == dict(reused=True, frames=4, points=2,
        track_sha256=runner.sha(folder/'tracking/tracks.npz'))
    (masks/'f000002.png').write_bytes(b'changed support mask')
    with pytest.raises(ValueError, match='Segmentation masks differ'):
        runner.verify_saved_tracks(case)


def make_case(root, name):
    return dict(name=name, tracks_root=root/'tracks', measurements_root=root/'measurements',
                review_root=root/'review', stroke_status='assumed')


@pytest.mark.parametrize('action', ['prepare', 'track', 'tracking_video', 'measure', 'review', 'dual_contact'])
def test_batch_commands_reach_actual_cli_parsers_without_losing_inputs(monkeypatch, tmp_path, action):
    import softcut
    from src.workflows import tracking, measurement, motion, sliding, sliding_render
    case = make_case(tmp_path, '1014fixture')
    case['contact_analysis_stop_mm'] = .85
    output = tmp_path/'batch'
    handlers = {}
    for module, names in ((tracking, ('prepare', 'track', 'render')),
                          (measurement, ('measure',)), (motion, ('measure', 'render')),
                          (sliding, ('calculate',)), (sliding_render, ('render',))):
        for name in names:
            handler = Mock()
            handlers[module.__name__+'.'+name] = handler
            monkeypatch.setattr(module, name, handler)
    monkeypatch.setattr(motion, 'SOURCE', tmp_path/'unused-source')
    monkeypatch.setattr(motion, 'MEASUREMENTS', tmp_path/'unused-measurements')
    command = runner.command_for(case, action, output)
    softcut.main(command[3:])
    if action in ('prepare', 'track', 'tracking_video'):
        key = 'render' if action == 'tracking_video' else action
        call = handlers['src.workflows.tracking.'+key].call_args
        assert call.args[0] == case['name']
        assert call.args[1] == case['tracks_root']/case['name']
        assert call.args[2].target_mm == 3. and call.args[2].query_scope == 'full_material'
    elif action == 'measure':
        args = handlers['src.workflows.measurement.measure'].call_args.args[1]
        assert args.source == case['tracks_root'] and args.out == case['measurements_root']
    elif action == 'review':
        assert motion.SOURCE == case['tracks_root'] and motion.MEASUREMENTS == case['measurements_root']
        assert handlers['src.workflows.motion.measure'].call_args.args[1].out == case['review_root']
        handlers['src.workflows.motion.render'].assert_called_once()
    else:
        call = handlers['src.workflows.sliding.calculate'].call_args
        assert call.args[0]['cutoff']['depth_mm'] == .85
        assert Path(call.args[0]['tracks_root']) == case['tracks_root']
        assert Path(call.args[0]['measurements_root']) == case['measurements_root']
        assert call.args[1] == output/'dual_contact'
        assert call.kwargs['estimator'] == 'bounded' and call.kwargs['overwrite']
        render = handlers['src.workflows.sliding_render.render'].call_args
        assert render.args[0] == output/'dual_contact'/case['name']
        assert render.kwargs['maximum_depth_mm'] == .85


def test_manifest_uses_local_roots_and_keeps_external_reference_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    manifest = tmp_path/'cases.json'
    manifest.write_text(json.dumps(dict(cases=[dict(name='0930l60c2up', microscopy_offset_um=430,
        microscopy_measurement_state='unconfirmed_recovery_state', tracks_root='results/reused')])))
    case = runner.load_cases(manifest, tmp_path/'results/new')[0]
    assert case['tracks_root'] == tmp_path/'results/reused'
    assert case['review_root'] == tmp_path/'results/new/review'
    assert case['microscopy_offset_um'] == 430
    assert 'fracture_depth_mm' not in case
    manifest.write_text(json.dumps(dict(cases=[dict(name='0930l60c2up', tracks_root='../outside')])) )
    with pytest.raises(ValueError):
        runner.load_cases(manifest, tmp_path/'results/new')


@pytest.mark.parametrize('names', [['0930../escape'], ['0930l30a1.mp4'], ['0930a', '0930a'], ['0930:C'], []])
def test_invalid_or_duplicate_case_names_never_become_paths(names):
    with pytest.raises(ValueError):
        runner.validate_case_names(names)


@pytest.mark.parametrize('module_name', ['src.workflows.measurement', 'src.workflows.motion'])
def test_measurement_worker_cli_accepts_new_case_without_parent_state(monkeypatch, tmp_path, module_name):
    module = importlib.import_module(module_name)
    measure = Mock()
    monkeypatch.setattr(module, 'measure', measure)
    if module_name == 'src.workflows.motion':
        monkeypatch.setattr(module, 'SOURCE', tmp_path/'source')
        monkeypatch.setattr(module, 'MEASUREMENTS', tmp_path/'measurements')
    monkeypatch.setattr(sys, 'argv', [module_name+'.py', '--worker', '--stage', 'measure',
        '--names', '0930l60c2down', '--source', str(tmp_path/'source'), '--out', str(tmp_path/'review')])
    module.main()
    assert measure.call_args.args[0] == '0930l60c2down'


def test_stage_commands_preserve_accepted_settings_and_reuse_sources(tmp_path):
    case = make_case(tmp_path, '0930l60c2down')
    prepare = runner.command_for(case, 'prepare', tmp_path/'new')
    assert prepare[prepare.index('--names')+1] == '0930l60c2down'
    assert prepare[prepare.index('--query-scope')+1] == 'full_material'
    assert prepare[prepare.index('--target-mm')+1] == '3'
    assert prepare[prepare.index('--outer-pitch')+1] == '24'
    assert '--full-video' not in prepare
    review = runner.command_for(case, 'review', tmp_path/'new')
    assert review[review.index('--source')+1] == str(case['tracks_root'])
    assert review[review.index('--measurements')+1] == str(case['measurements_root'])
    assert review[review.index('--scope')+1] == 'interface'
    dual = runner.command_for(make_case(tmp_path, '0930l60a1'), 'dual_contact', tmp_path/'new')
    assert Path(dual[2]).name == 'softcut.py' and dual[3] == 'slip'
    assert dual[dual.index('--stop-mm')+1] == '1.4'
    assert '--worker' in dual and '--overwrite' in dual and '--render' in dual
    case['contact_analysis_stop_mm'] = 1.1
    dual = runner.command_for(case, 'dual_contact', tmp_path/'new')
    assert dual[dual.index('--stop-mm')+1] == '1.1'


def test_gallery_shows_only_successful_videos_with_cross_root_links(tmp_path):
    out = tmp_path/'new'
    out.mkdir()
    case = make_case(tmp_path/'reused with spaces', '0930l60a1')
    case['microscopy_offset_um'] = 430
    folder = case['review_root']/case['name']
    folder.mkdir(parents=True)
    (folder/'complete.json').write_text('{}')
    (folder/f'{case["name"]}_loading_interface_sliding.mp4').write_bytes(b'fixture')
    runner.write_index(out, [case], {case['name']:dict(stage='failed', error='source <changed>')}, 'completed_with_failures')
    page = (out/'index.html').read_text(encoding='utf-8')
    assert '<video ' not in page
    assert 'source &lt;changed&gt;' in page
    assert 'microscopy_reference' not in page and 'MMT-v1' not in page
    runner.write_index(out, [case], {case['name']:dict(stage='complete')}, 'complete')
    page = (out/'index.html').read_text(encoding='utf-8')
    assert '<video ' in page and '../reused%20with%20spaces/' in page
    assert 'microscopy_reference' not in page and 'MMT-v1' not in page


def test_scheduler_reuses_tracks_and_isolates_failure_before_dependent_review(monkeypatch, tmp_path):
    cases = [make_case(tmp_path, f'0930a{i}') for i in range(4)]
    cases[0]['contact_analysis_stop_mm'] = 1.3
    cases[2]['contact_analysis_stop_mm'] = 2.2
    existing = cases[0]['tracks_root']/cases[0]['name']/'tracking/tracks.npz'
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'existing saved trajectories')
    args = SimpleNamespace(out=tmp_path, jobs=2, tracking_videos=False)
    batch = runner.Batch(args, cases)
    calls, active, maximum = [], {'gpu':0, 'cpu':0}, {'gpu':0, 'cpu':0}
    lock = Lock()
    def preflight(*args, **kwargs):
        assert (tmp_path/'index.html').is_file()
        assert runner.read_json(tmp_path/'status.json')['stage'] == 'running'
    monkeypatch.setattr(runner, 'preflight', preflight)
    monkeypatch.setattr(runner, 'verify_saved_tracks', lambda case: dict(reused=True))
    monkeypatch.setattr(runner, 'verify_review_video', lambda case: dict(first_frame_decoded=True, last_frame_decoded=True))
    def stage(case, action):
        kind = 'gpu' if action == 'prepare' else 'cpu'
        with lock:
            calls.append((case['name'], action))
            active[kind] += 1
            maximum[kind] = max(maximum[kind], active[kind])
        time.sleep(.01)
        with lock:
            active[kind] -= 1
        if case['name'] == '0930a1' and action == 'measure':
            raise RuntimeError('deliberate measurement failure')
        if action == 'review':
            folder = case['review_root']/case['name']
            folder.mkdir(parents=True)
            (folder/'complete.json').write_text('{}')
            (folder/f'{case["name"]}_loading_interface_sliding.mp4').write_bytes(b'complete video fixture')
        if action == 'dual_contact':
            folder = tmp_path/'dual_contact'/case['name']
            folder.mkdir(parents=True)
            for file in ('complete.json', 'MMT-v2.npz', 'OBT-v2.npz', 'comparison.csv', 'repaired_contact.mp4'):
                (folder/file).write_bytes(b'complete fixture')
    monkeypatch.setattr(batch, 'stage', stage)
    assert batch.run() == 1
    assert ('0930a0', 'track') not in calls and ('0930a0', 'prepare') not in calls
    assert ('0930a1', 'review') not in calls
    assert ('0930a1', 'dual_contact') not in calls
    assert all((f'0930a{i}', 'dual_contact') in calls for i in (0, 2, 3))
    assert batch.states['0930a1']['stage'] == 'failed'
    assert all(batch.states[f'0930a{i}']['stage'] == 'complete' for i in (0, 2, 3))
    assert maximum == {'gpu':1, 'cpu':2}
    assert runner.read_json(tmp_path/'status.json')['stage'] == 'completed_with_failures'
    assert existing.read_bytes() == b'existing saved trajectories'
    # A batch invokes calculation workers directly, but later reporting must
    # still have the supervisor's explicit recipe without re-running tracking.
    from src.workflows.sliding import load_cases as load_slip_cases
    slip_cases = load_slip_cases(tmp_path/'dual_contact/manifest.json')
    assert [case['name'] for case in slip_cases] == [case['name'] for case in cases]
    assert [case['cutoff'] for case in slip_cases] == [
        dict(source='explicit_cli_cutoff', depth_mm=depth, depth_um=depth*1000.)
        for depth in (1.3, 3., 2.2, 3.)]
    assert all(Path(case['tracks_root']) == tmp_path/'tracks' for case in slip_cases)


def test_failed_import_is_recovered_and_recorded_before_processing(monkeypatch, tmp_path):
    case = dict(name='1014a1', tracks_root=tmp_path/'tracks')
    (tmp_path/'import_status.json').write_text(json.dumps(dict(pid=1, cases={'1014a1': dict(stage='failed', error='read stopped')})))
    (tmp_path/'import_manifest.json').write_text(json.dumps(dict(cases=[dict(name='1014a1', source='source.avi', data_directory='data')])) )
    job = runner.Batch(SimpleNamespace(out=tmp_path, jobs=2), [case])
    events = []
    monkeypatch.setattr(job, 'update', lambda case, **kw: events.append(kw))
    monkeypatch.setattr(runner, 'import_one', lambda *args: dict(frames=100, output_sha256='verified'))
    job.await_import(case)
    assert (tmp_path/'batch/1014a1/import_recovered.json').is_file()
    assert events[-1]['stage'] == 'retrying_import'


def test_failed_retry_does_not_mark_import_ready(monkeypatch, tmp_path):
    case = dict(name='1014a1', tracks_root=tmp_path/'tracks')
    (tmp_path/'import_status.json').write_text(json.dumps(dict(pid=1, cases={'1014a1': dict(stage='failed', error='read stopped')})))
    (tmp_path/'import_manifest.json').write_text(json.dumps(dict(cases=[dict(name='1014a1', source='source.avi', data_directory='data')])) )
    job = runner.Batch(SimpleNamespace(out=tmp_path, jobs=2), [case])
    monkeypatch.setattr(job, 'update', lambda *args, **kw: None)
    def fail(*args):
        raise RuntimeError('unreadable source frame')
    monkeypatch.setattr(runner, 'import_one', fail)
    with pytest.raises(RuntimeError, match='unreadable'):
        job.await_import(case)
    assert not (tmp_path/'batch/1014a1/import_recovered.json').exists()


def test_gallery_does_not_link_in_progress_encodes(tmp_path):
    case = dict(name='1014a1', tracks_root=tmp_path/'tracks', review_root=tmp_path/'review')
    folder = case['tracks_root']/case['name']
    folder.mkdir(parents=True)
    (folder/'1014a1_material_tracking.mp4').write_bytes(b'incomplete encoding')
    runner.write_index(tmp_path, [case], {'1014a1': dict(stage='tracking_video')}, 'running', {})
    assert '1014a1_material_tracking.mp4' not in (tmp_path/'index.html').read_text(encoding='utf-8')
    runner.write_index(tmp_path, [case], {'1014a1': dict(stage='measure')}, 'running', {})
    assert '1014a1_material_tracking.mp4' in (tmp_path/'index.html').read_text(encoding='utf-8')


@pytest.mark.parametrize('cutoff', [0, -1, 3.1, float('nan'), float('inf')])
def test_invalid_analysis_cutoff_cannot_enter_batch(monkeypatch, tmp_path, cutoff):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    manifest = tmp_path/'cases.json'
    manifest.write_text(json.dumps(dict(cases=[dict(name='1014l57a1', contact_analysis_stop_mm=cutoff)])))
    with pytest.raises(ValueError, match='cutoff'):
        runner.load_cases(manifest, tmp_path/'results/new')


def test_manifest_cutoff_is_not_replaced_by_case_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    manifest = tmp_path/'cases.json'
    manifest.write_text(json.dumps(dict(cases=[dict(name='0930l60a1', contact_analysis_stop_mm=2.2),
        dict(name='1014l57a1', contact_analysis_stop_mm=.55), dict(name='0930l30a2')])))
    cases = runner.load_cases(manifest, tmp_path/'results/new')
    assert [case['contact_analysis_stop_mm'] for case in cases] == [2.2, .55, 1.]
    assert [float(runner.command_for(case, 'dual_contact', tmp_path)[-3]) for case in cases] == [2.2, .55, 1.]


def test_check_only_never_creates_output_or_starts_import(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    manifest = tmp_path/'cases.json'
    manifest.write_text(json.dumps(dict(cases=[dict(name='1014l57a1')])))
    preflight = Mock()
    monkeypatch.setattr(runner, 'preflight', preflight)
    monkeypatch.setattr(runner.Batch, 'run', lambda _: pytest.fail('read-only check started batch'))
    target = tmp_path/'results/new'
    assert runner.main(['--manifest', str(manifest), '--out', str(target), '--check-only', '--await-import']) == 0
    assert preflight.call_count == 1
    assert not target.exists()
