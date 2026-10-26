"""Manifest batch: verified imports, segmentation, tracking and both slip methods.

One GPU preparation runs at a time; CPU case chains have bounded concurrency.
Saved tracks are reused only after provenance and material-ID verification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time
from urllib.parse import quote

import psutil

from src.io import ROOT, HIDDEN, read_json, write_json, sha
from src.video_sources import video_directory, validate_case_names
from src.workflows.tracking import preflight, mask_fingerprint

DEFAULT_OUT = ROOT / 'results/current_batch'
DEFAULT_CUTOFFS = {f'0930{kind}a{repeat}': depth
                  for kind, depth in (('l30', 1.), ('l60', 1.4), ('n60', 1.5))
                  for repeat in (1, 2)}


def results_path(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    path.relative_to((ROOT/'results').resolve())
    return path


def load_cases(manifest, out):
    record = read_json(manifest)
    cases = record.get('cases', [])
    validate_case_names([case['name'] for case in cases])
    result = []
    for entry in cases:
        if entry.get('contact_to_reversal_mm', 3) != 3:
            raise ValueError('This accepted batch requires approximately 3 mm contact-to-reversal calibration')
        if entry.get('stroke_status', 'assumed') not in ('assumed', 'user_confirmed'):
            raise ValueError('stroke_status must be assumed or user_confirmed')
        case = dict(entry, contact_to_reversal_mm=3, stroke_status=entry.get('stroke_status', 'assumed'))
        for field, child in [('tracks_root', 'tracks'), ('measurements_root', 'measurements'), ('review_root', 'review')]:
            case[field] = results_path(entry.get(field, out/child))
        if len({case[key] for key in ('tracks_root', 'measurements_root', 'review_root')}) != 3:
            raise ValueError('Tracking, measurement and review roots must be different')
        cutoff = case.get('contact_analysis_stop_mm', DEFAULT_CUTOFFS.get(case['name'], 3.))
        if not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) or not 0 < cutoff <= 3:
            raise ValueError('Contact analysis cutoff must be a finite depth in (0, 3] mm')
        case['contact_analysis_stop_mm'] = float(cutoff)
        result.append(case)
    return result


def verify_saved_tracks(case):
    """A declared reuse source must contain the accepted full-material stroke."""
    import numpy as np
    from dataclasses import asdict
    from src.tracking.material_patch_tracker import MaterialPatchConfig
    folder = case['tracks_root']/case['name']
    plan = read_json(folder/'plan.json')
    complete = read_json(folder/'tracking/complete.json')
    imported = read_json(video_directory(case['name'])/f'{case["name"]}.json')
    if (plan.get('source_sha256') != imported['output_sha256'] or plan.get('target_depth_mm') != 3
            or plan.get('contact_to_reversal_mm') != 3 or plan.get('query_scope') != 'full_material'
            or plan.get('full_material_coarse_pitch_px') != 24):
        raise ValueError('Existing tracks do not match the accepted local 3 mm/full-material inputs')
    if complete.get('method') != 'material_independent' or complete.get('neural_proposals'):
        raise ValueError('Only independent material trajectories can be reused')
    expected_code = sha(ROOT/'src/tracking/material_patch_tracker.py')
    if complete.get('code_sha256') != expected_code:
        raise ValueError('Saved material tracker differs from the current accepted implementation')
    if complete.get('config') != asdict(MaterialPatchConfig()):
        raise ValueError('Saved tracker parameters differ from the accepted defaults')
    if sha(folder/'normalized_lossless.mkv') != complete['specification']['video_sha256']:
        raise ValueError('Saved normalization differs from the tracked image source')
    with np.load(folder/'queries.npz', allow_pickle=False) as queries, np.load(folder/'tracking/tracks.npz', allow_pickle=False) as tracks:
        ids, mapping = tracks['point_ids'], tracks['source_frame_indices']
        if not np.array_equal(ids, queries['point_ids']):
            raise ValueError('Original material IDs differ between tracks and queries')
        if (hashlib.sha256(queries['seed_points'].tobytes()).hexdigest() != complete['specification']['seed_sha256']
                or hashlib.sha256(ids.tobytes()).hexdigest() != complete['specification']['ids_sha256']):
            raise ValueError('Saved seed coordinates or identity bytes changed')
        if not np.array_equal(mapping, np.arange(len(mapping))) or len(mapping) <= plan['reversal_frame']:
            raise ValueError('Saved trajectories do not cover a contiguous complete loading stroke')
        if tracks['positions'].shape != (len(mapping), len(ids), 2):
            raise ValueError('Saved trajectory dimensions disagree with frame and identity mappings')
    masks = mask_fingerprint(folder/'segmentation'/case['name']/'semantic_masks', mapping)
    if masks != complete['specification'].get('semantic_masks_sha256'):
        raise ValueError('Segmentation masks differ from the tracked material/blade support')
    return dict(reused=True, track_sha256=sha(folder/'tracking/tracks.npz'), frames=len(mapping), points=len(ids))


def command_for(case, action, out):
    name = case['name']
    base = [sys.executable, '-u', str(ROOT/'softcut.py')]
    if action in ('prepare', 'track', 'tracking_video'):
        stage = 'render' if action == 'tracking_video' else action
        return [*base, 'track', '--worker', '--names', name, '--stage', stage,
            '--out', str(case['tracks_root']), '--reuse-root', str(out/'calibration'),
            '--target-mm', '3', '--query-scope', 'full_material', '--outer-pitch', '24']
    if action == 'measure':
        return [*base, 'measure', '--worker', '--names', name, '--stage', 'measure',
            '--source', str(case['tracks_root']), '--out', str(case['measurements_root'])]
    if action == 'review':
        return [*base, 'motion', '--worker', '--names', name,
            '--source', str(case['tracks_root']), '--measurements', str(case['measurements_root']),
            '--out', str(case['review_root']), '--scope', 'interface', '--stage', 'all']
    if action == 'dual_contact':
        cutoff = case.get('contact_analysis_stop_mm', DEFAULT_CUTOFFS.get(name, 3.))
        return [*base, 'slip', '--worker', '--names', name,
            '--source', str(case['tracks_root']), '--measurements', str(case['measurements_root']),
            '--out', str(out/'dual_contact'), '--stop-mm', str(cutoff), '--render', '--overwrite']
    raise ValueError(f'Unknown batch stage: {action}')


def link(path, out):
    return html.escape(quote(os.path.relpath(path, out).replace('\\', '/'), safe='/'), quote=True)


def write_index(out, cases, states, overall, imports=None):
    """One status page for existing inputs and concurrently imported inputs."""
    def anchor(path, label):
        return f'<a href="{link(path, out)}">{html.escape(label)}</a>' if path.is_file() else ''
    cards = []
    for case in cases:
        name = case['name']
        state = states.get(name, dict(stage='pending'))
        stage = state['stage']
        source, review = (case[key]/name for key in ('tracks_root', 'review_root'))
        dual = out/'dual_contact'/name
        links, media = [], ''
        if (source/'tracking/complete.json').is_file():
            links.append(anchor(source/'tracking/tracks.npz', 'Fixed material-ID tracks'))
        if stage in ('tracking_video_complete', 'measure', 'measure_complete', 'review', 'review_complete',
                     'dual_contact', 'dual_contact_complete', 'complete'):
            links.extend(anchor(source/f'{name}_{suffix}.mp4', title) for suffix, title in
                         (('material_tracking', 'Full material view'), ('tip_tracking', 'Blade-tip view')))
        if stage == 'complete' and (review/'complete.json').is_file():
            video = review/f'{name}_loading_interface_sliding.mp4'
            if video.is_file():
                media = f'<video controls preload="none" src="{link(video, out)}"></video>'
            links.extend(anchor(review/file, label) for file, label in (
                ('surface_sliding.csv', 'Surface address'), ('loading_trends.csv', 'Relative motion'),
                ('interface_segments.csv', 'Blade-interface segments'), ('measurement_summary.json', 'Measurement summary')))
            if (dual/'complete.json').is_file():
                links.extend(anchor(dual/f'{method}.npz', method+' NPZ') for method in ('MMT-v2', 'OBT-v2'))
                links.extend((anchor(dual/'comparison.csv', 'Both slip curves'),
                              anchor(dual/'complete.json', 'Slip provenance')))
                links.append(anchor(dual/'repaired_contact.mp4', 'Both slip methods video'))
        detail = f'<p>Status: {html.escape(stage)}</p>'
        if imports is not None:
            detail += f'<p>Import: {html.escape(imports.get(name, {}).get("stage", "pending"))}</p>'
        cutoff = case.get('contact_analysis_stop_mm', DEFAULT_CUTOFFS.get(name, 3.))
        detail += f'<p>Analysis cutoff: {cutoff:g} mm; contact-to-reversal calibration: approximately 3 mm.</p>'
        if state.get('error'):
            detail += f'<pre>{html.escape(state["error"])}</pre>'
        cards.append(f'<section><h2>{html.escape(name)}</h2>{detail}{media}<p>{" ? ".join(filter(None, links))}</p></section>')
    refresh = '<meta http-equiv="refresh" content="45">' if overall == 'running' else ''
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        + refresh + '<title>Material tracking and surface slip</title><style>'
        'body{max-width:1500px;margin:28px auto;padding:0 22px;background:#121a1d;color:#e1ecec;font:17px/1.7 system-ui}'
        'video{width:100%;background:#080d10}a{color:#77d7e9}section{margin:34px 0;border-top:1px solid #385059}'
        'pre{white-space:pre-wrap;color:#ffb3aa}</style><h1>Material tracking and surface slip</h1>'
        f'<p>{len(cases)} cases; batch status: {html.escape(overall)}.</p>'
        '<p>Loading only, first contact = 0. Mixed-model transport and Objective boundary transport '
        'retain separate estimates. Unavailable observations are not validated measurements; model sensitivity '
        'is not a statistical confidence interval. Playback frame rate is not the acquisition clock.</p>'
        '<p><a href="status.json">Status</a> ? <a href="manifest.json">Inputs and analysis cutoffs</a></p>'
        + ''.join(cards) + '</html>')
    temporary = out/'index.encoding.html'
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(out/'index.html')


def verify_review_video(case):
    """Check actual H.264 stream metadata and decode both endpoint frames."""
    import cv2
    folder = case['review_root']/case['name']
    video = folder/f'{case["name"]}_loading_interface_sliding.mp4'
    result = read_json(folder/'complete.json')
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
               'stream=codec_name,width,height,nb_frames,r_frame_rate', '-of', 'json', str(video)]
    probe = subprocess.run(command, check=True, capture_output=True, text=True, creationflags=HIDDEN)
    stream = json.loads(probe.stdout)['streams'][0]
    count = int(stream['nb_frames'])
    numerator, denominator = map(float, stream['r_frame_rate'].split('/'))
    if (stream['codec_name'] != 'h264' or count != result['frames'] or (stream['width'], stream['height']) != (1800, 1200)
            or abs(numerator/denominator-12) > 1e-5):
        raise ValueError('Review stream dimensions, codec or frame count disagree with the current complete record')
    cap = cv2.VideoCapture(str(video))
    try:
        first, _ = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, count-1)
        last, _ = cap.read()
    finally:
        cap.release()
    if not first or not last:
        raise ValueError('Review first/last frame is not decodable')
    return dict(**stream, first_frame_decoded=True, last_frame_decoded=True, bytes=video.stat().st_size)


def import_one(source, destination, report):
    # Importing the batch module itself must not access source AVI drives.
    from src.workflows.video_import import import_one as import_video
    return import_video(source, destination, report)


class Batch:
    def __init__(self, args, cases):
        self.args, self.cases = args, cases
        self.states = {case['name']: dict(stage='pending') for case in cases}
        self.lock = Lock()
        self.env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                        OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', OMP_NUM_THREADS='2')

    def snapshot(self, overall='running'):
        imports = None
        if getattr(self.args, 'await_import', False):
            imports = read_json(self.args.out/'import_status.json').get('cases', {})
            for name in imports:
                if (self.args.out/'batch'/name/'import_recovered.json').is_file():
                    imports[name] = dict(stage='complete', recovered=True)
        write_json(self.args.out/'status.json', dict(stage=overall, updated=time.time(), pid=os.getpid(),
            cases=self.states, target_mm=3, scope='loading_only_full_material_interface_sliding',
            gpu_preparation_jobs=1, cpu_case_jobs=self.args.jobs))
        write_index(self.args.out, self.cases, self.states, overall, imports)

    def update(self, case, **status):
        with self.lock:
            self.states[case['name']].update(status, updated=time.time())
            self.snapshot()

    def await_import(self, case):
        self.update(case, stage='waiting_for_import')
        while True:
            report = read_json(self.args.out/'import_status.json')
            imported = report['cases'][case['name']]
            if imported['stage'] == 'complete':
                return
            if imported['stage'] == 'failed':
                self.update(case, stage='retrying_import', original_import_error=imported.get('error'))
                inventory = read_json(self.args.out/'import_manifest.json')['cases']
                entry = next(item for item in inventory if item['name'] == case['name'])
                result = import_one(Path(entry['source']), Path(entry['data_directory']),
                    lambda state: self.update(case, stage='retrying_import', import_progress=state))
                destination = self.args.out/'batch'/case['name']/'import_recovered.json'
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_json(destination, result)
                return
            if not psutil.pid_exists(report['pid']):
                raise RuntimeError('Importer stopped before this case was completed')
            time.sleep(15)

    def stage(self, case, action):
        folder = self.args.out/'batch'/case['name']
        folder.mkdir(parents=True, exist_ok=True)
        command = command_for(case, action, self.args.out)
        with (folder/f'{action}.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=log,
                                       stderr=subprocess.STDOUT, creationflags=HIDDEN)
            self.update(case, stage=action, pid=process.pid, log=str(folder/f'{action}.log'))
            code = process.wait()
        if code:
            raise RuntimeError(f'{action} failed with exit code {code}; see {folder/action}.log')
        self.update(case, stage=action+'_complete', pid=None)

    def cpu_case(self, case, reused):
        try:
            if not reused:
                self.stage(case, 'track')
                self.update(case, track_verification=verify_saved_tracks(case), tracks_reused=False)
            if self.args.tracking_videos:
                self.stage(case, 'tracking_video')
            self.stage(case, 'measure')
            self.stage(case, 'review')
            validation = verify_review_video(case)
            self.stage(case, 'dual_contact')
            dual = self.args.out/'dual_contact'/case['name']
            for file in ('complete.json', 'MMT-v2.npz', 'OBT-v2.npz', 'comparison.csv', 'repaired_contact.mp4'):
                if not (dual/file).is_file():
                    raise RuntimeError(f'Slip stage returned without {dual/file}')
            self.update(case, stage='complete', pid=None, video_validation=validation)
        except Exception as error:
            self.update(case, stage='failed', pid=None, error=str(error))

    def run(self):
        # Worker commands bypass the sliding supervisor. Preserve the same
        # explicit inputs/cutoffs so its later --report-only path is usable.
        slip_cases = []
        for case in self.cases:
            cutoff = float(case.get('contact_analysis_stop_mm', DEFAULT_CUTOFFS.get(case['name'], 3.)))
            slip_cases.append(dict(name=case['name'],
                tracks_root=str(case['tracks_root'].resolve()),
                measurements_root=str(case['measurements_root'].resolve()),
                cutoff=dict(source='explicit_cli_cutoff', depth_mm=cutoff, depth_um=cutoff*1000)))
        write_json(self.args.out/'dual_contact/manifest.json', dict(cases=slip_cases,
            estimator='bounded', audit_contact_zero=False))
        with self.lock:
            self.snapshot()
        with ThreadPoolExecutor(max_workers=self.args.jobs) as pool:
            futures = []
            for case in self.cases:
                try:
                    if getattr(self.args, 'await_import', False):
                        self.await_import(case)
                    self.update(case, stage='preflight')
                    preflight([case['name']], target_mm=3,
                              stroke_statuses={case['name']: case['stroke_status']})
                    reused = (case['tracks_root']/case['name']/'tracking/tracks.npz').is_file()
                    if reused:
                        self.update(case, stage='tracks_verified', tracks_reused=True,
                                    track_verification=verify_saved_tracks(case))
                    else:
                        self.stage(case, 'prepare')
                    futures.append(pool.submit(self.cpu_case, case, reused))
                except Exception as error:
                    self.update(case, stage='failed', pid=None, error=str(error))
            for future in futures:
                future.result()
        failed = any(state['stage'] != 'complete' for state in self.states.values())
        with self.lock:
            self.snapshot('completed_with_failures' if failed else 'complete')
        return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--tracking-videos', action='store_true', help='Also render separate full-field/tip videos')
    parser.add_argument('--check-only', action='store_true', help='Read-only source/manifest/model checks')
    parser.add_argument('--await-import', action='store_true', help='Wait for the importer in this output directory')
    args = parser.parse_args(argv)
    args.out = results_path(args.out)
    cases = load_cases(args.manifest, args.out)
    if args.check_only:
        for case in cases:
            preflight([case['name']], tracker='independent', target_mm=3,
                      stroke_statuses={case['name']:case['stroke_status']})
            if (case['tracks_root']/case['name']/'tracking/tracks.npz').is_file():
                print(json.dumps(dict(case=case['name'], **verify_saved_tracks(case))))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    previous = args.out/'status.json'
    if previous.is_file():
        status = read_json(previous)
        if status.get('stage') == 'running' and status.get('pid') and psutil.pid_exists(status['pid']):
            raise RuntimeError('This output already has an active batch supervisor')
    manifest = read_json(args.manifest)
    manifest['cases'] = [{key: str(value) if isinstance(value, Path) else value
        for key, value in case.items()} for case in cases]
    write_json(args.out/'manifest.json', manifest)
    return Batch(args, cases).run()


if __name__ == '__main__':
    raise SystemExit(main())
