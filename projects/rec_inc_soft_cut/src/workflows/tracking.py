"""Independent material tracking: segmentation, normalized frames and fixed IDs.

Segmentation uses the existing checkpoint. Tracking uses local affine patches,
multiple references and cycle checks; there is no neural-flow candidate path.
GPU preparation is serial while at most --jobs CPU tracking cases overlap.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Lock
import time

from src.io import ROOT, HIDDEN, atomic_write_json, sha as digest
from src.video_sources import video_directory, validate_case_names, input_profile

NAMES = ('0930l30a1', '0930l60a1', '0930n60a1')
DEFAULT_OUT = ROOT / 'results/current_material_loading'
FULL_OUT = ROOT / 'results/current_material_full_video'
REUSE_ROOT = ROOT / 'results/current_material_calibration'
STAGES = ('prepare', 'track', 'render', 'compress')


def save(path, value):
    atomic_write_json(Path(path), value, allow_nan=False)




def preflight(names, *, tracker='independent', target_mm=3., full_video=False, reuse_root=None,
              stroke_statuses=None):
    """Read-only checks, including preserved source-frame provenance."""
    import cv2
    import numpy as np

    if tracker != 'independent':
        raise ValueError('Only independent material tracking is supported')
    errors = []
    videos = []
    calibration_root = Path(reuse_root) if reuse_root is not None else REUSE_ROOT
    for name in names:
        if full_video:
            required = ('plan.json', 'geometry.json', 'geometry.npz', 'queries.npz',
                        'original_surface_reference.npz', 'normalized_lossless.mkv')
            missing_reuse = [str(calibration_root/name/p) for p in required
                             if not (calibration_root/name/p).is_file()]
            if missing_reuse:
                errors.append(dict(case=name, missing_verified_calibration=missing_reuse))
        video = video_directory(name) / f'{name}.mp4'
        record = video.with_suffix('.json')
        alignment = video.with_name(name + '_alignment.npz')
        missing = [str(p) for p in (video, record, alignment) if not p.is_file()]
        if missing:
            errors.append(dict(case=name, missing=missing))
            continue
        try:
            metadata = json.loads(record.read_text(encoding='utf-8'))
            profile = input_profile(name)
            if digest(video) != metadata['output_sha256']:
                raise ValueError('Imported video SHA256 differs from its provenance record')
            with np.load(alignment, allow_pickle=False) as mapping:
                source = mapping['source_indices']
                local = mapping['frame_indices']
                expected = (np.arange(len(source)) // 2) * 5 + (np.arange(len(source)) % 2) * 3
                np.testing.assert_array_equal(source, expected)
                np.testing.assert_array_equal(local, np.arange(len(source)))
            cap = cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened():
                    raise ValueError('Local video cannot be opened')
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                rate = float(cap.get(cv2.CAP_PROP_FPS))
                if size != (1024, 768) or abs(rate - 12) > 1e-4 or count != len(source):
                    raise ValueError('Unexpected local frame dimensions/rate/mapping length')
            finally:
                cap.release()
            videos.append(dict(case=name, frames=count, sha256=metadata['output_sha256'],
                               local_input=str(video), assumed_acquisition_fps=profile['acquisition_fps'],
                               physical_time_calibration=profile['acquisition_clock_status'],
                               input_profile=profile,
                               contact_to_reversal_mm=3,
                               target_display_mm=None if full_video else target_mm,
                               processing_scope='full_video' if full_video else 'loading_prefix',
                               stroke_status=(stroke_statuses or {}).get(name,
                                   'user_confirmed' if name in (*NAMES, '0930l60c1up', '0930l60c1down') else 'assumed')))
        except Exception as exc:
            errors.append(dict(case=name, error=str(exc)))
    segmentation = ROOT / 'models/segmentation/clean_roi_20260909/checkpoint_best.pt'
    if not segmentation.is_file():
        errors.append(dict(missing=str(segmentation)))
    for package in ('torch', 'numpy', 'cv2', 'scipy', 'matplotlib', 'psutil', 'yaml'):
        if importlib.util.find_spec(package) is None:
            errors.append(dict(missing_package=package))
    if shutil.which('ffmpeg') is None:
        errors.append(dict(missing_program='ffmpeg'))
    report = dict(ok=not errors, read_only=True, videos=videos, errors=errors,
                  external_drive_access=False, results_required=full_video,
                  calibration_reuse_root=str(calibration_root) if full_video else None,
                  segmentation_checkpoint=str(segmentation),
                  flow_checkpoint=None,
                  tracker='independent', neural_flow_proposals=False,
                  gpu_work_started=False,
                  gpu_required_for=('prepare',),
                  calibration='Approximately 3 mm contact-to-reversal stroke; depth zero is estimated first contact; physical time assumes 200 source fps and still requires acquisition-setting confirmation.')
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise RuntimeError('Local input/model preflight failed; see errors above')
    return report


def mask_fingerprint(directory, mapping):
    """Bind a full-field cache to the semantic images it actually consumed."""
    h = hashlib.sha256()
    for index in mapping:
        path = Path(directory) / f'f{int(index):06d}.png'
        h.update(path.name.encode('ascii'))
        h.update(bytes.fromhex(digest(path)))
    return h.hexdigest()


def validate_cohort(seed, ids, mapping):
    import numpy as np
    if seed.ndim != 2 or seed.shape[1] != 2 or not len(seed) or not np.isfinite(seed).all():
        raise ValueError('Queries must be nonempty, finite initial material coordinates')
    if ids.shape != (len(seed),) or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError('Exactly one unique material ID is required per query')
    if (mapping.ndim != 1 or len(mapping) < 2 or not np.issubdtype(mapping.dtype, np.integer)
            or mapping[0] < 0 or np.any(np.diff(mapping) != 1)):
        raise ValueError('Validation preserves every available consecutive local frame')


def label_reader(directory):
    import cv2
    def read(index):
        label = cv2.imread(str(directory / f'f{int(index):06d}.png'), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(directory / f'f{int(index):06d}.png')
        return label
    return read


def prepared(folder, name):
    import numpy as np
    plan = json.loads((folder / 'plan.json').read_text(encoding='utf-8'))
    video = folder / 'normalized_lossless.mkv'
    with np.load(folder / 'queries.npz', allow_pickle=False) as z:
        seed = z['seed_points'].copy()
        ids = z['point_ids'].copy() if 'point_ids' in z else np.arange(len(seed))
        mapping = z['source_frame_indices'].copy()
    validate_cohort(seed, ids, mapping)
    labels_dir = folder / 'segmentation' / name / 'semantic_masks'
    return plan, video, seed, ids, mapping, labels_dir


def prepare(name, folder, args):
    from src.workflows.tracking_inputs import prepare_case
    previous = args.reuse_root / name
    return prepare_case(name, folder, target_mm=args.target_mm,
                        full_video=args.full_video, query_scope=args.query_scope,
                        coarse_pitch_px=args.outer_pitch,
                        reuse_dir=previous if (previous / 'plan.json').is_file() else None)


def track(name, folder, args):
    import cv2
    import numpy as np
    from src.tracking.material_patch_tracker import MaterialPatchConfig, MaterialPatchTracker
    from src.tracking.fixed_grid import FixedGridProcessor, grid_0915_config
    plan, video, seed, ids, mapping, labels_dir = prepared(folder, name)
    grid = FixedGridProcessor(grid_0915_config()).mask()
    if not plan.get('input_profile', {}).get('grid_enabled', True):
        grid[:] = False
    code_hash = digest(ROOT / 'src/tracking/material_patch_tracker.py')
    variant = 'material_independent'
    dest = folder / 'tracking'
    dest.mkdir(exist_ok=True)
    config = MaterialPatchConfig()
    specification = dict(video_sha256=digest(video), seed_sha256=hashlib.sha256(seed.tobytes()).hexdigest(),
                         ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(), mapping=mapping.tolist(),
                         config=asdict(config), code_sha256=code_hash,
                         baseline_sha256=None)
    if plan.get('full_video', False) or plan.get('query_scope') == 'full_material':
        specification['semantic_masks_sha256'] = mask_fingerprint(labels_dir, mapping)
        specification['fixed_grid_sha256'] = hashlib.sha256(
            grid.tobytes()).hexdigest()
    if (dest / 'complete.json').exists() and (dest / 'tracks.npz').exists():
        old = json.loads((dest / 'complete.json').read_text(encoding='utf-8'))
        if old.get('specification') != specification:
            raise ValueError('Patch tracking cache inputs changed; use a new output root.')
        return old
    read = label_reader(labels_dir)
    cap = cv2.VideoCapture(str(video))
    rows = []
    started = time.perf_counter()
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mapping[0]))
        engine = None
        for t, source in enumerate(mapping):
            if t and source != mapping[t-1]+1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(source))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Decode failed at local frame {source}')
            labels = read(source)
            material = (labels == 1) & ~grid
            if engine is None:
                engine = MaterialPatchTracker(frame, material, seed, config=config, blade_mask=labels == 2)
                row = dict(positions=seed.copy(), supported=np.ones(len(seed), bool),
                           inferred=np.zeros(len(seed), bool), affine=np.tile(np.eye(2), (len(seed),1,1)),
                           cumulative_affine=np.tile(np.eye(2), (len(seed),1,1)),
                           residual=np.zeros(len(seed)), correlation=np.ones(len(seed)),
                           cycle_error=np.zeros(len(seed)), valid_fraction=np.ones(len(seed)),
                           reference_disagreement=np.zeros(len(seed)), gap_age=np.zeros(len(seed), int),
                           selected_reference=np.zeros(len(seed), int),
                           raw_residual=np.zeros(len(seed)),
                           photometric_inlier_fraction=np.ones(len(seed)),
                           reference_age=np.zeros(len(seed), int),
                           support_radius=np.zeros(len(seed)), ambiguous=np.zeros(len(seed), bool),
                           blade_occluded=np.zeros(len(seed), bool),
                           blade_boundary_uncertain=np.zeros(len(seed), bool),
                           candidate_rejected_by_blade=np.zeros(len(seed), bool),
                           candidate_crossed_blade=np.zeros(len(seed), bool))
            else:
                row = engine.step(frame, material, blade_mask=labels == 2)
            if np.shape(row['positions']) != seed.shape or not np.isfinite(row['positions']).all():
                raise ValueError('Patch tracker lost an ID or produced nonfinite coordinates')
            rows.append({key: np.asarray(value).copy() for key, value in row.items()
                         if key != 'point_ids' and np.shape(value)[:1] == (len(seed),)})
            if t % 24 == 0 or t == len(mapping)-1:
                save(dest / 'status.json', dict(stage='tracking', processed=t+1, total=len(mapping),
                                              seconds=time.perf_counter()-started,
                                              supported_fraction=float(np.mean(row['supported']))))
                print(f'{name} {variant}: {t+1}/{len(mapping)}', flush=True)
        keys = sorted(set.intersection(*(set(row) for row in rows)))
        arrays = {key: np.stack([row[key] for row in rows]) for key in keys}
        step_supported = np.zeros_like(arrays['supported'])
        step_supported[1:] = arrays['supported'][1:] & arrays['supported'][:-1]
        np.savez_compressed(dest / 'tracks.npz', **arrays, seed_points=seed, point_ids=ids,
                            source_frame_indices=mapping, frame_indices=mapping,
                            visible=arrays['supported'],
                            position_kind=np.where(arrays['supported'], 0, 2).astype(np.uint8),
                            motion_step_valid=step_supported, forecast_is_measurement=False,
                            support_semantics='local image agreement, not proven material identity',
                            method=variant, code_sha256=code_hash)
    finally:
        cap.release()
    result = dict(method=variant, output=str(dest), frames=len(mapping), points=len(seed),
                  seconds=time.perf_counter()-started, config=asdict(config), code_sha256=code_hash,
                  model_training=False, physical_truth_validated=False,
                  neural_proposals=False, supported_fraction=float(arrays['supported'].mean()),
                  specification=specification)
    save(dest / 'complete.json', result)
    save(folder / (variant + '.json'), result)
    return result


def encode_review_video(path, *, output_path=None, crf=20):
    """Produce a browser/PowerPoint-compatible review, then verify its timeline."""
    import cv2
    source = Path(path)
    output = Path(output_path) if output_path is not None else source
    temporary = output.with_name(output.stem + '.encoding.mp4')
    before = cv2.VideoCapture(str(source))
    count = int(before.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = before.get(cv2.CAP_PROP_FPS)
    before.release()
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(source), '-an',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', str(crf),
                    '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(temporary)], check=True, creationflags=HIDDEN)
    after = cv2.VideoCapture(str(temporary))
    new_count = int(after.get(cv2.CAP_PROP_FRAME_COUNT))
    new_fps = after.get(cv2.CAP_PROP_FPS)
    after.set(cv2.CAP_PROP_POS_FRAMES, max(0, new_count-1))
    ok, _ = after.read()
    after.release()
    if new_count != count or abs(new_fps-fps) > 1e-3 or not ok:
        raise ValueError('Review encoding changed frame count/rate or cannot decode final frame')
    temporary.replace(output)
    return dict(codec='h264', crf=crf, bytes=output.stat().st_size,
                encoded_frame_count=new_count, encoded_fps=new_fps, final_frame_decoded=True)


def render(name, folder, args):
    import numpy as np
    from src.analysis.material_tracking_review import MaterialReviewConfig, render_material_review, summarize_material_tracks
    plan = json.loads((folder / 'plan.json').read_text(encoding='utf-8'))
    tracking = json.loads((folder / 'tracking/complete.json').read_text(encoding='utf-8'))
    with np.load(folder / 'tracking/tracks.npz', allow_pickle=False) as stored:
        mapping = stored['source_frame_indices'].copy()
        count = int(np.searchsorted(mapping, plan['frame_end'], side='right'))
        mapping = mapping[:count]
        positions, supported = stored['positions'][:count].copy(), stored['supported'][:count].copy()
        ids = stored['point_ids'].copy()
    with np.load(folder / 'geometry.npz', allow_pickle=False) as stored:
        tips, depth = stored['tips_px'][mapping], stored['depth_mm'][mapping]
    with np.load(folder / 'frame_mapping.npz', allow_pickle=False) as stored:
        times = stored['physical_time_s'][mapping]
    full_video = bool(plan.get('full_video', False))
    if count != plan['display_frames']:
        raise ValueError('Saved tracking does not reach the requested display depth')
    if full_video:
        if not np.array_equal(mapping, np.arange(plan['full_video_frames'])):
            raise ValueError('Full-video tracking must include every frame through the last source frame')
        if not np.isclose(depth[plan['reversal_frame']], plan['contact_to_reversal_mm'], atol=.03):
            raise ValueError('Full-video tracking lost its calibrated reversal depth')
    elif not np.isclose(depth[-1], args.target_mm, atol=.03):
        raise ValueError('Saved loading prefix does not reach the requested depth')
    phases = dict(contact_frame=plan['contact_frame'], reversal_frame=plan['reversal_frame'])
    labels = label_reader(folder/'segmentation'/name/'semantic_masks')
    blade_masks = lambda frame: labels(frame) == 2
    whole_material = plan.get('query_scope') == 'full_material'
    review = folder / f'{name}_material_tracking.mp4'
    result = render_material_review(folder / 'normalized_lossless.mkv', positions, review,
        frame_indices=mapping, depth_mm=depth, tip_xy=tips, supported=supported,
        point_ids=ids, fps=12, overview_path=folder / f'{name}_overview.png',
        config=MaterialReviewConfig(view_mode='full' if whole_material else 'tip',
                                   point_radius=1 if whole_material else 2),
        blade_mask_reader=blade_masks, **phases)
    result.update(encode_review_video(review))
    tip_result = None
    if whole_material:
        tip_video = folder / f'{name}_tip_tracking.mp4'
        tip_result = render_material_review(folder / 'normalized_lossless.mkv', positions, tip_video,
            frame_indices=mapping, depth_mm=depth, tip_xy=tips, supported=supported,
            point_ids=ids, fps=12, overview_path=folder / f'{name}_tip_overview.png',
            blade_mask_reader=blade_masks, **phases)
        tip_result.update(encode_review_video(tip_video))
    summary = summarize_material_tracks(positions, supported, frame_indices=mapping,
        depth_mm=depth, times_s=times, point_ids=ids, **phases)
    save(folder / 'tracking_summary.json', summary)
    complete = dict(case=name, method='material_independent', neural_proposals=False,
        target_depth_mm=args.target_mm,
        actual_final_depth_mm=float(depth[-1]) if np.isfinite(depth[-1]) else None,
        full_video=full_video, query_scope=plan.get('query_scope'),
        frames=count, points=len(ids), video=str(review), tracks=str(folder / 'tracking/tracks.npz'),
        rendering=result, tip_rendering=tip_result,
        tracking_seconds=tracking['seconds'], code_sha256=tracking['code_sha256'],
        contact_to_reversal_mm=plan['contact_to_reversal_mm'], completed=time.time(),
        physical_truth_validated=False)
    save(folder / 'complete.json', complete)
    return complete


def compress(name, folder, args):
    source = folder / f'{name}_material_tracking.mp4'
    if not source.is_file():
        raise FileNotFoundError(source)
    output = folder / f'{name}_PPT.mp4'
    result = encode_review_video(source, output_path=output, crf=28)
    return dict(video=str(output), **result)


def write_index(root, names, states):
    cards = []
    for name in names:
        state = states.get(name, {}).get('stage', 'pending')
        title = html.escape(name)
        if state != 'complete' or not (root/name/'complete.json').is_file():
            cards.append(f'<section><h2>{title}</h2><p>{html.escape(state)}</p></section>')
            continue
        record = json.loads((root/name/'complete.json').read_text(encoding='utf-8'))
        src = html.escape(f'{name}/{Path(record["video"]).name}', quote=True)
        links = f'<a href="{src}">全场视频</a>'
        if record.get('tip_rendering'):
            tip = html.escape(f'{name}/{Path(record["tip_rendering"]["output_path"]).name}', quote=True)
            links += f' · <a href="{tip}">刀尖放大视频</a>'
        links += f' · <a href="{title}/tracking/tracks.npz">轨迹 NPZ</a> · <a href="{title}/tracking_summary.json">分阶段统计</a>'
        cards.append(f'<section><h2>{title}</h2><video controls preload="metadata" src="{src}"></video>'
                     f'<p>{record["frames"]} 帧 · {record["points"]} 个固定材料 ID</p><p>{links}</p></section>')
    document = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>独立材料跟踪结果</title><style>body{max-width:1320px;margin:32px auto;padding:0 20px;'
        'background:#111719;color:#e5efef;font:17px/1.6 system-ui}video{width:100%;background:#000}'
        'section{margin:36px 0}a{color:#82d9e9}</style><h1>独立材料跟踪结果</h1>'
        '<p>颜色表示固定材料 ID，空心点表示缺乏图像支持的预测。接触至开始回退按约 3 mm 定标；'
        '退刀深度可下降，不能将光滑轨迹或点数保留视为真实身份已被验证。</p>' + ''.join(cards) + '</html>')
    temporary = root/'index.encoding.html'
    temporary.write_text(document, encoding='utf-8')
    temporary.replace(root/'index.html')


def run_batch(args):
    states, failures, lock = {}, [], Lock()
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', OMP_NUM_THREADS='2')

    def snapshot():
        save(args.out / 'status.json', dict(method='material_independent', cases=args.names,
             target_mm=args.target_mm, full_video=args.full_video, query_scope=args.query_scope,
             active=states, failures=failures, updated=time.time()))

    def stage(name, action):
        folder = args.out / name
        folder.mkdir(exist_ok=True)
        command = [sys.executable, '-u', '-m', 'src.workflows.tracking', '--worker', '--names', name,
            '--stage', action, '--out', str(args.out), '--reuse-root', str(args.reuse_root),
            '--target-mm', str(args.target_mm), '--query-scope', args.query_scope,
            '--outer-pitch', str(args.outer_pitch)]
        if args.full_video:
            command.append('--full-video')
        with (folder / f'{action}.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, creationflags=HIDDEN)
            with lock:
                states[name] = dict(stage=action, pid=process.pid, started=time.time())
                snapshot()
            code = process.wait()
        with lock:
            states[name] = dict(stage=action+'_complete' if code == 0 else 'failed', exit_code=code)
            if code:
                failures.append(dict(case=name, stage=action, exit_code=code, log=str(folder/f'{action}.log')))
            snapshot()
        return code == 0

    def cpu_case(name):
        actions = ['track', 'render'] if args.stage == 'all' else [args.stage]
        if args.ppt and args.stage == 'all':
            actions.append('compress')
        for action in actions:
            if not stage(name, action):
                return
        with lock:
            states[name] = dict(stage='complete')
            snapshot()
            if args.stage in ('all', 'render'):
                write_index(args.out, args.names, states)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = []
        for name in args.names:
            if args.stage in ('all', 'prepare'):
                if not stage(name, 'prepare'):
                    continue
            if args.stage != 'prepare':
                futures.append(pool.submit(cpu_case, name))
        for future in futures:
            future.result()
    save(args.out / 'status.json', dict(stage='complete' if not failures else 'completed_with_failures',
        method='material_independent', cases=args.names, target_mm=args.target_mm,
        full_video=args.full_video, query_scope=args.query_scope,
        active=states, failures=failures, updated=time.time()))
    if failures:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--names', nargs='+', default=list(NAMES))
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--reuse-root', type=Path, default=None)
    parser.add_argument('--target-mm', type=float, default=3.)
    parser.add_argument('--full-video', action='store_true')
    parser.add_argument('--query-scope', choices=('dense', 'full_material'), default='full_material')
    parser.add_argument('--outer-pitch', type=int, default=24)
    parser.add_argument('--stage', choices=[*STAGES, 'all'], default='all')
    parser.add_argument('--jobs', type=int, choices=(1,2,3), default=2)
    parser.add_argument('--ppt', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 < args.target_mm <= 3:
        parser.error('target-mm must be in (0,3]')
    if args.full_video and args.target_mm != 3.:
        parser.error('Full videos retain a 3 mm contact-to-reversal calibration')
    if args.outer_pitch < 12:
        parser.error('Outer grid pitch must be at least 12 native pixels')
    if args.out is None:
        args.out = FULL_OUT if args.full_video else DEFAULT_OUT
    if args.reuse_root is None:
        args.reuse_root = REUSE_ROOT
    if len(set(args.names)) != len(args.names):
        parser.error('Each dataset may run only once per batch')
    if args.ppt and args.stage != 'all':
        parser.error('--ppt is used with --stage all; use --stage compress for an existing review')
    validate_case_names(args.names)
    if args.worker:
        if len(args.names) != 1 or args.stage == 'all':
            parser.error('A worker addresses one processing stage and case')
        import cv2
        cv2.setNumThreads(2)
        folder = args.out / args.names[0]
        folder.mkdir(parents=True, exist_ok=True)
        return globals()[args.stage](args.names[0], folder, args)
    preflight(args.names, tracker='independent', target_mm=args.target_mm, full_video=args.full_video,
              reuse_root=args.reuse_root)
    if args.check_only:
        return
    args.out = args.out.resolve()
    args.reuse_root = args.reuse_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    run_batch(args)


if __name__ == '__main__':
    main()
