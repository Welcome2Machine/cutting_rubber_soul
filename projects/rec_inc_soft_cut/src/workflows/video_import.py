"""Import dated camera AVI directories to verified, native-resolution 12 fps MP4.

The original acquisition frame indices are retained; 12 fps is the playback
rate, not a claim about the camera acquisition clock. Raw AVI files stay put.
"""
from __future__ import annotations

from src.io import sha as digest
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from src.io import atomic_write_json
from src.tracking.fixed_grid import FixedGridProcessor, grid_0915_config
from src.video_sources import validate_case_names

from src.io import ROOT
HIDDEN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def selected_frame(index):
    return index % 5 in (0, 3)


def import_one(source, destination, report):
    name = source.stem
    video = destination/f'{name}.mp4'
    record = video.with_suffix('.json')
    alignment = destination/f'{name}_alignment.npz'
    source_identity = identity(source)
    if record.exists() and video.exists() and alignment.exists():
        previous = json.loads(record.read_text(encoding='utf-8'))
        if previous['source'] != source_identity or digest(video) != previous['output_sha256']:
            raise ValueError('Existing import differs from source or recorded hash')
        return previous
    if video.exists():
        raise ValueError(f'Unverified existing destination: {video}')
    destination.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or (int(cap.get(3)), int(cap.get(4))) != (1024, 768) or abs(fps-30) > 1e-4:
        cap.release()
        raise ValueError('Expected native 1024x768 AVI with 30 fps container rate')
    partial = destination/f'{name}.encoding.mp4'
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', '1024x768', '-r', '12', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '2', str(partial)]
    log = (destination/f'{name}.import.log').open('wb')
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, creationflags=HIDDEN)
    mapping, samples, checks, decode_recoveries, unselected_read_failures = [], {}, [], [], []
    decoded_source = 0
    grid_fraction = None
    grid = FixedGridProcessor(grid_0915_config()).mask()
    try:
        while True:
            ok, image = cap.read()
            if not ok and decoded_source < header-2:
                # AVI sequential reads can stop transiently although the exact
                # indexed frame remains decodable. Never skip to a later frame.
                for attempt in range(3):
                    cap.release()
                    cap = cv2.VideoCapture(str(source))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, decoded_source)
                    ok, image = cap.read()
                    if ok and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == decoded_source+1:
                        decode_recoveries.append(dict(source_frame=decoded_source, attempt=attempt+1))
                        break
                    ok = False
            if not ok:
                if decoded_source < header-2 and not selected_frame(decoded_source):
                    # A broken AVI slot that the declared downsampling already
                    # excludes does not remove an output sample. Seek exactly
                    # to the next source slot; never substitute a selected one.
                    unselected_read_failures.append(decoded_source)
                    decoded_source += 1
                    cap.release()
                    cap = cv2.VideoCapture(str(source))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, decoded_source)
                    continue
                break
            if decoded_source == 0:
                grid_fraction = float((image[grid] == 85).mean())
                cv2.imwrite(str(destination/f'{name}.first_frame.png'), image)
            if selected_frame(decoded_source):
                local = len(mapping)
                mapping.append(decoded_source)
                process.stdin.write(image.tobytes())
                if local % 400 == 0 or (unselected_read_failures and decoded_source-unselected_read_failures[-1] <= 3):
                    samples[local] = image.copy()
            decoded_source += 1
            if decoded_source % 1000 == 0:
                report(dict(stage='encoding', source_frames=decoded_source, header_frames=header))
        process.stdin.close()
        if process.wait(timeout=180):
            raise RuntimeError('MP4 encoder failed')
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        cap.release()
        log.close()
    if decoded_source < header-2 or not mapping:
        raise ValueError(f'Source unexpectedly truncated: {decoded_source}/{header}')
    cap = cv2.VideoCapture(str(partial))
    decoded = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if image.shape != (768, 1024, 3):
                raise ValueError('Changed image dimensions')
            if decoded in samples:
                delta = image.astype(float)-samples[decoded]
                mae = float(abs(delta).mean())
                psnr = float(10*np.log10(255**2/max((delta*delta).mean(), 1e-12)))
                if mae > 5 or psnr < 34:
                    raise ValueError(f'Compression correspondence/quality failed: {mae}, {psnr}')
                checks.append(dict(local_frame=decoded, source_frame=mapping[decoded], mae=mae, psnr_db=psnr))
            decoded += 1
    finally:
        cap.release()
    if decoded != len(mapping) or identity(source) != source_identity:
        raise ValueError('Frame mapping mismatch or source changed during import')
    result = dict(source=source_identity, source_header_frames=header,
        source_sequentially_decoded_frames=decoded_source-len(unselected_read_failures),
        source_frame_positions_consumed=decoded_source, frames=decoded,
        width=1024, height=768, fps=12, source_header_fps=fps, output=str(video),
        output_sha256=digest(partial), output_bytes=partial.stat().st_size, crf=18,
        original_dimensions_preserved=True, raw_copy=False,
        physical_acquisition_time_calibration_used=False, frame_mapping='source n with n modulo 5 in (0,3)',
        raw_grid_0915_fraction=grid_fraction, grid_0915_verified=grid_fraction >= .999,
        decode_recoveries=decode_recoveries,
        unreadable_unselected_source_frames=unselected_read_failures,
        missing_selected_source_frames=[],
        checks=checks, all_later_processing_uses_local_mp4=True)
    np.savez_compressed(alignment, source_indices=np.asarray(mapping), frame_indices=np.arange(decoded))
    partial.replace(video)
    atomic_write_json(record, result, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--retry-after-pid', type=int, help='Wait for an existing importer before verifying/retrying its full manifest')
    args = parser.parse_args(argv)
    if args.retry_after_pid:
        import psutil
        while psutil.pid_exists(args.retry_after_pid):
            time.sleep(10)
    out = args.out.resolve()
    out.relative_to((ROOT/'results').resolve())
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for directory in args.sources:
        directory = directory.resolve()
        if not directory.name.isdigit():
            raise ValueError('Source directory must have a numeric date name')
        for source in sorted(directory.rglob('*')):
            if source.is_file() and source.suffix.lower() == '.avi':
                cases.append(dict(name=source.stem, source=str(source),
                    data_directory=str(ROOT/'data/videos'/directory.name)))
    validate_case_names([case['name'] for case in cases])
    atomic_write_json(out/'import_manifest.json', dict(cases=cases, created=time.time(), python=__import__('sys').executable), allow_nan=False)
    states = {case['name']: dict(stage='pending') for case in cases}
    def update(name, state):
        states[name] = dict(state, updated=time.time())
        atomic_write_json(out/'import_status.json', dict(pid=os.getpid(), cases=states, updated=time.time()), allow_nan=False)
    for case in cases:
        name = case['name']
        update(name, dict(stage='starting'))
        try:
            result = import_one(Path(case['source']), Path(case['data_directory']), lambda state: update(name, state))
            update(name, dict(stage='complete', frames=result['frames'], output_bytes=result['output_bytes'],
                video=result['output'], grid_0915_verified=result['grid_0915_verified']))
            print(f'{name}: imported {result["frames"]} frames, {result["output_bytes"]/2**20:.1f} MiB', flush=True)
        except Exception as error:
            update(name, dict(stage='failed', error=str(error)))
            print(f'{name}: FAILED {error}', flush=True)
    return int(any(state['stage'] != 'complete' for state in states.values()))


if __name__ == '__main__':
    raise SystemExit(main())
