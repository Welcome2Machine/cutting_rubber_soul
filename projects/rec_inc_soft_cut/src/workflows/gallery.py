"""Text-free saved-motion videos, static slip plots and compact PPT media."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import html
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.io import ROOT, HIDDEN, read_json, sha, write_json

WIDTH, HEIGHT, FPS = 1280, 480, 12
TEAL, GOLD = (195, 204, 70), (75, 190, 255)


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    cutoff_um: float
    estimate: Path
    prepared: Path
    tracks: Path
    measurements: Path


def load_cases(manifest):
    manifest = Path(manifest).resolve()
    rows = read_json(manifest)['cases']
    if not rows:
        raise ValueError('The gallery requires at least one case')
    cases = []
    for row in rows:
        name = str(row['name'])
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
            raise ValueError('Case names must be plain directory names')
        cutoff = float(row['cutoff_um'])
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError('cutoff_um must be positive and finite')
        paths = {}
        for key in ('estimate', 'prepared', 'tracks', 'measurements'):
            path = Path(row[key])
            paths[key] = (path if path.is_absolute() else manifest.parent/path).resolve()
        cases.append(Case(name, str(row.get('group', name)), cutoff, **paths))
    if len({case.name.casefold() for case in cases}) != len(cases):
        raise ValueError('Duplicate case names')
    return cases


def _load(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def series(case):
    """Read saved values; no gap filling, smoothing, extrapolation or refitting."""
    with np.load(case.estimate, allow_pickle=False) as archive:
        frames = np.asarray(archive['frames'] if 'frames' in archive else archive['frame_indices'])
        depth = np.asarray(archive['depth_mm'], float)
        field = next((key for key in ('display_offset_completed_um', 'display_offset_um')
                      if key in archive), 'offset_um')
        values = np.asarray(archive[field], float).copy()
        kind = np.asarray(archive['display_estimate_kind'], int) if 'display_estimate_kind' in archive else np.ones(len(frames), int)
        if field.startswith('display_'):
            values[kind == 0] = np.nan
        else:
            available = next((key for key in ('transport_offset_available', 'offset_available',
                                              'available', 'transport_available') if key in archive), None)
            if available:
                valid = np.asarray(archive[available], bool)
                if valid.shape != values.shape:
                    raise ValueError('Saved availability shape differs from offsets')
                values[~valid] = np.nan
        states = np.asarray(archive['transport_state'], int) if 'transport_state' in archive else np.zeros(len(frames), int)
    if (frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu'
            or frames[0] < 0 or np.any(np.diff(frames) <= 0)
            or any(value.shape != frames.shape for value in (depth, values, kind, states))):
        raise ValueError('Saved curve arrays must align with increasing integer frames')
    crossed = np.flatnonzero(np.isfinite(depth) & (depth > case.cutoff_um/1000 + 1e-9))
    stop = int(crossed[0]) if len(crossed) else len(frames)
    if not stop:
        raise ValueError('No saved samples before the requested cutoff')
    return dict(frames=frames[:stop], depth=depth[:stop], values=values[:stop],
                inferred=((kind == 2) | np.isin(states, (1, 4)))[:stop], field=field)


def prepare(case, out, overwrite=False):
    folder = Path(out)/case.name
    folder.mkdir(parents=True, exist_ok=True)
    data = series(case)
    provenance = {key: str(getattr(case, key)) for key in ('estimate', 'prepared', 'tracks', 'measurements')}
    digest = sha(case.estimate)
    path = folder/'calculation.json'
    if path.is_file() and not overwrite:
        previous = read_json(path)
        if (previous.get('estimate_sha256') != digest or previous.get('sources') != provenance
                or previous.get('cutoff_um') != case.cutoff_um):
            raise FileExistsError('Existing gallery uses different inputs; choose a new output or --overwrite')
    elif any(folder.iterdir()) and not overwrite:
        raise FileExistsError('Existing output has no matching calculation provenance')
    valid = np.flatnonzero(np.isfinite(data['depth']) & np.isfinite(data['values']))
    last = int(valid[-1]) if len(valid) else None
    result = dict(name=case.name, group=case.group, cutoff_um=case.cutoff_um,
        sources=provenance, estimate_sha256=digest, plotted_field=data['field'],
        last_saved_slip_um=None if last is None else float(data['values'][last]),
        last_saved_depth_mm=None if last is None else float(data['depth'][last]),
        frames=len(data['frames']), first_source_frame=int(data['frames'][0]),
        last_source_frame=int(data['frames'][-1]), estimates_modified=False,
        optical_flow_recomputed=False, surface_marker_is_inference=True)
    write_json(path, result)
    return data


def panel(canvas, image, rect, origin, width):
    x, y, w, h = rect
    factor = w/width
    matrix = np.array([[factor, 0, -factor*origin[0]], [0, factor, -factor*origin[1]]])
    canvas[y:y+h, x:x+w] = cv2.warpAffine(image, matrix, (w, h))
    return lambda xy: (np.asarray(xy)-origin)*factor+np.array([x, y])


def draw_points(canvas, xy, rect, supported):
    x, y, w, h = rect
    valid = np.isfinite(xy).all(1) & (xy[:, 0] > x+2) & (xy[:, 0] < x+w-2) & (xy[:, 1] > y+2) & (xy[:, 1] < y+h-2)
    overlay = canvas.copy()
    for k in np.flatnonzero(valid):
        cv2.circle(overlay, tuple(np.rint(xy[k]).astype(int)), 1,
                   (215, 215, 215) if supported[k] else (100, 100, 100),
                   -1 if supported[k] else 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, .68, canvas, .32, 0, dst=canvas)


def path_line(canvas, xy, rect, color):
    for a, b in zip(xy[:-1], xy[1:]):
        if np.isfinite([a, b]).all():
            visible, p, q = cv2.clipLine(rect, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)))
            if visible:
                cv2.line(canvas, p, q, color, 1, cv2.LINE_AA)


def interpolation_value(frames, values, frame):
    """Display between adjacent finite samples only; missing intervals stay empty."""
    j = int(np.searchsorted(frames, frame))
    if j < len(frames) and frames[j] == frame:
        return float(values[j])
    if not 0 < j < len(frames) or not np.isfinite(values[j-1:j+1]).all():
        return np.nan
    weight = (frame-frames[j-1])/(frames[j]-frames[j-1])
    return float((1-weight)*values[j-1]+weight*values[j])


def rows_at(mapping, requested):
    mapping = np.asarray(mapping)
    if mapping.ndim != 1 or mapping.dtype.kind not in 'iu' or np.any(np.diff(mapping) <= 0):
        raise ValueError('Saved frame mapping must contain increasing integer IDs')
    rows = np.searchsorted(mapping, requested)
    if np.any(rows >= len(mapping)) or not np.array_equal(mapping[rows], requested):
        raise ValueError('Requested display frame lacks saved tracking or measurement data')
    return rows


def video_info(path, *, expected_fps=None, expected_size=None, expected_duration=None, tolerance=.01):
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        first, _ = cap.read()
        if n == 1:
            last = first
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n-1)
            last, _ = cap.read()
    finally:
        cap.release()
    if n <= 0 or not np.isfinite(fps) or fps <= 0 or not first or not last:
        raise ValueError('Encoded first/last video frame cannot be decoded: '+str(path))
    if expected_fps is not None and not np.isclose(fps, expected_fps):
        raise ValueError('Encoded frame rate differs from requested rate')
    if expected_size is not None and size != expected_size:
        raise ValueError('Encoded image dimensions differ from requested dimensions')
    duration = n/fps
    if expected_duration is not None and abs(duration-expected_duration) > tolerance:
        raise ValueError('Encoded duration changed beyond frame-rounding tolerance')
    return dict(frames=n, fps=fps, width=size[0], height=size[1], duration_s=duration,
                bytes=Path(path).stat().st_size, first_and_last_frame_decoded=True)


def render(case, out, data, overwrite=False):
    folder = Path(out)/case.name
    destination = folder/'review.mp4'
    if destination.is_file() and not overwrite:
        return
    prepared = _load(case.prepared/'input.npz', ('query__point_ids', 'query__seed_points',
        'initial__boundary_xy', 'initial__surface_s_px', 'initial__contact_s_px', 'scale'))
    track = _load(case.tracks/'tracking/tracks.npz', ('positions', 'point_ids', 'source_frame_indices'))
    support = _load(case.measurements/'measurement_support.npz', ('supported', 'point_ids', 'frame_indices'))
    blade = _load(case.measurements/'blade_contact_geometry.npz', ('on_blade', 'point_ids', 'frame_indices'))
    guard = _load(case.prepared/'blade_reference_guard.npz', ('tips_xy', 'source_frames'))
    for item in (track, support, blade):
        if not np.array_equal(item['point_ids'], prepared['query__point_ids']):
            raise ValueError('Material identities differ across rendering inputs')
    frames = np.arange(data['frames'][0], data['frames'][-1]+1)
    tr = rows_at(track['source_frame_indices'], frames)
    sr, br = rows_at(support['frame_indices'], frames), rows_at(blade['frame_indices'], frames)
    known = np.isfinite(guard['tips_xy']).all(1)
    if known.sum() < 2:
        raise ValueError('Insufficient saved blade reference for the display camera')
    rows_at(guard['source_frames'], guard['source_frames'])
    guide = np.column_stack([np.interp(guard['source_frames'], guard['source_frames'][known],
                                      guard['tips_xy'][known, axis]) for axis in range(2)])
    smooth = gaussian_filter1d(guide, 3, axis=0, mode='nearest')
    camera = np.column_stack([np.interp(frames, guard['source_frames'], smooth[:, axis]) for axis in range(2)])
    surface, arc = prepared['initial__boundary_xy'], prepared['initial__surface_s_px']
    baseline, scale = float(prepared['initial__contact_s_px']), float(prepared['scale'])
    if (surface.shape != (len(arc), 2) or len(arc) < 2 or not np.isfinite(surface).all()
            or not np.isfinite(arc).all() or np.any(np.diff(arc) <= 0)
            or not np.isfinite(scale) or scale <= 0 or not arc[0] <= baseline <= arc[-1]):
        raise ValueError('Invalid saved initial surface or micron-per-pixel calibration')
    at_s = lambda value: np.array([np.interp(value, arc, surface[:, axis]) for axis in range(2)])
    zero = at_s(baseline)
    source = case.tracks/'normalized_lossless.mkv'
    cap = cv2.VideoCapture(str(source))
    ok, initial = cap.read()
    if not ok:
        cap.release()
        raise OSError('Cannot decode initial source image: '+str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    pending = folder/'review.encoding.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '24', '-threads', '2', '-pix_fmt', 'yuv420p', '-tag:v', 'avc1',
        '-movflags', '+faststart', str(pending)]
    # A temporary encoder log prevents a full stderr pipe from blocking writes.
    with (folder/'encode.log').open('wb') as log:
        try:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log, creationflags=HIDDEN)
        except BaseException:
            cap.release()
            raise
        try:
            for i, frame_id in enumerate(frames):
                ok, frame = cap.read()
                if not ok:
                    raise OSError(f'Cannot decode source frame {frame_id}')
                canvas = np.full((HEIGHT, WIDTH, 3), (28, 24, 15), np.uint8)
                left_rect, right_rect = (12, 8, 620, 464), (648, 8, 620, 464)
                left = panel(canvas, frame, left_rect, camera[i]-[135., 155.], 270.)
                right = panel(canvas, initial, right_rect, zero-[95., 90.], 300.)
                current = track['positions'][tr[i]].copy()
                current[blade['on_blade'][br[i]]] = np.nan
                draw_points(canvas, left(current), left_rect, support['supported'][sr[i]])
                seeds = prepared['query__seed_points']
                draw_points(canvas, right(seeds), right_rect, np.ones(len(seeds), bool))
                path_line(canvas, right(surface), right_rect, (140, 150, 150))
                cv2.circle(canvas, tuple(np.rint(right(zero)).astype(int)), 5, (145, 220, 125), 2, cv2.LINE_AA)
                j = np.searchsorted(guard['source_frames'], frame_id)
                if j < len(known) and guard['source_frames'][j] == frame_id and known[j]:
                    cv2.drawMarker(canvas, tuple(np.rint(left(guard['tips_xy'][j])).astype(int)),
                                   TEAL, cv2.MARKER_CROSS, 18, 1)
                value = interpolation_value(data['frames'], data['values'], frame_id)
                address = baseline+value/scale
                if np.isfinite(address) and arc[0] <= address <= arc[-1]:
                    cv2.circle(canvas, tuple(np.rint(right(at_s(address))).astype(int)), 7, GOLD, 2, cv2.LINE_AA)
                proc.stdin.write(np.ascontiguousarray(canvas).tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError('FFmpeg encoding failed; see '+str(folder/'encode.log'))
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            raise
        finally:
            if proc.stdin is not None and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            cap.release()
    info = video_info(pending, expected_fps=FPS, expected_size=(WIDTH, HEIGHT),
                      expected_duration=len(frames)/FPS)
    pending.replace(destination)
    (folder/'encode.log').unlink(missing_ok=True)
    return info


def curves(case, out, data, overwrite=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(out)/case.name
    if all((folder/('curve.'+suffix)).is_file() for suffix in ('png', 'svg')) and not overwrite:
        return
    x, y = data['depth'], data['values']
    fig, ax = plt.subplots(figsize=(10.5, 2.65), layout='constrained')
    for i in range(len(x)-1):
        if np.isfinite(x[i:i+2]).all() and np.isfinite(y[i:i+2]).all():
            ax.plot(x[i:i+2], y[i:i+2], color='#168b8f', lw=2,
                    ls='--' if data['inferred'][i:i+2].any() else '-')
    if len(x) == 1 and np.isfinite([x[0], y[0]]).all():
        ax.plot(x, y, '.', color='#168b8f')
    finite = y[np.isfinite(y)]
    low = min(0., float(finite.min())) if len(finite) else 0.
    high = max(50., float(finite.max())*1.12) if len(finite) else 50.
    ax.set(xlim=(0, case.cutoff_um/1000), ylim=(low, high), xlabel='Displacement (mm)', ylabel='Slip (μm)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.14)
    for suffix in ('png', 'svg'):
        fig.savefig(folder/('curve.'+suffix), dpi=160)
    plt.close(fig)


def compress_ppt(case, out, overwrite=False):
    source = Path(out)/case.name/'review.mp4'
    destination = Path(out)/'ppt_videos'/f'{case.name}.mp4'
    if destination.is_file() and not overwrite:
        return
    master = video_info(source, expected_fps=FPS, expected_size=(WIDTH, HEIGHT))
    digest = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix('.encoding.mp4')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
        '-vf', 'fps=3,scale=1024:-2:flags=lanczos', '-an', '-c:v', 'libx264', '-preset', 'slow',
        '-crf', '28', '-threads', '2', '-pix_fmt', 'yuv420p', '-tag:v', 'avc1', '-movflags', '+faststart',
        str(pending)], check=True, capture_output=True, creationflags=HIDDEN)
    info = video_info(pending, expected_fps=3, expected_size=(1024, 384),
                      expected_duration=master['duration_s'], tolerance=1/3+.01)
    if sha(source) != digest:
        raise RuntimeError('Source review video changed during compression')
    pending.replace(destination)
    return info


def report(cases, out):
    out = Path(out)
    rows, cards = [], []
    fmt = lambda value: '—' if value is None else f'{value:.3f}'
    for case in cases:
        summary = read_json(out/case.name/'calculation.json')
        name, group = html.escape(case.name), html.escape(case.group)
        key = quote(case.name)
        video = f'ppt_videos/{key}.mp4' if (out/'ppt_videos'/f'{case.name}.mp4').is_file() else f'{key}/review.mp4'
        rows.append(f'<tr><td>{group}</td><td>{name}</td><td>{case.cutoff_um/1000:.3f}</td>'
                    f'<td>{fmt(summary["last_saved_slip_um"])}</td><td>{fmt(summary["last_saved_depth_mm"])}</td></tr>')
        media = ''
        if (out/video).is_file():
            media += f'<video controls preload="metadata" playsinline src="{video}"></video><a href="{video}">Download video</a>'
        if (out/case.name/'curve.png').is_file():
            media += f'<img src="{key}/curve.png" alt="Displacement versus slip"><a href="{key}/curve.png">PNG</a> · <a href="{key}/curve.svg">SVG</a>'
        cards.append(f'<section><h2>{group} <small>{name}</small></h2>{media}</section>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Surface slip observations</title><style>body{margin:0;background:#eef2f3;color:#21343d;font:16px/1.5 sans-serif}main{max-width:1280px;margin:auto;padding:24px}section{background:white;padding:20px;margin:24px 0;border-radius:10px}video,img{display:block;width:100%;height:auto}small{font-weight:normal}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde4e6}a{color:#087d83}</style><main>
<h1>Surface slip observations</h1><p>Saved estimates along the initial surface. Displacement starts at the saved first-contact reference. Dashed portions retain the source's inferred or interpolated status. Missing intervals are left empty.</p>
<table><tr><th>Group</th><th>Case</th><th>Cutoff (mm)</th><th>Last saved slip (μm)</th><th>Corresponding displacement (mm)</th></tr>'''
    (out/'index.html').write_text(document+''.join(rows)+'</table>'+''.join(cards)+'</main></html>', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('all', 'video', 'plot', 'ppt', 'index'), default='all')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if any(out.is_relative_to(ROOT/name) for name in ('data', 'models', '.git')):
        parser.error('Gallery output cannot be inside data, models or .git')
    cases = load_cases(args.manifest)
    data = [prepare(case, out, args.overwrite) for case in cases]
    if args.stage in ('all', 'video'):
        cv2.setNumThreads(1)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda pair: render(pair[0], out, pair[1], args.overwrite), zip(cases, data)))
    if args.stage in ('all', 'plot'):
        # Matplotlib uses process-global state; plots remain sequential.
        for case, values in zip(cases, data):
            curves(case, out, values, args.overwrite)
    if args.stage in ('all', 'ppt'):
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda case: compress_ppt(case, out, args.overwrite), cases))
    if args.stage in ('all', 'index'):
        report(cases, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
