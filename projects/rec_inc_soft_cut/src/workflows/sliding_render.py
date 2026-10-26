"""English current/original-view review with explicit missing/held states."""
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from src.io import load_npz as load
from src.io import write_json
from src.analysis.contact_transport_filter import WINDOWED_STATES as STATES

W, H, FPS = 1280, 800, 12
COLORS = ((200, 130, 245), (80, 180, 255))  # BGR


def text(image, caption, xy, scale=.52, color=(225, 232, 237)):
    cv2.putText(image, caption, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def panel(canvas, source, rect, origin, width):
    x, y, w, h = rect
    factor = w/width
    M = np.array([[factor, 0, -factor*origin[0]], [0, factor, -factor*origin[1]]], float)
    canvas[y:y+h, x:x+w] = cv2.warpAffine(source, M, (w, h))
    def map_points(points):
        return (np.asarray(points)-origin)*factor+np.array([x, y])
    return map_points


def points(canvas, xy, rect, support=None):
    x, y, w, h = rect
    valid = np.isfinite(xy).all(1)&(xy[:, 0] >= x+2)&(xy[:, 0] < x+w-2)&(xy[:, 1] >= y+2)&(xy[:, 1] < y+h-2)
    for i in np.flatnonzero(valid):
        point = tuple(np.rint(xy[i]).astype(int))
        filled = support is None or support[i]
        cv2.circle(canvas, point, 1, (165, 165, 165) if filled else (85, 85, 85), -1 if filled else 1, cv2.LINE_AA)


def plot_background(arrays, methods, maximum_depth_mm=3.):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.8, 2.4), dpi=100, facecolor='#101c24')
    ax = fig.add_axes([.065, .23, .90, .68], facecolor='#101c24')
    for label, a, color in zip(methods, arrays, ('#f582c8', '#ffb450')):
        # Unknown gaps remain gaps; they are never joined by the renderer.
        ax.plot(a['depth_mm'], a['offset_um'], color=color, lw=1.8, label=label)
        conditional = np.where(a['transport_state'] == 4, a['offset_um'], np.nan)
        ax.plot(a['depth_mm'], conditional, color='#d3dce1', lw=2, ls='--')
    ax.axhline(0, color='#4c6574', lw=.7)
    ax.set_xlim(0, maximum_depth_mm)
    ax.set_xlabel('Indentation from first contact (mm)', color='white')
    ax.set_ylabel('Surface offset (um)', color='white')
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#687d88')
    ax.grid(alpha=.15)
    ax.legend(loc='upper left', fontsize=8)
    fig.canvas.draw()
    image = np.array(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def render(folder, maximum_depth_mm=3., overwrite=False):
    if not np.isfinite(maximum_depth_mm) or maximum_depth_mm <= 0:
        raise ValueError('Positive finite plotting depth required')
    started = time.perf_counter()
    cv2.setNumThreads(1)
    meta = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    source = Path(meta['input_video'])
    old_folder = Path(meta['blade_guard']['old_contact_source'])
    d = load(old_folder/'input.npz')
    repaired = load(folder/'input.npz')
    guard = load(folder/'blade_reference_guard.npz')
    methods = [item['method'] for item in meta['methods']]
    if len(methods) != 2:
        raise ValueError('Expected the paired MMT and OBT outputs')
    arrays = [load(folder/(method+'.npz')) for method in methods]
    row_map = {int(f): i for i, f in enumerate(repaired['frames'])}
    guide = guard['tips_xy'].copy()
    known = np.isfinite(guide).all(1)
    if known.sum() < 2:
        raise ValueError('No reliable blade coordinates for review')
    for j in range(2):
        guide[:, j] = np.interp(guard['source_frames'], guard['source_frames'][known], guide[known, j])
    camera = cv2.GaussianBlur(guide, (1, 15), 0)  # camera only; never measurement
    cap = cv2.VideoCapture(str(source/'normalized_lossless.mkv'))
    ok, initial = cap.read()
    cap.release()
    if not ok:
        raise OSError('Cannot decode initial material image')
    surface, arc = d['initial__boundary_xy'], d['initial__surface_s_px']
    baseline = float(d['initial__contact_s_px'])
    def at_s(value):
        if not np.isfinite(value) or value < arc.min() or value > arc.max():
            return np.full(2, np.nan)
        return np.array([np.interp(value, arc, surface[:, j]) for j in range(2)])
    zero = at_s(baseline)
    right_origin = zero-np.array([160., 135.])
    background = plot_background(arrays, methods, maximum_depth_mm=maximum_depth_mm)
    destination = folder/'repaired_contact.mp4'
    if destination.exists() and not overwrite:
        return
    temporary = folder/'repaired_contact.encoding.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-threads', '2', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(temporary)]
    cap = cv2.VideoCapture(str(source/'normalized_lossless.mkv'))
    current_frame = -1
    previews = {0: 'start', len(d['frames'])//2: 'middle', len(d['frames'])-1: 'endpoint'}
    with (folder/'encode.log').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            for i, target in enumerate(d['frames']):
                while current_frame < target:
                    ok, frame = cap.read()
                    current_frame += 1
                    if not ok:
                        raise OSError(f'Missing source image {target}')
                k = int(target-guard['source_frames'][0])
                q = row_map.get(int(target))
                canvas = np.full((H, W, 3), (36, 28, 16), np.uint8)
                tip = guard['tips_xy'][k]
                valid_tip = np.isfinite(tip).all()
                depth = (tip[1]-guard['tips_xy'][0, 1])*float(d['scale'])/1000 if valid_tip else np.nan
                text(canvas, folder.name+' | Corrected contact transport | Original material IDs', (18, 24), .65)
                caption = f'Indentation: {depth:.3f} mm | first contact = 0' if valid_tip else 'Indentation unavailable | blade reference missing'
                text(canvas, caption, (18, 49), .55, (240, 210, 60))
                text(canvas, 'Current view | gray material tracks', (18, 75))
                text(canvas, 'Original view | surface address estimates', (652, 75))
                left_rect, right_rect = (12, 86, 620, 375), (648, 86, 620, 375)
                left = panel(canvas, frame, left_rect, camera[k]-np.array([150., 125.]), 340.)
                right = panel(canvas, initial, right_rect, right_origin, 440.)
                points(canvas, left(d['positions'][i]), left_rect, d['support'][i])
                points(canvas, right(d['query__seed_points']), right_rect)
                boundary = right(surface)
                for a0, a1 in zip(boundary[:-1], boundary[1:]):
                    if not np.isfinite([a0, a1]).all():
                        continue
                    visible, u, v = cv2.clipLine(right_rect, tuple(np.rint(a0).astype(int)), tuple(np.rint(a1).astype(int)))
                    if visible:
                        cv2.line(canvas, u, v, (175, 185, 185), 1, cv2.LINE_AA)
                if valid_tip:
                    xy = tuple(np.rint(left(tip)).astype(int))
                    color = (235, 210, 40) if guard['state'][k] == 2 else (130, 130, 130)
                    cv2.drawMarker(canvas, xy, color, cv2.MARKER_CROSS, 18, 2)
                z = right(zero)
                if np.isfinite(z).all():
                    cv2.circle(canvas, tuple(np.rint(z).astype(int)), 5, (135, 220, 120), 2)
                for j, a in enumerate(arrays):
                    state = int(a['transport_state'][q]) if q is not None else 0
                    value = float(a['offset_um'][q]) if q is not None else np.nan
                    number = f'{value:+.1f} um' if np.isfinite(value) else 'unavailable'
                    label = methods[j]+': '+number+' | '+STATES[state]
                    text(canvas, label, (18, 485+j*24), .51, COLORS[j])
                    if q is not None and a['transport_offset_available'][q]:
                        point = right(at_s(float(a['reference_s_px'][q])))
                        x, y, w, h = right_rect
                        if np.isfinite(point).all() and x+8 <= point[0] < x+w-8 and y+8 <= point[1] < y+h-8:
                            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 7+j*4, COLORS[j], 2)
                text(canvas, 'Gray tip = bounded interpolation | Gaps = unknown | Dashed holds remain conditional', (18, 539), .49)
                canvas[560:800] = background
                if valid_tip:
                    cursor = int(1280*(.065+.90*np.clip(depth/maximum_depth_mm, 0, 1)))
                    cv2.line(canvas, (cursor, 581), (cursor, 745), (210, 220, 225), 1)
                if i in previews:
                    cv2.imwrite(str(folder/(previews[i]+'.png')), canvas)
                process.stdin.write(np.ascontiguousarray(canvas).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('Video encoding failed')
        except BaseException:
            if process.poll() is None:
                process.kill()
            raise
        finally:
            cap.release()
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-xerror', '-i',
        str(temporary), '-f', 'null', '-'], check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    temporary.replace(destination)
    write_json(folder/'video_metadata.json', dict(width=W, height=H, fps=FPS, frames=len(d['frames']),
        source_frames=d['frames'], duration_s=len(d['frames'])/FPS,
        seconds=time.perf_counter()-started, bytes=destination.stat().st_size,
        unknown_measurement_frames_retained_in_video=True, optical_flow_recomputed=False,
        camera_interpolation_is_display_only=True, language='English',
        plot_maximum_depth_mm=float(maximum_depth_mm)))
