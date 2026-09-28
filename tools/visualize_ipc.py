"""Render saved IPC meshes and maximum principal Cauchy tensile stress.

No solves, displacement amplification, nodal averaging, or temporal interpolation.
All three principal stresses are used, including the plane-strain transverse stress.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors, font_manager
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prototype.material import response
from prototype.model_config import FEMConfig, local_to_lab
from run_support import write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frames(source, *, allow_failed_prefix=False):
    source = Path(source).resolve()
    report = json.loads((source / "result.json").read_text(encoding="utf-8"))
    complete = report.get('completed') is True and report.get('status') == 'completed'
    if not (complete or (allow_failed_prefix and report.get('completed') is False
                         and report.get('status') == 'failed')):
        raise ValueError('Require completed history or --allow-failed-prefix for an ended failed run; '
                         'running and inconsistent histories are not accepted')
    if report.get('simplifications', {}).get('fracture') is not False:
        raise ValueError('This visualization requires intact accepted states, without damage')
    steps = report.get('steps')
    if not isinstance(steps, list) or not steps:
        raise ValueError('No accepted states to visualize')
    increment, target = report['loading_increment_mm'], report['target_mm']
    if not np.isfinite([increment, target]).all() or increment <= 0 or target <= 0:
        raise ValueError('Invalid loading increment or target')
    expected = np.r_[0., np.arange(increment, target-1e-12, increment), target]
    if len(steps) > len(expected) or (complete and len(steps) != len(expected)):
        raise ValueError('Accepted history does not have the required loading coverage')
    depths = np.array([step['depth_mm'] for step in steps])
    if not np.allclose(depths, expected[:len(steps)], atol=1e-12, rtol=0):
        raise ValueError('Accepted history is not a continuous loading prefix')
    if {p.name for p in (source/'states').glob('*.npz')} != {f'{i:04d}.npz' for i in range(len(steps))}:
        raise ValueError('Missing or unreported accepted state files')
    if 'last_depth_mm' in report and abs(report['last_depth_mm']-depths[-1]) > 1e-12:
        raise ValueError('Last accepted depth mismatch')
    if 'last_checkpoint' in report and report['last_checkpoint'] != f'states/{len(steps)-1:04d}.npz':
        raise ValueError('Last accepted checkpoint mismatch')
    tolerance = report['force_tolerance_n']
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Invalid source force tolerance')
    # Legacy element runs recompute local stresses; mixed runs use the accepted
    # stress including the nonlocal pressure correction saved by the solver.
    for name in ("prototype/material.py", "mechanics/finite_strain.py"):
        if digest(ROOT / name) != report["source_sha256"][name]:
            raise ValueError(f"Material source differs from saved run: {name}")
        frozen = source/'source'/name
        if frozen.exists() and digest(frozen) != report['source_sha256'][name]:
            raise ValueError(f'Frozen material source differs from saved run: {name}')
    config = FEMConfig(**report["config"])
    volume = report.get("volumetric_discretization", "element")
    if volume not in ("element", "macro_mixed"):
        raise ValueError("Unknown saved volumetric discretization")
    frames, hashes = [], {}
    reference = triangles = inverse_edges = previous = body_count = None
    for i, step in enumerate(steps):
        if (not np.isfinite(step['residual_n']) or not 0 <= step['residual_n'] <= tolerance
                or not np.isfinite(step['reaction_fz_n'])
                or (step.get('reaction_fx_n') is not None and not np.isfinite(step['reaction_fx_n']))):
            raise ValueError('Unaccepted force residual or non-finite force')
        path = source / "states" / f"{i:04d}.npz"
        hashes[path.relative_to(source).as_posix()] = digest(path)
        with np.load(path, allow_pickle=False) as state:
            count = int(state["body_nodes"])
            xy = state["positions"].copy()
            if (xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all()
                    or xy.shape != state['reference_xy'].shape or not 0 < count < len(xy)):
                raise ValueError('Invalid body/blade coordinates')
            if 'damage' in state.files:
                raise ValueError('Intact history contains damage')
            if i == 0:
                reference = state["reference_xy"].copy()
                triangles = state["triangles"].copy()
                if (not np.isfinite(reference).all() or triangles.ndim != 2
                        or triangles.shape[1] != 3 or not len(triangles)
                        or not np.issubdtype(triangles.dtype, np.integer)
                        or triangles.min() < 0 or triangles.max() >= count):
                    raise ValueError('Invalid reference coordinates or topology')
                previous, body_count = reference, count
                X = reference[:count]
                edges = np.stack((X[triangles[:, 1]] - X[triangles[:, 0]],
                                  X[triangles[:, 2]] - X[triangles[:, 0]]), axis=-1)
                inverse_edges = np.linalg.inv(edges)
            else:
                np.testing.assert_array_equal(state["reference_xy"], reference)
                np.testing.assert_array_equal(state["triangles"], triangles)
            if count != body_count or not np.array_equal(state['previous_positions'], previous):
                raise ValueError('Broken accepted-state chain or changed body node count')
            np.testing.assert_allclose(float(state["depth_mm"]), step["depth_mm"], atol=1e-12, rtol=0)
            saved_stress = saved_j = None
            if volume == "macro_mixed":
                if str(state['volumetric_discretization']) != volume:
                    raise ValueError("State/report volumetric discretization mismatch")
                saved_stress, saved_j = state['cauchy'].copy(), state['J'].copy()
                if saved_stress.shape != (len(triangles), 3, 3) or saved_j.shape != (len(triangles),):
                    raise ValueError("Invalid saved mixed stress/J shape")
        x = xy[:count]
        edges = np.stack((x[triangles[:, 1]] - x[triangles[:, 0]],
                          x[triangles[:, 2]] - x[triangles[:, 0]]), axis=-1)
        F = edges @ inverse_edges
        if volume == "macro_mixed":
            np.testing.assert_allclose(saved_j, np.linalg.det(F), rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(saved_stress, saved_stress.swapaxes(-1, -2), atol=1e-9)
            stress, J = saved_stress, saved_j
        else:
            fields = response(F, config, tangent=False)
            stress, J = fields['cauchy'], fields['J']
        if not np.isfinite(stress).all() or not np.isfinite(J).all() or np.any(J <= 0):
            raise ValueError("Invalid saved stress or deformation")
        np.testing.assert_allclose(J.min(), step["min_j"], atol=1e-10)
        tensile = np.maximum(np.linalg.eigvalsh(stress)[:, -1], 0.)
        if not np.isfinite(tensile).all():
            raise ValueError("Non-finite stress")
        frames.append(dict(xy=xy, tensile=tensile, depth=step["depth_mm"],
                           force=step["reaction_fz_n"], force_x=step.get('reaction_fx_n'), min_J=step["min_j"],
                           peak_mpa=float(tensile.max())))
        previous = xy
    return report, config, reference, triangles, count, frames, hashes


def load_online_onset(source, report, frames, reference, count):
    """Tie the online primary event to its accepted frame, not a plotted peak."""
    path = Path(source)/'initiation/result.json'
    if not path.exists():
        return None, None
    audit = json.loads(path.read_text(encoding='utf-8'))
    if audit.get('mechanics_feedback') is not False or audit.get('fracture_evolution') is not False:
        raise ValueError('Expected independent intact-branch initiation diagnostic')
    event = audit.get('primary_event')
    if event is None:
        return None, digest(path)
    if audit.get('events', {}).get(audit.get('primary_criterion')) != event:
        raise ValueError('Primary initiation event and criterion record disagree')
    if 'initiation' in report and report['initiation'].get('primary_event') != event:
        raise ValueError('Initiation diagnostic and parent source history disagree')
    lower, upper = event['depth_bracket_mm']
    matches = [i for i, frame in enumerate(frames) if abs(frame['depth']-upper) <= 1e-12]
    if len(matches) != 1:
        raise ValueError('Initiation event is not in the accepted prefix')
    i = matches[0]
    prior = None if i == 0 else frames[i-1]['depth']
    if (lower is None) != (prior is None) or (lower is not None and abs(lower-prior) > 1e-12):
        raise ValueError('Initiation interval is not bounded by consecutive accepted states')
    if event.get('checkpoint') != f'states/{i:04d}.npz':
        raise ValueError('Initiation checkpoint does not match accepted event frame')
    evidence = event['evidence']
    if 'peak_node' in evidence:
        node = evidence['peak_node']
        if not isinstance(node, int) or not 0 <= node < count:
            raise ValueError('Invalid initiation body node')
        if not (np.allclose(evidence['reference_xz_mm'], reference[node], atol=1e-12, rtol=0)
                and np.allclose(evidence['current_xz_mm'], frames[i]['xy'][node], atol=1e-12, rtol=0)):
            raise ValueError('Initiation material point differs from accepted checkpoint')
    return event, digest(path)


def render_force_summary(report, frames, output, event=None):
    """Small signed-force plot; after-onset forces remain visibly conditional."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    depths = np.array([f['depth'] for f in frames])
    first = event['depth_bracket_mm'][1] if event else None
    force_x = np.array([np.nan if f['force_x'] is None else f['force_x'] for f in frames])
    with plt.rc_context({'font.size': 10}):
        fig, ax = plt.subplots(figsize=(8., 5.), layout='constrained')
        for forces, color, label in [(force_x, '#D55E00', 'Fx'),
                                     (np.array([f['force'] for f in frames]), '#0072B2', 'Fz')]:
            if not np.isfinite(forces).any():
                continue
            before = np.ones(len(depths), bool) if first is None else depths <= first+1e-12
            ax.plot(depths[before], forces[before], color=color, lw=1.7, label=label)
            if first is not None:
                after = depths >= first-1e-12
                ax.plot(depths[after], forces[after], color=color, ls='--', lw=1.7)
        if event:
            lower, upper = event['depth_bracket_mm']
            ax.axvspan(upper if lower is None else lower, upper, color='#777777', alpha=.22,
                       label='Conditional onset bracket')
            ax.axvline(upper, color='#777777', lw=.7)
        ax.axhline(0, color='#777777', lw=.6)
        ax.set(xlabel='Solver depth (mm; downward)', ylabel='Rubber force on knife (N)')
        config = FEMConfig(**report['config'])
        angle, mu = config.inclination_deg, config.friction_coefficient
        scope = ('Completed numerical loading' if report['completed'] else
                 f'FAILED run: accepted prefix only ({depths[-1]:g}/{report["target_mm"]:g} mm)')
        ax.set_title(f'{angle:g} deg, mu={mu:g} | {scope}\n'
                     +('Dashed: intact continuation after conditional onset' if event else
                       'Intact loading; no validated physical crack event'), fontsize=10)
        ax.grid(alpha=.2); ax.legend(fontsize=9)
        if not np.isfinite(force_x).all():
            fig.supxlabel('Missing native Fx is omitted; no zero value or support recovery is substituted.', fontsize=8)
        fig.savefig(output/'force_depth.png', dpi=160, facecolor='white')
        plt.close(fig)
    return dict(file='force_depth.png', sha256=digest(output/'force_depth.png'),
                force='signed native reaction_fx_n/reaction_fz_n; unavailable Fx omitted',
                post_onset_style='dashed intact counterfactual',
                onset_depth_bracket_mm=event['depth_bracket_mm'] if event else None,
                source_completed=report['completed'], source_status=report['status'],
                last_accepted_depth_mm=float(depths[-1]))


def render_observables(source, output, onset_result=None):
    """Plot exported material trajectories and signed forces, without IPC."""
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('export_complete'):
        raise ValueError('Expected completed observable export')
    for name, value in manifest['files_sha256'].items():
        if digest(source/name) != value:
            raise ValueError(f'Export changed after manifest: {name}')
    with (source/'forces.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with np.load(source/'surface_history.npz', allow_pickle=False) as saved:
        depths = saved['depths_mm']; X = saved['reference_xy']
        positions = saved['surface_positions']; ids = saved['surface_node_ids']
    fz = np.array([float(row['reaction_fz_n']) for row in rows])
    fx = np.array([float(row.get('reaction_fx_n') or row.get('recovered_fx_n') or 'nan') for row in rows])
    limit, event = depths[-1], None
    if onset_result:
        audit = json.loads(Path(onset_result).read_text(encoding='utf-8'))
        if audit.get('source_result_sha256') != manifest['source_result_sha256']:
            raise ValueError('Onset analysis and observable export must use the same source history')
        event = audit.get('events', {}).get('directional_at1_local_stability')
        if event:
            limit = event['depth_bracket_mm'][1]
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    before = depths <= limit+1e-12
    cfg = FEMConfig(**manifest['source_config'])
    contact = local_to_lab(np.array([[cfg.drop_s_mm, cfg.height_mm]]), cfg)[0]
    theta = np.deg2rad(cfg.inclination_deg)
    local_s = X @ np.array([np.cos(theta), -np.sin(theta)])
    tracked = np.unique([int(np.argmin(abs(local_s-s))) for s in cfg.drop_s_mm+np.linspace(-.6, .6, 7)])
    stages = np.unique([int(np.argmin(abs(depths-d))) for d in np.linspace(depths[0], limit, 4)])
    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    styles = ['-', '--', '-.', ':']
    with plt.rc_context({'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), layout='constrained')
        for force, label, color, style in [(fz, 'Fz (native)', palette[0], '-'),
                (fx, 'Fx (native / support recovery)', palette[1], '--')]:
            axes[0].plot(depths[before], force[before], color=color, ls=style, label=label)
            if event:
                tail = depths >= depths[np.flatnonzero(before)[-1]]
                axes[0].plot(depths[tail], force[tail], color=color, ls=':', alpha=.5)
        if event:
            lower, upper = event['depth_bracket_mm']
            axes[0].axvspan(upper if lower is None else lower, upper, color='#888888', alpha=.3,
                           label='Conditional diffuse onset bracket')
        axes[0].axhline(0, color='#777777', lw=.6)
        axes[0].set(xlabel='Solver depth (mm; downward)', ylabel='Rubber force on knife (N)',
                    title='Signed knife reaction')
        axes[0].legend(fontsize=8)
        for j, index in enumerate(stages):
            xy = positions[index]-contact
            axes[1].plot(xy[:, 0], xy[:, 1], color=palette[j], ls=styles[j],
                         label=f'{depths[index]:.2f} mm')
        if event:
            point = np.asarray(event['evidence']['current_xz_mm'])-contact
            axes[1].plot(*point, '*', color='black', ms=9, label='Conditional onset node')
        for j, index in enumerate(tracked):
            xy = positions[before, index]-contact
            color = palette[j % len(palette)]
            axes[2].plot(xy[:, 0], xy[:, 1], color=color, ls=styles[j % len(styles)],
                         label=f'node {ids[index]}')
            axes[2].plot(xy[0, 0], xy[0, 1], 'o', color=color, ms=3)
            axes[2].plot(xy[-1, 0], xy[-1, 1], 'x', color=color, ms=4)
        for ax, title in zip(axes[1:], ['Original top material boundary', 'Fixed material point trajectories']):
            ax.set(xlabel='x - reference drop point (mm)', ylabel='z - reference drop point (mm)',
                   title=title, xlim=(-1.5, 1.5), ylim=(-limit-1.5, 1.5), aspect='equal')
            ax.legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
        scope = ('Dotted force tail: intact counterfactual after conditional onset. '
                 if event else 'No validated physical crack event assigned. ')
        fig.suptitle(f'{cfg.inclination_deg:g} deg, mu={cfg.friction_coefficient:g}: accepted numerical history\n'
                     +scope+'Motion unscaled; no experimental validation.', fontsize=11)
        fig.savefig(output/'cutting_observables.png', dpi=160, facecolor='white')
        fig.savefig(output/'cutting_observables.svg', facecolor='white')
        plt.close(fig)
    write_json(output/'manifest.json', dict(source_export=str(source),
        source_manifest_sha256=digest(source/'manifest.json'), script_sha256=digest(Path(__file__)),
        onset_result=str(onset_result) if onset_result else None,
        onset_sha256=digest(Path(onset_result)) if onset_result else None,
        stages=stages.tolist(), tracked_material_nodes=ids[tracked].tolist(), motion_max_depth_mm=float(limit),
        transformations='No smoothing or amplification. Coordinates translated by fixed undeformed drop point.',
        depth='Original solver coordinate; geometric plane offset is available separately in source export.',
        force='Native Fz; Fx native where available, otherwise explicitly verified support recovery.',
        matplotlib=matplotlib.__version__, numpy=np.__version__,
        alt_text='Signed knife force versus solver depth, original top boundary at four depths, and seven material trajectories.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument('--observables', action='store_true', help='Plot export_cutting_history output without ffmpeg')
    parser.add_argument('--onset-result', type=Path, help='Optional same-source conditional onset report')
    parser.add_argument('--allow-failed-prefix', action='store_true',
                        help='Render only accepted states from an ended failed run; never an active or rejected trial')
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("fps must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty")
    if args.observables:
        render_observables(args.source, args.output, args.onset_result)
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("Install ffmpeg and put it on PATH to export MP4")
    report, config, reference, triangles, count, frames, hashes = load_frames(
        args.source, allow_failed_prefix=args.allow_failed_prefix)
    onset_event, onset_hash = load_online_onset(args.source, report, frames, reference, count)
    args.output.mkdir(parents=True, exist_ok=True)
    force_summary = render_force_summary(report, frames, args.output, onset_event)
    max_stress = max(f["peak_mpa"] for f in frames)
    model_label = (f"有限应变平面应变 · 倾角 {config.inclination_deg:g}° · μ={config.friction_coefficient:g} · "
                   f"刀尖半径 {report['simplifications']['tip_radius_mm']:g} mm · 位移不放大")
    if report.get('volumetric_discretization', 'element') == 'macro_mixed':
        model_label += ' · 混合位移/压力'
    coverage_label = ('已完成数值加载' if report['completed'] else
                      f"失败算例的已接受前段：{frames[-1]['depth']:g}/{report['target_mm']:g} mm")
    def branch_label(frame):
        if onset_event and frame['depth'] >= onset_event['depth_bracket_mm'][1]-1e-12:
            return '条件性起裂候选后：完整材料反事实续算'
        return '完整材料分支；未确认物理裂纹'
    friction_label = "无摩擦" if config.friction_coefficient == 0 else f"μ={config.friction_coefficient:g}"
    upper = max(.5, np.ceil(max_stress * 2) / 2)
    norm = colors.Normalize(0, upper)
    corners = local_to_lab(np.array([[0., 0.], [config.length_mm, 0.],
                                    [config.length_mm, config.height_mm], [0., config.height_mm]]), config)
    contact = local_to_lab(np.array([[config.drop_s_mm, config.height_mm]]), config)[0]
    zoom = (contact[0]-3., contact[0]+3., contact[1]-report["target_mm"]-2., contact[1]+2.)
    body_all = np.concatenate([f["xy"][:count] for f in frames])
    overall = (float(body_all[:, 0].min())-1., float(body_all[:, 0].max())+1.,
               float(body_all[:, 1].min())-1., float(body_all[:, 1].max())+1.)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = "Microsoft YaHei"
    else:
        # Matplotlib can select any installed CJK font from this fallback list.
        family = ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]

    def axes_style(ax, limits):
        ax.set(xlim=limits[:2], ylim=limits[2:], aspect="equal", xlabel="实验室 x (mm)", ylabel="实验室 z (mm)")
        ax.set_facecolor("#f7f8fa")
        ax.spines[["top", "right"]].set_visible(False)

    def draw_body(ax, frame, stress):
        if stress:
            mesh = PolyCollection(frame["xy"][:count][triangles], cmap="viridis", norm=norm,
                                  edgecolors=(1., 1., 1., .3), linewidths=.25, rasterized=True)
            mesh.set_array(frame["tensile"])
        else:
            mesh = PolyCollection(frame["xy"][:count][triangles], facecolors="#e0eef7",
                                  edgecolors="#37657d", linewidths=.27, rasterized=True)
        ax.add_collection(mesh)
        knife = Polygon(frame["xy"][count:], facecolor="#8b9099", edgecolor="#333b43", linewidth=.9)
        ax.add_patch(knife)
        return mesh, knife

    def tip_window(frame):
        knife = frame["xy"][count:]
        tip = knife[np.argmin(knife[:, 1])]
        return (tip[0]-.9, tip[0]+.9, tip[1]-1.3, tip[1]+1.1)

    with plt.rc_context({"font.family": family, "font.size": 11, "axes.unicode_minus": False,
                         "animation.ffmpeg_path": ffmpeg}):
        fig = plt.figure(figsize=(15.2, 7.5), dpi=120)
        grid = fig.add_gridspec(1, 3, left=.045, right=.915, bottom=.20, top=.77,
                               width_ratios=[.85, 1.1, 1.05], wspace=.30)
        left, middle, right = [fig.add_subplot(grid[i]) for i in range(3)]
        fig.suptitle("无断裂压入过程：变形网格与最大主拉应力", x=.06, y=.96, ha="left", fontsize=18)
        fig.text(.06, .90, model_label, color="#444444")
        status = fig.text(.06, .845, "", fontsize=12)
        branch_status = fig.text(.06, .125, '', fontsize=10, color='#8a4300')
        axes_style(left, overall)
        axes_style(middle, zoom)
        axes_style(right, tip_window(frames[0]))
        left.set_title("整体变形网格")
        middle.set_title("压入区域网格（固定视窗）")
        right.set_title("拉应力放大（视窗随刀尖移动）", fontsize=11)
        left.add_patch(Polygon(corners, closed=True, fill=False, linestyle="--", edgecolor="#999999", linewidth=1.))
        left.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1]-zoom[0], zoom[3]-zoom[2],
                                 fill=False, edgecolor="#b45c00", linewidth=1.2))
        bodies = [draw_body(left, frames[0], False), draw_body(middle, frames[0], False),
                  draw_body(right, frames[0], True)]
        window = tip_window(frames[0])
        zoom_box = Rectangle((window[0], window[2]), window[1]-window[0], window[3]-window[2],
                             fill=False, edgecolor="#b45c00", linewidth=1.2)
        middle.add_patch(zoom_box)
        cax = fig.add_axes([.935, .24, .012, .46])
        fig.colorbar(bodies[2][0], cax=cax, label=r"最大主拉应力 $\sigma_1^+$ (MPa)")
        fig.text(.06, .075, r"全程固定线性色标；$\sigma_1^+$ = max(三维 Cauchy 应力最大主值, 0)，包含出平面应力。", fontsize=10)
        fig.text(.06, .028, "P1 单元内常值，未做节点平均或空间平滑；动画按记录步播放，不代表实际加载速度。", fontsize=10)
        writer = animation.FFMpegWriter(fps=args.fps, codec="libx264", bitrate=2400,
                                        extra_args=["-pix_fmt", "yuv420p", "-threads", "2"],
                                        metadata={"title": "IPC unfractured indentation: mesh and principal tensile stress"})
        with writer.saving(fig, str(args.output / "indentation_tensile.mp4"), dpi=120):
            for i, frame in enumerate(frames):
                for mesh, knife in bodies:
                    mesh.set_verts(frame["xy"][:count][triangles])
                    knife.set_xy(frame["xy"][count:])
                bodies[2][0].set_array(frame["tensile"])
                window = tip_window(frame)
                right.set(xlim=window[:2], ylim=window[2:])
                zoom_box.set_xy((window[0], window[2]))
                status.set_text(f"下压 {frame['depth']:.3f} mm     Fz = {frame['force']:.2f} N     全域峰值 = {frame['peak_mpa']:.2f} MPa     [{i+1}/{len(frames)}]")
                branch_status.set_text(coverage_label+' · '+branch_label(frame))
                writer.grab_frame(facecolor="white")
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(15.2, 5.4))
        fig.subplots_adjust(left=.045, right=.91, bottom=.2, top=.76, wspace=.30)
        fig.suptitle("压入过程的最大主拉应力与变形网格", x=.045, y=.965, ha="left", fontsize=17)
        fig.text(.045, .87, f"{friction_label}、无断裂；{coverage_label}；真实位移与统一色标", fontsize=11)
        keyframes = []
        for ax, depth in zip(axes, np.linspace(0, frames[-1]['depth'], 4)):
            i = int(np.argmin([abs(f["depth"]-depth) for f in frames]))
            frame = frames[i]
            keyframes.append(i)
            axes_style(ax, tip_window(frame))
            mesh, _ = draw_body(ax, frame, True)
            ax.set_title(f"下压 {frame['depth']:.2f} mm\n全域峰值 {frame['peak_mpa']:.2f} MPa", fontsize=11)
            if onset_event and frame['depth'] >= onset_event['depth_bracket_mm'][1]-1e-12:
                ax.text(.5, -.25, '候选后：完整材料续算', transform=ax.transAxes,
                        ha='center', fontsize=9, color='#8a4300')
        fig.colorbar(mesh, cax=fig.add_axes([.935, .24, .012, .45]), label=r"最大主拉应力 $\sigma_1^+$ (MPa)")
        fig.text(.045, .09, "云图含平面应变出平面应力；颜色仅表示拉应力大小，不表示损伤或已形成裂纹。", fontsize=10)
        fig.savefig(args.output / "tensile_stages.png", dpi=160, facecolor="white")
        plt.close(fig)

    metadata = dict(source=str(args.source),result_sha256_at_export_start=digest(args.source/"result.json"),
                    checkpoints_sha256=hashes,script_sha256=digest(Path(__file__)),
                    material_sha256={n:digest(ROOT/n) for n in ("prototype/material.py","mechanics/finite_strain.py")},
                    stress="max(maximum eigenvalue of full 3D Cauchy stress, 0)",units="MPa",sampling="P1 elementwise constant",
                    scale=[0,float(upper)],scale_type="linear, fixed over all frames",displacement_scale=1,
                    camera="overview and indentation window fixed; tensile-stress close-up follows blade tip with fixed 1.8 x 2.4 mm span",
                    smoothing=False,frames=len(frames),fps=args.fps,playback_seconds=len(frames)/args.fps,
                    keyframe_indices=keyframes,matplotlib=matplotlib.__version__,numpy=np.__version__,
                    peaks=[dict(depth_mm=f["depth"],maximum_tensile_mpa=f["peak_mpa"]) for f in frames],
                    friction_coefficient=config.friction_coefficient,
                    source_completed=report['completed'], source_status=report['status'],
                    coverage='complete source loading' if report['completed'] else 'accepted prefix only; source failed',
                    last_accepted_depth_mm=frames[-1]['depth'], target_mm=report['target_mm'],
                    initiation_result_sha256=onset_hash, primary_onset_event=onset_event,
                    force_summary=force_summary,
                    note="Numerical unfractured indentation; not a crack prediction or experimental validation")
    write_json(args.output / "manifest.json", metadata)
    print(json.dumps(dict(frames=len(frames),maximum_tensile_mpa=max_stress,output=str(args.output))))


if __name__ == "__main__":
    main()
