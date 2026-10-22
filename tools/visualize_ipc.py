"""Render saved IPC meshes and maximum principal Cauchy tensile stress.

No solves, displacement amplification, nodal averaging, or temporal interpolation.
All three principal stresses are used, including the plane-strain transverse stress.
Tip contact is read from accepted-state diagnostics, never inferred from a picture.
"""
import argparse
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
from prototype.contact_trace import validate_loading_schedule
from prototype.material import response
from mechanics.finite_strain import Material, response as maxwell_response
from prototype.model_config import FEMConfig, local_to_lab
from run_support import write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tip_contact_display(contact):
    """Return the saved contact location and label without choosing a candidate."""
    if not isinstance(contact, dict):
        return None, "Lowest blade point: not recorded"
    status = contact.get("status", "not_recorded")
    if status != "unique":
        labels = {"no_tip_contact": "no contact", "ambiguous": "ambiguous"}
        nose = contact.get("nose_status")
        suffix = "; rounded nose: contact" if nose in ("connected", "disconnected") else ""
        return None, "Lowest blade point: " + labels.get(status, str(status)) + suffix
    point = np.asarray(contact.get("current_point_mm"), dtype=float)
    offset = contact.get("reference_offset_mm")
    if point.shape != (2,) or not np.isfinite(point).all() or offset is None or not np.isfinite(offset):
        raise ValueError("A unique tip contact requires a finite location and reference offset")
    return point, rf"Lowest blade point contact: $\Delta s_0$ = {float(offset):+.4f} mm"


def _make_tip_contact_artist(ax, *, show_label=True, fontsize=9):
    marker, = ax.plot([], [], linestyle="none", marker="D", markersize=6,
                      markerfacecolor="#a00074", markeredgecolor="white",
                      markeredgewidth=1., zorder=8, label="Unique lowest-point contact")
    label = (ax.text(.02, .025, "", transform=ax.transAxes, fontsize=fontsize,
                     va="bottom", color="#222222", zorder=9,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=.93, pad=2.))
             if show_label else None)
    return marker, label


def _update_tip_contact_artist(artist, contact):
    marker, label = artist
    point, text = _tip_contact_display(contact)
    marker.set_data([], []) if point is None else marker.set_data([point[0]], [point[1]])
    if label is not None:
        label.set_text(text)


def plot_tip_contact(records, outputpath):
    """Plot saved material offsets and separate nose-contact intervals versus depth.

    ``records`` are accepted result steps with ``depth_mm`` and ``tip_contact``.
    Only unique tip contacts form the offset curve. Missing/non-unique steps break
    that curve. Each nose interval is a separate vertical segment at its recorded
    depth; these segments are contact extents, not confidence intervals. The input
    order is preserved and no spatial or temporal smoothing is performed.
    """
    records = list(records)
    if not records:
        raise ValueError("At least one accepted step is required")
    depths = np.asarray([r["depth_mm"] for r in records], dtype=float)
    if not np.isfinite(depths).all() or np.any(np.diff(depths) <= 0):
        raise ValueError("Accepted depths must be finite and strictly increasing")
    offsets = np.full(len(records), np.nan)
    statuses, segments = [], []
    for i, record in enumerate(records):
        contact = record.get("tip_contact")
        _point, _label = _tip_contact_display(contact)
        status = contact.get("status", "not_recorded") if isinstance(contact, dict) else "not_recorded"
        statuses.append(status)
        if status == "unique":
            offsets[i] = float(contact["reference_offset_mm"])
        if not isinstance(contact, dict):
            continue
        for interval in contact.get("nose_offset_intervals_mm", []) or []:
            pair = np.asarray(interval, dtype=float)
            if pair.shape != (2,) or not np.isfinite(pair).all() or pair[1] < pair[0]:
                raise ValueError("Nose-contact intervals must be ordered finite pairs")
            segments.append((depths[i], float(pair[0]), float(pair[1])))

    outputpath = Path(outputpath)
    outputpath.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, (ax, state_ax) = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True,
                                         gridspec_kw={"height_ratios": [4, 1.15]},
                                         layout="constrained")
        ax.axhline(0., color="#555555", linestyle="--", linewidth=.8,
                   label="Nominal initial nose-contact material coordinate")
        if segments:
            ds, lo, hi = np.asarray(segments).T
            ax.vlines(ds, lo, hi, color="#0072b2", linewidth=1.4, alpha=.8,
                      label="Nose-contact intervals (each depth)")
            point_intervals = lo == hi
            ax.scatter(ds[point_intervals], lo[point_intervals], color="#0072b2",
                       marker="_", s=18, zorder=2)
        ax.plot(depths, offsets, color="#a00074", marker="D", markersize=3,
                linewidth=1.2, label="Unique lowest-point contact; gaps retained")
        ax.set(ylabel=r"Reference surface offset $\Delta s_0$ (mm)",
               title="Unfractured indentation: material coordinate at the lowest blade point")
        ax.grid(color="#dddddd", linewidth=.6)
        ax.legend(loc="best", fontsize=9)
        status_styles = (("unique", "Unique", "#a00074", "D"),
                         ("no_tip_contact", "No lowest-point contact", "#555555", "x"),
                         ("ambiguous", "Ambiguous", "#b55c00", "^"),
                         ("not_recorded", "Not recorded", "#666666", "s"))
        statuses = np.asarray(statuses)
        for j, (status, label, color, marker) in enumerate(status_styles):
            mask = statuses == status
            if status == "not_recorded":
                mask |= ~np.isin(statuses, [s[0] for s in status_styles])
            state_ax.scatter(depths[mask], np.full(mask.sum(), j), marker=marker,
                             s=16, color=color)
        state_ax.set(yticks=np.arange(4), yticklabels=[s[1] for s in status_styles],
                     ylim=(-.5, 3.5), xlabel="Downward blade depth (mm)")
        state_ax.grid(axis="x", color="#dddddd", linewidth=.6)
        for axes in (ax, state_ax):
            axes.spines[["top", "right"]].set_visible(False)
        fig.get_layout_engine().set(rect=(0., .08, 1., .92))
        fig.text(.5, .025,
                 "Signed distance along the ORIGINAL material surface; positive follows increasing s.\n"
                 "Separate nose-contact segments are not uncertainty bounds. No crack is predicted.",
                 ha="center", va="bottom", fontsize=9, color="#333333")
        fig.savefig(outputpath, dpi=160, facecolor="white", metadata={
            "Title": "Blade-tip contact material offset versus indentation depth",
            "Description": "Accepted-state diagnostics only; no smoothing; non-unique or missing tip contacts remain gaps. "
                           "Nose-contact intervals are displayed separately at each depth, not merged or filled.",
            "Software": f"Matplotlib {matplotlib.__version__}; {Path(__file__).name}",
        })
        plt.close(fig)
    return outputpath


def load_frames(source):
    report = json.loads((source / "result.json").read_text(encoding="utf-8"))
    if report.get('discretization', {}).get('element', 'p1') != 'p1':
        raise ValueError('This renderer supports native P1 states only')
    if not report.get("completed"):
        raise ValueError("This visualization expects a completed, continuous run")
    validate_loading_schedule(report, require_complete=True)
    is_maxwell = 'material' in report.get('signature', {})
    material = None
    if is_maxwell:
        from tools.qualify_intact_baseline import verify_case
        verify_case(source)
        values = dict(report['signature']['material'])
        for key in ('mu_eq','alpha_eq','mu_ov','alpha_ov'):values[key]=tuple(values[key])
        material = Material(**values)
    # Maxwell postprocessing calls the shared kernel directly; a retired elastic
    # adapter is not a source dependency. A changed kernel still fails closed.
    source_names = ('mechanics/finite_strain.py',) if is_maxwell else ('prototype/material.py','mechanics/finite_strain.py')
    for name in source_names:
        if digest(ROOT / name) != report["source_sha256"][name]:
            raise ValueError(f"Material source differs from saved run: {name}")
    config = FEMConfig(**report.get("config",report.get('signature',{}).get('config',{})))
    frames, hashes = [], {}
    reference = triangles = inverse_edges = None
    for i, step in enumerate(report["steps"]):
        path = source / "states" / f"{i:04d}.npz"
        hashes[path.relative_to(source).as_posix()] = digest(path)
        with np.load(path,allow_pickle=False) as state:
            count = int(state["body_nodes"])
            xy = state["positions"].copy()
            if i == 0:
                reference = state["reference_xy"].copy()
                triangles = state["triangles"].copy()
                X = reference[:count]
                edges = np.stack((X[triangles[:, 1]] - X[triangles[:, 0]],
                                  X[triangles[:, 2]] - X[triangles[:, 0]]), axis=-1)
                inverse_edges = np.linalg.inv(edges)
            else:
                np.testing.assert_array_equal(state["reference_xy"], reference)
                np.testing.assert_array_equal(state["triangles"], triangles)
            np.testing.assert_allclose(float(state["depth_mm"]), step["depth_mm"], atol=1e-12)
            if is_maxwell:
                previous_metric=state['previous_material_metric'].copy()
                saved_metric=state['material_metric'].copy()
                material_dt=float(state['material_dt_s'])
                expected_metric=np.tile(np.eye(3),(len(triangles),1,1)) if i==0 else last_metric
                np.testing.assert_array_equal(previous_metric,expected_metric)
                last_metric=saved_metric

        x = xy[:count]
        edges = np.stack((x[triangles[:, 1]] - x[triangles[:, 0]],
                          x[triangles[:, 2]] - x[triangles[:, 0]]), axis=-1)
        F=edges @ inverse_edges
        if is_maxwell:
            fields=maxwell_response(F,previous_metric,material_dt,material,tangent=False,kinematics='plane_strain')
            np.testing.assert_allclose(fields['metric'],saved_metric,rtol=1e-10,atol=1e-11)
            cauchy=np.zeros((len(F),3,3))
            cauchy[:,:2,:2]=(fields['P'] @ F.transpose(0,2,1))/fields['J'][:,None,None]
            cauchy[:,2,2]=fields['plane_stress_residual']/fields['J']
            fields['cauchy']=cauchy
        else:fields=response(F,config,tangent=False)
        np.testing.assert_allclose(fields["J"].min(), step["min_j"], atol=1e-10)
        tensile = np.maximum(np.linalg.eigvalsh(fields["cauchy"])[:, -1], 0.)
        if not np.isfinite(tensile).all():
            raise ValueError("Non-finite stress")
        frames.append(dict(xy=xy, tensile=tensile, depth=step["depth_mm"],
                           force=step["reaction_fz_n"], min_J=step["min_j"],
                           peak_mpa=float(tensile.max()), tip_contact=step.get("tip_contact")))
    return report, config, reference, triangles, count, frames, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("fps must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("Install ffmpeg and put it on PATH to export MP4")
    report, config, reference, triangles, count, frames, hashes = load_frames(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    max_stress = max(f["peak_mpa"] for f in frames)
    model_label = (f"有限应变平面应变 · 倾角 {config.inclination_deg:g}° · μ={config.friction_coefficient:g} · "
                   "原始接受网格 · 位移不放大")
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
        contact_artists = [_make_tip_contact_artist(ax, show_label=(ax is right))
                           for ax in (left, middle, right)]
        window = tip_window(frames[0])
        zoom_box = Rectangle((window[0], window[2]), window[1]-window[0], window[3]-window[2],
                             fill=False, edgecolor="#b45c00", linewidth=1.2)
        middle.add_patch(zoom_box)
        cax = fig.add_axes([.935, .24, .012, .46])
        fig.colorbar(bodies[2][0], cax=cax, label=r"最大主拉应力 $\sigma_1^+$ (MPa)")
        fig.text(.06, .125, r"Diamond: lowest-point contact; $\Delta s_0$: ORIGINAL surface offset from nominal initial nose contact (not accumulated slip).", fontsize=10)
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
                for artist in contact_artists:
                    _update_tip_contact_artist(artist, frame["tip_contact"])
                status.set_text(f"下压 {frame['depth']:.3f} mm     Fz = {frame['force']:.2f} N     全域峰值 = {frame['peak_mpa']:.2f} MPa     [{i+1}/{len(frames)}]")
                writer.grab_frame(facecolor="white")
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(15.2, 5.4))
        fig.subplots_adjust(left=.045, right=.91, bottom=.2, top=.76, wspace=.30)
        fig.suptitle("压入过程的最大主拉应力与变形网格", x=.045, y=.965, ha="left", fontsize=17)
        fig.text(.045, .87, f"{friction_label}、无断裂；视窗随刀尖移动，跨度与色标统一；真实位移，P1 单元应力未平滑", fontsize=11)
        keyframes = []
        for ax, depth in zip(axes, np.linspace(0, report["target_mm"], 4)):
            i = int(np.argmin([abs(f["depth"]-depth) for f in frames]))
            frame = frames[i]
            keyframes.append(i)
            axes_style(ax, tip_window(frame))
            mesh, _ = draw_body(ax, frame, True)
            _update_tip_contact_artist(_make_tip_contact_artist(ax, fontsize=8), frame["tip_contact"])
            ax.set_title(f"下压 {frame['depth']:.2f} mm\n全域峰值 {frame['peak_mpa']:.2f} MPa", fontsize=11)
        fig.colorbar(mesh, cax=fig.add_axes([.935, .24, .012, .45]), label=r"最大主拉应力 $\sigma_1^+$ (MPa)")
        fig.text(.045, .135, r"Magenta diamond: unique tip contact; $\Delta s_0$: signed ORIGINAL surface offset from initial contact.", fontsize=9)
        fig.text(.045, .09, "云图含平面应变出平面应力；颜色仅表示拉应力大小，不表示损伤或已形成裂纹。", fontsize=10)
        fig.savefig(args.output / "tensile_stages.png", dpi=160, facecolor="white")
        plt.close(fig)

    if any(isinstance(step.get("tip_contact"), dict) for step in report["steps"]):
        plot_tip_contact(report["steps"], args.output / "tip_contact_offset.png")
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
                    tip_contact="Saved accepted-state tracker only; diamond shown only for unique status; old runs may have no diagnostic",
                    tip_contact_offset="Signed reference surface coordinate minus initial contact coordinate; not a crack prediction",
                    note="Numerical unfractured indentation; not a crack prediction or experimental validation")
    write_json(args.output / "manifest.json", metadata)
    print(json.dumps(dict(frames=len(frames),maximum_tensile_mpa=max_stress,output=str(args.output))))


if __name__ == "__main__":
    main()
