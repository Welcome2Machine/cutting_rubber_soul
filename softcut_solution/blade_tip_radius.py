"""Image-only local circle estimates, with explicit segmentation sensitivity.

No force input or FE loading. A fitted circle is not automatically a physical
blade radius. Source pixels and diagnostic outputs remain outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .blade_profiles import scale_bar


def fit_circle(points):
    """Orthogonal-distance circle fit; NumPy LM, multiple geometric starts.

    Coordinates are translated before fitting. Report conditioning and bounds:
    a near-straight arc cannot establish a radius from a small residual alone.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 6 or not np.isfinite(points).all():
        raise ValueError('Need at least six finite contour points')
    origin = points.mean(axis=0)
    p = points-origin
    if np.linalg.matrix_rank(p) < 2:
        raise ValueError('Collinear contour does not determine a circle')
    a = np.column_stack((2*p, np.ones(len(p))))
    v = np.linalg.lstsq(a, np.sum(p*p, axis=1), rcond=None)[0]
    front = p[:, 0].min()
    lower = np.array([front, -200., .1])
    upper = np.array([front+200, 200., 200.])
    starts = [np.r_[v[:2], np.sqrt(max(.01, v[2]+v[:2]@v[:2]))]]
    starts += [np.array([front+r, 0., r]) for r in (2., 5., 15., 60.)]
    best = None
    for initial in starts:
        z = np.clip(initial, lower, upper)
        damping = .001
        for _ in range(120):
            delta = z[:2]-p
            distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
            residual = distance-z[2]
            jac = np.column_stack((delta/distance[:, None], -np.ones(len(p))))
            h = jac.T@jac
            step = np.linalg.solve(h+damping*np.diag(np.maximum(np.diag(h), 1.)), -jac.T@residual)
            trial = np.clip(z+step, lower, upper)
            new = np.linalg.norm(p-trial[:2], axis=1)-trial[2]
            if new@new < residual@residual:
                z = trial
                damping = max(damping/3, 1e-12)
                if np.linalg.norm(step) < 1e-9:
                    break
            else:
                damping *= 10
                if damping > 1e12:
                    break
        residual = np.linalg.norm(p-z[:2], axis=1)-z[2]
        score = float(residual@residual)
        if best is None or score < best[0]:
            best = score, z.copy(), residual.copy()
    _, z, residual = best
    delta = z[:2]-p
    jac = np.column_stack((delta/np.linalg.norm(delta, axis=1)[:, None], -np.ones(len(p))))
    angles = np.sort(np.mod(np.arctan2(p[:, 1]-z[1], p[:, 0]-z[0]), 2*np.pi))
    coverage = 2*np.pi-np.diff(np.r_[angles, angles[0]+2*np.pi]).max()
    return dict(centre_px=(z[:2]+origin).tolist(), radius_px=float(z[2]),
                rms_px=float(np.sqrt(np.mean(residual**2))), max_error_px=float(abs(residual).max()),
                condition=float(np.linalg.cond(jac)), angular_coverage_deg=float(np.degrees(coverage)),
                bound_active=bool(np.any(np.isclose(z, lower, atol=1e-5)) or np.any(np.isclose(z, upper, atol=1e-5))),
                contour_point_count=len(p))


def subtract_background(gray, strips, shift=0):
    """Interpolate per-column background medians from two clean horizontal strips."""
    (a, b), (c, d) = np.asarray(strips, int)+shift
    if not (0 <= a < b < c < d <= len(gray)):
        raise ValueError('Invalid background strips')
    y1, y2 = (a+b-1)/2, (c+d-1)/2
    blend = (np.arange(len(gray))-y1)/(y2-y1)
    background = (1-blend[:, None])*np.median(gray[a:b], axis=0)+blend[:, None]*np.median(gray[c:d], axis=0)
    return gray-background


def edge_contour(contrast, roi, level, seed):
    from matplotlib.figure import Figure
    x0, y0, x1, y1 = roi
    ax = Figure().subplots()
    cs = ax.contour(np.arange(x0, x1), np.arange(y0, y1), contrast[y0:y1, x0:x1], levels=[level])
    # Copy before any labels: inline contour labels can remove path segments.
    candidates = [s.copy() for s in cs.allsegs[0] if len(s) > 8 and s[:, 0].max() >= x1-1.01]
    if not candidates:
        raise ValueError('No connected blade contour reaches the right ROI boundary')
    return min(candidates, key=lambda s: np.linalg.norm(s-np.asarray(seed), axis=1).min())


def resample_arc(contour, spacing=.5):
    keep = np.r_[True, np.linalg.norm(np.diff(contour, axis=0), axis=1) > 1e-10]
    p = contour[keep]
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.linspace(0, length[-1], int(np.ceil(length[-1]/spacing))+1)
    return np.column_stack([np.interp(s, length, p[:, k]) for k in (0, 1)])


def select_cap(contour, yrange, depth):
    p = contour[(contour[:, 1] >= yrange[0]) & (contour[:, 1] <= yrange[1])]
    if not len(p):
        raise ValueError('Empty feature window')
    return p[p[:, 0] <= p[:, 0].min()+depth]


def analyze(rgb, spec, method):
    scale = scale_bar(rgb, spec['scale']['length_um'])
    gray = rgb[:, :, :3].mean(axis=2)*255
    records, failures = [], []
    for shift in method['background_shifts_px']:
        contrast = subtract_background(gray, method['background_strips_y'], shift)
        for level in method['contrast_levels_255']:
            contour = edge_contour(contrast, spec['roi_xyxy'], level, spec['contour_seed_xy'])
            for sampling, sampled in [('vertices', contour), ('arc_0.5px', resample_arc(contour))]:
                for feature in spec['features']:
                    for depth in feature['fit_depths_px']:
                        identity = dict(feature=feature['id'], background_shift_px=shift, level=level,
                                        sampling=sampling, depth_px=depth)
                        try:
                            fit = fit_circle(select_cap(sampled, feature['y_range_px'], depth))
                            records.append(dict(**identity, **fit, radius_um=fit['radius_px']*scale['um_per_pixel']))
                        except ValueError as error:
                            failures.append(dict(**identity, error=str(error)))
    summaries = []
    for feature in spec['features']:
        fits = [r for r in records if r['feature'] == feature['id']]
        if not fits:
            raise ValueError('No fits for '+feature['id'])
        radii = [r['radius_um'] for r in fits]
        scale_range = scale['endpoint_sensitivity_um_per_pixel']
        summaries.append(dict(feature=feature['id'], interpretation=feature['interpretation'],
            radius_median_um=float(np.median(radii)), image_processing_range_um=[min(radii), max(radii)],
            with_scale_endpoint_range_um=[min(radii)*min(scale_range)/scale['um_per_pixel'],
                                          max(radii)*max(scale_range)/scale['um_per_pixel']],
            rms_range_px=[min(r['rms_px'] for r in fits), max(r['rms_px'] for r in fits)],
            bound_active_count=sum(r['bound_active'] for r in fits), fit_count=len(fits),
            physical_prediction_qualified=False))
    return dict(id=spec['id'], image_number=spec['image_number'], source_sha256=spec['source_sha256'],
                scale=scale, summaries=summaries, fits=records, failures=failures)


def plot_review(rgb, spec, method, result, destination):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    contrast = subtract_background(rgb[:, :, :3].mean(axis=2)*255, method['background_strips_y'])
    level = 35
    contour = edge_contour(contrast, spec['roi_xyxy'], level, spec['contour_seed_xy'])
    x0, y0, x1, y1 = spec['roi_xyxy']
    um_per_pixel = result['scale']['um_per_pixel']
    for ax in axes:
        ax.imshow(rgb, interpolation='nearest')
        ax.set(xlim=(x0, min(x1, x0+34)), ylim=(y1, y0), xlabel=f'x [pixel]; {um_per_pixel:g} um/pixel', ylabel='y [pixel]')
    axes[0].set_title(f"Image {spec['image_number']}: {spec['performance']}")
    axes[1].set_title('Contrast-level sensitivity')
    for lev in method['contrast_levels_255']:
        p = edge_contour(contrast, spec['roi_xyxy'], lev, spec['contour_seed_xy'])
        axes[1].plot(*p.T, lw=.9, label=f'{lev}')
    axes[1].legend(fontsize=8)
    axes[2].set_title('Local equivalent circles; not FE geometry')
    for feature in spec['features']:
        if feature.get('diagnostic_only'):
            continue
        p = select_cap(resample_arc(contour), feature['y_range_px'], 3)
        q = fit_circle(p)
        cx, cy = q['centre_px']; r = q['radius_px']
        theta = np.linspace(0, 2*np.pi, 300)
        line, = axes[2].plot(cx+r*np.cos(theta), cy+r*np.sin(theta), lw=1,
                           label=f"{feature['id']}: {um_per_pixel*r:.1f} um")
        axes[2].scatter(*p.T, s=7, color=line.get_color())
    axes[2].legend(fontsize=8)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def extract(root, manifest_path, output):
    import matplotlib.image as mpimg
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous extraction output')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    images = []
    for spec in manifest['observations']:
        path = root/spec['local_source']
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['source_sha256']:
            raise ValueError('Source hash mismatch; review image identity before extraction')
        images.append(mpimg.imread(path)[:, :, :3])
    output.mkdir(parents=True)
    result = dict(schema='softcut-image-radius-estimates/1', interpreter=sys.executable,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        method=manifest['radius_method'], experimental_force_read=False, loading_started=False,
        uncertainty_kind='Processing and scale sensitivity; NOT statistical CI or optical-resolution uncertainty', observations=[])
    for rgb, spec in zip(images, manifest['observations']):
        q = analyze(rgb, spec, manifest['radius_method'])
        result['observations'].append(q)
        plot_review(rgb, spec, manifest['radius_method'], q, output/(spec['id']+'.png'))
    (output/'estimates.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps([{k: q[k] for k in ('id', 'summaries')} for q in result['observations']], indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--manifest', type=Path, default=Path('configs/blade_tip_closeups.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.root, args.manifest, args.output)


if __name__ == '__main__':
    main()
