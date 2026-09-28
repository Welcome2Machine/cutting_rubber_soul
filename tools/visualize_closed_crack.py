"""Actual accepted meshes/stresses for the prescribed closed-crack component test."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def render(source):
    source = Path(source).resolve()
    report_path = source / 'result.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['status'] == 'running':
        raise ValueError('Render only a finished or failed component trial')
    steps = report['steps']
    if not steps:
        raise ValueError('No accepted states')
    # Stage endpoints, including the most open state (before reclosure).
    wanted = ('compress', 'slide_closed', 'open', 'knife')
    selected = [max((s['index'] for s in steps if s['stage'] == label), default=None)
                for label in wanted]
    selected = [i for i in selected if i is not None]
    states = []
    hashes = {report_path.name: hashlib.sha256(report_path.read_bytes()).hexdigest()}
    for index in selected:
        path = source / 'states' / f'{index:04d}.npz'
        hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            state = {key: z[key].copy() for key in z.files}
        if not np.isfinite(state['positions']).all() or np.any(state['J'] <= 0):
            raise ValueError('Invalid accepted geometry')
        state['tensile'] = np.maximum(np.linalg.eigvalsh(state['cauchy'])[:, -1], 0.)
        if not np.isfinite(state['tensile']).all():
            raise ValueError('Invalid saved stress')
        states.append((index, state))
    vmax = max(1e-6, max(s['tensile'].max() for _, s in states))
    styles = {'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
              'savefig.facecolor': 'white', 'pdf.fonttype': 42}
    output = source / 'visualization'
    output.mkdir(exist_ok=False)
    with plt.rc_context(styles):
        fig, axes = plt.subplots(2, len(states), figsize=(4.1 * len(states), 7.5),
                                 squeeze=False, layout='constrained')
        for column, (index, state) in enumerate(states):
            x, tri, seams = state['positions'], state['triangles'], state['seams']
            step = steps[index]
            title = f"{step['stage']} | accepted step {index + 1}"
            for row in range(2):
                ax = axes[row, column]
                artist = ax.tripcolor(x[:, 0], x[:, 1], tri, facecolors=state['tensile'],
                                      cmap='cividis', vmin=0., vmax=vmax, shading='flat')
                ax.triplot(x[:, 0], x[:, 1], tri, color='#777777', lw=.35)
                ax.plot(*x[seams[0]].T, color='#0072B2', lw=1.5, label='Left crack face')
                ax.plot(*x[seams[1]].T, color='#D55E00', lw=1.5, ls='--', label='Right crack face')
                # Exact circular tip + tangent straight sides; only rendering
                # the arc uses line segments. Solver geometry is analytic.
                radius = .02
                alpha = np.deg2rad(11.5)
                tip = float(state['tip_y'])
                angles = np.linspace(np.pi + alpha, 2 * np.pi - alpha, 100)
                arc = np.column_stack((radius * np.cos(angles), tip + radius + radius * np.sin(angles)))
                top = 1.5
                half = ((top - tip - radius) * np.sin(alpha) + radius) / np.cos(alpha)
                knife = np.vstack(([-half, top], arc, [half, top]))
                ax.add_patch(Polygon(knife, facecolor='#dddddd', edgecolor='#333333', lw=1., zorder=4))
                ax.set(aspect='equal', xlabel='x (mm)', ylabel='y (mm)')
                if row == 0:
                    ax.set(xlim=(-1.1, 1.1), ylim=(-.1, 1.3), title=title)
                else:
                    ax.set(xlim=(-.16, .16), ylim=(.7, 1.13), title='Interface detail (common axes)')
            axes[0, column].text(.02, .03,
                f"min J = {step['min_J']:.3f}\nresidual = {step['force_residual_n']:.2g} N",
                transform=axes[0, column].transAxes, fontsize=9,
                bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none'})
        axes[0, 0].legend(loc='upper left', fontsize=8)
        fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.65,
                     label='Positive part of maximum principal Cauchy stress (MPa)')
        fig.suptitle(f"Prescribed, already-debonded seam: {report['status']}\n"
                     'Actual displacement (x1); no crack nucleation or propagation prediction', fontsize=13)
        fig.savefig(output / 'summary.png', dpi=180)
        plt.close(fig)
    manifest = dict(source=str(source), source_sha256=hashes, accepted_state_indices=selected,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
        transformations=['No displacement magnification or coordinate transform',
            'Elementwise max(largest eigenvalue of saved Cauchy stress, 0); no smoothing',
            'Shared stress scale over displayed stages; common limits within each row',
            'Analytic radius 0.02 mm, angle 23 degree wedge; arc polygon for display only'],
        stress_limits_mpa=[0., float(vmax)], uncertainty='Single numerical realization; no experimental uncertainty',
        alt_text='Independent mesh faces remain closed under compression and slide, separate during opening, '
                 'then contact both sides of the knife. The upper row shows whole blocks and the lower row '
                 'a common interface detail. This is a prescribed seam test, not predicted fracture.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(output / 'summary.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    render(parser.parse_args().source)
