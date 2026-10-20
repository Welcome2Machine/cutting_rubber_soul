"""Amplitude-only diagnostic against manual screenshot readings over 0--1 mm.

This does not run finite elements, identify a material modulus, or validate the
model. The input ranges are manual reading ranges, never confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DEPTH_ROUNDOFF_MM = 1e-12
READING_FIELDS = ('depth_mm', 'force_n', 'lower_n', 'upper_n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not np.isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number


def validate_points(points):
    if len(points) < 2:
        raise ValueError('At least two manual reading points are required')
    cleaned = []
    for index, row in enumerate(points):
        if any(key not in row for key in READING_FIELDS):
            raise ValueError(f'Reading {index} is missing required fields')
        item = {key: finite(row[key], f'reading {index} {key}') for key in READING_FIELDS}
        if not 0 <= item['depth_mm'] <= 1:
            raise ValueError('This amplitude diagnostic is restricted to 0--1 mm')
        if item['force_n'] <= 0 or not 0 <= item['lower_n'] <= item['force_n'] <= item['upper_n']:
            raise ValueError('Read forces must be positive and inside nonnegative ordered bounds')
        if cleaned and item['depth_mm']-cleaned[-1]['depth_mm'] <= DEPTH_ROUNDOFF_MM:
            raise ValueError('Reading depths must be strictly increasing and distinct')
        cleaned.append(item)
    return cleaned


def load_points(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(READING_FIELDS):
            raise ValueError('CSV header must be depth_mm,force_n,lower_n,upper_n')
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError('CSV row contains unexpected extra fields')
    return validate_points(rows)


def validate_history(report):
    """Check reported acceptance metadata for the entire completed history.

    This is not a fresh equilibrium assembly or a checkpoint geometry audit.
    """
    if report.get('completed') is not True or report.get('status') != 'completed':
        raise ValueError('Only a completed run may be used')
    if report.get('simplifications', {}).get('fracture') is not False:
        raise ValueError('Only an explicitly intact, no-fracture run may be used')
    tolerance = finite(report.get('force_tolerance_n'), 'force_tolerance_n')
    target = finite(report.get('target_mm'), 'target_mm')
    last_depth = finite(report.get('last_depth_mm'), 'last_depth_mm')
    if not 0 < tolerance <= 1e-4 or target <= 0:
        raise ValueError('Require positive target and force tolerance no looser than 1e-4 N')
    steps = report.get('steps')
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError('Completed history must contain at least two accepted depths')
    fields = ('depth_mm', 'reaction_fz_n', 'residual_n', 'min_j',
              'minimum_distance_mm', 'friction_work_n_mm')
    rows = []
    for index, step in enumerate(steps):
        row = {key: finite(step.get(key), f'step {index} {key}') for key in fields}
        if row['depth_mm'] < 0 or row['reaction_fz_n'] < 0:
            raise ValueError(f'Step {index} has negative depth or downward force')
        if rows and row['depth_mm']-rows[-1]['depth_mm'] <= DEPTH_ROUNDOFF_MM:
            raise ValueError('Accepted depths must be strictly increasing and distinct')
        if not 0 <= row['residual_n'] <= tolerance:
            raise ValueError(f'Step {index} failed reported force equilibrium')
        if row['min_j'] <= 0 or row['minimum_distance_mm'] <= 0:
            raise ValueError(f'Step {index} failed reported Jacobian or gap acceptance')
        if row['friction_work_n_mm'] < -1e-12:
            raise ValueError(f'Step {index} failed reported friction work acceptance')
        rows.append(row)
    if (abs(rows[0]['depth_mm']) > DEPTH_ROUNDOFF_MM
            or abs(rows[-1]['depth_mm']-target) > DEPTH_ROUNDOFF_MM
            or abs(last_depth-target) > DEPTH_ROUNDOFF_MM):
        raise ValueError('Completed history does not span zero through its reported target')
    provenance = report.get('source_sha256')
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError('Run source fingerprints are required')
    audit = dict(
        scope='all reported accepted states; no fresh equilibrium or checkpoint reassembly',
        accepted_states=len(rows), depth_range_mm=[rows[0]['depth_mm'], rows[-1]['depth_mm']],
        force_tolerance_n=tolerance, max_residual_n=max(r['residual_n'] for r in rows),
        min_j=min(r['min_j'] for r in rows),
        minimum_distance_mm=min(r['minimum_distance_mm'] for r in rows),
        minimum_friction_work_n_mm=min(r['friction_work_n_mm'] for r in rows),
        friction_work_roundoff_tolerance_n_mm=1e-12,
    )
    return rows, audit


def fit_scale(history, points):
    """Fit c >= 0 to sum_i (c F_FE,i - F_read,i)^2, without intercept."""
    points = validate_points(points)
    matched = []
    depths = np.array([r['depth_mm'] for r in history])
    for item in points:
        ids = np.flatnonzero(np.abs(depths-item['depth_mm']) <= DEPTH_ROUNDOFF_MM)
        if len(ids) != 1:
            raise ValueError(f'No unique exact accepted depth for {item["depth_mm"]} mm; interpolation is forbidden')
        index = int(ids[0])
        simulation = finite(history[index]['reaction_fz_n'], 'matched simulation force')
        if simulation <= 0:
            raise ValueError('Matched simulation forces must be positive')
        matched.append(dict(item, accepted_step_index=index,
                            accepted_depth_mm=float(depths[index]), simulation_force_n=simulation))
    x = np.array([r['simulation_force_n'] for r in matched])
    y = np.array([r['force_n'] for r in matched])
    # Normalize internally to avoid overflow for otherwise finite inputs.
    xs, ys = float(x.max()), float(y.max())
    xn, yn = x/xs, y/ys
    scale = float(max(0., float(xn@yn)/float(xn@xn)*(ys/xs)))
    if not np.isfinite(scale):
        raise ValueError('Amplitude scale is not representable')
    for row in matched:
        simulated = row['simulation_force_n']
        scaled = scale*simulated
        row.update(scaled_force_n=scaled,
                   original_error_n=simulated-row['force_n'],
                   scaled_error_n=scaled-row['force_n'],
                   scaled_error_lower_n=scaled-row['upper_n'],
                   scaled_error_upper_n=scaled-row['lower_n'],
                   allowed_scale_lower=row['lower_n']/simulated,
                   allowed_scale_upper=row['upper_n']/simulated)
        if not all(np.isfinite(v) for v in row.values()):
            raise ValueError('Derived force or interval is not finite')
        row['scaled_inside_reading_range'] = bool(row['lower_n'] <= scaled <= row['upper_n'])
    lower = max(r['allowed_scale_lower'] for r in matched)
    upper = min(r['allowed_scale_upper'] for r in matched)
    feasible = lower <= upper
    rmse = lambda values: float(np.linalg.norm(values)/np.sqrt(len(values)))
    return dict(
        scale=scale, estimator='equal-weight nonnegative least squares, zero intercept',
        formula='c = max(0, dot(F_sim, F_read) / dot(F_sim, F_sim))',
        fitted_points=len(matched),
        original_rmse_n=rmse(x-y), scaled_rmse_n=rmse(scale*x-y),
        uniform_scale_feasible=bool(feasible),
        allowed_scale_intersection=([lower, upper] if feasible else None),
        interval_intersection_bounds=dict(max_lower=lower, min_upper=upper),
        interval_conclusion=('A uniform scale exists within all manual reading ranges.' if feasible else
                             'No single amplitude scale lies within all manual reading ranges.'),
        points=matched,
    )


def plot_diagnostic(history, fit, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    curve = [row for row in history if row['depth_mm'] <= 1+DEPTH_ROUNDOFF_MM]
    d = np.array([r['depth_mm'] for r in curve])
    f = np.array([r['reaction_fz_n'] for r in curve])
    points = fit['points']
    xp = np.array([r['depth_mm'] for r in points])
    yp = np.array([r['force_n'] for r in points])
    ranges = np.array([[r['force_n']-r['lower_n'] for r in points],
                       [r['upper_n']-r['force_n'] for r in points]])
    error = np.array([r['scaled_error_n'] for r in points])
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, layout='constrained')
        axes[0].plot(d, f, '--', color='#0072B2', label='Existing FE curve (accepted states)')
        axes[0].plot(d, fit['scale']*f, '-', color='#D55E00',
                     label=f'Scaled existing curve, not a new FE solve (c={fit["scale"]:.4g})')
        axes[0].errorbar(xp, yp, yerr=ranges, fmt='o', color='#222222', capsize=4,
                         label='Manual readings; bars = reading ranges, not CI')
        axes[0].set(ylabel='Downward force Fz (N)', ylim=(0, None),
                    title='Amplitude-only diagnostic from manual screenshot readings')
        axes[0].legend(fontsize=9, loc='upper left')
        axes[1].axhline(0, color='#666666', linewidth=1)
        # Subtracting a range reverses its lower/upper distances.
        axes[1].errorbar(xp, error, yerr=ranges[::-1], fmt='s', color='#D55E00', capsize=4,
                         label='Scaled FE minus manual reading; bars = reading range, not CI')
        axes[1].plot(xp, [r['original_error_n'] for r in points], 'x', color='#0072B2',
                     label='Original FE minus manual reading')
        # Leave room for the endpoint marker/caps; data still end at 1 mm.
        axes[1].set(xlabel='Downward depth (mm)', ylabel='Force difference (N)', xlim=(-.01, 1.02))
        axes[1].legend(fontsize=9)
        for axis in axes:
            axis.grid(alpha=.2)
        fig.savefig(destination, dpi=180, facecolor='white')
        plt.close(fig)
    return dict(path=Path(destination).name, backend='Agg', matplotlib_version=matplotlib.__version__,
                dpi=180, size_inches=[9, 8], manual_observations_connected=False,
                smoothing=False, error_bars='manual reading ranges, not confidence intervals',
                alt_text='Existing and amplitude-scaled FE force curves with discrete manual readings; '
                         'the lower panel shows signed force differences at those readings.')


def diagnose(run_folder, points_path, output, *, plot=False):
    run_folder, points_path, output = Path(run_folder), Path(points_path), Path(output)
    if output.exists():
        raise ValueError('Output directory must be new; existing artifacts are never overwritten')
    source = run_folder/'result.json'
    report = json.loads(source.read_text(encoding='utf-8-sig'))
    history, audit = validate_history(report)
    points = load_points(points_path)
    result = fit_scale(history, points)
    result.update(
        status='amplitude_diagnostic_only', new_fe_solve=False, material_calibrated=False,
        observation_kind='manual screenshot estimates; not original experimental data',
        range_meaning='manual reading ranges; not statistical confidence intervals',
        validity='No material parameter is inferred. c*G is not a new FE solution or calibrated modulus.',
        matching=dict(method='unique accepted depth only; no interpolation or extrapolation',
                      floating_point_equality_tolerance_mm=DEPTH_ROUNDOFF_MM),
        fit_window_mm=[0, 1], history_acceptance_audit=audit,
        provenance=dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            run_result_path=str(source.resolve()), run_result_sha256=sha256(source),
            manual_points_path=str(points_path.resolve()), manual_points_sha256=sha256(points_path),
            tool_path=Path(__file__).name, tool_sha256=sha256(__file__),
            numpy_version=np.__version__, python_version=sys.version,
            run_source_sha256=report['source_sha256'],
            run_config=report.get('config'), run_mesh_settings=report.get('mesh_settings'),
            run_simplifications=report['simplifications'],
            run_ipc_binary_sha256=report.get('ipc_binary_sha256'),
            source_fingerprints_status='copied from run report, not independently reassembled',
        ),
    )
    output.mkdir(parents=True)
    with (output/'fit_points.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['points'][0]))
        writer.writeheader()
        writer.writerows(result['points'])
    if plot:
        result['figure'] = plot_diagnostic(history, result, output/'force_scale.png')
    (output/'fit_points.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='Write a headless two-panel PNG')
    args = parser.parse_args(argv)
    try:
        result = diagnose(args.run, args.points, args.output, plot=args.plot)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Amplitude diagnostic rejected: {exc}\n')
    print(json.dumps({key: result[key] for key in (
        'status', 'scale', 'original_rmse_n', 'scaled_rmse_n', 'uniform_scale_feasible')}))


if __name__ == '__main__':
    main()
