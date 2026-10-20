"""Fresh history audit and comparisons for the isolated cross-mode study."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from tools.check_cross_mode import Block, KINEMATICS, block_path, invariant_checks, save_json


def check_sources(out):
    source_hashes = json.loads((out/'sources.json').read_text())
    for path, digest in source_hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((out/'source'/path).read_bytes()).hexdigest() == digest
    return len(source_hashes)


def key(row):
    return (row['n'], row['kind'], row['kinematics'], round(row['kfactor'], 6),
            row['angle_deg'], row['step_mm'], row['order'])


def comparison(a, b):
    """Compare b to a only at the SAME completed endpoint, never failed forces."""
    if a['status'] != 'completed' or b['status'] != 'completed':
        return dict(available=False, a_status=a['status'], b_status=b['status'])
    x, y = a['rows'][-1], b['rows'][-1]
    assert abs(x['depth_mm']-y['depth_mm']) < 1e-12
    # Both spaces have a piecewise-linear exterior. Evaluate the difference at
    # the union of boundary knots: the norm of a linear vector on each interval
    # is convex, so its maximum is at an endpoint, not an arbitrary sparse probe.
    surfaces = []
    for path in (a, b):
        with np.load(path['checkpoint']) as data:
            xy = data['reference_xy']; u = data['displacement'][-1].reshape(-1, 2)
            top = np.flatnonzero(np.isclose(xy[:, 1], 2.))
            top = top[np.argsort(xy[top, 0])]
            surfaces.append((xy[top, 0], u[top]))
    knots = np.union1d(surfaces[0][0], surfaces[1][0])
    knots = knots[(knots <= 1.5)|(knots >= 2.5)]
    motions = [np.column_stack([np.interp(knots, xx, uu[:, k]) for k in range(2)])
               for xx, uu in surfaces]
    shift = motions[1]-motions[0]
    return dict(available=True, a=a['id'], b=b['id'],
        a_force_n=x['force_n'], b_force_n=y['force_n'],
        force_change_pct=100*(y['force_n']/x['force_n']-1),
        max_surface_vector_change_mm=float(np.linalg.norm(shift, axis=1).max()),
        surface_comparison='entire free top piecewise-linear trace, union of knots',
        a_min_j=x['min_j'], b_min_j=y['min_j'],
        a_true_inplane_min_j=x['true_inplane_min_j'], b_true_inplane_min_j=y['true_inplane_min_j'])


def refine(out):
    r = json.loads((out/'blocks.json').read_text()); assert r['status'] == 'bounded_checks_ended'
    by = {key(x): x for x in r['paths']}; selected = []
    for kin in KINEMATICS:
        for angle in (33, 57):
            pair = comparison(by[8, 'p1', kin, 1., angle, .025, 3],
                              by[8, 'p2', kin, 1., angle, .025, 3])
            mesh = [comparison(by[4, kind, kin, 1., angle, .025, 3],
                               by[8, kind, kin, 1., angle, .025, 3]) for kind in ('p1', 'p2')]
            if any(x['available'] and abs(x['force_change_pct'])>5 for x in [pair, *mesh]):
                selected.extend([(kind, kin, angle) for kind in ('p1', 'p2')])
    target = out/'refinement.json'
    if target.exists():
        raise FileExistsError('Preserve previous refinement evidence')
    report = dict(status='running', trigger='>5% mesh/enrichment endpoint force difference', paths=[])
    for kind, kin, angle in selected:
        row = block_path(out, 12, kind, kin, 1., angle)
        report['paths'].append(row); save_json(target, report)
        print('REFINE', row['id'], row['status'], row['elapsed_s'], flush=True)
    report['status'] = 'bounded_checks_ended'; save_json(target, report)


def fresh_audit(out, paths):
    audits = []; max_residual = 0.; max_metric = 0.; count = 0
    start = time.perf_counter()
    for path in paths:
        saved = Path(path['checkpoint'])
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == path['checkpoint_sha256']
        b = Block(path['n'], path['kind'], path['kinematics'], path['kfactor'], path['order'])
        with np.load(saved) as data:
            us = data['displacement']; metrics = data['metric']
            np.testing.assert_array_equal(data['reference_xy'], b.mesh.reference_xy)
            np.testing.assert_array_equal(data['triangles'], b.base.triangles)
        assert len(us) == len(metrics) == len(path['rows'])+1
        np.testing.assert_array_equal(us[0], 0.)
        np.testing.assert_array_equal(metrics[0], np.tile(np.eye(3), (len(metrics[0]), 1, 1)))
        angle = np.deg2rad(path['angle_deg']); direction = np.array([np.sin(angle), -np.cos(angle)])
        rows = []
        for i, row in enumerate(path['rows'], 1):
            _, g, _, q = b.evaluate(us[i], metrics[i-1], path['step_mm']/.2, False)
            invariant_checks(q, b.kin)
            residual = float(np.linalg.norm(g[b.free])); assert residual <= 1e-7
            error = float(np.max(np.abs(q['metric']-metrics[i]))); assert error < 1e-9
            force = float(g.reshape(-1, 2)[b.patch].sum(axis=0)@direction)
            assert abs(force-row['force_n']) < 1e-8
            np.testing.assert_allclose(us[i].reshape(-1, 2)[b.patch],
                np.broadcast_to(row['depth_mm']*direction, (len(b.patch), 2)), atol=1e-13, rtol=0)
            np.testing.assert_array_equal(us[i].reshape(-1, 2)[b.bottom], 0.)
            max_residual = max(max_residual, residual); max_metric = max(max_metric, error); count += 1
            rows.append(dict(depth_mm=row['depth_mm'], residual_n=residual, history_max_error=error,
                force_n=force, true_inplane_min_j=b.min_j(us[i]), min_J=float(q['J'].min())))
        audits.append(dict(id=path['id'], status=path['status'], states=len(rows), passed=True, rows=rows))
    result = dict(passed=True, states=count, paths=len(paths), max_residual_n=max_residual,
        max_history_error=max_metric, elapsed_s=time.perf_counter()-start, audits=audits)
    save_json(out/'fresh_audit.json', result)
    return {k: v for k, v in result.items() if k != 'audits'}


def summarize(out, paths):
    by = {key(p): p for p in paths}
    r = dict(scope='nonuniform bonded block, not cutting or experimental validation',
             comparisons=[], angle_ratios=[], failed=[], material=[])
    for path in paths:
        if path['status'] != 'completed':
            r['failed'].append(dict(id=path['id'], error=path.get('error'),
                accepted_depth=path['rows'][-1]['depth_mm'] if path['rows'] else 0))
    def add(kind, a, b):
        if a in by and b in by:
            r['comparisons'].append(dict(comparison=kind, **comparison(by[a], by[b])))
    for kin in KINEMATICS:
        for angle in (33, 57):
            for n in (4, 8, 12):
                for k in (round(1/3, 6), 1., 3.):
                    add('p1_to_p2', (n, 'p1', kin, k, angle, .025, 3), (n, 'p2', kin, k, angle, .025, 3))
            for kind in ('p1', 'p2'):
                add('mesh_4_to_8', (4, kind, kin, 1., angle, .025, 3), (8, kind, kin, 1., angle, .025, 3))
                add('mesh_8_to_12', (8, kind, kin, 1., angle, .025, 3), (12, kind, kin, 1., angle, .025, 3))
                add('time_halving', (8, kind, kin, 1., angle, .025, 3), (8, kind, kin, 1., angle, .0125, 3))
                for factor in (round(1/3, 6), 3.):
                    add('K_change', (8, kind, kin, 1., angle, .025, 3), (8, kind, kin, factor, angle, .025, 3))
            add('quadrature', (8, 'p2', kin, 1., angle, .025, 3), (8, 'p2', kin, 1., angle, .025, 4))
    for n in (4, 8, 12):
        for kind in ('p1', 'p2'):
            for angle in (33, 57):
                add('PE_to_PS', (n, kind, 'plane_strain', 1., angle, .025, 3),
                    (n, kind, 'plane_stress', 1., angle, .025, 3))
            for kin in KINEMATICS:
                for k in (round(1/3, 6), 1., 3.):
                    a, b = by.get((n, kind, kin, k, 33, .025, 3)), by.get((n, kind, kin, k, 57, .025, 3))
                    if a and b:
                        available = a['status'] == b['status'] == 'completed'
                        r['angle_ratios'].append(dict(n=n, kind=kind, kinematics=kin, kfactor=k,
                            available=available, ratio33_57=a['rows'][-1]['force_n']/b['rows'][-1]['force_n'] if available else None))
    mat = json.loads((out/'material.json').read_text()); assert mat['status'] == 'completed'
    for kin in KINEMATICS:
        for mode in ('tension', 'compression', 'constrained_compression', 'shear'):
            selection = [x for x in mat['paths'] if x['mode']==mode and x['kinematics']==kin]
            base = next(x for x in selection if x['ramp_s']==2.5 and x['dt_s']==.05)
            fine = next(x for x in selection if x['ramp_s']==2.5 and x['dt_s']==.025)
            r['material'].append(dict(mode=mode, kinematics=kin,
                ramp_stress_mpa=base['end_ramp']['nominal_stress_mpa'],
                hold_stress_mpa=base['end_hold']['nominal_stress_mpa'],
                ramp_J=base['end_ramp']['J'], held_J=base['end_hold']['J'],
                relaxation_pct=100*(1-base['end_hold']['nominal_stress_mpa']/base['end_ramp']['nominal_stress_mpa']),
                time_force_change_pct=100*(fine['end_ramp']['nominal_stress_mpa']/base['end_ramp']['nominal_stress_mpa']-1),
                rate_controls=[dict(ramp_s=x['ramp_s'], stress_mpa=x['end_ramp']['nominal_stress_mpa']) for x in selection if x['dt_s']==.05]))
    controls = [x for x in r['comparisons'] if x['comparison'] in ('time_halving', 'quadrature')]
    r['time_quadrature_gate'] = dict(count=len(controls), passed=all(x['available'] and
        abs(x['force_change_pct'])<=2 and x['max_surface_vector_change_mm']<=.002 for x in controls))
    r['counts'] = dict(attempted=len(paths), completed=sum(x['status']=='completed' for x in paths),
        accepted_states=sum(len(x['rows']) for x in paths), solve_elapsed_s=sum(x['elapsed_s'] for x in paths))
    return r, mat


def figures(out, paths, r, mat):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, mode in zip(axes.flat, ('tension', 'compression', 'constrained_compression', 'shear')):
        for kin, color, style in zip(KINEMATICS, ('#0072B2', '#D55E00'), ('-', '--')):
            q = next(x for x in mat['paths'] if (x['mode'], x['kinematics'], x['ramp_s'], x['dt_s'])==(mode, kin, 2.5, .025))
            ax.plot([0]+[x['time_s'] for x in q['rows']], [0]+[x['nominal_stress_mpa'] for x in q['rows']],
                    style, color=color, label=kin.replace('_', ' '))
        ax.axvline(2.5, color='.55', lw=.8, ls=':'); ax.set_title(mode.replace('_', ' '))
        ax.set(xlabel='Time (s)', ylabel='Nominal stress (MPa)'); ax.grid(alpha=.2)
    axes[0, 0].legend(); fig.suptitle('Common material: prescribed ramp and hold (model predictions)\nEach deformation mode uses its own stress scale')
    fig.savefig(out/'material_modes.png', dpi=180); fig.savefig(out/'material_modes.svg'); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True, sharex=True, sharey=True)
    ceiling = 1.12*max(x['rows'][-1]['force_n'] for x in paths if x['n']==8 and
                      x['step_mm']==.025 and x['order']==3 and x['status']=='completed')
    for i, kin in enumerate(KINEMATICS):
        for j, angle in enumerate((33, 57)):
            ax = axes[i, j]
            for kind, color, marker in (('p1', '#0072B2', 'o'), ('p2', '#D55E00', 's')):
                q = sorted([x for x in paths if x['n']==8 and x['kind']==kind and x['kinematics']==kin
                    and x['angle_deg']==angle and x['step_mm']==.025 and x['order']==3], key=lambda x:x['kfactor'])
                ax.plot([x['kfactor'] for x in q], [x['rows'][-1]['force_n'] if x['status']=='completed' else np.nan for x in q],
                        marker=marker, color=color, label=kind.upper() if kind=='p1' else 'interior P2')
                for x in q:
                    if x['status']!='completed':
                        ax.annotate('failed; no endpoint', (x['kfactor'], .08), xycoords=('data', 'axes fraction'),
                                    ha='right', color=color, fontsize=8)
            ax.set_title(f'{kin.replace("_", " ")}, direction {angle} deg')
            ax.set(xlabel='Bulk modulus / original K', ylabel='Loading-direction reaction (N)', ylim=(0, ceiling))
            ax.set_xscale('log'); ax.set_xticks([1/3, 1, 3], ['1/3', '1', '3']); ax.grid(alpha=.2)
    axes[0, 0].legend(); fig.suptitle('Bonded block at 0.2 mm: constraint / volume / discretization\nNot blade contact or a cutting-force prediction')
    fig.savefig(out/'coupling.png', dpi=180); fig.savefig(out/'coupling.svg'); plt.close(fig)
    rows = []
    for p in paths:
        for row in p['rows']:
            rows.append(dict(case=p['id'], status=p['status'], depth_mm=row['depth_mm'],
                reaction_n=row['force_n'], min_J=row['min_j'], true_inplane_min_J=row['true_inplane_min_j']))
    with (out/'block_curves.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    save_json(out/'figure_manifest.json', dict(inputs=['material.json', 'blocks.json', 'refinement.json'],
        plots=['material_modes.png', 'material_modes.svg', 'coupling.png', 'coupling.svg'],
        experimental_data_used=False, smoothing=False, failed_endpoints='NaN, never plotted as complete',
        matplotlib=matplotlib.__version__, numpy=np.__version__, dpi=180))


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--refine', action='store_true'); args = p.parse_args(); out = args.output.resolve()
    nsrc = check_sources(out)
    with threadpool_limits(2):
        if args.refine:
            refine(out); return
        base = json.loads((out/'blocks.json').read_text()); assert base['status']=='bounded_checks_ended'
        paths = base['paths']
        if (out/'refinement.json').exists():
            extra = json.loads((out/'refinement.json').read_text()); assert extra['status']=='bounded_checks_ended'
            paths += extra['paths']
        audit = fresh_audit(out, paths)
        result, mat = summarize(out, paths)
        result.update(source_hashes_checked=nsrc, fresh_audit=audit,
            review_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save_json(out/'summary.json', result)
        figures(out, paths, result, mat)
        print(json.dumps(dict(counts=result['counts'], audit=audit, gates=result['time_quadrature_gate'],
                              failed=result['failed'], material=result['material'])), flush=True)


if __name__ == '__main__':
    main()
