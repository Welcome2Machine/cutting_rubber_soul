"""Common Ogden shape screen on saved native loading histories.

Each candidate integrates its own material history from rest. Virtual-work
proxies rank directions only: they are never re-equilibrated FE reactions.
No angle-specific parameters, force rescaling or automatic production launch.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from mechanics.finite_strain import response
from run_support import compute_process_policy, write_json
from tools.run_material_candidate import make_problem, load_force_csv


ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (1., 1.5, 2., 3., 4.)
ANCHORS = (.25, .5)
COMMON = dict(equilibrium_shear=.14, maxwell_shear=1.24, relaxation_s=1.7,
              loading_rate=.2, bulk_modulus=23.34333333333331, mu=.35)


def shape_material(material, alpha):
    """Keep both infinitesimal shear moduli, K and actual small-shear tau."""
    if not np.isfinite(alpha) or not 1 <= alpha <= 4:
        raise ValueError('Bounded positive Ogden shape exponent must be in [1, 4]')
    ge, _ = material.moduli(); gv, _ = material.moduli(True)
    return replace(material, mu_eq=(2*ge/alpha,), alpha_eq=(float(alpha),),
                   mu_ov=(2*gv/alpha,), alpha_ov=(float(alpha),))


def decide(rows, experiment):
    """Predeclared two-angle gate; central screenshot values are estimates."""
    values = {a: {angle: np.array([next(r for r in rows if r['angle'] == angle
        and abs(r['depth_mm']-d) < 1e-10)['candidates'][str(a)]['work_proxy_n']
        for d in ANCHORS]) for angle in (33, 57)} for a in ALPHAS}
    e = {k: np.asarray(v) for k, v in experiment.items()}
    def metrics(v):
        return dict(rmse33_n=float(np.sqrt(np.mean((v[33]-e[33])**2))),
                    rmse57_n=float(np.sqrt(np.mean((v[57]-e[57])**2))),
                    ratio_error_rms=float(np.sqrt(np.mean((v[33]/v[57]-e[33]/e[57])**2))))
    base = metrics(values[2.]); decisions = {}
    for a in ALPHAS:
        m = metrics(values[a])
        gate = a != 2. and (m['rmse33_n'] <= .85*base['rmse33_n'] and
            m['rmse57_n'] <= base['rmse57_n']+.1 and
            m['ratio_error_rms'] <= .85*base['ratio_error_rms'])
        decisions[str(a)] = dict(**m, promising_direction=bool(gate))
    return dict(control=base, candidates=decisions,
        rule='33 RMSE and angle-ratio RMS each improve >=15%; 57 RMSE worsens <=0.1 N. '
             'Two anchors .25/.5 only, no rescaling. NOT experimental or FE validation.')


def run(out, seconds):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started+seconds
    study = ROOT/'outputs/native_angles_2mm_20261016'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    sources = {name: sha(ROOT/name) for name in ('mechanics/finite_strain.py',
        'prototype/ipc_viscoelastic.py', 'prototype/ipc_reference.py')}
    experiment_path = ROOT/'docs/1014n57a1__Fz__curve_0.csv'
    ed, ef, speed = load_force_csv(experiment_path)
    if speed != COMMON['loading_rate']: raise ValueError('Experiment rate mismatch')
    experiment = {33: [4., 8.5], 57: np.interp(ANCHORS, ed, ef).tolist()}
    report = dict(status='running', scope='Saved displacement path, own material histories; '
        'work proxies are NOT candidate equilibrium forces', alphas=ALPHAS, common=COMMON,
        seconds_budget=seconds, source_sha256=sources, input_sha256={str(experiment_path): sha(experiment_path)},
        experiment_anchors_n=experiment, screenshot_33_reading_range_n=1., rows=[],
        gate='33 RMSE and ratio RMS improve >=15%; 57 RMSE degradation <=0.1 N at .25/.5; no rescaling',
        fracture=False, new_fe_solve=False)
    write_json(out/'screen.json', report)
    try:
        with threadpool_limits(2), compute_process_policy() as policy:
            report['runtime_policy'] = policy
            for angle, folder in ((33, '01_angle33'), (57, '02_angle57')):
                case = study/folder; result = read(case/'result.json'); audit = read(case/'audit.json')
                for name, digest in sources.items():
                    if result['source_sha256'][name] != digest:
                        raise ValueError('Control physical source changed: '+name)
                p = make_problem(angle=angle, **COMMON)
                area, grad, dofs = p.mesh.bulk_data; vol = area*p.config.width_mm
                materials = {a: shape_material(p.material, a) for a in ALPHAS}
                metrics = {a: np.tile(np.eye(3), (len(area), 1, 1)) for a in ALPHAS}
                previous_x = p.rest; previous_metric = metrics[2.].copy()
                for i, record in enumerate(result['steps']):
                    if record['depth_mm'] > .5+1e-12: break
                    path = case/'states'/f'{i:04d}.npz'; digest = sha(path)
                    if digest != audit['rows'][i]['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
                    report['input_sha256'][str(path)] = digest
                    with np.load(path, allow_pickle=False) as z: s = {k: z[k] for k in z.files}
                    np.testing.assert_array_equal(s['reference_xy'], p.rest)
                    np.testing.assert_array_equal(s['triangles'], p.mesh.triangles)
                    np.testing.assert_array_equal(s['previous_positions'], previous_x)
                    np.testing.assert_array_equal(s['previous_material_metric'], previous_metric)
                    x = s['positions']; dt = float(s['material_dt_s'])
                    F = np.eye(2)+np.einsum('eai,eaJ->eiJ', (x[:p.n]-p.rest[:p.n])[p.mesh.triangles], grad)
                    v = (x-previous_x)/(dt*p.loading_rate)
                    dF = np.einsum('eai,eaJ->eiJ', v[:p.n][p.mesh.triangles], grad)
                    control = response(F, metrics[2.], dt, materials[2.], tangent=False, kinematics='plane_strain')
                    np.testing.assert_allclose(control['metric'], s['material_metric'], rtol=2e-9, atol=2e-9)
                    row = dict(angle=angle, depth_mm=record['depth_mm'], candidates={},
                        control_fz_n=record['reaction_fz_n'],
                        control_metric_error=float(np.max(abs(control['metric']-s['material_metric']))))
                    anchor = any(abs(record['depth_mm']-d)<1e-10 for d in ANCHORS)
                    if anchor:
                        p.previous = previous_x.copy(); p.previous_material_metric = previous_metric.copy(); p.material_dt = dt
                        _, gbase, _, _ = p.evaluate(x, p.lag(x), tangent=False)
                        if np.linalg.norm(gbase[p.free])>1e-4: raise RuntimeError('Control does not equilibrate')
                    for a, material in materials.items():
                        if time.monotonic()>deadline: raise TimeoutError('Material shape screen budget')
                        q = control if a == 2. else response(F, metrics[a], dt, material, tangent=False, kinematics='plane_strain')
                        metrics[a] = q['metric']
                        if np.max(abs(np.linalg.det(q['metric'])-1))>2e-8 or q['viscous_rate'].min() < -1e-12:
                            raise RuntimeError('Candidate material admissibility failed')
                        deltaP = q['P']-control['P']; work = float(vol@np.einsum('eiJ,eiJ->e', deltaP, dF))
                        local = np.einsum('e,eiJ,eaJ->eai', vol, deltaP, grad).reshape(-1,6)
                        dg = np.zeros(p.total); np.add.at(dg, dofs.ravel(), local.ravel())
                        if abs(dg@v.ravel()-work)>2e-8: raise RuntimeError('Virtual work closure failed')
                        row['candidates'][str(a)] = dict(work_change_n=work,
                            work_proxy_n=record['reaction_fz_n']+work,
                            free_residual_change_n=float(np.linalg.norm(dg[p.free])),
                            actual_fixed_field_residual_n=float(np.linalg.norm((gbase+dg)[p.free])) if anchor else None)
                    report['rows'].append(row); previous_x=x; previous_metric=s['material_metric']
                    if i%10 == 0:
                        write_json(out/'screen.json', report)
                        print('SHAPE', angle, record['depth_mm'], time.monotonic()-started, flush=True)
            report.update(status='completed', decision=decide(report['rows'], experiment))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_s'] = time.monotonic()-started
        report['sources_unchanged'] = all(sha(ROOT/n)==v for n,v in sources.items())
        report['inputs_unchanged'] = all(sha(Path(n))==v for n,v in report['input_sha256'].items())
        write_json(out/'screen.json', report)
        (out/'screen_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps({k: report.get(k) for k in ('status','elapsed_s','decision','error')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=600.)
    args = parser.parse_args()
    run(args.output, args.seconds)
