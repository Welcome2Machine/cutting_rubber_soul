"""Bounded verification of zero-gap ties, edge bias and inertial energy balance.

Not a cutting run. Independent synthetic inputs only. Reports limited scope;
unit-scale operator equivalence or scalar dynamics never qualify full cutting.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay

from prototype.extrinsic_topology import partition, transfer_release, lumped_mass
from prototype.cohesive_dynamics import advance
from prototype.cohesive_mixed import MixedModeLaw
from prototype.material import bulk
from prototype.model_config import FEMConfig, FEMMesh
from tools.verify_network_compliance import square_mesh


def tick(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Resolution verification budget exhausted')


def tie_case(cells, deadline):
    tick(deadline)
    c = FEMConfig(length_mm=1., height_mm=1., width_mm=2., inclination_deg=0.)
    m = square_mesh(cells, c)
    t = partition(m.reference_xy, m.triangles)
    empty = np.empty(0, int)
    tied = FEMMesh(t.reference, t.reference.copy(), t.triangles, np.empty((0, 2), int),
                   np.empty(0), empty, empty, np.zeros(len(t.reference), int))
    x = m.reference_xy
    u = np.column_stack((.03*x[:, 0]*x[:, 1], -.01*x[:, 1]+.02*x[:, 0]**2))
    a = bulk(m, u.ravel(), c)
    b = bulk(tied, u[t.original_vertex_ids].ravel(), c)
    # A complete boundary-to-boundary release; identical element identities.
    midpoint = cells//2
    edges = [(j*(cells+1)+midpoint, (j+1)*(cells+1)+midpoint) for j in range(cells)]
    new = partition(x, m.triangles, edges)
    rng = np.random.default_rng(314)
    velocity = rng.normal(size=t.reference.shape)
    position = t.reference+u[t.original_vertex_ids]
    xp, vp = transfer_release(t, new, position, velocity)
    mass0, mass1 = (lumped_mass(top, .001, 2.) for top in (t, new))
    row = dict(cells=cells, elements=len(m.triangles), tied_nodes=len(t.reference),
               released_nodes=len(new.reference), retained_area_ratio=1., reference_gap_mm=0.,
               maximum_element_force_difference_n=float(np.max(abs(a[0]-b[0]))),
               maximum_element_tangent_difference_n_mm=float(np.max(abs(a[1]-b[1]))),
               stored_energy_difference_n_mm=float(abs(a[3]-b[3])),
               release_position_difference_mm=float(np.max(abs(position[t.triangles]-xp[new.triangles]))),
               release_mass_difference=float(abs(mass0.sum()-mass1.sum())),
               release_momentum_difference=float(np.linalg.norm(mass0@velocity-mass1@vp)),
               release_kinetic_difference_n_mm=float(abs(.5*mass0@(velocity**2).sum(1)-.5*mass1@(vp**2).sum(1))))
    row['operator_equivalence_passed'] = all(row[k] < 1e-11 for k in row if 'difference' in k)
    return row


def edge_metric(cells, family, seed=0):
    """Lower bound on isotropic fracture cost restricted to candidate edges.

    Path-length/chord >= 1. This is a geometric counterexample, not prediction
    of the mechanically selected crack. No direction-dependent Gc correction.
    """
    c = FEMConfig(length_mm=1., height_mm=1., inclination_deg=0.)
    m = square_mesh(cells, c)
    x, tri = m.reference_xy.copy(), m.triangles.copy()
    if family == 'jittered':
        inner = np.all((x > 0.) & (x < 1.), axis=1)
        x[inner] += np.random.default_rng(seed).uniform(-.25/cells, .25/cells, (sum(inner), 2))
        tri = Delaunay(x).simplices
    elif family != 'aligned':
        raise ValueError('Unknown mesh family')
    edges, counts = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
    edges = edges[counts == 2]
    length = np.linalg.norm(x[edges[:, 1]]-x[edges[:, 0]], axis=1)
    graph = coo_matrix((np.r_[length, length], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(x), len(x))).tocsr()
    start = int(np.argmin(np.linalg.norm(x-[.5, .5], axis=1)))
    distances = dijkstra(graph, indices=start)
    rows = []
    for theta in range(0, 180, 15):
        direction = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
        end = int(np.argmin(np.linalg.norm(x-(x[start]+.3*direction), axis=1)))
        chord = np.linalg.norm(x[end]-x[start])
        rows.append(dict(cells=cells, family=family, seed=seed, angle_deg=theta,
                         actual_angle_deg=float(np.rad2deg(np.arctan2(*(x[end]-x[start])[::-1]))),
                         endpoint_error_mm=float(np.linalg.norm(x[end]-x[start]-.3*direction)),
                         chord_mm=float(chord), edge_path_mm=float(distances[end]),
                         cost_ratio=float(distances[end]/chord)))
    return rows


def dynamics_case(dt, damping, deadline):
    law = MixedModeLaw(normal_stiffness_mpa_per_mm=50., compression_stiffness_mpa_per_mm=50.)
    mass, spring, speed, duration = .001, 1., .2, 2.
    n = round(duration/dt)
    if not np.isclose(n*dt, duration):
        raise ValueError('Duration must be an integer number of steps')
    u = v = k = cumulative_work = cumulative_fracture = cumulative_damping = 0.
    rows = []
    for i in range(1, n+1):
        tick(deadline)
        row = advance(law, displacement=u, velocity=v, history=k, dt=dt,
                      support_before=(i-1)*dt*speed, support_after=i*dt*speed,
                      mass=mass, spring=spring, damping=damping)
        u, v, k = (row[name] for name in ('displacement', 'velocity', 'history'))
        cumulative_work += row['work']; cumulative_fracture += row['fracture']; cumulative_damping += row['damping']
        row.update(time_s=i*dt, support_mm=i*dt*speed, cumulative_work=cumulative_work,
                   cumulative_fracture=cumulative_fracture, cumulative_damping=cumulative_damping,
                   total_stored=.5*spring*(u-i*dt*speed)**2+float(law.response(u, 0., k)['stored']))
        rows.append(row)
    summary = dict(dt_s=dt, damping_n_s_mm=damping, mass_n_s2_mm=mass, spring_n_mm=spring,
                   steps=n, loading_rate_mm_s=speed, parameters='synthetic verification, not rubber measurements',
                   fully_failed=rows[-1]['damage'] == 1.,
                   onset_time_bracket_s=next(([r['time_s']-dt, r['time_s']] for r in rows if r['damage'] > 0.), None),
                   maximum_increment_energy_remainder_n_mm=max(abs(r['energy_remainder']) for r in rows),
                   total_energy_remainder_n_mm=sum(r['energy_remainder'] for r in rows),
                   maximum_force_residual_n=max(r['force_residual'] for r in rows),
                   maximum_kinetic_n_mm=max(r['kinetic'] for r in rows),
                   final_fracture_n_mm=cumulative_fracture, final_damping_n_mm=cumulative_damping,
                   minimum_fracture_increment_n_mm=min(r['fracture'] for r in rows))
    summary['energy_identity_passed'] = (summary['maximum_increment_energy_remainder_n_mm'] < 1e-10
        and abs(summary['total_energy_remainder_n_mm']) < 1e-9 and summary['minimum_fracture_increment_n_mm'] > -1e-12)
    return summary, rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(output, seconds):
    if not np.isfinite(seconds) or not 1 <= seconds <= 300:
        raise ValueError('Budget must be in [1,300] s')
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started+seconds
    root = Path(__file__).resolve().parents[1]
    names = ['tools/verify_fracture_resolution.py', 'prototype/extrinsic_topology.py',
             'prototype/cohesive_dynamics.py', 'prototype/cohesive_mixed.py', 'prototype/material.py',
             'prototype/model_config.py', 'mechanics/finite_strain.py', 'tools/verify_network_compliance.py']
    hashes = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    report = dict(schema='softcut-fracture-resolution/1', generated_utc=datetime.now(timezone.utc).isoformat(),
                  interpreter=sys.executable, source_sha256=hashes, budget_s=seconds,
                  experimental_inputs=[], no_inverse_fitting=True, full_cutting_qualified=False,
                  zero_gap_contact_integrated=False, free_path_direction_qualified=False,
                  material_parameters_qualified=False, bulk_dynamic_cutting_integrated=False,
                  scope='Independent operator/topology, geometric edge-path bias, scalar inertial fracture only')
    report['ties'] = [tie_case(n, deadline) for n in (2, 4, 8, 16)]
    paths = []
    for n in (16, 32, 64):
        for family, seed in [('aligned', 0), ('jittered', 17), ('jittered', 43), ('jittered', 91)]:
            tick(deadline); paths.extend(edge_metric(n, family, seed))
    report['edge_bias'] = paths
    dynamic = []; curves = {}
    for damping in (0., .02):
        for dt in (.004, .002, .001, .0005, .00025, .000125):
            summary, rows = dynamics_case(dt, damping, deadline)
            key = f'dynamics_c{damping:g}_dt{dt:g}'
            write_csv(out/f'{key}.csv', rows)
            curves[(damping, dt)] = rows
            summary['csv'] = key+'.csv'; dynamic.append(summary)
    for damping in (0., .02):
        fine = curves[(damping, .000125)]
        tf = np.array([r['time_s'] for r in fine])
        uf = np.array([r['displacement'] for r in fine])
        for dt in (.004, .002, .001, .0005, .00025):
            rows = curves[(damping, dt)]
            error = np.array([r['displacement'] for r in rows])-np.interp([r['time_s'] for r in rows], tf, uf)
            next(s for s in dynamic if s['damping_n_s_mm'] == damping and s['dt_s'] == dt).update(
                displacement_rms_vs_dt000125_mm=float(np.sqrt(np.mean(error**2))),
                displacement_max_vs_dt000125_mm=float(np.max(abs(error))))
    report['scalar_dynamics'] = dynamic
    report['trajectory_refinement'] = [dict(damping_n_s_mm=c,
        comparison='dt 0.0005 and 0.00025 against 0.000125',
        rms_reduction=(next(r['displacement_rms_vs_dt000125_mm'] for r in dynamic if r['dt_s']==.00025 and r['damping_n_s_mm']==c)/
                       next(r['displacement_rms_vs_dt000125_mm'] for r in dynamic if r['dt_s']==.0005 and r['damping_n_s_mm']==c))) for c in (0., .02)]
    report['source_stable'] = all(hashlib.sha256((root/n).read_bytes()).hexdigest() == h for n, h in hashes.items())
    report['local_verification_passed'] = (report['source_stable'] and all(r['operator_equivalence_passed'] for r in report['ties'])
                                         and all(r['energy_identity_passed'] and r['fully_failed'] for r in dynamic))
    report['elapsed_s'] = time.monotonic()-started
    write_csv(out/'ties.csv', report['ties']); write_csv(out/'edge_bias.csv', paths)
    (out/'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for n in (16, 32, 64):
        rows = [r for r in paths if r['cells'] == n and r['family'] == 'aligned']
        ax[0].plot([r['angle_deg'] for r in rows], [r['cost_ratio'] for r in rows], 'o-', label=f'aligned h=1/{n}')
    ax[0].axhline(1., color='black', linestyle='--'); ax[0].set(xlabel='Requested direction (deg)', ylabel='Minimum edge path / chord')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)
    rows = curves[(.02, .001)]; t = [r['time_s'] for r in rows]
    for name, label in [('cumulative_work', 'External work'), ('total_stored', 'Stored'), ('kinetic', 'Kinetic'),
                        ('cumulative_fracture', 'Fracture'), ('cumulative_damping', 'Physical dashpot')]:
        ax[1].plot(t, [r[name] for r in rows], label=label)
    ax[1].set(xlabel='Time (s)', ylabel='Energy (N mm)', title='Scalar demonstration only'); ax[1].legend(fontsize=8)
    for suffix in ('png', 'svg'): fig.savefig(out/f'verification.{suffix}', dpi=160)
    plt.close(fig)
    content = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>断裂改进：独立验证</title>
<style>body{max-width:1100px;margin:32px auto;font:16px/1.7 system-ui;padding:0 20px}img{width:100%}pre{white-space:pre-wrap;background:#f4f6f8;padding:16px}.notice{border-left:5px solid #c65;padding:12px}</style>
<h1>零间隙连接、裂路方向与失稳能量：独立验证</h1>
<p class="notice">局部验证通过不能代替完整切割。零间隙拓扑尚未接入裂后接触；动力学为单自由度测试，未接入二维 Maxwell/摩擦切割。材料参数均为数值示例，没有读取实验切割力。</p>
<img src="verification.png" alt="网格边方向偏差与惯性断裂能量">
<p>左图检验沿网格边行进的最短长度，表明固定方向网格仅缩小尺寸仍可能保留方向偏差；不是实际裂路预测。右图检验动能、恢复能和耗散的分账恒等式，不把缺项填入断裂能。</p>
<p><a href="verification.json">完整机器结果</a> · <a href="ties.csv">零间隙算子/释放验证</a> · <a href="edge_bias.csv">网格与方向矩阵</a></p>'''
    content += '<pre>'+html.escape(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'edge_bias')}, indent=2, ensure_ascii=False))+'</pre></html>'
    (out/'index.html').write_text(content, encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path); p.add_argument('--seconds', type=float, default=90.)
    a = p.parse_args(); r = run(a.output, a.seconds)
    print(json.dumps({k: r[k] for k in ('local_verification_passed', 'full_cutting_qualified', 'elapsed_s')}))
    return 0 if r['local_verification_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
