"""Bounded saved-state contact/transport audit; no load solving or calibration.

Blade nodal force regions are exact bookkeeping, not a unique face traction.
Proximity residence is NOT contact duration, pressure, or cumulative slip.
"""
import argparse
import contextlib
import csv
from dataclasses import replace
import hashlib
from html import escape
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.ipc_reference import IPCProblem
from prototype.model_config import FEMConfig, _rotation_vectors
from prototype.tip_contact import TipContactTracker
from tools.run_material_candidate import make_problem
from tools.render_solver_dashboard import shell, atomic_write
from run_support import write_json, compute_process_policy, source_hashes

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {
    0: 'outputs/relaxation_partition_20261015/angles30_0_2mm/02_angle0',
    33: 'outputs/native_angles_2mm_20261016/01_angle33',
    57: 'outputs/native_angles_2mm_20261016/02_angle57'}


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def closest_blade(points, blade):
    """Vectorized closest points to the CLOSED original knife polygon."""
    a = blade
    v = np.roll(blade, -1, axis=0)-a
    delta = points[:, None, :]-a
    alpha = np.clip(np.einsum('nki,ki->nk', delta, v)/np.sum(v*v, axis=1), 0., 1.)
    closest = a[None, :, :]+alpha[:, :, None]*v
    d2 = np.sum((points[:, None, :]-closest)**2, axis=2)
    k = np.argmin(d2, axis=1)
    return np.sqrt(d2[np.arange(len(points)), k]), k


def weighted_quantiles(values, weights, probabilities=(.05, .5, .95)):
    if weights.sum() <= 0: return [None]*len(probabilities)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])/weights.sum()
    return [float(values[order[min(np.searchsorted(cumulative, p), len(order)-1)]])
            for p in probabilities]


def construct(r):
    c = r.get('signature', {}).get('config', r.get('config'))
    m = r.get('signature', {}).get('material')
    if m and m.get('mu_ov'):
        K = 2*c['shear_modulus_n_mm2']*(1+c['poisson_ratio'])/(3*(1-2*c['poisson_ratio']))
        p = make_problem(angle=c['inclination_deg'], equilibrium_shear=c['shear_modulus_n_mm2'],
            maxwell_shear=m['mu_ov'][0], relaxation_s=m['relaxation_s'],
            loading_rate=r['signature']['loading_rate'], bulk_modulus=K)
    else:
        p = IPCProblem(case='oblique', mu=.35, shear_modulus=c['shear_modulus_n_mm2'],
            inclination=c['inclination_deg'], local_spacing=.05, surface_spacing=.005,
            fine_halfwidth=3., tip_radius=.02, nose_facets=64, dhat=.0025,
            hessian_mode='regularized', predictor='secant', max_iterations=500)
        p.config = FEMConfig(**c)
    p.ipc.set_num_threads(4)
    return p


def anchor(p, state, row, tracker, nodes, residence, active, force_residence):
    x = state['positions']
    np.testing.assert_array_equal(state['reference_xy'], p.rest)
    np.testing.assert_array_equal(state['triangles'], p.mesh.triangles)
    if 'material_metric' in state and hasattr(p, 'audit_state'):
        fresh = p.audit_state(state, row)
        p.previous_material_metric = state['previous_material_metric'].copy()
        p.material_dt = float(state['material_dt_s'])
    else:
        np.testing.assert_array_equal(x[p.mesh.bottom_nodes], p.rest[p.mesh.bottom_nodes])
        np.testing.assert_allclose(x[p.n:], p.prescribed_blade_positions(row['depth_mm']),
                                   rtol=0, atol=p.blade_boundary_roundoff_tolerance(row['depth_mm']))
        p.previous = state['previous_positions'].copy()
        fresh = p.acceptance(x, row['depth_mm'], 0, 0)
        np.testing.assert_allclose(fresh['reaction_fz_n'], row['reaction_fz_n'], rtol=1e-9, atol=1e-8)
    p.previous = state['previous_positions'].copy()
    ng = p.barrier.gradient(p.collisions(x), p.cmesh, x[p.surface]).reshape(-1, 2)
    fg = p.friction.gradient(p.lag(x), p.cmesh, (x-p.previous)[p.surface]).reshape(-1, 2)
    blade_n, blade_f = -ng[p.surface >= p.n], -fg[p.surface >= p.n]
    np.testing.assert_allclose((blade_n+blade_f)[:, 1].sum(), row['reaction_fz_n'], rtol=1e-9, atol=1e-8)
    pole = 2+p.nose_facets//2
    groups = {'left_shank_shoulder_nodes': np.arange(2), 'left_nose_nodes': np.arange(2, pole),
              'pole_node': np.array([pole]), 'right_nose_nodes': np.arange(pole+1, p.nose_facets+3),
              'right_shank_shoulder_nodes': np.arange(p.nose_facets+3, p.nose_facets+5)}
    force = {name: dict(normal_fz_n=float(blade_n[idx, 1].sum()),
                      friction_fz_n=float(blade_f[idx, 1].sum()),
                      total_fz_n=float((blade_n+blade_f)[idx, 1].sum())) for name, idx in groups.items()}
    # Extract cross-body supports only, with a fresh native collision mesh.
    original = p.cmesh
    cross = p.ipc.CollisionMesh(p.rest[p.surface], np.asarray(original.edges))
    cross.can_collide = p.ipc.make_vertex_patches_filter((p.surface >= p.n).astype(np.int32))
    try:
        p.cmesh = cross
        crossg = p.barrier.gradient(p.collisions(x), cross, x[p.surface]).reshape(-1, 2)
    finally:
        p.cmesh = original
    np.testing.assert_allclose(crossg[p.surface >= p.n], ng[p.surface >= p.n], rtol=1e-9, atol=1e-9)
    body_normal = crossg[np.searchsorted(p.surface, nodes)]
    weights = np.linalg.norm(body_normal, axis=1)
    es, en = _rotation_vectors(p.config.inclination_deg)
    ref_s = p.rest[nodes] @ es-p.config.drop_s_mm
    # Kinematic ratio relative to the rigid blade displacement; no claim of total slip.
    blade_delta = (x[p.n:]-p.previous[p.n:]).mean(axis=0)
    increment = -float(blade_delta[1])
    relative = x[nodes]-p.previous[nodes]-blade_delta
    mids = (x[tracker.edges[:, 0]]+x[tracker.edges[:, 1]])/2
    stretch = np.linalg.norm(x[tracker.edges[:, 1]]-x[tracker.edges[:, 0]], axis=1) / np.linalg.norm(
        p.rest[tracker.edges[:, 1]]-p.rest[tracker.edges[:, 0]], axis=1)
    d = row['depth_mm']; area, grad, _ = p.mesh.bulk_data
    u = x[:p.n]-p.rest[:p.n]
    F = np.eye(2)+np.einsum('eai,eaJ->eiJ', u[p.mesh.triangles], grad)
    strain_measure = np.sum((F.transpose(0, 2, 1)@F-np.eye(2))**2, axis=(1, 2))
    centroid = p.rest[:p.n][p.mesh.triangles].mean(axis=1)
    center = p.rest[nodes[np.argmin(abs(ref_s))]]
    radius = np.linalg.norm(centroid-center, axis=1)
    measures = area*strain_measure
    deformation_radius = weighted_quantiles(radius, measures, (.5, .9))
    field = dict(depth_mm=d, fz_n=row['reaction_fz_n'], fresh_residual_n=fresh['residual_n'],
        fresh_min_j=fresh['min_j'], nodal_force_partition=force,
        reference_normal_force_quantiles_mm=weighted_quantiles(ref_s, weights),
        normal_force_weighted_proximity_residence_depth_mm=float(weights@residence/weights.sum()) if weights.sum() else None,
        normal_force_weighted_continuous_support_depth_mm=[float(weights@v/weights.sum()) if weights.sum() else None for v in force_residence.T],
        normal_force_weighted_relative_material_motion_per_depth=(weights@relative/weights.sum()/increment).tolist() if weights.sum() and increment>0 else None,
        force_support_thresholds=[dict(relative_to_max=t, nodes=int(np.sum(weights>t*weights.max())),
                reference_span_mm=float(np.ptp(ref_s[weights>t*weights.max()])) if weights.max()>0 else None)
                for t in (1e-8, 1e-6, 1e-4)],
        geometric_proximity_nodes=int(active.sum()),
        geometric_footprint=tracker.observe_footprint(x, d),
        tip_contact=tracker.observe(x, d),
        deformation_reference_radius_quantiles_mm=deformation_radius,
        deformation_r90_over_depth=deformation_radius[1]/d if d>0 else None,
        support_stretch_max=float(stretch[closest_blade(mids, x[p.n:])[0]<p.dhat].max(initial=1.)),
        below_halfheight_strain_measure_fraction=float(measures[centroid@en<p.config.height_mm/2].sum()/measures.sum()),
        interpretation='Force regions are nodal bookkeeping; proximity ages are depth intervals, not contact residence time.')
    return field


def inspect_case(folder, label, end=1., *, anchor_depths=(.25, .5, 1.)):
    anchor_depths=tuple(float(d) for d in anchor_depths)
    if any(not np.isfinite(d) or d<0 for d in anchor_depths):
        raise ValueError('Field anchors must be finite and nonnegative')
    folder=Path(folder); r=read(folder/'result.json'); p=construct(r)
    tracker=TipContactTracker(p.rest, p.mesh.triangles, p.n, p.config, p.dhat, p.nose_facets)
    nodes=np.unique(tracker.edges); es, en=_rotation_vectors(p.config.inclination_deg)
    full=p.cmesh;cross=p.ipc.CollisionMesh(p.rest[p.surface],np.asarray(full.edges))
    cross.can_collide=p.ipc.make_vertex_patches_filter((p.surface>=p.n).astype(np.int32))
    force_residence=np.zeros((len(nodes),3));previous_force_active=np.zeros_like(force_residence,dtype=bool)
    source_hashes={}
    for name, digest in r.get('source_sha256', {}).items():
        f=folder/'source'/name
        if sha(f)!=digest: raise ValueError('Changed frozen source: '+str(f))
        source_hashes[str(f.resolve())]=digest
    fingerprints={str((folder/'result.json').resolve()):sha(folder/'result.json'), **source_hashes}
    audited=read(folder/'audit.json') if (folder/'audit.json').exists() else None
    residence=np.zeros(len(nodes)); previous_active=np.zeros(len(nodes),dtype=bool)
    rows=[]; fields=[]; profiles=[]; old_depth=0.; previous_x=None
    for i, row in enumerate(r['steps']):
        d=row['depth_mm']
        if d>end+1e-10: break
        path=folder/'states'/f'{i:04d}.npz'; digest=sha(path)
        if audited and digest!=audited['rows'][i]['checkpoint_sha256']:raise ValueError('Changed audited state')
        fingerprints[str(path.resolve())]=digest
        is_anchor=any(abs(d-z)<1e-10 for z in anchor_depths)
        with np.load(path,allow_pickle=False) as z:
            x=z['positions']; previous=z['previous_positions']
            if previous_x is not None:np.testing.assert_array_equal(previous,previous_x)
            distance,_=closest_blade(x[nodes],x[p.n:]); active=distance<p.dhat
            residence=np.where(active & previous_active,residence+(d-old_depth),0.)
            # Same observation code for all generations, including old runs
            # whose stored records did not yet export unique pole identities.
            tip=tracker.observe(x,d)
            try:
                p.cmesh=cross
                g=p.barrier.gradient(p.collisions(x),cross,x[p.surface]).reshape(-1,2)
            finally:p.cmesh=full
            weight=np.linalg.norm(g[np.searchsorted(p.surface,nodes)],axis=1)
            force_active=weight[:,None]>weight.max()*np.array([1e-8,1e-6,1e-4])[None,:]
            force_residence=np.where(force_active & previous_force_active,force_residence+(d-old_depth),0.)
            rows.append(dict(depth_mm=d, fz_n=row['reaction_fz_n'], proximity_nodes=int(active.sum()),
                newly_near_nodes=int(np.sum(active & ~previous_active)),
                departed_nodes=int(np.sum(~active & previous_active)),
                unique_tip_offset_mm=tip.get('reference_offset_mm') if tip.get('status')=='unique' else None,
                tip_status=tip.get('status','missing'), max_proximity_residence_depth_mm=float(residence.max()),
                force_support_nodes=int(force_active[:,1].sum()),
                force_support_depth_mm=float(weight@force_residence[:,1]/weight.sum()) if weight.sum()>0 else None))
            if is_anchor:
                state={k:z[k] for k in z.files}
                fields.append(anchor(p,state,row,tracker,nodes,residence,active,force_residence))
                profiles.append(dict(depth_mm=d, reference_offset_mm=(p.rest[nodes]@es-p.config.drop_s_mm).tolist(),
                    tangential_u_mm=((x[nodes]-p.rest[nodes])@es).tolist(),
                    normal_u_mm=((x[nodes]-p.rest[nodes])@en).tolist()))
            previous_x=x;previous_active=active;previous_force_active=force_active;old_depth=d
    print(f'CONTACT_AUDIT label={label} states={len(rows)} fresh_fields={len(fields)}',flush=True)
    return dict(label=label,angle=p.config.inclination_deg,path=str(folder.resolve()),
                rows=rows,fields=fields,profiles=profiles,input_sha256=fingerprints)


def render(out, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={0:'#222222',30:'#0072B2',60:'#D55E00',33:'#0072B2',57:'#D55E00'}
    stages=list(dict.fromkeys(c['label'].split('_')[0] for c in cases))
    fig,axes=plt.subplots(3,len(stages),figsize=(4.2*len(stages),9),layout='constrained',sharex='row',sharey='row',squeeze=False)
    for c in cases:
        j=stages.index(c['label'].split('_')[0]);rows=c['rows'];f=c['fields'];color=colors[c['angle']]
        axes[0,j].plot([r['depth_mm'] for r in rows],[r['fz_n'] for r in rows],color=color,label=f"{c['angle']:g} deg")
        axes[1,j].plot([r['depth_mm'] for r in rows],[r['unique_tip_offset_mm'] if r['unique_tip_offset_mm'] is not None else np.nan for r in rows],color=color)
        axes[2,j].plot([r['depth_mm'] for r in rows],[r['force_support_depth_mm'] if r['force_support_depth_mm'] is not None else np.nan for r in rows],color=color)
    for j,stage in enumerate(stages):axes[0,j].set_title(stage);axes[0,j].legend();axes[2,j].set_xlabel('Vertical depth (mm)')
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Fz (N)');axes[1,0].set_ylabel('Unique pole material offset (mm)\nMissing observations are gaps')
    axes[2,0].set_ylabel('Continuous nodal load support\n(in depth mm; threshold 1e-6 max force)')
    fig.savefig(out/'contact_transport.png',dpi=155);plt.close(fig)
    body='<h1>保存路径：承载与材料迁移</h1><p>无新加载。逐态核对输入指纹，锚点按准确历史重装配；原生模型的力学验收不是场精度证明。旧30/60与当前33/57分别标记，不混为同角度。</p>'
    body+='<img src="contact_transport.png" style="max-width:100%" alt="各历史阶段的反力、唯一刀尖材料偏移和连续节点承载深度；缺失身份留空。">'
    body+='<p>刀具节点分区反力相加严格还原Fz，但肩部节点不能唯一归属于相邻某一刀面。屏障邻近长度/持续深度不是承压长度/接触时间，参考坐标偏移不是累计滑程。没有由几何长度推算压力。</p>'
    body+='<p>下排按每个保存态的刀具—材料法向节点力判断连续承载，主阈值为该态最大节点力的1e−6；1e−8和1e−4敏感性另存锚点字段。边接触力分配到端点，故这是离散承载历史，仍非唯一接触点的真实停留时间。所有历史版本都用同一观测器重新计算刀尖身份。</p>'
    body+='<table><tr><th>模型/角度</th><th>d mm</th><th>Fz N</th><th>承载材料5/50/95%坐标 mm</th><th>邻近持续深度 mm</th><th>minJ</th></tr>'
    for c in cases:
        for f in c['fields']:
            body+=f'<tr><td>{escape(c["label"])}</td><td>{f["depth_mm"]:.2f}</td><td>{f["fz_n"]:.4f}</td><td>'+escape(str(f['reference_normal_force_quantiles_mm']))+f'</td><td>{f["normal_force_weighted_proximity_residence_depth_mm"]:.4f}</td><td>{f["fresh_min_j"]:.4f}</td></tr>'
    body+='</table><p><a href="report.json">完整字段、分区力与全部输入哈希</a> · <a href="history.csv">历史表</a></p>'
    atomic_write(out/'index.html',shell('接触与材料迁移',body))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--refresh',action='store_true');args=ap.parse_args()
    out=args.output
    if args.refresh:
        old=read(out/'report.json')
        if old.get('new_loading_solves')!=0 or (out/'previous_report.json').exists():raise ValueError('Only one explicit own-report revision allowed')
        write_json(out/'previous_report.json',old)
    else:out.mkdir(parents=True,exist_ok=False)
    cases=[];start=time.monotonic()
    old={0:'outputs/ipc_contact_study_20261006/06_angle0_mu0.35',30:'outputs/ipc_contact_study_20261006/05_angle30_mu0.35',60:'outputs/ipc_friction_unlimited_20260912/01_angle60_mu0.35'}
    first={0:'outputs/material_angles_2mm_20261014/03_angle0',30:'outputs/material_angles_2mm_20261014/02_angle30',60:'outputs/material_angles_2mm_20261014/01_angle60'}
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy():
        for stage, paths in [('elastic066',old),('firstMaxwell',first),('currentMaxwell',BASELINES)]:
            for angle,path in paths.items():
                cases.append(inspect_case(ROOT/path,f'{stage}_{angle}'))
                write_json(out/'report.json',dict(status='running',cases=cases))
    write_json(out/'report.json',dict(status='completed',cases=cases,elapsed_s=time.monotonic()-start,new_loading_solves=0,
        analysis_sha256=sha(__file__),fresh_assembly_source_sha256=source_hashes(),
        revision='Common pole observer and force-supported nodal history; previous proximity-only report retained' if args.refresh else None))
    with (out/'history.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=['label','angle',*cases[0]['rows'][0]];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for c in cases:
            for r in c['rows']:writer.writerow(dict(label=c['label'],angle=c['angle'],**r))
    render(out,cases)
    print(json.dumps(dict(status='completed',cases=len(cases),fresh_fields=sum(len(c['fields']) for c in cases),elapsed_s=time.monotonic()-start)))


if __name__=='__main__':main()
