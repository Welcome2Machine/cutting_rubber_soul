"""Shared intact-Maxwell case construction, exact-history IO and audits.

No CLI side effects. Experimental forces retain the exported units and zero.
Changing module boundaries does not permit resuming mismatched frozen physics.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
from dataclasses import replace
import numpy as np
from run_support import write_json
from .ipc_reference import BASELINE
from .ipc_viscoelastic import ViscoelasticProblem


def signature(p):
    value = dict(config=p.config.to_dict(),material=p.material.to_dict(),loading_rate=p.loading_rate,
                mesh_sha256=hashlib.sha256(p.rest.tobytes()+p.mesh.triangles.tobytes()).hexdigest(),
                contact_formulation=p.contact_formulation,dhat=p.dhat,
                physical_force_tolerance_n=BASELINE['force_tolerance_n'])
    if getattr(p,'friction_smoothing_mm',BASELINE['friction_smoothing_mm']) != BASELINE['friction_smoothing_mm']:
        value['friction_smoothing_mm'] = p.friction_smoothing_mm
    if p.enforce_physical_chord:
        value["enforce_physical_chord"] = True
    return value


def resolved_model(p):
    """Actual constructed geometry and time/friction meaning, not input echo."""
    blade = p.rest[p.n:]
    arc = blade[2:p.nose_facets+3]
    origin = arc.mean(axis=0); z = arc-origin
    center_local = np.linalg.lstsq(2*z, np.sum(z*z,axis=1)-np.mean(np.sum(z*z,axis=1)), rcond=None)[0]
    center = origin+center_local; radii = np.linalg.norm(arc-center,axis=1)
    left = blade[1]-arc[0]; right = blade[p.nose_facets+3]-arc[-1]
    angle = np.rad2deg(np.arccos(np.clip(left@right/np.linalg.norm(left)/np.linalg.norm(right),-1,1)))
    return dict(config=p.config.to_dict(), material=p.material.to_dict(),
        loading_rate_mm_s=p.loading_rate, initial_depth_mm=p.start,
        physical_time='dt=(depth-previous_depth)/loading_rate; elapsed=(depth-initial_depth)/loading_rate',
        integrator=('full-domain backward Euler momentum, implicit Maxwell and CZM' if getattr(p,'event_mode',False)
            else 'quasistatic force balance, implicit Maxwell material update; PTC is not physical time'),
        inertia=(dict(density_kg_m3=p.event_density_kg_m3,total_mass_n_s2_mm=float(p.mass.sum()),
            mirror_factor=1,blade_mass=0) if getattr(p,'event_mode',False) else None),
        friction=dict(coefficient=p.config.friction_coefficient,
            smoothing_kind=('fixed_velocity' if getattr(p,'event_mode',False) else 'fixed_increment_displacement'),
            epsilon_u_mm=p.friction_smoothing_mm,
            epsilon_v_mm_s=(p.event_friction_velocity_mm_s if getattr(p,'event_mode',False) else None),
            scope='same coefficient for blade and body self-contact; closure penalty excluded from IPC friction load'),
        contact=dict(formulation=p.contact_formulation,dhat_mm=p.dhat,
            enforce_physical_chord=p.enforce_physical_chord,
            ccd=getattr(p,'network_ccd_policy',{'mode':'toolkit_default'})),
        geometry=dict(actual_wedge_included_angle_deg=float(angle),
            actual_blade_thickness_mm=float(np.ptp(blade[:,0])),
            actual_arc_radius_mm=float(radii.mean()),arc_fit_max_error_mm=float(np.max(abs(radii-radii.mean()))),
            radius_status='assumed, not independently measured',
            blade_reference_sha256=hashlib.sha256(blade.tobytes()).hexdigest(),
            body_reference_sha256=hashlib.sha256(p.mesh.reference_xy.tobytes()).hexdigest(),
            triangles_sha256=hashlib.sha256(p.mesh.triangles.tobytes()).hexdigest(),
            nodes=p.n,triangles=len(p.mesh.triangles),nose_facets=p.nose_facets,
            blade_spacing_mm=p.blade_spacing,reference_volume_mm3=float(p.mesh.bulk_data[0].sum()*p.config.width_mm)),
        damage_off_semantics=('finite_stiffness_finite_gap_elastic_interface' if hasattr(p,'cohesive_law')
            and not p.cohesive_law.damage_enabled else 'continuous_intact' if not hasattr(p,'cohesive_law') else 'damage_enabled'))


def failure_checkpoint(p, x):
    """Pair a rejected iterate with its actual pre-increment dual history.

    The load controller may already have rolled ``previous`` back. Material
    integration scratch still contains the attempted dt and pre-step metric.
    A saved PTC direction belongs to its own base position, not the last trial.
    """
    data = dict(positions=getattr(p, 'trial', x),
                previous_positions=getattr(p, 'failed_previous', p.previous),
                **p.checkpoint_data())
    context = getattr(p, 'failed_context', {}).copy()
    context.update(history_pairing='actual_attempt_start',
                   phase_seconds=getattr(p, 'phase_seconds', {}).copy())
    return data, context


def probe_ptc_state(p,state):
    """Probe without changing either pre-increment history of the caller."""
    before=(p.previous,p.previous_material_metric,p.material_dt)
    try:return _probe_ptc_state(p,state)
    finally:p.previous,p.previous_material_metric,p.material_dt=before


def _probe_ptc_state(p,state):
    """Inspect an actual saved direction; no history commit or loading solve.

    Rebuild the recorded shift/regularization, test the ORIGINAL linear system,
    then sample complete fresh EV/VV forces along the same safe ray. Additional
    Newton and metric-residual-gradient rays are explicitly labelled probes.
    """
    from scipy.sparse import coo_matrix,diags
    p.previous=state['previous_positions'].copy()
    p.previous_material_metric=state['previous_material_metric'].copy()
    p.material_dt=float(state['material_dt_s'])
    x=state['positions'];lag=p.lag(x)
    _,g,H,_,metric=p.evaluate(x,lag,projected=False,with_metric=True)
    free=p.free;r=g[free];norm=float(np.linalg.norm(r))
    A=p.coupled_friction_tangent(x,lag,H)[free][:,free]
    M=metric[free][:,free]+diags(np.full(len(free),BASELINE['hessian_regularization_n_mm']))
    def embed(matrix):
        q=matrix.tocoo()
        return coo_matrix((q.data,(p.surface_dofs[q.row],p.surface_dofs[q.col])),shape=H.shape).tocsc()[free][:,free]
    def parts(position,total,friction):
        normal=p.collisions(position);sx=position[p.surface]
        gn=np.zeros(p.total);gf=gn.copy()
        gn[p.surface_dofs]=p.barrier.gradient(normal,p.cmesh,sx)
        gf[p.surface_dofs]=p.friction.gradient(friction,p.cmesh,(position-p.previous)[p.surface])
        return dict(bulk=(total-gn-gf)[free],normal=gn[free],friction=gf[free]),normal
    base,c=parts(x,g,lag)
    projection=p.ipc.PSDProjectionMethod.NONE
    N=embed(p.barrier.hessian(c,p.cmesh,x[p.surface],projection))
    frozen_f=embed(p.friction.hessian(lag,p.cmesh,(x-p.previous)[p.surface],projection))
    B=H[free][:,free]-N-frozen_f
    jacobians=dict(bulk=B,normal=N,friction=A-B-N)
    def keys(collisions):
        return ({('vv',int(s.vertex0_id),int(s.vertex1_id)) for s in collisions.vv_collisions}
            |{('ev',int(s.edge_id),int(s.vertex_id)) for s in collisions.ev_collisions})
    direction=state['direction'].ravel()[free]
    native_direction=state.get('native_direction',state['direction']).ravel()[free]
    shift=float(state['shift_used'])
    result=dict(depth_mm=float(state['depth_mm']),material_dt_s=p.material_dt,
        residual_n=norm,saved_residual_difference_n=float(np.linalg.norm(r-state['residual'])),
        recorded_linear_relative_residual=float(np.linalg.norm((A+shift*M)@native_direction+r)/norm),
        regularization_n_mm=BASELINE['hessian_regularization_n_mm'],shift_used=shift,rays={})
    nodal=np.zeros(p.total);nodal[free]=r
    nodal_norm=np.linalg.norm(nodal.reshape(-1,2),axis=1)
    top=np.argsort(nodal_norm[:p.n])[-5:][::-1]
    result['dominant_nodes']=[dict(id=int(i),residual_n=float(nodal_norm[i]),
        position_mm=x[i].tolist(),reference_local_sn_mm=p.mesh.local_sn[i].tolist()) for i in top]
    result['top5_residual_squared_fraction']=float(nodal_norm[top]@nodal_norm[top]/(norm*norm))
    result['normal_feature_counts']=dict(ev=len(c.ev_collisions),vv=len(c.vv_collisions))
    result['dominant_normal_stencils']=[]
    for kind,stencils in (('ev',c.ev_collisions),('vv',c.vv_collisions)):
        for stencil in stencils:
            local_ids=np.array(stencil.vertex_ids(p.cmesh.edges,p.cmesh.faces),dtype=int)
            local_ids=local_ids[local_ids>=0];global_ids=p.surface[local_ids]
            if not set(top[:3])&set(global_ids):continue
            local=x[global_ids].ravel();gradient=p.barrier.gradient(stencil,local)
            ldofs=(2*global_ids[:,None]+np.arange(2)).ravel()
            row=dict(kind=kind,global_node_ids=global_ids.tolist(),weight=float(stencil.weight),
                gap_mm=float(np.sqrt(stencil.compute_distance(local))),
                free_gradient_norm_n=float(np.linalg.norm(gradient[np.isin(ldofs,free)])))
            if kind=='ev':
                edge=p.cmesh.edges[stencil.edge_id];a,b=x[p.surface[edge]]
                v=x[p.surface[stencil.vertex_id]]
                row['edge_projection']=float((v-a)@(b-a)/((b-a)@(b-a)))
            else:
                row['connected_body_boundary']=bool(np.all(global_ids<p.n) and
                    tuple(sorted(local_ids)) in p._connected_boundary_pairs)
            result['dominant_normal_stencils'].append(row)
    candidates={'recorded':direction,'newton':p.linear_direction(A,-r),
        'metric_residual_gradient':p.linear_direction(M,-A.T@r)}
    for name,du in candidates.items():
        action=A@du;slope=float(r@action);d=np.zeros(p.total);d[free]=du;d=d.reshape(-1,2)
        cap=min(1.,.5*p.options.tip_radius_mm/np.max(np.abs(du)),float(p.safe_step(x,d)))
        if name=='metric_residual_gradient' and slope<0:
            cap=min(cap,-slope/float(action@action))
        samples=[]
        for fraction in (1.,.25,.0625,.015625,.00390625):
            alpha=cap*fraction;y=x+alpha*d;ylag=p.lag(y)
            gy=p.evaluate(y,ylag,tangent=False)[1];component,cy=parts(y,gy,ylag)
            errors={k:component[k]-base[k]-alpha*(jacobians[k]@du) for k in base}
            samples.append(dict(alpha=alpha,residual_n=float(np.linalg.norm(gy[free])),
                predicted_residual_n=float(np.linalg.norm(r+alpha*action)),
                model_error_n=float(np.linalg.norm(gy[free]-r-alpha*action)),
                component_model_error_n={k:float(np.linalg.norm(v)) for k,v in errors.items()},
                changed_features=len(keys(c)^keys(cy))))
        result['rays'][name]=dict(residual_squared_slope=slope,metric_quadratic=float(du@(M@du)),samples=samples)
    return result

def make_problem(*,angle,equilibrium_shear,maxwell_shear,relaxation_s,loading_rate,bulk_modulus=None,mu=.35,
                 equilibrium_alpha=2.,maxwell_alpha=2.,
                 volumetric='quadratic_j',blade_spacing=None,blade_thickness=None,blade_angle=None,
                 tip_radius=.02,local_spacing=.05,surface_spacing=.005,nose_facets=64,
                 fine_halfwidth=3.,coarse_spacing=None,drop_s=None,dhat=.0025,
                 preparation_clearance_mm=None,friction_smoothing_mm=1e-5,enforce_physical_chord=False):
    """Shared construction for actual-angle runs and completed-reference audits."""
    if not all(np.isfinite(a) and 1 <= a <= 4 for a in (equilibrium_alpha,maxwell_alpha)):
        raise ValueError('Explicit candidate exponents must be in [1, 4]')
    p=ViscoelasticProblem(maxwell_shear=maxwell_shear,relaxation_s=relaxation_s,
        loading_rate=loading_rate,bulk_modulus=bulk_modulus,volumetric=volumetric,case='oblique',mu=mu,
        shear_modulus=equilibrium_shear,ogden_alpha=equilibrium_alpha,dhat=dhat,local_spacing=local_spacing,
        surface_spacing=surface_spacing,nose_facets=nose_facets,inclination=angle,tip_radius=tip_radius,
        fine_halfwidth=fine_halfwidth,max_iterations=500,predictor='secant',hessian_mode='regularized',element='p1',
        blade_spacing=blade_spacing,blade_thickness=blade_thickness,blade_angle=blade_angle,
        coarse_spacing=coarse_spacing,drop_s=drop_s,preparation_clearance_mm=preparation_clearance_mm,
        friction_smoothing_mm=friction_smoothing_mm,enforce_physical_chord=enforce_physical_chord)
    p.stagnation_window=40
    if maxwell_alpha != 2.:
        p.material=replace(p.material,mu_ov=(2*maxwell_shear/maxwell_alpha,),alpha_ov=(float(maxwell_alpha),))
    return p

def audit(p,out,report,*,destination=None,progress=None,deadline=None,initial_state=None):
    rows=[];prior_x=p.rest;prior_metric=np.tile(np.eye(3),p.material_metric.shape[:1]+(1,1))
    prior_depth=p.start;total_dissipation=0.
    if initial_state is not None:
        if not getattr(p,'event_mode',False):raise ValueError('Explicit initial state only for conditional BE audit')
        prior_x=initial_state['positions'].copy();prior_metric=initial_state['material_metric'].copy()
        prior_depth=float(initial_state['depth_mm'])
        total_dissipation=float(initial_state['cumulative_viscous_dissipation_n_mm'])
        prior_velocity=initial_state['velocity'].copy();prior_time=float(initial_state['event_time_s'])
        prior_cohesive=initial_state['cohesive_maximum'].copy()
    for i,record in enumerate(report['steps']):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Independent audit wall budget')
        path=out/'states'/f'{i:04d}.npz'
        with np.load(path,allow_pickle=False) as archive:
            state={key:archive[key] for key in archive.files}
        np.testing.assert_array_equal(state['previous_positions'],prior_x)
        np.testing.assert_array_equal(state['previous_material_metric'],prior_metric)
        np.testing.assert_allclose(float(state['depth_mm']),record['depth_mm'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(float(state['material_dt_s']),
            (record['depth_mm']-prior_depth)/p.loading_rate,rtol=1e-12,atol=1e-12)
        if initial_state is not None:
            np.testing.assert_array_equal(state['previous_velocity'],prior_velocity)
            np.testing.assert_array_equal(state['previous_cohesive_maximum'],prior_cohesive)
            np.testing.assert_allclose(float(state['previous_event_time_s']),prior_time,rtol=0,atol=1e-14)
            np.testing.assert_allclose(float(state['event_time_s']),prior_time+float(state['material_dt_s']),rtol=0,atol=1e-14)
            np.testing.assert_allclose(record['event']['time_s'],float(state['event_time_s']),rtol=0,atol=1e-14)
            prior_velocity=state['velocity'].copy();prior_time=float(state['event_time_s'])
            prior_cohesive=state['cohesive_maximum'].copy()
        row=p.audit_state(state,record)
        total_dissipation+=record['material']['viscous_dissipation_increment_n_mm']
        np.testing.assert_allclose(float(state['cumulative_viscous_dissipation_n_mm']),
                                   total_dissipation,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(record['material']['cumulative_viscous_dissipation_n_mm'],
                                   total_dissipation,rtol=1e-10,atol=1e-10)
        prior_x=state['positions'].copy();prior_metric=state['material_metric'].copy()
        prior_depth=record['depth_mm']
        row['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(row)
        if progress:progress(row,len(rows),len(report['steps']))
        if len(rows)%25==0:print(f'AUDIT accepted_states={len(rows)}/{len(report["steps"])}',flush=True)
    prefix=next((i for i,row in enumerate(rows) if not row['passed']),len(rows))
    result=dict(passed=bool(rows) and prefix==len(rows),states=len(rows),rows=rows,
                qualified_prefix_states=prefix,
                endpoint_checks_passed=all(row.get('endpoint_passed',row['passed']) for row in rows),
                trajectory_checked=bool(rows) and all('trajectory_passed' in row for row in rows),
                scope='fresh force/geometry/history checks; cohesive paths also check each physical chord; not space/time or experimental convergence')
    write_json(destination or out/'audit.json',result)
    return result

def resume_compatibility(old,report):
    """An audited CLI/reporting revision may differ; physical sources may not."""
    if old['signature']!=report['signature']:
        raise ValueError('Resume requires identical material, angle, mesh, contact and acceptance')
    native_policy=dict(version=1,predictor='secant',affine_stagnation_window=0)
    if old.get('solver_policy',native_policy)!=report.get('solver_policy',native_policy):
        raise ValueError('Resume nonlinear solver policy differs; start an independent path')
    a,b=old['source_sha256'],report['source_sha256']
    changed=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
    allowed={'tools/run_material_candidate.py'}
    if set(a)!=set(b) or set(changed)-allowed:
        raise ValueError('Resume physical source fingerprints differ: '+', '.join(changed))
    return dict(changed_runner_files=changed,physical_sources_identical=True,
        previous_source_sha256=a,
        policy='Only explicitly listed CLI/reporting study runners may differ; exact signature and every physical source must match')

def resume(p,source,out,report):
    old=json.loads((source/'result.json').read_text(encoding='utf-8'))
    provenance=resume_compatibility(old,report)
    root=Path(__file__).resolve().parents[1]
    for name,digest in report['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Current source fingerprint differs: '+name)
    if not old['steps']:raise ValueError('No accepted candidate prefix')
    print(f'RESUME_AUDIT source={source} accepted_states={len(old["steps"])}',flush=True)
    checked=audit(p,source,old,destination=out/'resume_audit.json')
    requested=len(old['steps'])
    if (checked.get('passed') is not True or checked.get('states') != requested
            or checked.get('qualified_prefix_states') != requested):
        raise ValueError('Resume requires the complete requested prefix to pass independent audit')
    for i in range(len(old['steps'])):
        f=source/'states'/f'{i:04d}.npz';shutil.copy2(f,out/'states'/f.name)
    report['resume_provenance']=provenance
    report['steps']=old['steps'];report['resumed_from']=str(source.resolve())
    report['prior_elapsed_s']=old.get('elapsed_s',0.)+old.get('prior_elapsed_s',0.)
    report['rejected_load_attempts']=old['rejected_load_attempts'].copy()
    last=report['steps'][-1]['depth_mm']
    with np.load(out/'states'/f"{len(report['steps'])-1:04d}.npz") as state:
        x=state['positions'].copy();previous=state['previous_positions'].copy()
        p.material_metric=state['material_metric'].copy()
        p.material_dissipation=float(state['cumulative_viscous_dissipation_n_mm'])
    p.previous=x.copy()
    history=([(report['steps'][-2]['depth_mm'],previous)] if len(report['steps'])>1 else [])
    p.seed_predictor_history(history+[(last,x)])
    report['last_depth_mm']=last
    print(f'RESUME_READY depth_mm={last:.6f} exact_material_and_friction_history=True',flush=True)
    return x,last

def load_force_csv(path):
    """Explicit experiment export schema; units already N, no sensor rescaling."""
    path=Path(path)
    if path.stat().st_size>20_000_000:raise ValueError('Force CSV exceeds bounded reader size')
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream,strict=True)
        required={'Depth (\u03bcm)','Fz (N)','depth_unit','force_unit','curve_type','component','speed_um_s'}
        if not required<=set(reader.fieldnames or []):raise ValueError('Unexpected force CSV schema')
        rows=[]
        for row in reader:
            if len(rows)>=100000:raise ValueError('Too many force samples')
            if (row.get('depth_unit')!='\u03bcm' or row.get('force_unit')!='N'
                    or row.get('curve_type')!='experiment' or row.get('component')!='Fz'):
                raise ValueError('Force CSV units/type/component inconsistent')
            rows.append(row)
    d=np.array([float(r['Depth (\u03bcm)'])/1000 for r in rows])
    f=np.array([float(r['Fz (N)']) for r in rows])
    rates={float(r['speed_um_s'])/1000 for r in rows}
    if (len(d)<2 or not np.isfinite(d).all() or not np.isfinite(f).all()
            or not np.all(np.diff(d)>0) or abs(d[0])>1e-12
            or len(rates)!=1 or not np.isfinite(next(iter(rates))) or next(iter(rates))<=0):
        raise ValueError('Require finite ordered initial-contact loading data and one positive speed')
    return d,f,rates.pop()
