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
    if getattr(p, 'actual_affine_policy', False):
        value['predictor_policy'] = 'actual-affine-stagnation-rule'
    if getattr(p, 'shear_limit_mpa', None) is not None:
        value['interface_law'] = dict(name='nodal-current-area-smooth-shear-limit',
            strength_mpa=p.shear_limit_mpa,smooth_min_power=4,area='current rubber boundary dual area')
    return value

def make_problem(*,angle,equilibrium_shear,maxwell_shear,relaxation_s,loading_rate,bulk_modulus=None,mu=.35,
                 equilibrium_alpha=2.,maxwell_alpha=2.,actual_affine_policy=False,shear_limit_mpa=None):
    """Shared construction for actual-angle runs and completed-reference audits."""
    if not all(np.isfinite(a) and 1 <= a <= 4 for a in (equilibrium_alpha,maxwell_alpha)):
        raise ValueError('Explicit candidate exponents must be in [1, 4]')
    p=ViscoelasticProblem(maxwell_shear=maxwell_shear,relaxation_s=relaxation_s,
        loading_rate=loading_rate,bulk_modulus=bulk_modulus,case='oblique',mu=mu,
        shear_modulus=equilibrium_shear,ogden_alpha=equilibrium_alpha,dhat=.0025,local_spacing=.05,
        surface_spacing=.005,nose_facets=64,inclination=angle,tip_radius=.02,
        fine_halfwidth=3.,max_iterations=500,predictor='secant',hessian_mode='regularized',element='p1')
    p.stagnation_window=40
    if maxwell_alpha != 2.:
        p.material=replace(p.material,mu_ov=(2*maxwell_shear/maxwell_alpha,),alpha_ov=(float(maxwell_alpha),))
    mixins=[]
    if actual_affine_policy:
        from tools.replay_affine_policy import ActualAffinePolicy
        mixins.append(ActualAffinePolicy)
        p.actual_affine_policy=True
    if shear_limit_mpa is not None:
        from prototype.shear_limited_contact import ShearLimitedContact
        mixins.append(ShearLimitedContact)
    if mixins:
        p.__class__=type('ExplicitMaterialInterfaceCandidate',tuple(mixins)+ (ViscoelasticProblem,),{})
    if shear_limit_mpa is not None:p.enable_shear_limit(shear_limit_mpa)
    return p

def audit(p,out,report,*,destination=None,progress=None,deadline=None):
    rows=[];prior_x=p.rest;prior_metric=np.tile(np.eye(3),p.material_metric.shape[:1]+(1,1))
    prior_depth=p.start;total_dissipation=0.
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
        row=p.audit_state(state,record)
        total_dissipation+=record['material']['viscous_dissipation_increment_n_mm']
        np.testing.assert_allclose(float(state['cumulative_viscous_dissipation_n_mm']),
                                   total_dissipation,rtol=1e-10,atol=1e-10)
        prior_x=state['positions'].copy();prior_metric=state['material_metric'].copy()
        prior_depth=record['depth_mm']
        row['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(row)
        if progress:progress(row,len(rows),len(report['steps']))
        if len(rows)%25==0:print(f'AUDIT accepted_states={len(rows)}/{len(report["steps"])}',flush=True)
    result=dict(passed=True,states=len(rows),rows=rows,
                scope='fresh force/geometry/history checks; not space/time or experimental convergence')
    write_json(destination or out/'audit.json',result)
    return result

def resume_compatibility(old,report):
    """An audited CLI/reporting revision may differ; physical sources may not."""
    if old['signature']!=report['signature']:
        raise ValueError('Resume requires identical material, angle, mesh, contact and acceptance')
    a,b=old['source_sha256'],report['source_sha256']
    changed=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
    allowed={'tools/run_material_candidate.py','tools/run_common_material_study.py'}
    if set(a)!=set(b) or set(changed)-allowed:
        raise ValueError('Resume physical source fingerprints differ: '+', '.join(changed))
    return dict(changed_runner_files=changed,physical_sources_identical=True,
        previous_source_sha256=a,
        policy='Only explicitly listed CLI/reporting study runners may differ; exact signature and every physical source must match')

def resume(p,source,out,report):
    old=json.loads((source/'result.json').read_text(encoding='utf-8'))
    provenance=resume_compatibility(old,report)
    for name,digest in old['source_sha256'].items():
        if hashlib.sha256((source/'source'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Saved source snapshot differs: '+name)
    if not old['steps']:raise ValueError('No accepted candidate prefix')
    print(f'RESUME_AUDIT source={source} accepted_states={len(old["steps"])}',flush=True)
    audit(p,source,old,destination=out/'resume_audit.json')
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
