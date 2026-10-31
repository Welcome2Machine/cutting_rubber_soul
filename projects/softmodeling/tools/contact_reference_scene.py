"""Conditional frozen-force hard-contact free-flight probe on an audited C1 state.

Uses a declared source checkout to reconstruct the parent. The planar event
kernel advances exact mirror constraints; full native contact and a conservative
motion envelope certify that no other hard contact is reached. If that
certificate fails, this limited reference STOPS instead of ignoring contacts.
No FE history is committed and no production continuation is generated.
"""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',type=Path);parser.add_argument('--parent-audit',type=Path,required=True);parser.add_argument('--checkout-fe',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--new-rejection',action='store_true');parser.add_argument('--seconds',type=float,default=120);a=parser.parse_args()
    if a.out.exists() or not 0<a.seconds<=540:raise ValueError('New bounded conditional output required')
    sys.path.insert(0,str(a.checkout_fe.resolve()))
    from numerical_runtime import numerical_runtime
    from tools import run_endpoint_face_route as route
    from tools import run_fracture_route as base
    from tools.face_events import endpoint_forces
    spec=importlib.util.spec_from_file_location('round7_impulse_reference',Path(__file__).resolve().parents[1]/'mechanics/impulse_reference.py');kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)
    start=time.monotonic()
    with numerical_runtime(2,1,True) as runtime:
        raw=json.loads((a.case/'result.json').read_text());audit=json.loads(a.parent_audit.read_text())
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if not audit['passed'] or audit['result_sha256']!=digest(a.case/'result.json') or audit['source_hashes']!=raw['source_hashes']:raise ValueError('Original complete prefix audit required')
        if route.sources()!=raw['source_hashes'] or not raw['source_unchanged']:raise ValueError('Parent source must match its declared checkout')
        trial=None
        if a.new_rejection:
            candidates=[]
            for path in (a.case/'witnesses/rejected').glob('*.npz'):
                with np.load(path,allow_pickle=False) as z:
                    m=json.loads(str(z['metadata_json']));e=m.get('energy')
                    if e and e['friction_algorithmic_work_n_mm']<0:candidates.append((abs(e['friction_time_defect_n_mm']),path,m))
            _,trial,meta=max(candidates,key=lambda v:v[0]);parent_name=meta['attempt']['parent']['file'];h=meta['attempt']['physical_dt_s']
            rec=next(r for r in raw['steps'] if r['state_file']==parent_name);index=raw['steps'].index(rec)
        else:
            rec=raw['steps'][-1];index=len(raw['steps'])-1
            h=next(r['dt_s'] for r in reversed(raw['attempts']) if 'Negative endpoint friction work' in r['error'])
        parent=a.case/rec['state_file']
        if digest(parent)!=rec['state_sha256']:raise ValueError('Parent NPZ identity')
        with np.load(parent,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        p=route.problem(raw['config']);fresh=base.restore_audit(p,s,rec);x=s['positions'];v=s['velocity'];fields=endpoint_forces(p,x)
        if not fresh['passed'] or not fresh['accepted_chord_ccd_safe']:raise ValueError('Parent reassembly failed')
        force=-(fields['bulk']+fields['cohesive']).reshape(-1,2)[:p.n]
        y=x.copy();endv=v.copy();impulse=np.zeros((p.n,2));loss=0.;frozen_work=0.;motion=np.zeros(len(x));events=[]
        tied=set(p.seam[~s['released']].tolist());bottom=set(p.mesh.bottom_nodes.tolist());centre=p.config.drop_s_mm
        for i in range(p.n):
            if time.monotonic()-start>a.seconds:raise TimeoutError('Conditional reference budget')
            if i in bottom:impulse[i]=-h*force[i];continue
            f=force[i].copy()
            if i in tied:f[0]=0
            q=kernel.advance([x[i,1],x[i,0]-centre],[v[i,1],v[i,0]],0.,h,mass=p.mass[i],force=f[::-1],mu=0.)
            y[i]=np.array(q['position'])[::-1]+[centre,0.];endv[i]=np.array(q['velocity'])[::-1];impulse[i]=q['contact_impulse_n_s'][::-1]
            if i in tied:impulse[i,0]=-h*force[i,0]
            loss+=2*q['normal_impact_loss_n_mm'];frozen_work+=2*float(force[i]@(y[i]-x[i]));motion[i]=np.linalg.norm(v[i])*h+.5*np.linalg.norm(f)*h*h/p.mass[i]
            events.extend(dict(node=i,**e) for e in q['events'])
        y[p.n:]=p.prescribed_blade_positions(rec['depth_mm']+p.loading_rate*h);endv[p.n:]=[0.,-p.loading_rate];motion[p.n:]=p.loading_rate*h
        gap=float(np.sqrt(p.collisions(x).compute_minimum_distance(p.cmesh,x[p.surface])))
        # Every potential contact distance is 2-Lipschitz in maximum vertex
        # motion. Initial candidates outside dhat stay safe if 2R<dhat.
        envelope=2*float(np.max(motion));distance_certificate=envelope<min(gap,p.dhat)
        F0=p.deformation(x);grad=p.mesh.bulk_data[1]
        dF_bound=np.sum(motion[p.mesh.triangles]*np.linalg.norm(grad,axis=2),axis=1)
        singular=np.linalg.svd(F0,compute_uv=False)[:,-1];j_certificate=bool(np.all(dF_bound<singular))
        chord_safe=bool(p.safe_step(x,y-x)>=1-1e-10)
        geometry_safe=bool(distance_certificate and j_certificate and chord_safe)
        report=dict(schema='softcut-conditional-hard-flight/1',scope='Conditional contact model change with frozen noncontact forces; never a production suffix',
          parent_source_hashes=raw['source_hashes'],parent_checkout=str(a.checkout_fe.resolve()),parent_result_sha256=digest(a.case/'result.json'),parent_audit_sha256=digest(a.parent_audit),parent_state_sha256=digest(parent),parent_index=index,parent_depth_mm=rec['depth_mm'],
          new_rejection_sha256=digest(trial) if trial else None,current_reference_sha256=digest(spec.origin),runner_sha256=digest(Path(__file__)),thread_runtime=runtime,physical_dt_s=h,
          parent_reassembly=fresh,initial_native_contact_gap_mm=gap,motion_envelope_mm=envelope,full_native_chord_ccd=chord_safe,
          curved_motion_distance_certificate=bool(distance_certificate),whole_interval_positive_j_certificate=j_certificate,geometry_safe=geometry_safe,
          contact_model='hard e=0 mirror; no blade/self hard contact reached only if full motion envelope proves it; IPC forces removed, never superposed',
          initial_ipc_barrier_energy_n_mm=2*float(p.barrier(p.collisions(x),p.cmesh,x[p.surface])),
          original_contact_force_norm_n=float(np.sqrt(2)*np.linalg.norm((fields['normal']+fields['friction'])[:2*p.n])),events=events,
          new_blade_impulse_n_s=[0.,0.],new_friction_work_n_mm=0.,new_wall_work_n_mm=0.,mirror_impact_loss_n_mm=loss,
          frozen_noncontact_work_n_mm=frozen_work,kinetic_delta_n_mm=float(p.mass@np.sum(endv[:p.n]**2-v[:p.n]**2,axis=1)),
          full_contact_coupled_reference_qualified=False,history_committed=False,blade_entered=False)
        report['model_conversion_work_n_mm']=-report['initial_ipc_barrier_energy_n_mm']
        report['barrier_storage_delta_on_conversion_n_mm']=-report['initial_ipc_barrier_energy_n_mm']
        report['model_conversion_scope']='Explicit external removal to define a separate hard-model baseline, not energy released by rubber; no position projection or continuous physical handoff claimed'
        delta=p.mass[:,None]*(endv[:p.n]-v[:p.n]);residual=delta-h*force-impulse
        report['frozen_equivalent_force_residual_n']=float(np.sqrt(2)*np.linalg.norm(residual)/h)
        report['frozen_energy_residual_n_mm']=frozen_work-report['kinetic_delta_n_mm']-loss
        if geometry_safe:
            p.previous=x.copy();p.previous_velocity=v.copy();p.previous_material_metric=s['material_metric'].copy();p.previous_history=s['history'].copy();p.material_dt=h;p.set_primary_increment(y,y-x)
            r,_,dofs,_,_=p.material_bulk((y[:p.n]-p.mesh.reference_xy).ravel(),with_tangent=False)
            bulk=np.zeros(p.total);np.add.at(bulk,dofs.ravel(),r.ravel());coh=np.zeros(p.total);coh[2*p.seam]=p.face_force(p.cohesive_algorithmic(y))[0]
            nonlinear=delta+h*(bulk+coh).reshape(-1,2)[:p.n]-impulse
            free=p.base_free.copy();closed=np.flatnonzero((y[:p.n,0]<=centre+1e-12)&(endv[:p.n,0]<=1e-12))
            free=np.setdiff1d(free,2*np.union1d(closed,list(tied)))
            norm=float(np.sqrt(2)*np.linalg.norm(nonlinear.ravel()[free])/h)
            report['restored_nonlinear_bulk_cohesive_force_residual_n']=norm
            report['nonlinear_force_gate_passed']=norm<=1e-4
            report['maxwell_recovery']='not advanced: coupled nonlinear force must pass first'
            report['stop_reason']='nonlinear_force_gate' if norm>1e-4 else 'conditional-only: hard-contact activation coupling and Maxwell still unvalidated'
        else:report['stop_reason']='full_geometry_certificate_failed; no isolated contact trajectory accepted'
        a.out.mkdir(parents=True);np.savez_compressed(a.out/'conditional_trial.npz',parent_positions=x,parent_velocity=v,positions=y,velocity=endv,mass=p.mass,noncontact_force=force,mirror_constraint_impulse=impulse,displacement_bound=motion,bulk_positive_j_bound=dF_bound,parent_material_metric=s['material_metric'],parent_history=s['history'])
        report.update(elapsed_s=time.monotonic()-start,source_unchanged=route.sources()==raw['source_hashes'])
        (a.out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report.get(k) for k in ('parent_depth_mm','physical_dt_s','initial_native_contact_gap_mm','motion_envelope_mm','geometry_safe','frozen_equivalent_force_residual_n','restored_nonlinear_bulk_cohesive_force_residual_n','stop_reason')}))
if __name__=='__main__':main()
