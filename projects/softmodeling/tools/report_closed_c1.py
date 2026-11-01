"""Bound accepted-prefix C1 geometry, force/constraint and growth evidence.

No interpolation, force fitting, or inference of true closure from solver stops.
All force components come from impulses independently verified by the audit.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from .run_coupled_growth import problem,sources
from .report_coupled_growth import face_contacts,connected_faces
from .frozen_scene import digest
from .face_witness import native
from prototype.zero_gap_cut import extrinsic_response


@runtime_entry
def run(case,audit_file,out,seconds=500,*,runtime=None):
    case=Path(case);out=Path(out);start=time.monotonic();q=json.loads((case/'result.json').read_text());a=json.loads(Path(audit_file).read_text())
    if not a['passed'] or a['result_sha256']!=digest(case/'result.json') or len(a['states'])!=len(q['steps']):raise ValueError('Bound complete accepted-prefix audit required')
    if q['config']['initial_slit_halfwidth_mm']!=0 or q['source_hashes']!=sources():raise ValueError('Current closed C1 equations required')
    out.mkdir(exist_ok=False,parents=True);p=problem(q['config']);threshold=2*p.toughness/p.strength;top=int(p.seam[-1]);tip=int(p.seam[np.flatnonzero(p.precracked_faces)[0]])
    summary=dict(scope='Closed prescribed 0.5 mm C1 exploration; not intact C2 or time/grid/experimental qualification',
        result_sha256=digest(case/'result.json'),audit_sha256=digest(Path(audit_file)),source_hashes=q['source_hashes'],reporter_sha256=digest(Path(__file__)),config=q['config'],thread_runtime=runtime,
        initial=dict(full_mass_n_s2_mm=float(2*p.mass.sum()),precrack_area_mm2=float(p.rule['face_area_mm2'][p.precracked_faces].sum()),precrack_gc_excluded_n_mm=p.initial_fracture_energy_n_mm,
            top_node=top,tip_node=tip,top_released=bool(p.released[-1]),top_full_opening_mm=0.,maximum_initial_speed_mm_s=float(np.linalg.norm(p.velocity,axis=1).max())),
        status=q['status'],failure_kind=q.get('failure_kind'),error=q.get('error'),accepted_states=len(q['steps']),first_blade_contact=None,first_internal_contact=None,first_internal_outside_endpoint_neighborhood=None,first_new_failed_connected_contact=None,first_new_fully_failed=None,
        first_activation=a.get('first_activation_bracket'),first_new_damage=a.get('first_new_damage'),time_converged=False,mesh_converged=False)
    rows=[];points=[];contacts=[];selected={0,len(q['steps'])-1};last=None
    for i,(row,check) in enumerate(zip(q['steps'],a['states'])):
        if time.monotonic()-start>seconds:raise TimeoutError('C1 report budget')
        path=case/row['state_file']
        if digest(path)!=check['state_sha256'] or row['state_sha256']!=check['state_sha256']:raise ValueError('State binding changed')
        with np.load(path,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        h=float(s['material_dt_s']);x=s['positions'];p.active_faces=s['active_faces'].copy();p.released=s['released'].copy();p.rebuild_contact()
        failed=p.active_faces&np.all(s['history']>=threshold,axis=1);connected=connected_faces(failed);cs=face_contacts(p,x,failed)
        tick=dict(index=i,state_file=row['state_file'],state_sha256=row['state_sha256'],time_s=check['time_s'],depth_mm=check['depth_mm'])
        internal=[c for c in cs if c['strict_face_interior']];away=[c for c in internal if c['outside_endpoint_neighborhood']]
        new_contacts=[c for c in away if c['classification']=='new_fully_failed' and c['connected']]
        for key,candidates in [('first_blade_contact',cs),('first_internal_contact',internal),('first_internal_outside_endpoint_neighborhood',away),('first_new_failed_connected_contact',new_contacts)]:
            if candidates and summary[key] is None:
                summary[key]=dict(**tick,contacts=candidates,previous_time_s=0. if i==0 else a['states'][i-1]['time_s'],previous_depth_mm=p.start if i==0 else a['states'][i-1]['depth_mm'])
                selected.update([max(0,i-1),i])
        if row.get('face_events'):selected.update([max(0,i-1),i])
        if np.any(failed&~p.precracked_faces) and summary['first_new_fully_failed'] is None:
            summary['first_new_fully_failed']=dict(**tick,faces=np.flatnonzero(failed&~p.precracked_faces).tolist());selected.update([max(0,i-1),i])
        contacts.append(dict(**tick,contacts=cs,mirror_active_nodes=np.flatnonzero(s['active_contact']).tolist(),active_faces=np.flatnonzero(p.active_faces).tolist(),fully_failed_faces=np.flatnonzero(failed).tolist()))
        opening=p.opening_at_points(x);cq=extrinsic_response(opening,s['history'],p.strength,p.toughness)
        for face in np.flatnonzero(p.active_faces&~p.precracked_faces):
            for iq in range(len(p.rule['shape'])):
                points.append(dict(**tick,face=int(face),point=iq,reference_y_mm=float(p.rule['shape'][iq]@p.rest[p.seam[face:face+2],1]),reference_area_mm2=float(p.rule['face_area_mm2'][face]),opening_mm=float(opening[face,iq]),history_mm=float(s['history'][face,iq]),delta_f_mm=threshold,traction_mpa=float(cq['traction'][face,iq]),damage_fraction=float(min(s['history'][face,iq]/threshold,1.)),dissipated_n_mm=float(p.rule['weights_mm2'][face,iq]*cq['dissipated'][face,iq]),all_points_failed=bool(failed[face]),connected=bool(connected[face])))
        force={name:s['impulse_'+name]/h for name in ('bulk','normal','friction','cohesive')}
        inertia=p.mass_dof.reshape(-1,2)*(s['velocity']-s['previous_velocity'])/h
        residual=inertia-sum(force.values());mirror=s['active_contact'].copy();mirror[p.seam[~p.released]]=False
        r=dict(**tick,dt_s=h,fz_n=check['independent_fz_n'],normal_fz_n=check['blade_normal_impulse_n_s']/h,friction_fz_n=check['blade_friction_impulse_n_s']/h,
            top_released=bool(s['released'][-1]),top_mirror_active=bool(mirror[top]),top_full_opening_mm=float(2*(x[top,0]-15)),
            mirror_active_released_nodes=int(mirror.sum()),mirror_reaction_half_sum_n=float(residual[:p.n,0][mirror].sum()),
            new_active_faces=int((s['active_faces']&~p.precracked_faces).sum()),new_fully_failed_faces=int((failed&~p.precracked_faces).sum()),
            new_connected_length_mm=float(np.sum(np.diff(p.rest[p.seam,1])[connected&~p.precracked_faces])),new_connected_area_mm2=float(p.rule['face_area_mm2'][connected&~p.precracked_faces].sum()),
            maximum_speed_mm_s=float(np.linalg.norm(s['velocity'][:p.n],axis=1).max()),kinetic_energy_n_mm=float(p.mass@np.sum(s['velocity'][:p.n]**2,axis=1)),
            blade_tip_minus_current_top_y_mm=float(x[p.n:,1].min()-x[top,1]),producer_entry_flag=row['blade_entered_surface_channel'],producer_new_face_entry_flag=row['blade_contacted_strict_new_face'],contact_count=len(cs),strict_internal_count=len(internal),outside_endpoint_neighborhood_count=len(away))
        for node,label in ((top,'top'),(tip,'initial_tip')):
            for name,f in {**force,'inertia':inertia,'constraint_residual':residual}.items():
                for dof,axis in enumerate('xy'):r[label+'_'+name+'_'+axis+'_half_n']=float(f[node,dof])
        r.update({k:v for k,v in check.items() if k.endswith('_n_mm') and np.isscalar(v)})
        rows.append(r);last=s
    summary.update(terminal=rows[-1] if rows else None,selected_state_indices=sorted(selected),new_fracture_n_mm=sum(s['independent_fracture_n_mm'] for s in a['states']),absolute_numerical_n_mm=a['absolute_numerical_work_n_mm'],
        force_convention='Physical half-domain nodal force=stored impulse/h, independently verified; mirror reaction=inertia minus physical forces. Blade reaction and energies use full mirror domain.',
        damage_fraction_definition='min(kappa/delta_f,1), envelope degradation progress for this extrinsic secant law; not stiffness-normalized damage',
        terminal_new_face_points=[v for v in points if v['index']==len(rows)-1],
        endpoint_policy='Strict EV projection interior reported separately from projection farther than existing dhat from both edge endpoints; original top/VV contacts remain explicit.')
    (out/'summary.json').write_text(json.dumps(native(summary),indent=2));(out/'contacts.json').write_text(json.dumps(native(contacts),indent=2))
    if points:
        with (out/'face_points.csv').open('w') as f:w=csv.DictWriter(f,list(points[0]));w.writeheader();w.writerows(points)
    if rows:
        with (out/'raw.csv').open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8));d=[r['depth_mm'] for r in rows]
    for k in ('fz_n','normal_fz_n','friction_fz_n'):axes[0,0].plot(d,[r[k] for r in rows],label=k)
    for k in ('top_bulk_x_half_n','top_normal_x_half_n','top_friction_x_half_n','top_cohesive_x_half_n','top_constraint_residual_x_half_n'):axes[0,1].plot(d,[r[k] for r in rows],label=k)
    for ax,key in [(axes[0,2],'top_full_opening_mm'),(axes[1,0],'maximum_speed_mm_s'),(axes[1,1],'new_connected_length_mm')]:ax.plot(d,[r[key] for r in rows],label=key)
    for key in ('independent_fracture_n_mm','absolute_numerical_including_equation_work_n_mm','independent_friction_n_mm','independent_maxwell_n_mm'):
        if key in rows[0]:axes[1,2].plot(d,np.cumsum([r[key] for r in rows]),label=key)
    for ax in axes.flat:ax.set_xlabel('absolute blade depth (mm)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'closed_c1.png',dpi=140);plt.close(fig)
    for i in selected:
        with np.load(case/q['steps'][i]['state_file'],allow_pickle=False) as z:x=z['positions'];active=z['active_faces'];failed=active&np.all(z['history']>=threshold,axis=1)
        fig,ax=plt.subplots(figsize=(6,7))
        for face in range(len(active)):
            nodes=p.seam[face:face+2];color='tab:blue' if p.precracked_faces[face] else 'tab:red' if failed[face] else 'tab:orange' if active[face] else '0.65'
            for sign in (1,-1):ax.plot(15+sign*(x[nodes,0]-15),x[nodes,1],color=color)
        ax.scatter(x[p.n:,0],x[p.n:,1],s=3,c='black');ax.set(xlim=(14.6,15.4),ylim=(4.1,5.2),aspect='equal',title=f"Closed C1 {q['steps'][i]['depth_mm']:.9g} mm",xlabel='x (mm)',ylabel='y (mm)')
        fig.tight_layout();fig.savefig(out/f'geometry_{i:05d}.png',dpi=140);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Closed C1 exploratory accepted prefix</h1><img src="closed_c1.png" width="1100"><p><a href="raw.csv">Raw forces/constraints/energy</a> <a href="contacts.json">Independent contact identity</a></p><pre>'+json.dumps(native(summary),indent=2)+'</pre>')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('audit');p.add_argument('out');p.add_argument('--seconds',type=float,default=500);add_thread_arguments(p);a=p.parse_args()
    q=run(a.case,a.audit,a.out,a.seconds,**thread_arguments(a));print(json.dumps({k:q[k] for k in ('status','failure_kind','accepted_states','terminal')}))
