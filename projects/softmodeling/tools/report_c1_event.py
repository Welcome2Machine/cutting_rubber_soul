"""Bound closed-C1 event evidence in absolute time; no smoothing or resampling."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from .run_coupled_growth import problem,sources
from .report_coupled_growth import face_contacts,connected_faces
from .frozen_scene import digest
from .face_witness import native
from prototype.zero_gap_cut import extrinsic_response


def event_energy_scale(initial_kinetic,ledger):
    """One endpoint reservoir scale, never release plus its dissipation twice."""
    terms=dict(initial_kinetic_n_mm=float(initial_kinetic),positive_external_work_n_mm=float(ledger['positive_external_work_n_mm']))
    for name in ('bulk','cohesive','barrier'):
        terms[name+'_net_recoverable_release_n_mm']=max(0.,-float(ledger['independent_'+name+'_delta_n_mm']))
    if min(terms.values())<0 or not all(np.isfinite(v) for v in terms.values()):raise ValueError('Invalid event scale')
    total=sum(terms.values())
    if total<=0:raise ValueError('No positive resolved event energy scale')
    return dict(definition='Initial kinetic + positive external work + positive net recoverable bulk/cohesive/IPC release; dissipations excluded',components=terms,value_n_mm=total,strict_error_bound=False)



def nearest_new_face(p,x,failed):
    edges=p.surface[np.asarray(p.cmesh.edges,dtype=int)]
    blade=edges[np.all(edges>=p.n,axis=1)];records=[]
    if not len(blade):return []
    def distances(points,a,b):
        d=b-a;lam=np.sum((points-a)*d,axis=-1)/np.sum(d*d,axis=-1)
        return np.linalg.norm(points-(a+np.clip(lam,0,1)[...,None]*d),axis=-1)
    for f in np.flatnonzero(connected_faces(failed)&~p.precracked_faces):
        a,b=x[p.seam[f:f+2]];u,v=x[blade[:,0]],x[blade[:,1]]
        # The independent full CCD/J audit has already excluded intersections.
        # For disjoint 2-D segments the closest pair includes an endpoint.
        ds=np.minimum.reduce([distances(a,u,v),distances(b,u,v),distances(u,a,b),distances(v,a,b)])
        i=int(np.argmin(ds));records.append(dict(face=int(f),minimum_blade_distance_mm=float(ds[i]),blade_edge=blade[i].tolist(),current_face_endpoints_mm=[a.tolist(),b.tolist()],reference_face_y_mm=p.rest[p.seam[f:f+2],1].tolist(),ipc_activation_distance_mm=float(p.dhat),scope='geometric distance of CCD-verified disjoint segments; not a contact-force or entry certificate'))
    return records


def original_node_contacts(p,contacts,precrack_mm):
    """A uses the independent auditor's nested contact, only a single body node."""
    out=[]
    for item in contacts:
        nodes=item['contact']['body_nodes']
        if len(nodes)!=1:continue
        y=p.rest[nodes[0],1]
        if p.config.height_mm-precrack_mm+1e-9<y<p.config.height_mm-1e-9:out.append(item)
    return out


def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def bracket(previous,now,**extra):return dict(previous=previous,first=now,**extra)


@runtime_entry
def run(case,audit_file,out,start_index=0,seconds=500,*,runtime=None):
    case=Path(case);out=Path(out);q=json.loads((case/'result.json').read_text());a=json.loads(Path(audit_file).read_text());start=time.monotonic()
    if not a['passed'] or a['result_sha256']!=digest(case/'result.json') or len(q['steps'])!=len(a['states']):raise ValueError('Bound accepted-state audit required')
    if q['config']['initial_slit_halfwidth_mm']!=0:raise ValueError('Closed C1 geometry required')
    current=sources()
    if any(q['source_hashes'].get(k)!=v for k,v in current.items()):raise ValueError('Physical source identity mismatch')
    if not 0<=start_index<len(q['steps']):raise ValueError('Nonempty event suffix required')
    parent_file=case/(q['steps'][start_index-1]['state_file'] if start_index else 'parent_state.npz')
    parent=read(parent_file);p=problem(q['config']);top=int(p.seam[-1]);threshold=2*p.toughness/p.strength
    t0=(float(parent['depth_mm'])-p.start)/p.loading_rate;previous=dict(time_s=t0,depth_mm=float(parent['depth_mm']),state_file=parent_file.name,state_sha256=digest(parent_file))
    events={};rows=[];contacts=[];points=[];selected=set();active_old=parent['active_faces'];failed_old=active_old&np.all(parent['history']>=threshold,axis=1)
    parent_top_active=bool(parent['active_contact'][top]);K0=float(p.mass@np.sum(parent['velocity'][:p.n]**2,axis=1));energy={}
    for local,(row,check) in enumerate(zip(q['steps'][start_index:],a['states'][start_index:])):
        if time.monotonic()-start>seconds:raise TimeoutError('Event report budget')
        path=case/row['state_file']
        if row['state_sha256']!=check['state_sha256'] or digest(path)!=row['state_sha256']:raise ValueError('Event state binding mismatch')
        s=read(path);x=s['positions'];p.active_faces=s['active_faces'].copy();p.released=s['released'].copy();p.rebuild_contact()
        now=dict(time_s=check['time_s'],depth_mm=check['depth_mm'],state_file=row['state_file'],state_sha256=row['state_sha256'])
        failed=p.active_faces&np.all(s['history']>=threshold,axis=1);connected=connected_faces(failed);cs=face_contacts(p,x,failed)
        A=original_node_contacts(p,check['strict_contacts'],q['config']['precrack_mm'])
        B=[c for c in cs if c['strict_face_interior']]
        C=[c for c in B if c['classification']=='new_fully_failed' and c['connected']]
        Caway=[c for c in C if c['outside_endpoint_neighborhood']]
        newly_failed=np.flatnonzero(failed&~failed_old&~p.precracked_faces).tolist();new_active=np.flatnonzero(p.active_faces&~active_old).tolist()
        tick=dict(**now,offset_us=(check['time_s']-t0)*1e6)
        for key,yes,extra in [('top_unlock',parent_top_active and not s['active_contact'][top],{}),('A_original_interior_node_contact',bool(A),dict(contacts=A)),('B_P1_interior_projection',bool(B),dict(contacts=B)),('C_new_failed_connected_projection',bool(C),dict(contacts=C)),('C_away_from_endpoints',bool(Caway),dict(contacts=Caway))]:
            if yes and key not in events:events[key]=bracket(previous,now,**extra);selected.update((max(0,local-1),local))
        if newly_failed:events.setdefault('full_failures',[]).append(bracket(previous,now,faces=newly_failed));selected.update((max(0,local-1),local))
        if new_active:events.setdefault('activations',[]).append(bracket(previous,now,faces=new_active));selected.update((max(0,local-1),local))
        cq=extrinsic_response(p.opening_at_points(x),s['history'],p.strength,p.toughness)
        for face in np.flatnonzero(p.active_faces&~p.precracked_faces):
            for iq in range(p.rule['shape'].shape[0]):
                points.append(dict(**tick,face=int(face),point=iq,reference_y_mm=float(p.rule['shape'][iq]@p.rest[p.seam[face:face+2],1]),reference_area_mm2=float(p.rule['face_area_mm2'][face]),opening_mm=float(p.opening_at_points(x)[face,iq]),history_mm=float(s['history'][face,iq]),delta_f_mm=threshold,traction_mpa=float(cq['traction'][face,iq]),damage_progress=float(min(s['history'][face,iq]/threshold,1)),dissipated_n_mm=float(p.rule['weights_mm2'][face,iq]*cq['dissipated'][face,iq]),all_points_failed=bool(failed[face]),connected=bool(connected[face])))
        rr=dict(**tick,dt_s=check['dt_s'],fz_n=check['independent_fz_n'],normal_fz_n=check['blade_normal_impulse_n_s']/check['dt_s'],friction_fz_n=check['blade_friction_impulse_n_s']/check['dt_s'],maximum_speed_mm_s=float(np.linalg.norm(s['velocity'][:p.n],axis=1).max()),mass_velocity_norm_sqrt_nmm=float(np.sqrt(p.mass@np.sum(s['velocity'][:p.n]**2,axis=1))),kinetic_energy_n_mm=float(p.mass@np.sum(s['velocity'][:p.n]**2,axis=1)),top_full_opening_mm=float(2*(x[top,0]-p.config.drop_s_mm)),top_mirror_active=bool(s['active_contact'][top]),A_contact=bool(A),B_contact=bool(B),C_contact=bool(C),C_away_contact=bool(Caway),new_active_faces=int((p.active_faces&~p.precracked_faces).sum()),new_failed_faces=int((failed&~p.precracked_faces).sum()),new_connected_length_mm=float(np.diff(p.rest[p.seam,1])[connected&~p.precracked_faces].sum()),new_connected_area_mm2=float(p.rule['face_area_mm2'][connected&~p.precracked_faces].sum()),force_residual_n=check['residual_n'])
        rr.update({k:float(v) for k,v in check.items() if k.endswith('_n_mm') and np.isscalar(v)})
        budget=check.get('nonlinear_error_budget',{});rr.update(nonlinear_budget_stop=not row.get('nonlinear_accuracy',{}).get('preferred_goal_met',True),residual_impulse_n_s=check['impulse_residual_n_s'],residual_impulse_mass_norm_sqrt_nmm=budget.get('residual_impulse_mass_norm_sqrt_nmm',0.))
        rows.append(rr);contacts.append(dict(**tick,A=A,B=B,C=C,all_contacts=cs,mirror_nodes=np.flatnonzero(s['active_contact']).tolist(),failed_faces=np.flatnonzero(failed&~p.precracked_faces).tolist(),connected_faces=np.flatnonzero(connected&~p.precracked_faces).tolist()))
        active_old=p.active_faces.copy();failed_old=failed;previous=now
    energy={k:sum(r[k] for r in rows) for k in rows[0] if k.endswith('_n_mm') and k!='kinetic_energy_n_mm'}
    energy['positive_external_work_n_mm']=sum(max(0,r['independent_external_work_n_mm']) for r in rows)
    scale=event_energy_scale(K0,energy);entry=events.get('A_original_interior_node_contact');selected.update((0,len(rows)-1));peak=int(np.argmax([z['maximum_speed_mm_s'] for z in rows]));fpeak=int(np.argmax([z['fz_n'] for z in rows]));fmin=fpeak+int(np.argmin([z['fz_n'] for z in rows[fpeak:]]));selected.update((peak,fpeak,fmin))
    summary=dict(schema='softcut-closed-c1-event/1',result_sha256=digest(case/'result.json'),audit_sha256=digest(Path(audit_file)),reporter_sha256=digest(Path(__file__)),source_hashes=q['source_hashes'],config=q['config'],thread_runtime=runtime,scope='Closed prescribed C1 same-source event evidence; no automatic temporal, grid or experimental qualification',parent=previous if not rows else dict(file=str(parent_file.resolve()),sha256=digest(parent_file),time_s=t0,depth_mm=float(parent['depth_mm'])),start_index=start_index,window_start_s=t0,window_end_s=rows[-1]['time_s'],duration_s=rows[-1]['time_s']-t0,accepted_states=len(rows),production_status=q['status'],terminal=rows[-1],terminal_new_face_distances=nearest_new_face(p,x,failed),events=events,post_entry_duration_s=None if entry is None else rows[-1]['time_s']-entry['first']['time_s'],initial_precrack_fracture_energy_excluded_n_mm=p.initial_fracture_energy_n_mm,energy_n_mm=energy,event_energy_reference=scale,absolute_numerical_over_event_scale=energy['absolute_numerical_including_equation_work_n_mm']/scale['value_n_mm'],old_absolute_numerical_over_blade_work=energy['absolute_numerical_including_equation_work_n_mm']/max(abs(energy['independent_external_work_n_mm']),1e-300),absolute_equation_work_n_mm=sum(abs(r['algebraic_residual_work_n_mm']) for r in rows),residual_impulse_n_s=sum(r['residual_impulse_n_s'] for r in rows),residual_impulse_mass_norm_sum_sqrt_nmm=sum(r['residual_impulse_mass_norm_sqrt_nmm'] for r in rows),nonlinear_budget_stops=sum(r['nonlinear_budget_stop'] for r in rows),velocity_peak=rows[peak],force_drop=dict(peak=rows[fpeak],minimum=rows[fmin],total_n=rows[fpeak]['fz_n']-rows[fmin]['fz_n'],normal_n=rows[fpeak]['normal_fz_n']-rows[fmin]['normal_fz_n'],friction_n=rows[fpeak]['friction_fz_n']-rows[fmin]['friction_fz_n']),selected_local_indices=sorted(selected),event_complete=False,time_converged=False,mesh_converged=False)
    out.mkdir(parents=True,exist_ok=False)
    for name,table in [('raw.csv',rows),('face_points.csv',points)]:
        if table:
            with (out/name).open('w') as f:w=csv.DictWriter(f,list(table[0]));w.writeheader();w.writerows(table)
    for name,data in [('summary.json',summary),('contacts.json',contacts)]: (out/name).write_text(json.dumps(native(data),indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(14,11));t=np.array([z['time_s'] for z in rows])
    for key in ('fz_n','normal_fz_n','friction_fz_n'):axes[0,0].plot(t,[z[key] for z in rows],label=key)
    for key in ('maximum_speed_mm_s','mass_velocity_norm_sqrt_nmm'):axes[0,1].plot(t,[z[key] for z in rows],label=key)
    axes[1,0].plot(t,[z['top_full_opening_mm'] for z in rows],label='top full opening (mm)')
    for key in ('kinetic_energy_n_mm','top_mirror_active','A_contact','B_contact','C_contact'):axes[1,1].plot(t,[z[key] for z in rows],label=key)
    for key in ('independent_external_work_n_mm','independent_bulk_delta_n_mm','independent_cohesive_delta_n_mm','independent_barrier_delta_n_mm','independent_fracture_n_mm','independent_friction_n_mm','independent_maxwell_n_mm'):axes[2,0].plot(t,np.cumsum([z[key] for z in rows]),label=key)
    for key in ('absolute_numerical_including_equation_work_n_mm','net_physical_remainder_n_mm','numerical_inertia_n_mm','algebraic_residual_work_n_mm'):axes[2,1].plot(t,np.cumsum([z[key] for z in rows]),label=key)
    for ax in axes.flat:
        ax.set_xlabel('absolute physical time (s)');ax.ticklabel_format(useOffset=False);ax.legend(fontsize=6)
        ax.secondary_xaxis('top',functions=(lambda x:(x-t0)*1e6,lambda u:t0+u*1e-6)).set_xlabel('offset from parent (microseconds)')
    fig.tight_layout();fig.savefig(out/'event.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots();ax.plot([z['depth_mm'] for z in rows],[z['fz_n'] for z in rows]);ax.set(xlabel='absolute depth (mm)',ylabel='raw Fz (N)');fig.tight_layout();fig.savefig(out/'fz_depth.png',dpi=140);plt.close(fig)
    for local in [-1,*sorted(selected)]:
        s=parent if local==-1 else read(case/rows[local]['state_file']);x=s['positions'];failed=s['active_faces']&np.all(s['history']>=threshold,axis=1);fig,ax=plt.subplots(figsize=(6,6))
        for f in range(len(failed)):
            nodes=p.seam[f:f+2];color='tab:blue' if p.precracked_faces[f] else 'tab:red' if failed[f] else 'tab:orange' if s['active_faces'][f] else '0.6'
            for sign in (1,-1):ax.plot(p.config.drop_s_mm+sign*(x[nodes,0]-p.config.drop_s_mm),x[nodes,1],color=color)
        ax.plot(x[p.n:,0],x[p.n:,1],'k.',ms=2);ax.set(xlim=(14.6,15.4),ylim=(4.0,5.2),aspect='equal',title=f"t={t0 if local==-1 else rows[local]['time_s']:.12f}s");fig.tight_layout();fig.savefig(out/('geometry_parent.png' if local==-1 else f'geometry_{local:05d}.png'),dpi=140);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Closed C1 event evidence</h1><p>Blue: initial precrack; orange: new active face; red: new fully failed face.</p><img width="1100" src="event.png"><p><a href="raw.csv">Raw force and energy</a> <a href="face_points.csv">All cohesive points</a> <a href="contacts.json">Contact identities</a></p><pre>'+json.dumps(native(summary),indent=2)+'</pre>')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('audit');p.add_argument('out');p.add_argument('--start-index',type=int,default=0);p.add_argument('--seconds',type=float,default=500);add_thread_arguments(p);a=p.parse_args();q=run(a.case,a.audit,a.out,a.start_index,a.seconds,**thread_arguments(a));print(json.dumps({k:q[k] for k in ('window_start_s','window_end_s','post_entry_duration_s','absolute_numerical_over_event_scale')}))
