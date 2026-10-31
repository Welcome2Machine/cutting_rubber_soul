"""Audited face-point evolution and exact contact identity on a growth path.

No resampling or failure extrapolation. Fully failed means ALL quadrature
histories reach 2 Gc/Tmax; contact at a seam vertex is not face-interior entry.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from .run_coupled_growth import problem
from .audit_zero_gap_cut import blade_contacts
from .frozen_scene import digest
from .face_witness import native
from prototype.zero_gap_cut import extrinsic_response


def connected_faces(failed):
    connected=np.zeros_like(failed,dtype=bool)
    for i in range(len(failed)-1,-1,-1):
        if not failed[i]:break
        connected[i]=True
    return connected


def face_contacts(p,x,failed):
    """Strict interior requires an active body EDGE and interior projection.

    IPC barrier support is a regularized contact, not zero-gap hard impact.
    Blade-edge/body-vertex and VV cases are explicitly endpoint identities.
    """
    pairs={tuple(sorted(map(int,p.seam[i:i+2]))):i for i in range(len(p.active_faces))}
    connected=connected_faces(failed);records=[]
    for c in blade_contacts(p,x):
        body=c['body_nodes'];blade=c['blade_nodes'];face=pairs.get(tuple(sorted(body)))
        r=dict(c,face_index=face,strict_face_interior=False,classification='vertex_or_nonseam_contact')
        if face is not None and len(body)==2 and len(blade)==1:
            a,b=x[body];v=x[blade[0]];d=b-a;lam=float((v-a)@d/(d@d))
            gap=float(np.linalg.norm(v-(a+np.clip(lam,0,1)*d)))
            strict=1e-8<lam<1-1e-8
            r.update(edge_coordinate=lam,gap_mm=gap,strict_face_interior=strict,
                prescribed_face=bool(p.precracked_faces[face]),fully_failed=bool(failed[face]),connected=bool(connected[face]),
                classification=('initial_precrack' if p.precracked_faces[face] else 'new_fully_failed' if failed[face] else 'new_softening'))
        records.append(r)
    return records


@runtime_entry
def run(case,audit,out,*,runtime=None):
    case=Path(case);out=Path(out);out.mkdir(exist_ok=False,parents=True)
    raw=json.loads((case/'result.json').read_text());a=json.loads(Path(audit).read_text())
    if not a['passed'] or a['result_sha256']!=digest(case/'result.json') or len(a['states'])!=len(raw['steps']):
        raise ValueError('Complete bound accepted-prefix audit required')
    p=problem(raw['config']);delta_f=2*p.toughness/p.strength;initial=p.precracked_faces.copy()
    rows=[];points=[];events=[];contacts=[];previous_failed=initial.copy();selected={0,len(raw['steps'])-1}
    first_failed=first_entry=None;last=None
    for i,(row,verified) in enumerate(zip(raw['steps'],a['states'])):
        file=case/row['state_file']
        if digest(file)!=row['state_sha256'] or verified['state_sha256']!=row['state_sha256']:raise ValueError('Unbound state')
        with np.load(file,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        p.active_faces=s['active_faces'].copy();p.released=s['released'].copy();p.rebuild_contact()
        opening=p.opening_at_points(s['positions']);q=extrinsic_response(opening,s['history'],p.strength,p.toughness)
        np.testing.assert_array_equal(q['history'],s['history'])
        failed=p.active_faces&np.all(s['history']>=delta_f,axis=1);connected=connected_faces(failed)
        newfailed=failed&~initial;newconnected=connected&~initial;cs=face_contacts(p,s['positions'],failed)
        entering=[c for c in cs if c['strict_face_interior'] and c.get('classification')=='new_fully_failed' and c['connected']]
        tick=dict(index=i,state_file=row['state_file'],state_sha256=row['state_sha256'],time_s=row['time_s'],depth_mm=row['depth_mm'])
        contacts.append(dict(**tick,contacts=cs))
        just=failed&~previous_failed
        if just.any():
            event=dict(**tick,fully_failed_faces=np.flatnonzero(just).tolist(),previous_index=i-1,
                previous_depth_mm=p.start if i==0 else raw['steps'][i-1]['depth_mm'],previous_time_s=0 if i==0 else raw['steps'][i-1]['time_s'])
            events.append(event);selected.update([max(0,i-1),i,min(i+1,len(raw['steps'])-1)])
            if first_failed is None:first_failed=event
        if entering and first_entry is None:
            first_entry=dict(**tick,contacts=entering,previous_index=i-1);selected.update([max(0,i-1),i,min(i+1,len(raw['steps'])-1)])
        for event in verified['activation_chain']['events']:
            events.append(dict(**tick,activation=event));selected.update([max(0,i-1),i])
        for stage in [.385,.55,.65,1.]:
            if row['depth_mm']>=stage and (i==0 or raw['steps'][i-1]['depth_mm']<stage):selected.add(i)
        r=dict(**tick,fz_n=verified['independent_fz_n'],normal_fz_n=verified['blade_normal_impulse_n_s']/verified['dt_s'],
            friction_fz_n=verified['blade_friction_impulse_n_s']/verified['dt_s'],
            new_active_faces=int((p.active_faces&~initial).sum()),new_fully_failed_faces=int(newfailed.sum()),
            new_connected_length_mm=float(np.sum(np.diff(p.rest[p.seam,1])[newconnected])),
            new_connected_area_mm2=float(p.rule['face_area_mm2'][newconnected].sum()),blade_entered_new_failed_connected_face=bool(entering),
            new_fracture_dissipation_n_mm=float(np.sum(p.rule['weights_mm2']*q['dissipated']*(p.active_faces&~initial)[:,None])))
        for key,value in verified.items():
            if key.endswith('_n_mm') and np.isscalar(value):r[key]=value
        rows.append(r)
        for face in np.flatnonzero(p.active_faces&~initial):
            for k in range(opening.shape[1]):
                points.append(dict(**tick,face=int(face),point=k,reference_y_lower_mm=float(p.rest[p.seam[face],1]),
                    reference_y_upper_mm=float(p.rest[p.seam[face+1],1]),reference_area_mm2=float(p.rule['face_area_mm2'][face]),
                    quadrature_area_mm2=float(p.rule['weights_mm2'][face,k]),opening_mm=float(opening[face,k]),
                    maximum_opening_history_mm=float(s['history'][face,k]),failure_threshold_mm=delta_f,
                    history_to_failure_ratio=float(s['history'][face,k]/delta_f),remaining_history_to_failure_mm=float(max(0,delta_f-s['history'][face,k])),
                    envelope_damage_fraction=float(min(1,s['history'][face,k]/delta_f)),traction_mpa=float(q['traction'][face,k]),
                    dissipated_n_mm=float(p.rule['weights_mm2'][face,k]*q['dissipated'][face,k]),face_fully_failed=bool(failed[face]),face_connected=bool(connected[face])))
        previous_failed=failed.copy();last=(s,q,opening)
    lastpoints=[z for z in points if z['index']==len(rows)-1]
    faces=sorted(set(z['face'] for z in lastpoints));closest=None
    if faces:
        closest=max(faces,key=lambda f:min(z['history_to_failure_ratio'] for z in lastpoints if z['face']==f))
    summary=dict(scope='Own-initial-state prescribed wide-notch diagnostic; no closed-C1, mesh or experiment prediction qualification',
        case=str(case.resolve()),result_sha256=digest(case/'result.json'),audit_sha256=digest(Path(audit)),reporter_sha256=digest(Path(__file__)),
        source_hashes=raw['source_hashes'],config_sha256=raw['config_sha256'],status=raw['status'],error=raw.get('error'),failure_kind=raw.get('failure_kind'),
        accepted_states=len(rows),terminal=rows[-1],activation_event_count=sum(bool(e.get('activation')) for e in events),
        first_new_full_failure=first_failed,first_entry_into_new_failed_connected_face=first_entry,
        closest_face_to_all_point_failure=closest,closest_face_points=[z for z in lastpoints if z['face']==closest],
        law=dict(strength_n_mm2=p.strength,gc_n_mm=p.toughness,delta_f_mm=delta_f,definition='delta_f=2 Gc/Tmax; mm; secant irreversible D per unit reference area=Tmax*min(kappa,delta_f)/2',
                 damage_definition='envelope strength-loss fraction min(kappa/delta_f,1); not an independently evolved damage variable',full_failure='all four quadrature histories >= delta_f; activation alone insufficient'),
        preexisting_dissipation_excluded_n_mm=p.initial_fracture_energy_n_mm,selected_state_indices=sorted(selected),thread_runtime=runtime,
        time_converged=False,mesh_converged=False,physical_prediction_validated=False)
    for name,table in [('raw_force_energy.csv',rows),('face_points.csv',points)]:
        if table:
            with (out/name).open('w') as f:w=csv.DictWriter(f,list(table[0]));w.writeheader();w.writerows(table)
    (out/'summary.json').write_text(json.dumps(native(summary),indent=2))
    (out/'events.json').write_text(json.dumps(native(events),indent=2))
    (out/'contact_identities.json').write_text(json.dumps(native(contacts),indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,3,figsize=(14,8));d=[r['depth_mm'] for r in rows]
    for key in ['fz_n','normal_fz_n','friction_fz_n']:ax[0,0].plot(d,[r[key] for r in rows],label=key)
    for face in faces:
        z=[v for v in points if v['face']==face]
        for point in range(p.rule['shape'].shape[0]):
            zz=[v for v in z if v['point']==point];xx=[v['depth_mm'] for v in zz];label=f'f{face} q{point}'
            for axes,key in [(ax[0,1],'opening_mm'),(ax[0,2],'traction_mpa'),(ax[1,0],'envelope_damage_fraction'),(ax[1,1],'dissipated_n_mm')]:axes.plot(xx,[v[key] for v in zz],label=label)
    ax[0,1].axhline(delta_f,color='k',ls='--',label='delta_f')
    ax[1,2].plot(d,[r['new_connected_length_mm'] for r in rows],label='new connected failed length')
    labels=['Fz (N)','opening (mm)','traction (MPa)','envelope loss fraction','point irreversible D (N mm)','new connected length (mm)']
    for axes,label in zip(ax.flat,labels):axes.set(xlabel='absolute blade depth (mm)',ylabel=label)
    ax[0,0].legend();fig.tight_layout();fig.savefig(out/'force_and_faces.png',dpi=150);plt.close(fig)
    for i in sorted(selected):
        row=raw['steps'][i]
        with np.load(case/row['state_file'],allow_pickle=False) as z:x=z['positions'];active=z['active_faces'];history=z['history']
        failed=active&np.all(history>=delta_f,axis=1)
        fig,ax=plt.subplots(figsize=(6,7))
        for face in range(len(active)):
            nodes=p.seam[face:face+2];color='tab:blue' if initial[face] else 'tab:red' if failed[face] else 'tab:orange' if active[face] else '0.6'
            ax.plot(x[nodes,0],x[nodes,1],color=color,lw=2)
            ax.plot(2*p.config.drop_s_mm-x[nodes,0],x[nodes,1],color=color,lw=2)
        ax.scatter(x[p.n:,0],x[p.n:,1],s=2,color='k');ax.set(xlim=(p.config.drop_s_mm-.65,p.config.drop_s_mm+.65),ylim=(p.config.height_mm-1.3,p.config.height_mm+.2),aspect='equal',title=f"{row['depth_mm']:.9f} mm; blue initial / orange softening / red failed",xlabel='x (mm)',ylabel='y (mm)')
        fig.tight_layout();fig.savefig(out/f'geometry_{i:05d}.png',dpi=140);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Audited prescribed-notch face growth</h1><img src="force_and_faces.png" width="1200"><p><a href="raw_force_energy.csv">Raw force/energy</a> <a href="face_points.csv">All new face quadrature histories</a> <a href="contact_identities.json">Independent contact identities</a></p><pre>'+json.dumps(native(summary),indent=2)+'</pre>')
    print(json.dumps(native({k:summary[k] for k in ['accepted_states','terminal','first_new_full_failure','first_entry_into_new_failed_connected_face','closest_face_to_all_point_failure']})))
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('case');ap.add_argument('audit');ap.add_argument('out');add_thread_arguments(ap);args=ap.parse_args()
    run(args.case,args.audit,args.out,**thread_arguments(args))
