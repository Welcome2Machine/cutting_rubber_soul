"""Common absolute-window time comparisons; no smoothing or event alignment."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .frozen_scene import digest
from .face_witness import native
from .coupled_audit_contract import common_inputs,bind_case,registered_offsets,common_offset_indices
from run_support import write_json


def cumulative_impulses(path,rows,*,key_by_state=False):
    """Integrate each case on its own accepted partition, then sample common times."""
    total=None;out={};normal=friction=0.
    for row in rows:
        with np.load(path/row['state_file'],allow_pickle=False) as z:
            n=len(z['mass']);I=z['impulse_normal'][:n]+z['impulse_friction'][:n]
            total=I.copy() if total is None else total+I
        normal+=row['blade_normal_impulse_n_s'];friction+=row['blade_friction_impulse_n_s']
        out[row['state_file'] if key_by_state else round(row['time_s'],12)]=(total.copy(),normal,friction)
    return out


def compare(cases,limits,allowed_variations=('dt_requested_s','time_schedule'),*,entry_required=True):
    identity=common_inputs(cases,allowed_variations)
    result=dict(input_identity=identity,allowed_variations=list(allowed_variations),passed=False,scope='this prescribed-notch conditional window only',pairs=[])
    if any(q['status']!='completed_window' for _,q,_ in cases):
        result['error']='Incomplete common physical window';return result
    fractions=[]
    for path,q,a in cases:
        if not a['passed']:result['error']='Independent audit failed';return result
        fractions.append(a['absolute_numerical_including_equation_work_n_mm']/abs(sum(s['independent_external_work_n_mm'] for s in a['states'])))
    for (path0,q0,a0),(path1,q1,a1) in zip(cases[:-1],cases[1:]):
        offsets0=registered_offsets(q0,a0);offsets1=registered_offsets(q1,a1)
        matches=common_offset_indices(offsets1,offsets0)
        rows0=a0['states'];rows1=[a1['states'][j] for _,j in matches]
        clock_roundoff=max(abs(r1['time_s']-r0['time_s']) for r0,r1 in zip(rows0,rows1))
        f0=np.array([r['independent_fz_n'] for r in rows0]);f1=np.array([r['independent_fz_n'] for r in rows1])
        ferror=float(max(abs(f0-f1))/max(abs(f1)))
        iforce=int(np.argmax(abs(f0-f1)));rf0,rf1=rows0[iforce],rows1[iforce]
        normal_difference=rf1['blade_normal_impulse_n_s']/rf1['dt_s']-rf0['blade_normal_impulse_n_s']/rf0['dt_s']
        friction_difference=rf1['blade_friction_impulse_n_s']/rf1['dt_s']-rf0['blade_friction_impulse_n_s']/rf0['dt_s']
        velocity_errors=[];velocity_scale=[];position_errors=[];impulse_errors=[];impulse_scales=[];blade_impulse_errors=[];blade_impulse_scales=[]
        impulses0=cumulative_impulses(path0,a0['states'],key_by_state=True);impulses1=cumulative_impulses(path1,a1['states'],key_by_state=True)
        for r0,r1 in zip(rows0,rows1):
            with np.load(path0/r0['state_file'],allow_pickle=False) as z:v0=z['velocity'];x0=z['positions'];mass=z['mass'];n=len(mass)
            with np.load(path1/r1['state_file'],allow_pickle=False) as z:v1=z['velocity'];x1=z['positions']
            I0,N0,T0=impulses0[r0['state_file']];I1,N1,T1=impulses1[r1['state_file']]
            impulse_errors.append(float(np.sqrt(np.sum((I1-I0)**2/mass[:,None]))))
            impulse_scales.append(float(np.sqrt(np.sum(I1**2/mass[:,None]))))
            blade_impulse_errors.append(abs((N1+T1)-(N0+T0)));blade_impulse_scales.append(abs(N1+T1))
            velocity_errors.append(float(np.sqrt(mass@np.sum((v1[:n]-v0[:n])**2,axis=1))))
            velocity_scale.append(float(np.sqrt(mass@np.sum(v1[:n]**2,axis=1))))
            position_errors.append(float(np.max(np.linalg.norm(x1[:n]-x0[:n],axis=1))))
        verror=max(velocity_errors)/max(velocity_scale)
        preexisting=bool(a0.get('entry_present_at_parent',False))
        if preexisting!=bool(a1.get('entry_present_at_parent',False)):raise ValueError('Same parent has inconsistent entry identity')
        no_entry=False
        if preexisting:
            lo=hi=q0['initial_depth_mm'];entry_difference=0.
        elif not entry_required and not a0.get('first_entry_bracket') and not a1.get('first_entry_bracket'):
            no_entry=True;lo=hi=q0['initial_depth_mm'];entry_difference=0.
        else:
            if not a0.get('first_entry_bracket') or not a1.get('first_entry_bracket'):
                raise ValueError('Required or inconsistent strict entry event in comparison')
            br0=a0['first_entry_bracket'];br1=a1['first_entry_bracket']
            lo=max(br0['previous_nonentry']['depth_mm'],br1['previous_nonentry']['depth_mm'])
            hi=min(br0['first_entry']['depth_mm'],br1['first_entry']['depth_mm'])
            entry_difference=abs(br0['first_entry']['depth_mm']-br1['first_entry']['depth_mm'])
        result['pairs'].append(dict(coarse_dt_s=q0['dt_requested_s'],fine_dt_s=q1['dt_requested_s'],
            matching_basis='same independently checked registered offset; no state interpolation or event alignment',maximum_reconstructed_clock_difference_s=clock_roundoff,
            maximum_fz_error_over_fine_peak=ferror,maximum_mass_velocity_error_over_fine_peak=verror,
            worst_force_time_s=rf0['time_s'],signed_force_difference_at_worst_n=float(f1[iforce]-f0[iforce]),
            signed_normal_force_difference_at_worst_n=normal_difference,signed_friction_force_difference_at_worst_n=friction_difference,
            maximum_mass_velocity_difference_sqrt_nmm=max(velocity_errors),fine_peak_mass_velocity_sqrt_nmm=max(velocity_scale),
            worst_velocity_time_s=rows0[int(np.argmax(velocity_errors))]['time_s'],
            maximum_cumulative_contact_impulse_mass_error_over_fine_peak=max(impulse_errors)/max(impulse_scales),
            maximum_cumulative_blade_impulse_error_over_fine_peak=max(blade_impulse_errors)/max(blade_impulse_scales),
            terminal_blade_normal_impulse_difference_n_s=N1-N0,terminal_blade_friction_impulse_difference_n_s=T1-T0,
            maximum_position_difference_mm=max(position_errors),entry_brackets_overlap=None if no_entry else lo<=hi+1e-14,
            entry_not_observed_in_either_window=no_entry,
            entry_depth_difference_mm=entry_difference,entry_preexists_parent=preexisting))
    result['absolute_numerical_fraction']=fractions
    fine=result['pairs'][-1]
    result['gates']=dict(force=fine['maximum_fz_error_over_fine_peak']<=limits['force_relative'],
        mass_velocity=fine['maximum_mass_velocity_error_over_fine_peak']<=limits['velocity_mass_relative'],
        event=all((not entry_required and p['entry_not_observed_in_either_window']) or (p['entry_brackets_overlap'] and p['entry_depth_difference_mm']<=limits['event_bracket_mm']) for p in result['pairs']),
        absolute_numerical=fractions[-1]<=limits['absolute_numerical_work_fraction'],
        numerical_decreases=all(a>b for a,b in zip(fractions[:-1],fractions[1:])),
        force_difference_decreases=result['pairs'][-1]['maximum_fz_error_over_fine_peak']<result['pairs'][0]['maximum_fz_error_over_fine_peak'],
        velocity_difference_decreases=result['pairs'][-1]['maximum_mass_velocity_error_over_fine_peak']<result['pairs'][0]['maximum_mass_velocity_error_over_fine_peak'])
    if 'impulse_relative' in limits:
        result['gates']['contact_impulse']=fine['maximum_cumulative_contact_impulse_mass_error_over_fine_peak']<=limits['impulse_relative']
        result['gates']['impulse_difference_decreases']=fine['maximum_cumulative_contact_impulse_mass_error_over_fine_peak']<result['pairs'][0]['maximum_cumulative_contact_impulse_mass_error_over_fine_peak']
    if cases[0][1]['config'].get('pure_precrack') is False:
        audits=[a for _,_,a in cases]
        result['new_fracture_dissipation_n_mm']=[sum(s['independent_fracture_n_mm'] for s in a['states']) for a in audits]
        events=[a.get('first_activation_bracket') for a in audits]
        result['activation_brackets']=events
        result['gates']['activation_replayed']=all(a.get('activation_witness_verified',False) for a in audits)
        result['gates']['positive_new_fracture_dissipation']=all(d>0 for d in result['new_fracture_dissipation_n_mm'])
        result['gates']['activation_brackets_overlap']=all(
            a and b and max(a['previous']['time_s'],b['previous']['time_s'])<=min(a['first']['time_s'],b['first']['time_s'])+1e-12 for a,b in zip(events[:-1],events[1:]))
        d=result['new_fracture_dissipation_n_mm']
        result['fracture_dissipation_relative_differences']=[abs(a-b)/b if b>0 else None for a,b in zip(d[:-1],d[1:])]
        result['gates']['fracture_dissipation']=d[-1]>0 and abs(d[-2]-d[-1])/d[-1]<=limits.get('fracture_relative',.05)
    result['passed']=all(result['gates'].values());return result


def run(root):
    root=Path(root);pre=json.loads((root/'preregistration.json').read_text());summary=dict(layers={},cases=[])
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    energy=[]
    for j,layer in enumerate(['frozen','maxwell']):
        cases=[]
        for suffix in ['h500','h250','h125']:
            name=layer+'_'+suffix;path=root/name
            if not (path/'result.json').exists() or not (root/(name+'.audit.json')).exists():continue
            q=json.loads((path/'result.json').read_text());a=json.loads((root/(name+'.audit.json')).read_text());bind_case(path,q,a);cases.append((path,q,a))
            summary['cases'].append(dict(name=name,status=q['status'],result_sha256=digest(path/'result.json'),audit_sha256=digest(root/(name+'.audit.json')),summary=q['summary'],audit_passed=a['passed']))
            t=[r['time_s'] for r in a['states']];label=str(q['dt_requested_s'])+' s'
            axes[j,0].plot(t,[r['independent_fz_n'] for r in a['states']],label=label)
            axes[j,1].plot(t,[r['maximum_speed_mm_s'] for r in a['states']],label=label)
            if a['passed']:axes[j,2].plot(t,np.cumsum([r['absolute_numerical_including_equation_work_n_mm'] for r in a['states']]),label=label)
            energy.append(dict(case=name,**{k:sum(z[k] for z in a['states']) for k in a['states'][0] if k.endswith('_n_mm')}))
        if len(cases)==3:summary['layers'][layer]=compare(cases,pre['coupled_window']['convergence_targets'])
        for i,y in enumerate(['raw Fz (N)','maximum speed (mm/s)','cumulative absolute numerical work (N mm)']):
            axes[j,i].set(xlabel='absolute time (s)',ylabel=y,title=layer);axes[j,i].legend()
    fig.tight_layout();fig.savefig(root/'coupled_window_raw.png',dpi=150);plt.close(fig)
    write_json(root/'window_comparison.json',native(summary))
    if energy:
        with (root/'energy_table.csv').open('w') as f:w=csv.DictWriter(f,list(energy[0]));w.writeheader();w.writerows(energy)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Nonlinear coupled contact: independently bound clock and inputs</h1><p>Conditional prescribed-notch windows. No 2 mm, new fracture or experiment qualification.</p><img src="coupled_window_raw.png" width="1100"><p><a href="energy_table.csv">Energy</a> <a href="window_comparison.json">Audits and preregistered common-window comparison</a></p><pre>'+json.dumps(summary['layers'],indent=2)+'</pre>')
    print(json.dumps(summary['layers'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)


def common_time_review(cases,legacy):
    """Three-level common clock/denominator, retaining finer samples and old gates.

    A difference between discrete solutions is not an upper error bound. The
    all-sample peak/extra-pair tables prevent shared coarse sampling concealing
    a finer transient. Inputs must already pass the independent identity audit.
    """
    import itertools
    identity=common_inputs(cases)
    if len(cases)!=3:raise ValueError('Exactly three bound levels required')
    offsets=[registered_offsets(q,a) for _,q,a in cases]
    maps=[{i:j for i,j in common_offset_indices(o,offsets[0],require_all=False)} for o in offsets]
    shared=sorted(set(maps[0]).intersection(maps[1],maps[2]))
    if not shared:raise ValueError('No three-way common physical times')
    indices=[[m[i] for i in shared] for m in maps]
    loaded=[]
    for path,raw,audit in cases:
        states=[];total=None
        for row in audit['states']:
            with np.load(path/row['state_file'],allow_pickle=False) as z:
                m=z['mass'].copy();v=z['velocity'][:len(m)].copy()
                impulse=z['impulse_normal'][:len(m)]+z['impulse_friction'][:len(m)]
                total=impulse.copy() if total is None else total+impulse
            states.append(dict(time_s=row['time_s'],state_file=row['state_file'],mass=m,v=v,I=total.copy(),
                velocity_norm=float(np.sqrt(np.sum(m[:,None]*v*v))),impulse_norm=float(np.sqrt(np.sum(total*total/m[:,None])))))
        loaded.append(states)
    m=loaded[0][0]['mass']
    for rows in loaded:
        for row in rows:np.testing.assert_array_equal(row['mass'],m)
    scales={key:max(loaded[2][i][field] for i in indices[2]) for key,field in [('velocity','velocity_norm'),('impulse','impulse_norm')]}
    if min(scales.values())<=0:raise ValueError('Positive common normalization required')
    def pair(i,j,ii,jj):
        errors=[];impulse_errors=[]
        for a,b in zip(ii,jj):
            dv=loaded[j][b]['v']-loaded[i][a]['v'];di=loaded[j][b]['I']-loaded[i][a]['I']
            errors.append(float(np.sqrt(np.sum(m[:,None]*dv*dv))));impulse_errors.append(float(np.sqrt(np.sum(di*di/m[:,None]))))
        if not errors:return None
        k=int(np.argmax(errors));a,b=ii[k],jj[k];dv=loaded[j][b]['v']-loaded[i][a]['v'];contribution=m[:,None]*dv**2
        ranking=np.argsort(contribution.ravel())[::-1][:10];den=float(contribution.sum())
        contributions=[dict(node=int(v//2),dof='xy'[v%2],delta_velocity_mm_s=float(dv.ravel()[v]),mass_n_s2_mm=float(m[v//2]),
            squared_norm_contribution_n_mm=float(contribution.ravel()[v]),fraction=float(contribution.ravel()[v]/den) if den else 0.) for v in ranking]
        ki=int(np.argmax(impulse_errors))
        return dict(levels=[i,j],sample_count=len(ii),maximum_velocity_difference_sqrt_nmm=max(errors),
            common_normalized_velocity_difference=max(errors)/scales['velocity'],worst_velocity_time_s=loaded[i][a]['time_s'],
            independently_reconstructed_other_time_s=loaded[j][b]['time_s'],largest_dof_contributions=contributions,
            maximum_contact_impulse_mass_difference_sqrt_nmm=max(impulse_errors),
            common_normalized_impulse_difference=max(impulse_errors)/scales['impulse'],
            worst_impulse_time_s=loaded[i][ii[ki]]['time_s'],
            times_s=[loaded[i][k]['time_s'] for k in ii],velocity_differences_sqrt_nmm=errors,impulse_differences_sqrt_nmm=impulse_errors)
    pairs=[pair(i,j,indices[i],indices[j]) for i,j in itertools.combinations(range(3),2)]
    original=[]
    for i,j in ((0,1),(1,2)):
        match=common_offset_indices(offsets[j],offsets[i]);ii=[a for a,b in match];jj=[b for a,b in match]
        q=pair(i,j,ii,jj);q['original_velocity_denominator_sqrt_nmm']=max(loaded[j][b]['velocity_norm'] for b in jj)
        q['original_impulse_denominator_sqrt_nmm']=max(loaded[j][b]['impulse_norm'] for b in jj)
        q['original_velocity_relative']=q['maximum_velocity_difference_sqrt_nmm']/q['original_velocity_denominator_sqrt_nmm']
        q['original_impulse_relative']=q['maximum_contact_impulse_mass_difference_sqrt_nmm']/q['original_impulse_denominator_sqrt_nmm']
        original.append(q)
    common_mid=set(indices[1]);extra=common_offset_indices(offsets[2],offsets[1]);extra=[(a,b) for a,b in extra if a not in common_mid]
    extra_pair=pair(1,2,[a for a,b in extra],[b for a,b in extra])
    peaks=[]
    for level,rows in enumerate(loaded):
        record=dict(level=level,all_sample_count=len(rows))
        for field in ('velocity_norm','impulse_norm'):
            for scope,ids in [('all',list(range(len(rows)))),('shared',indices[level]),('outside_shared',[i for i in range(len(rows)) if i not in set(indices[level])])]:
                k=max(ids,key=lambda i:rows[i][field]) if ids else None
                record[scope+'_'+field+'_peak']=None if k is None else dict(value=rows[k][field],time_s=rows[k]['time_s'],state_file=rows[k]['state_file'])
        peaks.append(record)
    p01,p02,p12=pairs
    return dict(schema='softcut-common-time-review/1',input_identity=identity,
        scope='Same registered absolute-time samples and one finest-level common peak; no event shifts, interpolation or continuous error bound. Legacy results retained.',
        common_sample_count=len(shared),common_times_s=[loaded[0][i]['time_s'] for i in indices[0]],
        common_velocity_scale_sqrt_nmm=scales['velocity'],common_impulse_scale_sqrt_nmm=scales['impulse'],pairs=pairs,
        velocity_difference_decreases=p12['maximum_velocity_difference_sqrt_nmm']<p01['maximum_velocity_difference_sqrt_nmm'],
        impulse_difference_decreases=p12['maximum_contact_impulse_mass_difference_sqrt_nmm']<p01['maximum_contact_impulse_mass_difference_sqrt_nmm'],
        fine_pair_velocity_5percent=p12['common_normalized_velocity_difference']<=.05,
        original_pairs_reconstructed=original,extra_middle_fine_common_times=extra_pair,all_sample_peaks=peaks,
        original_comparison=legacy,original_flags_preserved=True)
