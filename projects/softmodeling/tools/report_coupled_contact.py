"""Common absolute-window time comparisons; no smoothing or event alignment."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .frozen_scene import digest
from .face_witness import native
from .coupled_audit_contract import common_inputs,bind_case
from run_support import write_json


def cumulative_impulses(path,rows):
    """Integrate each case on its own accepted partition, then sample common times."""
    total=None;out={};normal=friction=0.
    for row in rows:
        with np.load(path/row['state_file'],allow_pickle=False) as z:
            n=len(z['mass']);I=z['impulse_normal'][:n]+z['impulse_friction'][:n]
            total=I.copy() if total is None else total+I
        normal+=row['blade_normal_impulse_n_s'];friction+=row['blade_friction_impulse_n_s']
        out[round(row['time_s'],12)]=(total.copy(),normal,friction)
    return out


def compare(cases,limits,allowed_variations=('dt_requested_s','time_schedule')):
    identity=common_inputs(cases,allowed_variations)
    result=dict(input_identity=identity,allowed_variations=list(allowed_variations),passed=False,scope='this prescribed-notch conditional window only',pairs=[])
    if any(q['status']!='completed_window' for _,q,_ in cases):
        result['error']='Incomplete common physical window';return result
    fractions=[]
    for path,q,a in cases:
        if not a['passed']:result['error']='Independent audit failed';return result
        fractions.append(a['absolute_numerical_including_equation_work_n_mm']/abs(sum(s['independent_external_work_n_mm'] for s in a['states'])))
    for (path0,q0,a0),(path1,q1,a1) in zip(cases[:-1],cases[1:]):
        bytime={round(r['time_s'],12):r for r in a1['states']}
        rows0=a0['states'];rows1=[]
        for row in rows0:
            key=round(row['time_s'],12)
            if key not in bytime:raise ValueError('No identical absolute-time comparison state; interpolation prohibited')
            rows1.append(bytime[key])
        f0=np.array([r['independent_fz_n'] for r in rows0]);f1=np.array([r['independent_fz_n'] for r in rows1])
        ferror=float(max(abs(f0-f1))/max(abs(f1)))
        velocity_errors=[];velocity_scale=[];position_errors=[];impulse_errors=[];impulse_scales=[];blade_impulse_errors=[];blade_impulse_scales=[]
        impulses0=cumulative_impulses(path0,a0['states']);impulses1=cumulative_impulses(path1,a1['states'])
        for r0,r1 in zip(rows0,rows1):
            with np.load(path0/r0['state_file'],allow_pickle=False) as z:v0=z['velocity'];x0=z['positions'];mass=z['mass'];n=len(mass)
            with np.load(path1/r1['state_file'],allow_pickle=False) as z:v1=z['velocity'];x1=z['positions']
            I0,N0,T0=impulses0[round(r0['time_s'],12)];I1,N1,T1=impulses1[round(r1['time_s'],12)]
            impulse_errors.append(float(np.sqrt(np.sum((I1-I0)**2/mass[:,None]))))
            impulse_scales.append(float(np.sqrt(np.sum(I1**2/mass[:,None]))))
            blade_impulse_errors.append(abs((N1+T1)-(N0+T0)));blade_impulse_scales.append(abs(N1+T1))
            velocity_errors.append(float(np.sqrt(mass@np.sum((v1[:n]-v0[:n])**2,axis=1))))
            velocity_scale.append(float(np.sqrt(mass@np.sum(v1[:n]**2,axis=1))))
            position_errors.append(float(np.max(np.linalg.norm(x1[:n]-x0[:n],axis=1))))
        verror=max(velocity_errors)/max(velocity_scale)
        preexisting=bool(a0.get('entry_present_at_parent',False))
        if preexisting!=bool(a1.get('entry_present_at_parent',False)):raise ValueError('Same parent has inconsistent entry identity')
        if preexisting:
            lo=hi=q0['initial_depth_mm'];entry_difference=0.
        else:
            br0=a0['first_entry_bracket'];br1=a1['first_entry_bracket']
            lo=max(br0['previous_nonentry']['depth_mm'],br1['previous_nonentry']['depth_mm'])
            hi=min(br0['first_entry']['depth_mm'],br1['first_entry']['depth_mm'])
            entry_difference=abs(br0['first_entry']['depth_mm']-br1['first_entry']['depth_mm'])
        result['pairs'].append(dict(coarse_dt_s=q0['dt_requested_s'],fine_dt_s=q1['dt_requested_s'],
            maximum_fz_error_over_fine_peak=ferror,maximum_mass_velocity_error_over_fine_peak=verror,
            maximum_mass_velocity_difference_sqrt_nmm=max(velocity_errors),fine_peak_mass_velocity_sqrt_nmm=max(velocity_scale),
            worst_velocity_time_s=rows0[int(np.argmax(velocity_errors))]['time_s'],
            maximum_cumulative_contact_impulse_mass_error_over_fine_peak=max(impulse_errors)/max(impulse_scales),
            maximum_cumulative_blade_impulse_error_over_fine_peak=max(blade_impulse_errors)/max(blade_impulse_scales),
            terminal_blade_normal_impulse_difference_n_s=N1-N0,terminal_blade_friction_impulse_difference_n_s=T1-T0,
            maximum_position_difference_mm=max(position_errors),entry_brackets_overlap=lo<=hi+1e-14,
            entry_depth_difference_mm=entry_difference,entry_preexists_parent=preexisting))
    result['absolute_numerical_fraction']=fractions
    fine=result['pairs'][-1]
    result['gates']=dict(force=fine['maximum_fz_error_over_fine_peak']<=limits['force_relative'],
        mass_velocity=fine['maximum_mass_velocity_error_over_fine_peak']<=limits['velocity_mass_relative'],
        event=all(p['entry_brackets_overlap'] and p['entry_depth_difference_mm']<=limits['event_bracket_mm'] for p in result['pairs']),
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
