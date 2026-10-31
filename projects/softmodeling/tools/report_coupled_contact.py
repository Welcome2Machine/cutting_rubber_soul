"""Common absolute-window time comparisons; no smoothing or event alignment."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .frozen_scene import digest
from .face_witness import native
from run_support import write_json


def compare(cases,limits):
    result=dict(passed=False,scope='this prescribed-notch conditional window only',pairs=[])
    if any(q['status']!='completed_window' for _,q,_ in cases):
        result['error']='Incomplete common physical window';return result
    fractions=[]
    for path,q,a in cases:
        if not a['passed']:result['error']='Independent audit failed';return result
        fractions.append(a['absolute_numerical_including_equation_work_n_mm']/abs(q['summary']['ledger_n_mm']['external_work_n_mm']))
    for (path0,q0,a0),(path1,q1,a1) in zip(cases[:-1],cases[1:]):
        stride=int(round(q0['dt_requested_s']/q1['dt_requested_s']))
        rows1=q1['steps'][stride-1::stride];rows0=q0['steps']
        np.testing.assert_allclose([r['time_s'] for r in rows0],[r['time_s'] for r in rows1],atol=1e-13,rtol=0)
        f0=np.array([r['reaction_fz_n'] for r in rows0]);f1=np.array([r['reaction_fz_n'] for r in rows1])
        ferror=float(max(abs(f0-f1))/max(abs(f1)))
        velocity_errors=[];velocity_scale=[];position_errors=[]
        for r0,r1 in zip(rows0,rows1):
            with np.load(path0/r0['state_file'],allow_pickle=False) as z:v0=z['velocity'];x0=z['positions'];mass=z['mass'];n=len(mass)
            with np.load(path1/r1['state_file'],allow_pickle=False) as z:v1=z['velocity'];x1=z['positions']
            velocity_errors.append(float(np.sqrt(mass@np.sum((v1[:n]-v0[:n])**2,axis=1))))
            velocity_scale.append(float(np.sqrt(mass@np.sum(v1[:n]**2,axis=1))))
            position_errors.append(float(np.max(np.linalg.norm(x1[:n]-x0[:n],axis=1))))
        verror=max(velocity_errors)/max(velocity_scale)
        br0=a0['first_entry_bracket'];br1=a1['first_entry_bracket']
        lo=max(br0['previous_nonentry']['depth_mm'],br1['previous_nonentry']['depth_mm'])
        hi=min(br0['first_entry']['depth_mm'],br1['first_entry']['depth_mm'])
        result['pairs'].append(dict(coarse_dt_s=q0['dt_requested_s'],fine_dt_s=q1['dt_requested_s'],
            maximum_fz_error_over_fine_peak=ferror,maximum_mass_velocity_error_over_fine_peak=verror,
            maximum_position_difference_mm=max(position_errors),entry_brackets_overlap=lo<=hi+1e-14,
            entry_depth_difference_mm=abs(br0['first_entry']['depth_mm']-br1['first_entry']['depth_mm'])))
    result['absolute_numerical_fraction']=fractions
    fine=result['pairs'][-1]
    result['gates']=dict(force=fine['maximum_fz_error_over_fine_peak']<=limits['force_relative'],
        mass_velocity=fine['maximum_mass_velocity_error_over_fine_peak']<=limits['velocity_mass_relative'],
        event=all(p['entry_brackets_overlap'] and p['entry_depth_difference_mm']<=limits['event_bracket_mm'] for p in result['pairs']),
        absolute_numerical=fractions[-1]<=limits['absolute_numerical_work_fraction'],
        numerical_decreases=all(a>b for a,b in zip(fractions[:-1],fractions[1:])),
        force_difference_decreases=result['pairs'][-1]['maximum_fz_error_over_fine_peak']<result['pairs'][0]['maximum_fz_error_over_fine_peak'],
        velocity_difference_decreases=result['pairs'][-1]['maximum_mass_velocity_error_over_fine_peak']<result['pairs'][0]['maximum_mass_velocity_error_over_fine_peak'])
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
            q=json.loads((path/'result.json').read_text());a=json.loads((root/(name+'.audit.json')).read_text());cases.append((path,q,a))
            summary['cases'].append(dict(name=name,status=q['status'],result_sha256=digest(path/'result.json'),audit_sha256=digest(root/(name+'.audit.json')),summary=q['summary'],audit_passed=a['passed']))
            t=[r['time_s'] for r in q['steps']];label=str(q['dt_requested_s'])+' s'
            axes[j,0].plot(t,[r['reaction_fz_n'] for r in q['steps']],label=label)
            axes[j,1].plot(t,[r['maximum_speed_mm_s'] for r in q['steps']],label=label)
            if a['passed']:axes[j,2].plot(t,np.cumsum([r['absolute_numerical_including_equation_work_n_mm'] for r in a['states']]),label=label)
            energy.append(dict(case=name,**q['summary']['ledger_n_mm'],absolute_numerical_including_equation_work_n_mm=a.get('absolute_numerical_including_equation_work_n_mm')))
        if len(cases)==3:summary['layers'][layer]=compare(cases,pre['coupled_window']['convergence_targets'])
        for i,y in enumerate(['raw Fz (N)','maximum speed (mm/s)','cumulative absolute numerical work (N mm)']):
            axes[j,i].set(xlabel='absolute time (s)',ylabel=y,title=layer);axes[j,i].legend()
    fig.tight_layout();fig.savefig(root/'coupled_window_raw.png',dpi=150);plt.close(fig)
    write_json(root/'window_comparison.json',native(summary))
    if energy:
        with (root/'energy_table.csv').open('w') as f:w=csv.DictWriter(f,list(energy[0]));w.writeheader();w.writerows(energy)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Round 8: nonlinear coupled contact</h1><p>Conditional prescribed-notch windows. No 2 mm, new fracture or experiment qualification.</p><img src="coupled_window_raw.png" width="1100"><p><a href="energy_table.csv">Energy</a> <a href="window_comparison.json">Audits and preregistered common-window comparison</a></p><pre>'+json.dumps(summary['layers'],indent=2)+'</pre>')
    print(json.dumps(summary['layers'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
