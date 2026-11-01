"""Common physical-clock closed-C1 comparisons with one bound event reservoir."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .coupled_audit_contract import common_inputs
from .report_coupled_contact import compare,common_time_review
from .force_sampling_review import review
from .frozen_scene import digest
from .face_witness import native


def overlap(brackets):
    return bool(all(brackets) and max(b['previous']['time_s'] for b in brackets)<=min(b['first']['time_s'] for b in brackets)+1e-12)


def run(root):
    root=Path(root);plan=json.loads((root/'preregistration.json').read_text())['coupled_window'];limits=plan['convergence_targets'];ref=plan['event_energy_reference']
    reference_file=Path(ref['report_file'])
    if digest(reference_file)!=ref['report_sha256']:raise ValueError('Reference pilot report changed')
    reference=json.loads(reference_file.read_text())
    if ref['value_n_mm']!=reference['event_energy_reference']['value_n_mm']:raise ValueError('Event scale changed')
    cases=[];reports=[]
    for name in ('coarse','middle','fine'):
        path=root/name;q=json.loads((path/'result.json').read_text());a=json.loads((root/(name+'.audit.json')).read_text());report=json.loads((root/(name+'_event')/'summary.json').read_text())
        if report['result_sha256']!=digest(path/'result.json') or report['audit_sha256']!=digest(root/(name+'.audit.json')):raise ValueError('Stale event report')
        cases.append((path,q,a));reports.append(report)
    identity=common_inputs(cases)
    if any(r['parent']['sha256']!=reference['parent']['sha256'] for r in reports):raise ValueError('Reference event has different parent')
    if any(r['source_hashes']!=reports[0]['source_hashes'] or r['config']!=reference['config'] for r in reports):raise ValueError('Reference/config mismatch')
    try:legacy=compare(cases,limits,entry_required=False)
    except ValueError as exc:legacy=dict(passed=False,error=str(exc),gates={})
    common=common_time_review(cases,legacy,allow_unmatched_partition_points=True);forces=review([a for _,_,a in cases],limits['mechanism_force_absolute_n'],raw_cases=[q for _,q,_ in cases])
    p01,p02,p12=common['pairs'];energy=[]
    for name,r in zip(('coarse','middle','fine'),reports):
        e=r['energy_n_mm'];energy.append(dict(case=name,**e,reference_scale_n_mm=ref['value_n_mm'],absolute_numerical_fraction=e['absolute_numerical_including_equation_work_n_mm']/ref['value_n_mm'],absolute_equation_work_n_mm=r['absolute_equation_work_n_mm'],absolute_equation_fraction=r['absolute_equation_work_n_mm']/ref['value_n_mm'],residual_impulse_mass_norm_sum_sqrt_nmm=r['residual_impulse_mass_norm_sum_sqrt_nmm'],residual_impulse_over_common_velocity=r['residual_impulse_mass_norm_sum_sqrt_nmm']/common['common_velocity_scale_sqrt_nmm'],old_absolute_numerical_over_blade_work=r['old_absolute_numerical_over_blade_work']))
    arrays={}
    for name in ('top_unlock','A_original_interior_node_contact','B_P1_interior_projection','C_new_failed_connected_projection'):
        brackets=[r['events'].get(name) for r in reports]
        arrays[name]=dict(brackets=brackets,presence_agrees=len({bool(b) for b in brackets})==1,all_present=all(brackets),overlap=overlap(brackets) if any(brackets) else None,interval_width_s=[None if b is None else b['first']['time_s']-b['previous']['time_s'] for b in brackets],successive_upper_time_difference_s=[None if a is None or b is None else abs(a['first']['time_s']-b['first']['time_s']) for a,b in zip(brackets[:-1],brackets[1:])])
    failure_maps=[{face:b for b in r['events'].get('full_failures',[]) for face in b['faces']} for r in reports];allfaces=sorted(set().union(*[set(m) for m in failure_maps]));failures={str(f):dict(brackets=[m.get(f) for m in failure_maps],overlap=overlap([m.get(f) for m in failure_maps])) for f in allfaces}
    activation_sequences=[[b['faces'] for b in r['events'].get('activations',[])] for r in reports]
    force_pairs=forces['pairs'];force_scale=abs(forces['peaks'][-1]['common_peak']['fz_n']);f01,f02,f12=[z['maximum_absolute_difference_n'] for z in force_pairs]
    gates=dict(independent_equations_history_geometry=True,common_velocity_fine_5percent=common['fine_pair_velocity_5percent'],velocity_difference_decreases=common['velocity_difference_decreases'],common_impulse_fine_5percent=p12['common_normalized_impulse_difference']<=limits['impulse_relative'],impulse_difference_decreases=common['impulse_difference_decreases'],force_fine_relative=f12/force_scale<=limits['force_relative'],force_difference_decreases=f12<f01,
        top_unlock_present_and_brackets_overlap=arrays['top_unlock']['all_present'] and arrays['top_unlock']['overlap'],original_internal_contact_present_and_brackets_overlap=arrays['A_original_interior_node_contact']['all_present'] and arrays['A_original_interior_node_contact']['overlap'],new_failure_observed=bool(allfaces),full_failure_faces_agree=all(set(m)==set(failure_maps[0]) for m in failure_maps),full_failure_intervals_overlap=bool(failures) and all(z['overlap'] for z in failures.values()),activation_sequences_agree=all(z==activation_sequences[0] for z in activation_sequences),contact_B_presence_agrees=arrays['B_P1_interior_projection']['presence_agrees'],contact_C_presence_agrees=arrays['C_new_failed_connected_projection']['presence_agrees'],event_energy_fine_1percent=energy[-1]['absolute_numerical_fraction']<=.01,absolute_numerical_decreases=all(a['absolute_numerical_fraction']>b['absolute_numerical_fraction'] for a,b in zip(energy[:-1],energy[1:])),equation_work_allocated_10percent=all(z['absolute_equation_fraction']<=.001 for z in energy),residual_impulse_allocated_10percent=all(z['residual_impulse_over_common_velocity']<=.005 for z in energy),fracture_dissipation=energy[-1]['independent_fracture_n_mm']>0 and abs(energy[-2]['independent_fracture_n_mm']-energy[-1]['independent_fracture_n_mm'])/energy[-1]['independent_fracture_n_mm']<=limits['fracture_relative'])
    retained={k:v for k,v in gates.items() if k!='force_difference_decreases'}
    summary=dict(schema='softcut-closed-c1-event-comparison/1',input_identity=identity,reference_energy=ref,legacy_comparison=legacy,common_time_review=common,force_sampling=forces,energy=energy,events=arrays,full_failures=failures,activation_sequences=activation_sequences,gates=gates,strict_time_comparison_passed=all(gates.values()),mechanism_force_consistency_passed=bool(forces['all_three_sampled_force_consistent'] and all(retained.values())),complete_event_verified=False,mesh_converged=False,scope='Same absolute clock; no shifted events, smoothed forces, interpolated states or rigorous upper error bounds',per_level_event_reports=reports)
    (root/'event_comparison.json').write_text(json.dumps(native(summary),indent=2))
    with (root/'event_energy_table.csv').open('w') as f:w=csv.DictWriter(f,list(energy[0]));w.writeheader();w.writerows(energy)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8));t0=reports[0]['window_start_s']
    for name,r,(_,_,a) in zip(('coarse','middle','fine'),reports,cases):
        with (root/(name+'_event')/'raw.csv').open() as f:rows=list(csv.DictReader(f))
        t=np.array([float(z['time_s']) for z in rows]);
        for ax,key in [(axes[0,0],'fz_n'),(axes[0,1],'maximum_speed_mm_s'),(axes[0,2],'top_full_opening_mm'),(axes[1,0],'kinetic_energy_n_mm')]:ax.plot(t,[float(z[key]) for z in rows],label=name);ax.set_ylabel(key)
        axes[1,1].plot(t,np.cumsum([float(z['absolute_numerical_including_equation_work_n_mm']) for z in rows])/ref['value_n_mm'],label=name);axes[1,1].set_ylabel('absolute numerical / fixed event scale')
        axes[1,2].plot(t,np.cumsum([z['blade_normal_impulse_n_s']+z['blade_friction_impulse_n_s'] for z in a['states']]),label=name);axes[1,2].set_ylabel('blade impulse (N s)')
    for ax in axes.flat:
        ax.set_xlabel('absolute physical time (s)');ax.ticklabel_format(useOffset=False);ax.legend();ax.secondary_xaxis('top',functions=(lambda x:(x-t0)*1e6,lambda u:t0+u*1e-6)).set_xlabel('offset (microseconds)')
    fig.tight_layout();fig.savefig(root/'event_comparison.png',dpi=140);plt.close(fig)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Closed C1 common event window</h1><img src="event_comparison.png" width="1100"><p><a href="event_comparison.json">Full evidence and original metrics</a> <a href="event_energy_table.csv">Energy table</a></p><pre>'+json.dumps(gates,indent=2)+'</pre>')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');q=run(parser.parse_args().root);print(json.dumps(q['gates']))
