"""Absolute-time comparison of independently audited, exactly linked event phases.

Every original result/audit is immutable. Virtual rows below are only a reporting
view; no merged production trajectory, rewritten witness or history transplant.
"""
import argparse,copy,csv,json
from pathlib import Path
import numpy as np
from .coupled_audit_contract import bind_case,common_inputs,registered_offsets,close
from .report_coupled_contact import common_time_data
from .force_sampling_review import review
from .report_c1_event import event_energy_scale
from .compare_c1_event import overlap
from .face_witness import native
from .frozen_scene import digest


def verify_link(previous,current,previous_result_sha,previous_audit_sha):
    if current['parent']['result_sha256']!=previous_result_sha or current['parent']['audit_sha256']!=previous_audit_sha or current['parent']['state_sha256']!=previous['steps'][-1]['state_sha256']:
        raise ValueError('Conditional phase has the wrong accepted parent')
    if current['source_hashes']!=previous['source_hashes'] or current['config']!=previous['config'] or current['material_layer']!=previous['material_layer'] or current['solve_tolerance_n']!=previous['solve_tolerance_n']:
        raise ValueError('Phase equations/configuration changed')
    if not current['conditional_comparison'] or current['production_continuation']:raise ValueError('False production continuation claim')
    close(current['initial_time_s'],previous['initial_time_s']+previous['window_s'],'Phase clock')
    close(current['initial_depth_mm'],previous['steps'][-1]['depth_mm'],'Phase depth')


def chain(root,spec):
    cases=[];reports=[];raw_rows=[];combined_states=[];combined_rows=[];offsets=[];cut=0.;events={};energy={};links=[]
    for item in spec:
        path=root/item['case'];af=root/item['audit'];rf=root/item['report'];q=json.loads((path/'result.json').read_text());a=json.loads(af.read_text());r=json.loads(rf.read_text())
        bind_case(path,q,a)
        if r['result_sha256']!=digest(path/'result.json') or r['audit_sha256']!=digest(af):raise ValueError('Unbound phase report')
        if cases:
            prev=cases[-1];verify_link(prev[1],q,digest(prev[0]/'result.json'),digest(root/spec[len(cases)-1]['audit']))
        for key,value in r['events'].items():
            if key in ('activations','full_failures'):events.setdefault(key,[]).extend(value)
            else:events.setdefault(key,value)
        for k,v in r['energy_n_mm'].items():energy[k]=energy.get(k,0.)+v
        for row in a['states']:
            z=copy.deepcopy(row);z['state_file']=str((path/row['state_file']).relative_to(root));combined_states.append(z)
        for row in q['steps']:
            z=copy.deepcopy(row);z['state_file']=str((path/row['state_file']).relative_to(root));combined_rows.append(z)
        offsets.extend(cut+registered_offsets(q,a));cut+=q['window_s']
        with rf.with_name('raw.csv').open() as f:
            for row in csv.DictReader(f):
                parsed={}
                for k,v in row.items():
                    if v in ('True','False'):parsed[k]=v=='True'
                    else:
                        try:parsed[k]=float(v)
                        except ValueError:parsed[k]=v
                raw_rows.append(parsed)
        cases.append((path,q,a));reports.append(r);links.append(dict(**item,result_sha256=digest(path/'result.json'),audit_sha256=digest(af),report_sha256=digest(rf)))
    q=copy.deepcopy(cases[0][1]);q.update(steps=combined_rows,window_s=cut,time_schedule=dict(end_offsets_s=list(offsets)),status='reporting_view_only')
    a=dict(states=combined_states)
    K0=reports[0]['event_energy_reference']['components']['initial_kinetic_n_mm'];ref=event_energy_scale(K0,energy)
    peak=max(raw_rows,key=lambda z:z['maximum_speed_mm_s']);ip=max(range(len(raw_rows)),key=lambda i:raw_rows[i]['fz_n']);im=min(range(ip,len(raw_rows)),key=lambda i:raw_rows[i]['fz_n'])
    start=reports[0]['window_start_s'];end=reports[-1]['window_end_s'];entry=events.get('A_original_interior_node_contact')
    summary=dict(phases=links,window_start_s=start,window_end_s=end,duration_s=end-start,events=events,energy_n_mm=energy,event_energy_reference=ref,velocity_peak=peak,terminal=raw_rows[-1],terminal_new_face_distances=reports[-1]['terminal_new_face_distances'],post_entry_duration_s=None if entry is None else end-entry['first']['time_s'],force_drop=dict(total_n=raw_rows[ip]['fz_n']-raw_rows[im]['fz_n'],normal_n=raw_rows[ip]['normal_fz_n']-raw_rows[im]['normal_fz_n'],friction_n=raw_rows[ip]['friction_fz_n']-raw_rows[im]['friction_fz_n'],peak_time_s=raw_rows[ip]['time_s'],minimum_time_s=raw_rows[im]['time_s']),absolute_equation_work_n_mm=sum(r['absolute_equation_work_n_mm'] for r in reports),residual_impulse_mass_norm_sum_sqrt_nmm=sum(r['residual_impulse_mass_norm_sum_sqrt_nmm'] for r in reports),peak_at_final_sample=peak is raw_rows[-1],terminal_speed_over_peak=raw_rows[-1]['maximum_speed_mm_s']/max(peak['maximum_speed_mm_s'],1e-300),terminal_kinetic_over_peak=raw_rows[-1]['kinetic_energy_n_mm']/max(max(z['kinetic_energy_n_mm'] for z in raw_rows),1e-300),complete_event_verified=False)
    return (root,q,a),summary,raw_rows,cases[0]


def run(root,manifest,out):
    root=Path(root).resolve();manifest=Path(manifest);out=Path(out);plan=json.loads(manifest.read_text());reference=plan['event_energy_reference'];file=root/reference['report_file']
    if digest(file)!=reference['report_sha256']:raise ValueError('Reference chain changed')
    bound=json.loads(file.read_text())
    if reference['value_n_mm']!=bound['event_energy_reference']['value_n_mm']:raise ValueError('Reference energy changed')
    levels=[chain(root,plan['levels'][name]) for name in ('coarse','middle','fine')]
    initial_identity=common_inputs([z[3] for z in levels]);cases=[z[0] for z in levels];reports=[z[1] for z in levels];rows=[z[2] for z in levels]
    if any(z['window_start_s']!=reports[0]['window_start_s'] or abs(z['window_end_s']-reports[0]['window_end_s'])>1e-10 for z in reports):raise ValueError('Different total event windows')
    if reports[0]['window_start_s']!=bound['window_start_s'] or abs(reports[0]['window_end_s']-bound['window_end_s'])>1e-10:raise ValueError('Reference covers another event window')
    identity=dict(common_initial_parent=initial_identity,phase_bindings=[z['phases'] for z in reports],manifest_sha256=digest(manifest))
    common=common_time_data(cases,{'scope':'Original first-phase comparator is retained separately'},identity,[np.asarray(q['time_schedule']['end_offsets_s']) for _,q,_ in cases],allow_unmatched_partition_points=True)
    force=review([a for _,_,a in cases],1e-5,raw_cases=[q for _,q,_ in cases]);p01,p02,p12=common['pairs'];f01,f02,f12=force['pairs'];fscale=abs(force['peaks'][-1]['common_peak']['fz_n'])
    energy=[]
    for name,r in zip(('coarse','middle','fine'),reports):
        energy.append(dict(level=name,**r['energy_n_mm'],event_reference_n_mm=reference['value_n_mm'],absolute_numerical_fraction=r['energy_n_mm']['absolute_numerical_including_equation_work_n_mm']/reference['value_n_mm'],absolute_equation_fraction=r['absolute_equation_work_n_mm']/reference['value_n_mm'],residual_impulse_over_common_velocity=r['residual_impulse_mass_norm_sum_sqrt_nmm']/common['common_velocity_scale_sqrt_nmm']))
    events={}
    for key in ('top_unlock','A_original_interior_node_contact','B_P1_interior_projection','C_new_failed_connected_projection'):
        bs=[r['events'].get(key) for r in reports];events[key]=dict(brackets=bs,all_present=all(bs),presence_agrees=len({bool(z) for z in bs})==1,overlap=overlap(bs) if any(bs) else None)
    maps=[{f:b for b in r['events'].get('full_failures',[]) for f in b['faces']} for r in reports];faces=sorted(set().union(*[set(m) for m in maps]));failures={str(f):dict(brackets=[m.get(f) for m in maps],overlap=overlap([m.get(f) for m in maps])) for f in faces}
    gates=dict(original_force_history_geometry_audits=True,velocity_fine_5percent=common['fine_pair_velocity_5percent'],velocity_difference_decreases=common['velocity_difference_decreases'],impulse_fine_5percent=p12['common_normalized_impulse_difference']<=.05,impulse_difference_decreases=common['impulse_difference_decreases'],force_fine_relative=f12['maximum_absolute_difference_n']/fscale<=.05,force_difference_decreases=f12['maximum_absolute_difference_n']<f01['maximum_absolute_difference_n'],unlock_brackets_overlap=events['top_unlock']['all_present'] and events['top_unlock']['overlap'],internal_contact_brackets_overlap=events['A_original_interior_node_contact']['all_present'] and events['A_original_interior_node_contact']['overlap'],new_failed_faces_agree=bool(faces) and all(set(m)==set(maps[0]) for m in maps),failure_intervals_overlap=bool(failures) and all(z['overlap'] for z in failures.values()),energy_fine_1percent=energy[-1]['absolute_numerical_fraction']<=.01,numerical_energy_decreases=all(a['absolute_numerical_fraction']>b['absolute_numerical_fraction'] for a,b in zip(energy[:-1],energy[1:])),nonlinear_work_share=all(z['absolute_equation_fraction']<=.001 for z in energy),nonlinear_impulse_share=all(z['residual_impulse_over_common_velocity']<=.005 for z in energy),peak_before_endpoint=all(not r['peak_at_final_sample'] for r in reports),B_presence_agrees=events['B_P1_interior_projection']['presence_agrees'],C_presence_agrees=events['C_new_failed_connected_projection']['presence_agrees'])
    result=dict(schema='softcut-staged-c1-event-review/1',identity=identity,common_time_review=common,force_sampling=force,energy=energy,events=events,full_failures=failures,gates=gates,strict_time_comparison_passed=all(gates.values()),mechanism_force_consistency_passed=force['all_three_sampled_force_consistent'] and all(v for k,v in gates.items() if k!='force_difference_decreases'),levels=reports,complete_event_verified=False,scope='Exactly linked independently audited conditional phases; not a fabricated merged production result or a rigorous time-error bound')
    out.mkdir(parents=True,exist_ok=False);(out/'summary.json').write_text(json.dumps(native(result),indent=2))
    with (out/'energy.csv').open('w') as f:w=csv.DictWriter(f,list(energy[0]));w.writeheader();w.writerows(energy)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8));t0=reports[0]['window_start_s']
    for name,table in zip(('coarse','middle','fine'),rows):
        t=np.asarray([z['time_s'] for z in table]);
        for ax,key in zip(axes.flat,('fz_n','maximum_speed_mm_s','top_full_opening_mm','kinetic_energy_n_mm','absolute_numerical_including_equation_work_n_mm','independent_fracture_n_mm')):
            y=[z[key] for z in table]
            if key in ('absolute_numerical_including_equation_work_n_mm','independent_fracture_n_mm'):y=np.cumsum(y)
            ax.plot(t,y,label=name);ax.set_ylabel(key)
    for ax in axes.flat:ax.set_xlabel('absolute time (s)');ax.ticklabel_format(useOffset=False);ax.legend();ax.secondary_xaxis('top',functions=(lambda x:(x-t0)*1e6,lambda u:t0+u*1e-6)).set_xlabel('microseconds from common parent')
    fig.tight_layout();fig.savefig(out/'event.png',dpi=150);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Staged closed C1 event</h1><img src="event.png" width="1100"><pre>'+json.dumps(native(gates),indent=2)+'</pre>')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('manifest');p.add_argument('out');a=p.parse_args();q=run(a.root,a.manifest,a.out);print(json.dumps(q['gates']))
