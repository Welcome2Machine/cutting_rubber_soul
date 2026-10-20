"""Frozen bounded native-contact P1/local-P2 and PE/PS study; never production.

Qualify quadrature, time and enrichment radius on independent .05 mm histories.
Only a JOINT pass permits own-history PE continuations to .25 mm. PS remains a
conditional 2-D constraint diagnostic, not a finite-width 3-D replacement.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from run_support import write_json, compute_process_policy
from tools import run_material_queue as queue
from tools.check_cutting_discretization import snapshot

ROOT=Path(__file__).resolve().parents[1]
ANGLES=(33,57)
TAGS=('pe_base','pe_order4','pe_halfstep','pe_radius1','ps_p1','ps_p2','pe_depth')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_job(out, angle, tag):
    if angle not in (0,33,57):raise ValueError('Only authorized 0/33/57 angles')
    j=dict(name=f'{tag}_{angle}',angle=angle,mu=.35,diagnostic='cutting_discretization',
        kind='p2',kinematics='plane_strain',radius=.5,order=3,increment=.01,
        target_mm=.05,solve_seconds=900.,status='pending',tag=tag)
    if tag=='pilot':
        j.update(name=f'pilot_p2_pe_{angle}',target_mm=.01,solve_seconds=300.)
    elif tag=='pe_base':j['resume']=str(out/f'pilot_p2_pe_{angle}')
    elif tag=='pe_order4':j['order']=4
    elif tag=='pe_halfstep':j['increment']=.005
    elif tag=='pe_radius1':j['radius']=1.
    elif tag in ('ps_p1','ps_p2'):
        j['kinematics']='plane_stress';j['kind']='p1' if tag=='ps_p1' else 'p2'
    elif tag=='pe_depth':
        j.update(target_mm=.25,solve_seconds=3600.,resume=str(out/f'pe_base_{angle}'))
    else:raise ValueError(tag)
    j['output']=str(out/j['name'])
    return j


def verify_case(folder):
    """Recheck saved source and every accepted checkpoint against its fresh audit."""
    folder=Path(folder);r=queue.read(folder/'result.json');a=queue.read(folder/'audit.json')
    if not a.get('passed') or a['states']!=len(r['steps']) or len(a['rows'])!=len(r['steps']):
        raise ValueError('Incomplete fresh audit')
    files={str(folder/'result.json'):digest(folder/'result.json'),
           str(folder/'audit.json'):digest(folder/'audit.json')}
    for name,h in r['source_sha256'].items():
        f=folder/'source'/name
        if digest(f)!=h:raise ValueError('Saved source changed: '+str(f))
        files[str(f)]=h
    for i,row in enumerate(a['rows']):
        f=folder/'states'/f'{i:04d}.npz'
        if row.get('passed') is not True or digest(f)!=row['checkpoint_sha256']:
            raise ValueError('Accepted checkpoint changed: '+str(f))
        files[str(f)]=row['checkpoint_sha256']
    return files


def rows_through(folder, end):
    r=queue.read(Path(folder)/'result.json')
    return r,[(i,s) for i,s in enumerate(r['steps']) if s['depth_mm']<=end+1e-10]


def compare_paths(first, second, geometry, end=.05):
    """Use actual matched accepted depths and identical original surface IDs only."""
    ra,a=rows_through(first,end);rb,b=rows_through(second,end)
    for key in ('config','material','loading_rate','contact_formulation','dhat','physical_force_tolerance_n'):
        if ra['signature'][key]!=rb['signature'][key]:raise ValueError('Physical mismatch: '+key)
    with np.load(geometry) as g:
        ids=g['top_nodes'];reference=g['reference_xy'];sref=g['top_s_from_contact_mm']
    matches=[];rows=[]
    for i,x in a:
        pair=next(((j,y) for j,y in b if abs(y['depth_mm']-x['depth_mm'])<1e-10),None)
        if pair is None:continue
        j,y=pair
        with np.load(Path(first)/'states'/f'{i:04d}.npz') as z1,\
             np.load(Path(second)/'states'/f'{j:04d}.npz') as z2:
            np.testing.assert_array_equal(z1['reference_xy'][ids],reference[ids])
            np.testing.assert_array_equal(z2['reference_xy'][ids],reference[ids])
            motion=np.linalg.norm(z2['positions'][ids]-z1['positions'][ids],axis=1)
        tx=x.get('tip_contact',{});ty=y.get('tip_contact',{})
        unique=tx.get('status')==ty.get('status')=='unique'
        f1=x['reaction_fz_n'];f2=y['reaction_fz_n']
        row=dict(depth_mm=x['depth_mm'],first_fz_n=f1,second_fz_n=f2,delta_fz_n=f2-f1,
            force_relative_difference=abs(f2-f1)/max(.1,abs(f1)),
            surface_max_difference_mm=float(motion.max()),surface_rms_difference_mm=float(np.sqrt(np.mean(motion**2))),
            near1mm_surface_max_difference_mm=float(motion[abs(sref)<=1].max()),
            far_surface_max_difference_mm=float(motion[abs(sref)>1].max()),
            first_min_inplane_j=x['min_j'],second_min_inplane_j=y['min_j'],
            first_min_volume_gauss_j=x['material'].get('min_volume_j',x['min_j']),
            second_min_volume_gauss_j=y['material'].get('min_volume_j',y['min_j']),
            first_tip_status=tx.get('status'),second_tip_status=ty.get('status'),
            reference_tip_difference_mm=ty['reference_offset_mm']-tx['reference_offset_mm'] if unique else None,
            first_nose_offset_intervals_mm=tx.get('nose_offset_intervals_mm'),
            second_nose_offset_intervals_mm=ty.get('nose_offset_intervals_mm'))
        rows.append(row);matches.append(x['depth_mm'])
    nominal=np.arange(0,end+1e-10,.01)
    covered=all(any(abs(d-z)<1e-9 for z in matches) for d in nominal)
    loaded=[r for r in rows if r['depth_mm']>0]
    fmax=max((r['force_relative_difference'] for r in loaded),default=None)
    umax=max((r['surface_max_difference_mm'] for r in rows),default=None)
    return dict(first=str(first),second=str(second),end_mm=end,rows=rows,all_nominal_depths_matched=covered,
        maximum_force_relative_difference=fmax,maximum_surface_difference_mm=umax,
        force_and_surface_gate_passed=bool(covered and loaded and fmax<=.02 and umax<=.002),
        unique_tip_comparison_count=sum(r['reference_tip_difference_mm'] is not None for r in rows),
        interpretation='Same reference material displacement, NOT cumulative interface slip; missing tip identities stay null. '
            'PE/PS comparisons change a constraint; radius/P2 comparisons are local spatial sensitivity, not global convergence.')


def depth_gate(manifest, jobs):
    by={(j['tag'],j['angle']):j for j in jobs};checks=[]
    for angle in manifest.get('angles',ANGLES):
        base=by['pe_base',angle]
        for tag in ('pe_order4','pe_halfstep','pe_radius1'):
            control=by[tag,angle]
            if not (base.get('mechanics_completed') and control.get('mechanics_completed')
                    and base.get('evidence_verified') and control.get('evidence_verified')):
                checks.append(dict(angle=angle,control=tag,passed=False,reason='Both audited complete paths required'))
                continue
            try:
                c=compare_paths(base['output'],control['output'],manifest['geometry'][str(angle)])
                checks.append(dict(angle=angle,control=tag,passed=c['force_and_surface_gate_passed'],comparison=c))
            except Exception as exc:checks.append(dict(angle=angle,control=tag,passed=False,error=repr(exc)))
    return dict(passed=all(c['passed'] for c in checks),checks=checks,
        limits=dict(force_fraction=.02,force_denominator_floor_n=.1,surface_max_mm=.002),
        scope='Post-zero shallow qualification only. No deep convergence or unique-contact qualification implied.')


def report(out,manifest,state):
    """Publish partial prefixes transparently; failed comparisons never pass a gate."""
    comparisons=[];jobs=state['jobs'];by={(j['tag'],j['angle']):j for j in jobs}
    for angle in manifest.get('angles',ANGLES):
        pairs=[('native_p1_vs_local_p2',manifest['references'][str(angle)],by['pe_base',angle]),
               ('p1_PE_vs_PS',manifest['references'][str(angle)],by['ps_p1',angle]),
               ('p2_PE_vs_PS',by['pe_base',angle],by['ps_p2',angle]),
               ('PS_p1_vs_p2',by['ps_p1',angle],by['ps_p2',angle]),
               ('native_p1_vs_local_p2_deep',manifest['references'][str(angle)],by['pe_depth',angle])]
        for label,first,second in pairs:
            if not (first.get('evidence_verified') and second.get('evidence_verified')):continue
            end=min(.25 if second['tag']=='pe_depth' else .05,
                    first['last_depth_mm'],second['last_depth_mm'])
            if end<=0:continue
            try:comparisons.append(dict(label=label,angle=angle,
                **compare_paths(first['output'],second['output'],manifest['geometry'][str(angle)],end)))
            except Exception as exc:comparisons.append(dict(label=label,angle=angle,error=repr(exc)))
    gate=depth_gate(manifest,jobs)
    result=dict(status=state['status'],updated_utc=queue.now(),depth_gate=gate,comparisons=comparisons,
        experimental_validation=False,finite_width_3d=False,
        limitations=['Shallow P2 is displacement enrichment, not mixed/locking-free.',
            'PE/PS bounds are different 2-D assumptions, not a finite-width solution.',
            'Time/quadrature/radius controls only qualify 0--.05 mm; deep paths remain provisional.',
            'No angle-specific material change or fracture; surface displacement is not slip.'])
    write_json(out/'study_report.json',result)
    lines=['# Actual-geometry discretization / constraint study','',f"Status: {state['status']}",
        '',f"Joint shallow gate permitting PE depth continuation: {gate['passed']}",'',
        '| Case | Status | Accepted depth mm | Fz N | Fresh audit and hashes |',
        '|---|---|---:|---:|---|']
    with (out/'accepted_curves.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['case','angle_deg','depth_mm','fz_n','min_inplane_J','tip_status','reference_offset_mm'])
        for j in jobs:
            lines.append(f"| {j['name']} | {j['status']} | {j.get('last_depth_mm')} | {j.get('endpoint_fz_n')} | {j.get('evidence_verified',False)} |")
            if not j.get('evidence_verified'):continue
            for s in queue.read(Path(j['output'])/'result.json')['steps']:
                tip=s['tip_contact'];w.writerow([j['name'],j['angle'],s['depth_mm'],s['reaction_fz_n'],s['min_j'],
                    tip['status'],tip.get('reference_offset_mm') if tip['status']=='unique' else None])
    lines+=['','Numerical qualification only. No result here establishes finite-width 3-D validity,',
            'experimental agreement, contact identity or accumulated slip. Full controls are in study_report.json.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def prepare(out, *, angles=ANGLES, fresh_pilots=False, after=None):
    from tools.check_native_mesh import baseline_problem, COMMON
    from tools.run_material_candidate import signature,audit
    from threadpoolctl import threadpool_limits
    if (out/'manifest.json').exists():raise RuntimeError('Study already prepared')
    if not angles or len(set(angles))!=len(angles) or any(a not in (0,33,57) for a in angles):
        raise ValueError('Require distinct authorized 0/33/57 angles')
    if after is not None and Path(after).resolve()==out.resolve():
        raise ValueError('Study cannot wait on itself')
    out.mkdir(parents=True,exist_ok=True)
    jobs=([case_job(out,a,'pilot') for a in angles] if fresh_pilots else [])
    jobs += [case_job(out,a,t) for t in TAGS for a in angles]
    inputs={};refs={};geometry={}
    # Actual .01 preflights already completed from independent rest states.
    frozen=snapshot(out)
    for a in (() if fresh_pilots else angles):
        pilot=out/f'pilot_p2_pe_{a}';j=case_job(out,a,'pe_base');j['target_mm']=.01
        s=queue.summarize_case(pilot,j,0)
        if not s['mechanics_completed']:raise ValueError('Unqualified pilot: '+str(pilot))
        r=queue.read(pilot/'result.json')
        if r['source_sha256']!=frozen:raise ValueError('Pilot physical/runner sources differ')
        inputs.update(verify_case(pilot))
    for name in ('tools/run_material_queue.py','tools/run_cutting_discretization.py'):
        dest=out/'source'/name;shutil.copy2(ROOT/name,dest);frozen[name]=digest(dest)
    with threadpool_limits(2),compute_process_policy():
        for a in angles:
            f=(ROOT/'outputs/relaxation_partition_20261015/angles30_0_2mm/02_angle0' if a==0 else
                ROOT/'outputs/native_angles_2mm_20261016'/('01_angle33' if a==33 else '02_angle57'))
            original=queue.read(f/'result.json');old_audit=queue.read(f/'audit.json')
            p=baseline_problem(angle=a,**COMMON)
            if json.loads(json.dumps(signature(p)))!=original['signature']:raise ValueError('P1 baseline physics differs')
            prefix=copy.deepcopy(original);prefix['steps']=[s for s in prefix['steps'] if s['depth_mm']<=.25+1e-12]
            dest=out/f'reference_{a}_fresh_audit.json';fresh=audit(p,f,prefix,destination=dest)
            inputs[str(dest)]=digest(dest);inputs[str(f/'result.json')]=digest(f/'result.json')
            inputs[str(f/'audit.json')]=digest(f/'audit.json')
            for name,h in original['source_sha256'].items():
                path=f/'source'/name
                if digest(path)!=h:raise ValueError('Original frozen source changed')
                inputs[str(path)]=h
            for i,row in enumerate(fresh['rows']):
                if row['checkpoint_sha256']!=old_audit['rows'][i]['checkpoint_sha256']:raise ValueError('Original state changed')
                inputs[str(f/'states'/f'{i:04d}.npz')]=row['checkpoint_sha256']
            nodes=np.flatnonzero(np.isclose(p.mesh.local_sn[:,1],p.config.height_mm,atol=1e-10))
            gp=out/f'geometry_{a}.npz';np.savez_compressed(gp,reference_xy=p.rest,
                top_nodes=nodes,top_s_from_contact_mm=p.mesh.local_sn[nodes,0]-p.config.drop_s_mm)
            geometry[str(a)]=str(gp);inputs[str(gp)]=digest(gp)
            refs[str(a)]=dict(output=str(f),last_depth_mm=.25,evidence_verified=True,
                audit=str(dest),scope='Only original prefix through .25 freshly reassembled; no new loading solve')
    predecessor=None
    if after is not None:
        after=Path(after).resolve()
        for name in ('manifest.json','launch.json'):
            path=after/name
            if not path.is_file():raise ValueError('Predecessor is not a launched study: '+str(path))
            inputs[str(path)]=digest(path)
        old_manifest=queue.read(after/'manifest.json')
        queue.verify_inputs(old_manifest)
        for name,h in old_manifest['source_sha256'].items():
            if digest(Path(old_manifest['source'])/name)!=h:raise ValueError('Predecessor source changed')
        predecessor=dict(output=str(after),wait_seconds=25200.,poll_seconds=30.,
            policy='Wait for terminal finished queue and all terminal jobs; do not stop or restart predecessor')
    manifest=dict(python=sys.executable,source=str(out/'source'),source_sha256=frozen,input_sha256=inputs,
        runtime_path=str(ROOT/'.runtime/ipc-python'),solve_seconds_per_case=900.,audit_allowance_s=300.,
        references=refs,geometry=geometry,jobs=jobs,angles=list(angles),predecessor=predecessor,
        fresh_pilots=fresh_pilots,created_utc=queue.now(),
        plan='docs/history/topics/cutting_discretization.md',no_production_promotion=True)
    write_json(out/'manifest.json',manifest)
    state=dict(status='prepared',jobs=copy.deepcopy(jobs));write_json(out/'queue_status.json',state)
    report(out,manifest,state)
    return manifest


def wait_for_predecessor(out,m,state):
    """Background scheduling only; never mutate or revive the earlier queue."""
    dependency=m.get('predecessor')
    if not dependency:return True
    queue.verify_inputs(m)
    parent=Path(dependency['output']);began=time.monotonic()
    state.update(status='waiting_for_predecessor',predecessor=str(parent))
    write_json(out/'queue_status.json',state)
    print(f'WAIT predecessor={parent} maximum_seconds={dependency["wait_seconds"]}',flush=True)
    terminal={'completed','failed','budget_exhausted','audit_failed','failed_or_unverified',
        'process_timeout','process_failed','incomplete_or_unverified','skipped_shallow_gate','skipped_pilot_gate'}
    while True:
        try:prior=queue.read(parent/'queue_status.json')
        except (OSError,ValueError):prior={}
        jobs=prior.get('jobs',[])
        if prior.get('status')=='finished' and jobs and all(j.get('status') in terminal for j in jobs):
            queue.verify_inputs(m)
            state.update(status='running',predecessor_finished_observed_utc=queue.now(),
                dependency_wait_elapsed_s=time.monotonic()-began)
            write_json(out/'queue_status.json',state)
            print('PREDECESSOR_FINISHED starting independent new histories',flush=True)
            return True
        if time.monotonic()-began>=dependency['wait_seconds']:
            state.update(status='dependency_timeout',finished_utc=queue.now(),
                dependency_wait_elapsed_s=time.monotonic()-began)
            write_json(out/'queue_status.json',state)
            print('DEPENDENCY_TIMEOUT no new solves launched',flush=True)
            return False
        time.sleep(dependency['poll_seconds'])


def worker(out):
    lock=out/'worker_started.json'
    with lock.open('x',encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(),started_utc=queue.now()),stream)
    m=queue.read(out/'manifest.json');state=dict(status='running',jobs=copy.deepcopy(m['jobs']),started_utc=queue.now())
    if not wait_for_predecessor(out,m,state):
        report(out,m,state);return
    for j in state['jobs']:
        if j['tag']=='pe_base':
            pilot=next((s for s in state['jobs'] if s['tag']=='pilot' and s['angle']==j['angle']),None)
            if pilot and not (pilot.get('mechanics_completed') and pilot.get('evidence_verified')):
                j.update(status='skipped_pilot_gate',mechanics_completed=False)
                write_json(out/'queue_status.json',state);continue
        if j['tag']=='pe_depth':
            gate=depth_gate(m,state['jobs']);write_json(out/'depth_gate.json',gate)
            if not gate['passed']:
                j.update(status='skipped_shallow_gate',mechanics_completed=False)
                write_json(out/'queue_status.json',state);continue
        try:
            j.update(queue.run_case(out,m,j,state))
            evidence=verify_case(j['output'])
            write_json(Path(j['output'])/'evidence_hashes.json',evidence)
            j['evidence_verified']=True
        except Exception as exc:
            j.update(status='failed_or_unverified',evidence_verified=False,mechanics_completed=False,error=repr(exc))
            traceback.print_exc()
        try:report(out,m,state)
        except Exception as exc:
            state.setdefault('report_errors',[]).append(dict(case=j['name'],error=repr(exc)))
            traceback.print_exc()
        write_json(out/'queue_status.json',state)
    state.update(status='finished',finished_utc=queue.now(),
        all_cases_completed=all(j.get('mechanics_completed') for j in state['jobs']))
    try:
        queue.verify_inputs(m)
        for j in state['jobs']:
            if j.get('evidence_verified'):verify_case(j['output'])
        state['terminal_hash_check_passed']=True
    except Exception as exc:state.update(terminal_hash_check_passed=False,terminal_error=repr(exc))
    write_json(out/'queue_status.json',state);report(out,m,state)


def launch(out,m):
    queue.verify_inputs(m)
    for name,h in m['source_sha256'].items():
        if digest(Path(m['source'])/name)!=h:raise RuntimeError('Frozen source changed: '+name)
    if (out/'launch.json').exists() or (out/'worker_started.json').exists():raise RuntimeError('Already launched')
    command=[sys.executable,'-X','utf8','-u','-m','tools.run_cutting_discretization',
             '--output',str(out),'--phase','worker']
    env=os.environ.copy();env['PYTHONPATH']=m['runtime_path'];env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
    startup=None;flags=0
    if os.name=='nt':
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=subprocess.SW_HIDE
        flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        process=subprocess.Popen(command,cwd=m['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            startupinfo=startup,creationflags=flags,start_new_session=os.name!='nt')
    result=dict(worker_pid=process.pid,command=command,started_utc=queue.now())
    write_json(out/'launch.json',result);print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','launch','worker','report'),required=True)
    p.add_argument('--angles',type=int,nargs='+',choices=(0,33,57),default=list(ANGLES))
    p.add_argument('--fresh-pilots',action='store_true')
    p.add_argument('--after',type=Path,help='Wait for this launched queue to finish before new solves')
    a=p.parse_args();out=a.output.resolve()
    if a.phase=='prepare':prepare(out,angles=tuple(a.angles),fresh_pilots=a.fresh_pilots,after=a.after)
    elif a.phase=='launch':launch(out,queue.read(out/'manifest.json'))
    elif a.phase=='worker':worker(out)
    else:report(out,queue.read(out/'manifest.json'),queue.read(out/'queue_status.json'))


if __name__=='__main__':main()
