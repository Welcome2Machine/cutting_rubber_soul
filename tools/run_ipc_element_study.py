"""Frozen 0--1 mm P1/P2-interior comparison; preserves the completed P1 run.

The worker runs one new case then postprocesses its accepted prefix, even on
failure. No polling agent is required. Outputs are local, not publication data.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_support import source_hashes,write_json


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def accepted_prefix(report):
    steps=[]
    for step in report['steps']:
        d=step['depth_mm']
        if d>1+1e-12:break
        if (abs(d-len(steps)*.01)>1e-10 or not np.isfinite(step['reaction_fz_n']) or
            not 0<=step['residual_n']<=1e-4 or not step['min_j']>0 or
            step.get('minimum_distance_mm') is None or step['minimum_distance_mm']<=0 or
            step.get('friction_work_n_mm',0)<-1e-12):
            raise ValueError('Invalid/noncontiguous accepted prefix')
        steps.append(step)
    return steps


def check_pair(a,b,equivalence=None):
    if not isinstance(a['config'],dict) or not isinstance(b['config'],dict):
        raise ValueError('Comparison config must be a mapping')
    ca,cb=dict(a['config']),dict(b['config'])
    ca.setdefault('ogden_alpha',2.);cb.setdefault('ogden_alpha',2.)
    if ca!=cb:raise ValueError('Comparison changed config')
    if a.get('contact_formulation','improved-max')!=b.get('contact_formulation','improved-max'):
        raise ValueError('Comparison changed contact_formulation')
    for key in ('mesh_settings','dhat_mm','barrier_stiffness_mpa',
                'ipc_binary_sha256','loading_increment_mm','force_tolerance_n',
                'friction_solver','friction_solver_settings','hessian_mode'):
        if a[key]!=b[key]:raise ValueError(f'Comparison changed {key}')
    for key in ('tip_radius_mm','nose_facets','fracture','material','friction_smoothing_mm'):
        if a['simplifications'][key]!=b['simplifications'][key]:raise ValueError(f'Changed {key}')
    for name in ('prototype/material.py','mechanics/finite_strain.py','prototype/model_config.py','prototype/initiation.py'):
        old,new=a['source_sha256'][name],b['source_sha256'][name]
        if old!=new:
            allowed=(equivalence or {}).get('sources',{}).get(name,{})
            if not ((equivalence or {}).get('passed') and allowed==dict(baseline=old,current=new)):
                raise ValueError(f'Changed physical source without equivalence check: {name}')


def material_equivalence(baseline, report):
    """Verify explicit legacy parameters and unchanged constitutive responses.

    Shared defaults may change, but geometry code and explicitly reconstructed
    legacy parameter values must agree with saved old source.
    """
    from prototype.material import response,from_config
    from prototype.model_config import FEMConfig
    result=dict(passed=True,sources={},meaning='explicit legacy config and geometry mapping; same constitutive response')
    names=('prototype/material.py','mechanics/finite_strain.py','prototype/model_config.py')
    for name in names:
        old=report['source_sha256'][name];new=digest(ROOT/name)
        if old!=new:
            if digest(baseline/'source'/name)!=old:raise ValueError(f'Original source unavailable: {name}')
            result['sources'][name]=dict(baseline=old,current=new)
    if not result['sources']:return result
    old_path=baseline/'source/mechanics/finite_strain.py'
    if digest(old_path)!=report['source_sha256']['mechanics/finite_strain.py']:
        raise ValueError('Saved constitutive source fingerprint mismatch')
    spec=importlib.util.spec_from_file_location('_element_study_old_material',old_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    adapter_path=baseline/'source/prototype/material.py'
    if digest(adapter_path)!=report['source_sha256']['prototype/material.py']:
        raise ValueError('Saved material adapter fingerprint mismatch')
    adapter_spec=importlib.util.spec_from_file_location('_element_study_old_adapter',adapter_path)
    adapter=importlib.util.module_from_spec(adapter_spec);adapter_spec.loader.exec_module(adapter)
    config=FEMConfig(**report['config'])
    if 'prototype/model_config.py' in result['sources']:
        import ast
        import inspect
        import textwrap
        import prototype.model_config as current_config_module
        config_path=baseline/'source/prototype/model_config.py'
        config_spec=importlib.util.spec_from_file_location('_element_study_old_config',config_path)
        old_config_module=importlib.util.module_from_spec(config_spec)
        sys.modules[config_spec.name]=old_config_module
        config_spec.loader.exec_module(old_config_module)
        supplied=dict(report['config'])
        if supplied.pop('ogden_alpha',2.)!=2.:
            raise ValueError('Legacy material equivalence requires alpha=2')
        old_config=old_config_module.FEMConfig(**supplied)
        current_values=config.to_dict();current_values.pop('ogden_alpha',None)
        if old_config.to_dict()!=current_values:
            raise ValueError('Explicit legacy model configuration changed')
        if old_config.bulk_modulus_n_mm2!=config.bulk_modulus_n_mm2:
            raise ValueError('Legacy bulk modulus mapping changed')
        for name in ('FEMMesh','_rotation_vectors','local_to_lab'):
            old_ast=ast.dump(ast.parse(textwrap.dedent(inspect.getsource(getattr(old_config_module,name)))))
            new_ast=ast.dump(ast.parse(textwrap.dedent(inspect.getsource(getattr(current_config_module,name)))))
            if old_ast!=new_ast:raise ValueError('Geometry implementation changed: '+name)
        result['legacy_config_equivalence']=True
    old_parameters=adapter.from_config(config).to_dict()
    if old_parameters!=from_config(config).to_dict():raise ValueError('Material parameter mapping changed')
    material=module.Material(**old_parameters)
    result['material_parameters']=old_parameters
    rng=np.random.default_rng(61006)
    angle=rng.uniform(-np.pi,np.pi,128);R=np.zeros((128,2,2))
    R[:,0,0]=R[:,1,1]=np.cos(angle);R[:,0,1]=-np.sin(angle);R[:,1,0]=np.sin(angle)
    stretch=np.exp(rng.uniform(np.log(.3),np.log(5.),(128,2)))
    F=R*stretch[:,None,:];F[:,:,1]+=.3*F[:,:,0]
    old=module.elastic_response(F,material,tangent=True,kinematics='plane_strain')
    full=response(F,config,tangent=True);fast=response(F,config,tangent=True,diagnostics=False)
    differences={}
    for key in ('P','energy','J','tangent'):
        np.testing.assert_allclose(old[key],full[key],rtol=2e-13,atol=2e-12)
        np.testing.assert_allclose(old[key],fast[key],rtol=2e-13,atol=2e-12)
        differences[key]=float(np.max(np.abs(old[key]-fast[key])))
    result.update(samples=len(F),max_absolute_difference=differences)
    return result


def prepare(out, baseline, threads, resume=None):
    if out.exists():raise ValueError('Study output must not already exist')
    a=read(baseline/'result.json');prefix=accepted_prefix(a)
    if not a['completed'] or len(prefix)!=101:raise ValueError('Baseline must contain a completed 0--1 mm prefix')
    if a['config']['inclination_deg']!=60 or a['config']['friction_coefficient']!=.35:
        raise ValueError('Expected 60 degree/mu=0.35 baseline')
    if (a['config']['shear_modulus_n_mm2']!=.66 or a['config'].get('ogden_alpha',2.)!=2.
            or a.get('contact_formulation','improved-max')!='improved-max'):
        raise ValueError('Legacy element study requires G=.66, alpha=2, improved-max contact')
    if a['simplifications']['displacement_element']!='P1, fixed mesh':raise ValueError('Expected original P1 baseline')
    if digest(ROOT/'prototype/ipc_baseline.json')!=a['baseline_config_sha256']:
        raise ValueError('Baseline constants changed')
    for name in ('prototype/initiation.py',):
        if digest(ROOT/name)!=a['source_sha256'][name]:raise ValueError(f'Baseline source mismatch {name}')
    equivalence=material_equivalence(baseline,a)
    fingerprints={'result.json':digest(baseline/'result.json')}
    for i in range(101):
        name=f'states/{i:04d}.npz';fingerprints[name]=digest(baseline/name)
    out.mkdir(parents=True)
    frozen=out/'source';hashes=source_hashes(snapshot=frozen)
    for name in ('prototype/ipc_baseline.json','tools/run_ipc_element_study.py','tools/visualize_ipc.py',
                 'tests/test_ipc_p2.py','tests/test_ipc_reference.py','tests/test_ipc_p2_performance.py',
                 'tests/test_material_fast_path.py','tests/test_ipc_paired_assembly.py','tests/test_linear_solve.py',
                 'tests/test_ipc_p2_resume.py'):
        target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        hashes[name]=digest(target)
    m=a['mesh_settings'];s=a['simplifications']
    command=[sys.executable,'-B','-u','-m','prototype.ipc_reference','--case','oblique',
             '--output',str(out/'p2'),'--depth','1','--increment','.01','--friction','.35',
             '--inclination','60','--tip-radius',str(s['tip_radius_mm']),
             '--shear-modulus','.66','--ogden-alpha','2','--contact-formulation','improved-max',
             '--local-spacing',str(m['local_spacing_mm']),'--surface-spacing',str(m['surface_spacing_mm']),
             '--fine-halfwidth',str(m['fine_halfwidth_mm']),'--nose-facets',str(s['nose_facets']),
             '--dhat',str(a['dhat_mm']),'--threads',str(threads),'--seconds','0',
             '--max-iterations','500','--hessian-mode',a['hessian_mode'],
             '--element','p2-interior','--quadrature-order','4','--track-tip-contact','--no-visualize',
             '--linear-solver','reused-lu']
    if resume is not None:command += ['--resume',str(resume)]
    plan=dict(status='prepared',baseline=str(baseline),baseline_sha256=fingerprints,
              source_sha256=hashes,command=command,threads=threads,target_mm=1.,
              material_target_unchanged_mm=3.,runtime_pythonpath=str(ROOT/'.runtime/ipc-python'),
              material_equivalence=equivalence,resume=str(resume) if resume is not None else None)
    write_json(out/'study.json',plan)
    validation_command=[sys.executable,'-B','-m','unittest','tests.test_ipc_p2',
        'tests.test_ipc_p2_performance','tests.test_material_fast_path','tests.test_ipc_paired_assembly',
        'tests.test_linear_solve','tests.test_ipc_p2_resume',
        'tests.test_ipc_reference.DeterminantPathTests','tests.test_ipc_reference.BaselineCommandTests',
        'tests.test_ipc_reference.NativeIPCTests.test_unprojected_hessian_matches_force_derivative',
        'tests.test_ipc_reference.NativeIPCTests.test_coupled_friction_jacobian_matches_fresh_force_difference','-v']
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[key]=str(threads)
    validation=subprocess.run(validation_command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
    log=validation.stdout+'\n'+validation.stderr
    (out/'validation.log').write_text(log,encoding='utf-8')
    count=re.search(r'Ran (\d+) tests?',log)
    passed=validation.returncode==0 and count is not None and 'skipped=' not in log
    write_json(out/'validation.json',dict(passed=passed,returncode=validation.returncode,
        tests_run=int(count[1]) if count else None,command=validation_command,
        source_sha256=hashes,meaning='bounded numerical checks; not completed cutting validation'))
    if not passed:raise RuntimeError('Prelaunch validation failed or skipped; see validation.log')
    for name,value in hashes.items():
        if digest(ROOT/name)!=value:raise ValueError(f'Source changed during validation: {name}')
    return plan


def compare(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=read(out/'study.json');baseline=Path(plan['baseline'])
    for name,expected in plan['baseline_sha256'].items():
        if digest(baseline/name)!=expected:raise ValueError(f'Baseline changed during study: {name}')
    a=read(baseline/'result.json');b=read(out/'p2/result.json');check_pair(a,b,plan.get('material_equivalence'))
    curves=[accepted_prefix(a),accepted_prefix(b)]
    dest=out/'comparison';dest.mkdir(exist_ok=True)
    fields=['element','depth_mm','fz_N','residual_N','min_J','tip_reference_offset_mm']
    with (dest/'fz_depth.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(fields)
        for label,steps in zip(('P1','P2-interior'),curves):
            for st in steps:
                tip=st.get('tip_contact',{})
                writer.writerow([label,st['depth_mm'],st['reaction_fz_n'],st['residual_n'],st['min_j'],
                                 tip.get('reference_offset_mm') if tip.get('status')=='unique' else None])
    summary=dict(p2_status=b['status'],p2_error=b.get('error'),
        numerical_comparison_complete=b.get('completed',False) and len(curves[1])==101,
        interpretation='Interior discretization diagnostic; not a proof of locking-free contact or experimental validation',
        baseline_result_sha256=digest(baseline/'result.json'),p2_result_sha256=digest(out/'p2/result.json'),
        curves=[],fields=[],quadrature_audits=[])
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for label,steps,style in zip(('P1 reference','P2 interior / P1 boundary'),curves,('--','-')):
        depth=np.array([st['depth_mm'] for st in steps]);force=np.array([st['reaction_fz_n'] for st in steps])
        axes[0].plot(depth,force,style,label=f'{label} ({len(steps)}/101 states)')
        axes[1].plot((depth[1:]+depth[:-1])/2,np.diff(force)/np.diff(depth),style,label=label)
        normalized=force/force[-1] if len(steps)==101 else None
        if normalized is not None:axes[2].plot(depth,normalized,style,label=label)
        summary['curves'].append(dict(element=label,accepted_states=len(steps),last_depth_mm=float(depth[-1]) if len(depth) else None,
            half_depth_ratio=float(force[50]/force[100]) if len(steps)==101 else None))
    for ax in axes:
        ax.set_xlabel('Indentation (mm)');ax.set_xlim(0,1);ax.grid(alpha=.25)
    axes[0].set_ylabel('Fz (N)');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Incremental slope (N/mm), unsmoothed')
    axes[2].set_ylabel('Fz / Fz(1 mm), complete curves only')
    fig.suptitle(f'60 deg, mu=0.35; P2 status: {b["status"]}; no fracture')
    fig.savefig(dest/'fz_comparison.png',dpi=180);plt.close(fig)
    if curves[1]:field_comparison(out,baseline,a,b,curves,summary,dest)
    audits=summary['quadrature_audits']
    summary['quadrature_check_passed']=(all(v['energy_relative_difference']<.005 and
        v['bulk_force_difference_over_fz']<.01 for v in audits) if audits else None)
    write_json(dest/'summary.json',summary)


def field_comparison(out,baseline,a,b,curves,summary,dest):
    import matplotlib.pyplot as plt
    from prototype.model_config import FEMConfig,FEMMesh
    from prototype.ipc_p2 import InteriorP2
    from prototype.material import response
    with np.load(baseline/'states/0000.npz',allow_pickle=False) as st:
        n=int(st['body_nodes']);X=st['reference_xy'][:n].copy();tri=st['triangles'].copy()
    c=FEMConfig(**a['config']);angle=np.deg2rad(c.inclination_deg)
    rotation=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    sn=X@rotation
    mesh=FEMMesh(X,sn,tri,np.empty((0,2),int),np.empty(0),np.flatnonzero(abs(sn[:,1])<1e-9),
                 np.empty(0,int),np.zeros(n,int))
    p=InteriorP2(mesh,c,4)
    top=np.flatnonzero(abs(sn[:,1]-c.height_mm)<1e-9);top=top[np.argsort(sn[top,0])]
    visible=(sn[top,0]>=c.drop_s_mm-2)&(sn[top,0]<=c.drop_s_mm+2);top=top[visible]
    all_indices=[i for i in (0,20,50,100) if i<len(curves[1])]
    # Include the last accepted state on failure; clearly identify its depth.
    if len(curves[1])-1 not in all_indices:all_indices.append(len(curves[1])-1)
    for i in all_indices:
        paths=(baseline/'states'/f'{i:04d}.npz',out/'p2/states'/f'{i:04d}.npz')
        with np.load(paths[0],allow_pickle=False) as st:u1=st['positions'][:n]-X
        with np.load(paths[1],allow_pickle=False) as st:
            np.testing.assert_array_equal(st['reference_xy'][:p.mesh.node_count],p.mesh.reference_xy)
            np.testing.assert_array_equal(st['triangles'],tri)
            np.testing.assert_array_equal(st['p2_elements'],p.elements)
            u2=st['positions'][:p.mesh.node_count]-p.mesh.reference_xy
        depth=curves[1][i]['depth_mm']
        diff=u2[:n]-u1
        local=(abs(sn[:,0]-c.drop_s_mm)<.5)&(sn[:,1]>c.height_mm-.5)
        item=dict(depth_mm=depth,state_sha256=[digest(path) for path in paths],
            local_vertex_displacement_difference_max_mm=float(np.linalg.norm(diff[local],axis=1).max()),
            surface_displacement_difference_max_mm=float(np.linalg.norm(diff[top],axis=1).max()))
        F1=np.eye(2)+np.einsum('eai,eaJ->eiJ',u1[tri],mesh.bulk_data[1])
        F2=p.deformation(u2)
        for label,F in (('p1',F1),('p2',F2.reshape(-1,2,2))):
            q=response(F,c,tangent=False)
            item[label]=dict(min_sampled_J=float(q['J'].min()),
                             max_sampled_principal_stretch=float(np.linalg.svd(F,compute_uv=False)[:,0].max()))
        item['p2']['min_J_entire_elements']=p.minimum_j(u2)
        summary['fields'].append(item)
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for label,u,style in (('P1',u1,'--'),('P2 interior',u2,'-')):
            for j in range(2):axes[j].plot(sn[top,0]-c.drop_s_mm,u[top,j],style,label=label)
        for j,ax in enumerate(axes):
            ax.set(xlabel='Reference surface s - initial contact (mm)',ylabel=f'Lab {"x" if j==0 else "z"} displacement (mm)')
            ax.grid(alpha=.25);ax.legend()
        fig.suptitle(f'Same reference surface points, depth={depth:.2f} mm; no smoothing')
        fig.savefig(dest/f'surface_{i:04d}.png',dpi=160);plt.close(fig)
        # Reassembly only, not an additional high-order equilibrium solve.
        force=[];energy=[]
        for order in (4,6):
            rule=p if order==4 else InteriorP2(mesh,c,order)
            r,_,dofs,E,_=rule.bulk(u2,with_tangent=False)
            g=np.zeros(p.mesh.dof_count);np.add.at(g,dofs.ravel(),r.ravel())
            force.append(g);energy.append(E)
            if order==6:del rule
        summary['quadrature_audits'].append(dict(depth_mm=depth,
            energy_relative_difference=abs(energy[1]-energy[0])/max(abs(energy[1]),1e-10),
            bulk_force_difference_over_fz=float(np.linalg.norm(force[1]-force[0])/max(abs(curves[1][i]['reaction_fz_n']),.1)),
            interpretation='same P2 state reassembled with 16 vs 36 points; not a re-equilibrated Fz error estimate'))


def worker(out):
    plan=read(out/'study.json');env=os.environ.copy()
    env['PYTHONPATH']=plan['runtime_pythonpath']
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[key]=str(plan['threads'])
    for name,value in plan['source_sha256'].items():
        if digest(out/'source'/name)!=value:raise ValueError(f'Frozen source changed: {name}')
    plan.update(status='running',pid=os.getpid());write_json(out/'study.json',plan)
    with (out/'solver.stdout.log').open('wb') as stdout,(out/'solver.stderr.log').open('wb') as stderr:
        child=subprocess.Popen(plan['command'],cwd=out/'source',env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
        plan['solver_pid']=child.pid;write_json(out/'study.json',plan)
        code=child.wait()
    plan.update(solver_exit_code=code,status='completed' if code==0 else 'failed')
    try:
        compare(out);plan['postprocessing']='completed'
        plan['numerical_comparison_complete']=read(out/'comparison/summary.json')['numerical_comparison_complete']
    except Exception:
        plan['postprocessing']='failed';plan['postprocessing_error']=traceback.format_exc()
    write_json(out/'study.json',plan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--resume',type=Path,help='Continue an accepted P2 prefix; history and source validation in solver')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--compare-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.worker:return worker(out)
    if args.compare_only:return compare(out)
    if args.baseline is None or args.threads<1:parser.error('Positive thread count and --baseline required')
    prepare(out,args.baseline.resolve(),args.threads,args.resume.resolve() if args.resume else None)
    if not args.detach:return worker(out)
    command=[sys.executable,'-B','-u',str(out/'source/tools/run_ipc_element_study.py'),'--output',str(out),'--worker']
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        child=subprocess.Popen(command,cwd=out/'source',stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
    write_json(out/'launch.json',dict(pid=child.pid,command=command))
    print(json.dumps(dict(pid=child.pid,output=str(out))))


if __name__=='__main__':main()
