"""Bounded contact investigations, kept separate from the 3 mm experiment.

Examples:
  python -m prototype.contact_study --case coarse_path --output outputs/study_coarse
  python -m prototype.contact_study --case fine_path --output outputs/study_fine
  python -m prototype.contact_study --case replay_suite --output outputs/study_replay
"""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import numpy as np
from run_support import write_json,save_npz,source_hashes
from .model_config import FEMConfig,FEMMesh
from .initiation import refine_resolution, IntactProblem,InitiationOptions,ContactResolutionError,ContactSolveError
from .loading import loading_depths,recovery_targets,predictor_checkpoint,restore_predictor


def load_failed_problem(directory,config=None,options=None):
    bounded=not (directory/'config.json').exists()
    metadata=json.loads((directory/('result.json' if bounded else 'config.json')).read_text(encoding='utf-8'))
    p=IntactProblem.__new__(IntactProblem)
    p.config=config or FEMConfig(**metadata['config' if bounded else 'material_and_experiment'])
    saved_options=dict(metadata['options' if bounded else 'initiation'])
    # Retired exploratory pointwise projection gate; no material-law effect.
    saved_options.pop('volume_projection_tolerance_mpa',None)
    saved_options.pop('max_element_j_variation',None)
    p.options=options or InitiationOptions(**saved_options)
    if p.options.contact_quadrature_order!=saved_options.get('contact_quadrature_order',4):
        raise ValueError('Changing contact quadrature requires a new loading path')
    if p.options.volumetric_discretization in ('taylor_hood','quadratic_full'):
        from .quadratic import QuadraticProblem
        p.__class__=QuadraticProblem
    p.strength_surface=None
    p.config.validate();p.options.validate(p.config)
    with np.load(directory/('last_state.npz' if bounded else 'failed_attempt.npz')) as z:
        if int(z['contact_state_version']) != 2:
            raise ValueError('This study requires unique-node contact state version 2')
        saved_discretization=str(z['contact_discretization']) if 'contact_discretization' in z else metadata.get('options' if bounded else 'initiation',{}).get('contact_discretization','nodal')
        if saved_discretization!=p.options.contact_discretization:
            raise ValueError('Changing contact discretization requires a new loading path, not checkpoint continuation')
        saved_volume=metadata.get('options' if bounded else 'initiation',{}).get('volumetric_discretization','element')
        if saved_volume!=p.options.volumetric_discretization:
            raise ValueError('Changing volume discretization requires a new loading path')
        saved_stabilization=metadata.get('options' if bounded else 'initiation',{}).get('volume_stabilization_shear',0.)
        if saved_stabilization!=p.options.volume_stabilization_shear:
            raise ValueError('Changing volume stabilization requires a new loading path')
        xy,sn,tri=z['reference_xy'],z['local_sn'],z['triangles']
        p.mesh=FEMMesh(xy,sn,tri,np.empty((0,2),int),np.empty(0),
            np.flatnonzero(np.isclose(sn[:,1],0)),np.empty(0,int),np.zeros(len(xy),int))
        from .macro_element import restore
        restore(p.mesh,z)
        if p.options.volumetric_discretization=='macro_mixed' and not hasattr(p.mesh,'macro_triangles'):
            raise ValueError('Macro checkpoint is missing its pressure mesh')
        p._rebuild_boundary(z['boundary_edges'])
        u,state=z['displacement' if bounded else 'accepted_displacement'],z['contact_multipliers' if bounded else 'accepted_contact_multipliers']
        old=float(z['indentation_mm' if bounded else 'last_accepted_depth_mm'])
        restore_predictor(p,z,old)
    return p,u,state,old


def mesh_metrics(p):
    xy=p.mesh.reference_xy[p.mesh.triangles]
    v=np.roll(xy,-1,axis=1)-xy
    lengths=np.linalg.norm(v,axis=2)
    cos=np.sum(v*(-np.roll(v,1,axis=1)),axis=2)/(lengths*np.roll(lengths,1,axis=1))
    angles=np.rad2deg(np.arccos(np.clip(cos,-1,1)))
    return dict(nodes=p.mesh.node_count,elements=len(xy),
        min_angle_deg=float(angles.min()),max_angle_deg=float(angles.max()),
        min_area_mm2=float(p.mesh.bulk_data[0].min()))


def save_state(path,p,u,state,depth):
    from .macro_element import checkpoint
    save_npz(path,reference_xy=p.mesh.reference_xy,local_sn=p.mesh.local_sn,
        triangles=p.mesh.triangles,boundary_edges=p.edges,contact_nodes=p.contact_nodes,
        contact_reference_area_mm2=p.weights,contact_state_version=2,
        contact_discretization=p.options.contact_discretization,
        volumetric_discretization=p.options.volumetric_discretization,
        displacement=u,contact_multipliers=state,indentation_mm=depth,**checkpoint(p.mesh),**predictor_checkpoint(p,depth))


def mesh_fingerprint(p):
    import hashlib
    digest=hashlib.sha256()
    for array in (p.mesh.reference_xy,p.mesh.triangles,p.edges,getattr(p.mesh,'elements',np.empty(0))):
        value=np.ascontiguousarray(array);digest.update(str((value.shape,value.dtype)).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def flip_local_edges(base,boundary,config):
    """Flip disjoint convex interior pairs; preserve vertices and boundary.

    This is a controlled topology comparison, not adaptive mesh repair.
    Reject pairs with a new minimum angle below 25 degrees.
    """
    triangles=base.triangles.copy();lookup={}
    for index,tri in enumerate(triangles):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            lookup.setdefault(tuple(sorted((int(a),int(b)))),[]).append(index)
    used=set();flipped=[];xy=base.reference_xy
    def orient(tri):
        a,b,c=xy[tri];v,w=b-a,c-a;return v[0]*w[1]-v[1]*w[0]
    if any(len(lookup.get(tuple(sorted(edge)),[]))!=1 for edge in boundary):
        raise ValueError('Boundary does not match the reference triangulation')
    for edge,owners in sorted(lookup.items()):
        if len(owners)!=2 or any(i in used for i in owners):continue
        sn=base.local_sn[list(edge)].mean(axis=0)
        if not (config.drop_s_mm-.3<=sn[0]<=config.drop_s_mm+.6 and sn[1]>=config.height_mm-.3):continue
        i,j=owners;a,b=edge
        c=next(int(v) for v in triangles[i] if v not in edge)
        d=next(int(v) for v in triangles[j] if v not in edge)
        if tuple(sorted((c,d))) in lookup:continue
        if orient([c,d,a])*orient([c,d,b])>=0:continue
        proposed=np.array([[c,d,a],[d,c,b]])
        for tri in proposed:
            if orient(tri)<0:tri[1],tri[2]=tri[2],tri[1]
        points=xy[proposed];v=np.roll(points,-1,axis=1)-points;length=np.linalg.norm(v,axis=2)
        angles=np.rad2deg(np.arccos(np.clip(np.sum(v*(-np.roll(v,1,axis=1)),axis=2)/(length*np.roll(length,1,axis=1)),-1,1)))
        if angles.min()<25.:continue
        triangles[[i,j]]=proposed;used.update(owners);flipped.append([a,b,c,d])
    if not flipped:raise ValueError('No admissible local edge flips found')
    return replace(base,triangles=triangles),flipped


def zero_history_variant(source,order,flip=False):
    """Use saved reference geometry only; never transfer loaded fields."""
    from .quadratic import quadratic_mesh
    p,_,_,_=load_failed_problem(source)
    if p.options.volumetric_discretization!='quadratic_full' or p.options.contact_discretization!='quadratic_integral':
        raise ValueError('Discretization comparison requires full P2 and quadratic integral contact')
    changes=[]
    if flip:
        count=p.mesh.pressure_node_count
        base=FEMMesh(p.mesh.reference_xy[:count].copy(),p.mesh.local_sn[:count].copy(),
            p.mesh.triangles.copy(),np.empty((0,2),int),np.empty(0),
            np.flatnonzero(np.isclose(p.mesh.local_sn[:count,1],0)),np.empty(0,int),np.zeros(count,int))
        base,changes=flip_local_edges(base,p.edges,p.config)
        p.mesh=quadratic_mesh(base,p.edges)
    p.options=replace(p.options,contact_quadrature_order=order)
    p.options.validate(p.config);p._rebuild_boundary(p.edges.copy())
    for name in ('_accepted_velocity','_velocity_depth','_velocity_loading','_spatial_predictor'):
        if hasattr(p,name):delattr(p,name)
    return p,changes


def discretization_suite(source,out,end,increment,seconds):
    """Sequential fixed-mesh paths from zero; all physical gates unchanged."""
    summary=dict(status='running',scope='fixed-mesh discretization comparison, not full cutting experiment',
        source=str(source),pid=os.getpid(),runs=[],source_sha256=source_hashes(snapshot=out/'source_snapshot'))
    write_json(out/'suite.json',summary)
    for name,order,flip in [('q4',4,False),('q8',8,False),('q16',16,False),('flipped_q16',16,True)]:
        folder=out/name;folder.mkdir()
        try:
            p,changes=zero_history_variant(source,order,flip)
            write_json(folder/'provenance.json',dict(initialization='zero displacement and zero contact history',
                reference_geometry_source=str(source),mesh_sha256=mesh_fingerprint(p),flipped_edges=changes,
                quadrature_order_per_half_edge=order,wall_budget_s=seconds))
            targets=[d for d,_ in loading_depths(p.config,increment,stop_mm=end)]
            result=bounded_path(p,np.zeros(p.mesh.dof_count),p.initial_contact_state(),0.,targets,folder,
                adaptive=True,refine_mesh=False,record_history=True,time_limit_s=seconds)
            row=dict(name=name,status=result['status'],last_depth_mm=result['last_depth_mm'],
                error=result.get('error'),result=str(folder/'result.json'),audits=[])
            summary['runs'].append(row);write_json(out/'suite.json',summary)
            if result['steps']:
                from .audit_contact import audit
                depths=np.array([step['depth_mm'] for step in result['steps']])
                indices={len(depths)-1}
                for checkpoint in (.45,.49):
                    before=np.flatnonzero(depths<=checkpoint)
                    if len(before) and depths[-1]>=checkpoint:indices.add(int(before[-1]))
                for index in sorted(indices):
                    try:
                        report=audit(folder,increment=folder/'increments'/f'{index:05d}.npz',modes=3)
                        write_json(folder/f'audit_{index:05d}.json',report)
                        row['audits'].append(dict(index=index,depth_mm=float(depths[index]),
                            verification=report['original_increment_verification'],
                            negative_search_status=report['coupled_modes']['negative_search_status'],
                            negative_search=report['coupled_modes']['negative_branch_search']))
                    except Exception as exc:row['audits'].append(dict(index=index,error=repr(exc)))
        except Exception as exc:
            summary['runs'].append(dict(name=name,status='setup_failed',error=repr(exc)))
        write_json(out/'suite.json',summary)
    summary['status']='completed' if all(r['status']=='completed' for r in summary['runs']) else 'finished_with_incomplete_paths'
    write_json(out/'suite.json',summary)


def bounded_path(p,u,state,old,targets,out,*,adaptive,refine_mesh=True,record_history=False,time_limit_s=None):
    targets=np.asarray(targets,dtype=float)
    tolerance=8*np.finfo(float).eps*max(1.,abs(old),np.max(abs(targets),initial=0.))
    if targets.ndim!=1 or not len(targets) or not np.all(np.isfinite(targets)) or np.any(np.diff(np.r_[old,targets])<=tolerance):
        raise ValueError('Bounded loading targets must be finite and strictly exceed the preceding depth')
    if record_history and refine_mesh:raise ValueError('Recorded increment audits require a fixed mesh')
    if time_limit_s is not None and (not np.isfinite(time_limit_s) or time_limit_s<=0):raise ValueError('Invalid wall budget')
    started=time.monotonic();rows=[];refinements=0;subdivisions=0;attempted_depth=old
    report=dict(completed=False,scope='bounded numerical diagnostic; not full experiment',
        pid=os.getpid(),source_sha256=source_hashes(),
        start_depth_mm=old,target_depth_mm=float(targets[-1]),initial_mesh=mesh_metrics(p),
        config=p.config.to_dict(),options=vars(p.options),steps=rows,
        status='running',refine_mesh=refine_mesh,record_history=record_history)
    write_json(out/'result.json',report)
    save_state(out/'last_state.npz',p,u,state,old)
    def advance(depth):
        nonlocal u,state,old,refinements,subdivisions,attempted_depth
        attempted_depth=depth
        if time_limit_s is not None and time.monotonic()-started>=time_limit_s:
            raise TimeoutError('Bounded wall budget exhausted between increments')
        try:
            new,lam,d=p.step(u,old,depth,(depth-old)/p.config.speed_mm_s,state)
        except ContactResolutionError as exc:
            if not adaptive or not refine_mesh or refinements>=p.config.max_adaptive_substeps:
                raise
            u,state=refine_resolution(p,u,state,exc);refinements+=1
            print('REFINE',refinements,'boundary',len(exc.edges),'bulk',len(getattr(exc,'elements',[])),str(exc),flush=True)
            return advance(depth)
        except TimeoutError:raise
        except (RuntimeError,FloatingPointError):
            if not adaptive or (depth-old)/2 < p.config.min_adaptive_increment_mm or subdivisions>=64:
                raise
            subdivisions+=1
            advance((old+depth)/2)
            return advance(depth)
        if record_history:
            directory=out/'increments';directory.mkdir(exist_ok=True)
            save_npz(directory/f'{len(rows):05d}.npz',
                displacement=new,contact_multipliers=lam,previous_displacement=u,
                previous_multipliers=state,old_depth_mm=old,depth_mm=depth,
                dt_s=(depth-old)/p.config.speed_mm_s,mesh_sha256=mesh_fingerprint(p),
                config_json=json.dumps(p.config.to_dict()),options_json=json.dumps(vars(p.options)))
        u,state,old=new,lam,depth
        save_state(out/'last_state.npz',p,u,state,old)
        scalar_normal=p.weights*state[:,0]
        d['max_node_normal_load_fraction']=float(scalar_normal.max(initial=0)/max(scalar_normal.sum(),1e-30))
        rows.append(dict(depth_mm=depth,**d))
        report.update(last_depth_mm=old,refinements=refinements,subdivisions=subdivisions)
        write_json(out/'result.json',report)
        print('ACCEPT',depth,d['reaction_fz_n'],d['residual_n'],flush=True)
    with (out/'run.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        try:
            for depth in targets:advance(float(depth))
            report['completed']=True
        except Exception as exc:
            report.update(error=str(exc),error_type=type(exc).__name__,failed_depth_mm=attempted_depth)
            print('FAILED',repr(exc),flush=True)
            if isinstance(exc,ContactSolveError):
                save_state(out/'failed_trial.npz',p,exc.u,exc.multipliers,attempted_depth)
    report.update(last_depth_mm=old,elapsed_s=time.monotonic()-started,
        status='completed' if report['completed'] else 'failed',
        final_mesh=mesh_metrics(p),refinements=refinements,subdivisions=subdivisions)
    save_state(out/'last_state.npz',p,u,state,old)
    if report.get('error_type')=='TimeoutError':report['status']='budget_exhausted'
    write_json(out/'result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ('completed','last_depth_mm','elapsed_s','error')}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=('coarse_path','fine_path','base_path','replay_suite','continuation_suite','recovery','discretization_suite'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--formulation',choices=('alart_curnier','jean_moreau'))
    parser.add_argument('--source',type=Path,default=Path('outputs/current_failure'))
    parser.add_argument('--loading-end',type=float,default=.135,help='End of bounded recovery diagnostic; does not change experiment target')
    parser.add_argument('--max-increment',type=float,default=.0003125,help='Maximum loading increment for recovery diagnostic')
    parser.add_argument('--detach',action='store_true',help='Run a suite or recovery in a hidden background worker')
    parser.add_argument('--fixed-mesh',action='store_true',help='Recovery: prohibit remeshing and save actual increment histories')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--case-seconds',type=float,default=1200.,help='Per-path wall budget checked between increments')
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('Study output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    if args.fixed_mesh and args.case!='recovery':parser.error('--fixed-mesh requires recovery')
    if args.case in ('discretization_suite','recovery'):
        if not 0<args.loading_end<=3 or not np.isfinite(args.max_increment) or args.max_increment<=0 or args.threads<1 or not np.isfinite(args.case_seconds) or args.case_seconds<=0:
            parser.error('Invalid suite end, increment, thread count or budget')
        if args.detach:
            import subprocess,sys
            command=[sys.executable,'-u','-m','prototype.contact_study','--case',args.case,'--source',str(args.source.resolve()),
                '--output',str(out.resolve()),'--loading-end',str(args.loading_end),'--max-increment',str(args.max_increment),
                '--case-seconds',str(args.case_seconds),'--threads',str(args.threads)]
            if args.fixed_mesh:command.append('--fixed-mesh')
            if args.formulation:command.extend(['--formulation',args.formulation])
            env=os.environ.copy()
            for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]=str(args.threads)
            startup=None;flags=0
            if os.name=='nt':
                startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=subprocess.SW_HIDE
                flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
            # Worker owns the result directory; launcher logs are beside it.
            with out.with_suffix('.stdout.log').open('wb') as log,out.with_suffix('.stderr.log').open('wb') as error:
                process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=error,startupinfo=startup,creationflags=flags,start_new_session=os.name!='nt')
            record=dict(pid=process.pid,command=command,threads=args.threads,output=str(out.resolve()))
            write_json(out.with_suffix('.launch.json'),record);print(json.dumps(record));return
        if args.case=='discretization_suite':
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=args.threads):discretization_suite(args.source,out,args.loading_end,args.max_increment,args.case_seconds)
            return
    if args.detach:parser.error('--detach is supported for discretization_suite and recovery only')
    bounded_source=not (args.source/'config.json').exists()
    metadata=json.loads((args.source/('result.json' if bounded_source else 'config.json')).read_text(encoding='utf-8'))
    config=FEMConfig(**metadata['config' if bounded_source else 'material_and_experiment'])
    saved_options=dict(metadata['options' if bounded_source else 'initiation'])
    saved_options.pop('volume_projection_tolerance_mpa',None)
    saved_options.pop('max_element_j_variation',None)
    options=InitiationOptions(**saved_options)
    if bounded_source and args.case!='recovery':parser.error('A bounded checkpoint is supported by --case recovery')
    if args.formulation:
        options=replace(options,coulomb_formulation=args.formulation)
    if args.case=='recovery':
        diagnostic_end=args.loading_end
        if not 0<diagnostic_end<=config.max_indentation_mm or not np.isfinite(args.max_increment) or args.max_increment<=0:
            raise ValueError('Recovery diagnostic requires a positive increment and an end within the experiment')
    elif args.case.endswith('_path'):
        diagnostic_end=.125
    elif args.case=='continuation_suite':
        diagnostic_end=.13
    else:
        diagnostic_end=json.loads((args.source/'status.json').read_text(encoding='utf-8'))['failed_attempt']['depth_mm']
    write_json(out/'provenance.json',dict(source_run=str(args.source),case=args.case,
        source_sha256=source_hashes(snapshot=out/'source_snapshot'),
        experiment_target_mm=config.max_indentation_mm,diagnostic_end_mm=diagnostic_end))
    if args.case=='recovery':
        p,u,state,old=load_failed_problem(args.source,config,options)
        failed=(metadata.get('failed_depth_mm',min(old+args.max_increment,diagnostic_end)) if bounded_source else
                json.loads((args.source/'status.json').read_text(encoding='utf-8'))['failed_attempt']['depth_mm'])
        targets=recovery_targets(config,args.max_increment,old,diagnostic_end,failed)
        if diagnostic_end<failed:raise ValueError('Recovery end must reach the saved failed increment')
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            bounded_path(p,u,state,old,targets,out,adaptive=True,refine_mesh=not args.fixed_mesh,
                         record_history=args.fixed_mesh,time_limit_s=args.case_seconds)
    elif args.case.endswith('_path'):
        spacing={'coarse_path':.005,'fine_path':.00125,'base_path':.0025}[args.case]
        p=IntactProblem(config,replace(options,surface_spacing_mm=spacing))
        targets=[depth for depth,_ in loading_depths(config,.00125,stop_mm=diagnostic_end)]
        bounded_path(p,np.zeros(p.mesh.dof_count),p.initial_contact_state(),0.,targets,out,adaptive=True)
    else:
        cases=[('base',{}, {},1),('half_steps',{}, {},2),('quarter_steps',{}, {},4),
            ('normal_25',dict(contact_surface_penalty_n_mm3=25.),{},1),
            ('normal_2500',dict(contact_surface_penalty_n_mm3=2500.),{},1),
            ('tangent_25',{},dict(tangential_stiffness_mpa_per_mm=25.),1),
            ('tangent_2500',{},dict(tangential_stiffness_mpa_per_mm=2500.),1),
            ('backtrack_24',dict(max_line_search_iterations=24),{},1)]
        continuation=args.case=='continuation_suite'
        if continuation:
            config=replace(config,min_adaptive_increment_mm=.0000390625)
            cases=[('quarter_step',{}, {},64),
                ('normal_2500',dict(contact_surface_penalty_n_mm3=2500.),{},16),
                ('tangent_25',{},dict(tangential_stiffness_mpa_per_mm=25.),16)]
        target=diagnostic_end
        summary=[];seen={}
        for name,cc,oo,count in cases:
            case_config,case_options=replace(config,**cc),replace(options,**oo)
            signature=json.dumps(dict(config=case_config.to_dict(),options=vars(case_options),count=count),sort_keys=True)
            if signature in seen:
                summary.append(dict(case=name,duplicate_of=seen[signature],executed=False))
                write_json(out/'summary.json',summary)
                continue
            seen[signature]=name
            case_out=out/name;case_out.mkdir()
            p,u,state,old=load_failed_problem(args.source,case_config,case_options)
            if target<=old:raise ValueError('Diagnostic target must exceed the accepted source depth')
            result=bounded_path(p,u,state,old,np.linspace(old,target,count+1)[1:],case_out,adaptive=continuation)
            summary.append(dict(case=name,**{k:result[k] for k in ('completed','last_depth_mm','elapsed_s')},error=result.get('error')))
            write_json(out/'summary.json',summary)


if __name__=='__main__':main()
