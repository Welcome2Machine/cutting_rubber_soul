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


def bounded_path(p,u,state,old,targets,out,*,adaptive):
    targets=np.asarray(targets,dtype=float)
    tolerance=8*np.finfo(float).eps*max(1.,abs(old),np.max(abs(targets),initial=0.))
    if targets.ndim!=1 or not len(targets) or not np.all(np.isfinite(targets)) or np.any(np.diff(np.r_[old,targets])<=tolerance):
        raise ValueError('Bounded loading targets must be finite and strictly exceed the preceding depth')
    started=time.monotonic();rows=[];refinements=0;subdivisions=0;attempted_depth=old
    report=dict(completed=False,scope='bounded numerical diagnostic; not full experiment',
        pid=os.getpid(),source_sha256=source_hashes(),
        start_depth_mm=old,target_depth_mm=float(targets[-1]),initial_mesh=mesh_metrics(p),
        config=p.config.to_dict(),options=vars(p.options),steps=rows)
    write_json(out/'result.json',report)
    save_state(out/'last_state.npz',p,u,state,old)
    def advance(depth):
        nonlocal u,state,old,refinements,subdivisions,attempted_depth
        attempted_depth=depth
        try:
            new,lam,d=p.step(u,old,depth,(depth-old)/p.config.speed_mm_s,state)
        except ContactResolutionError as exc:
            if not adaptive or refinements>=p.config.max_adaptive_substeps:
                raise
            u,state=refine_resolution(p,u,state,exc);refinements+=1
            print('REFINE',refinements,'boundary',len(exc.edges),'bulk',len(getattr(exc,'elements',[])),str(exc),flush=True)
            return advance(depth)
        except (RuntimeError,FloatingPointError):
            if not adaptive or (depth-old)/2 < p.config.min_adaptive_increment_mm or subdivisions>=64:
                raise
            subdivisions+=1
            advance((old+depth)/2)
            return advance(depth)
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
        final_mesh=mesh_metrics(p),refinements=refinements,subdivisions=subdivisions)
    save_state(out/'last_state.npz',p,u,state,old)
    write_json(out/'result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ('completed','last_depth_mm','elapsed_s','error')}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=('coarse_path','fine_path','base_path','replay_suite','continuation_suite','recovery'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--formulation',choices=('alart_curnier','jean_moreau'))
    parser.add_argument('--source',type=Path,default=Path('outputs/current_failure'))
    parser.add_argument('--loading-end',type=float,default=.135,help='End of bounded recovery diagnostic; does not change experiment target')
    parser.add_argument('--max-increment',type=float,default=.0003125,help='Maximum loading increment for recovery diagnostic')
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('Study output must be empty')
    out.mkdir(parents=True,exist_ok=True)
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
        bounded_path(p,u,state,old,targets,out,adaptive=True)
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
