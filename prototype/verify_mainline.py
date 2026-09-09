"""Bounded intact indentation, hold and withdrawal checks for the shared kernel.

These are verification paths, not a replacement of the full 3 mm experiment.
"""
import argparse
import contextlib
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
from .model_config import FEMConfig
from .initiation import refine_resolution, make_problem, IntactProblem,InitiationOptions,ContactResolutionError
from .material import from_config
from .run_initiation import snapshot
from run_support import write_json,source_hashes
from .loading import loading_depths
from .contact_study import bounded_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--loading-end',type=float,help='Verify a prefix of the production loading schedule, without hold/retraction')
    parser.add_argument('--case',choices=('normal_frictionless','oblique_frictionless','oblique_frictional'))
    parser.add_argument('--contact-discretization',choices=('nodal','integral','macro','quadratic_nodes','quadratic_integral'),default='quadratic_integral')
    parser.add_argument('--mesh-grading',choices=('legacy','graded'),default='graded')
    parser.add_argument('--volumetric-discretization',choices=('element','nodal_volume','macro_mixed','taylor_hood','quadratic_full'),default='quadratic_full')
    parser.add_argument('--volume-stabilization-shear',type=float,default=0.,help='Experimental shear-scale volume-variance penalty; zero retains the comparison model')
    parser.add_argument('--surface-spacing',type=float,default=.0025)
    parser.add_argument('--max-increment',type=float,help='Bounded diagnostic increment; experiment target is unchanged')
    parser.add_argument('--tolerance',type=float,help='Set force (N) and projection (MPa) tolerances for a bounded sensitivity check')
    args=parser.parse_args();root=args.output
    options=InitiationOptions(contact_discretization=args.contact_discretization,mesh_grading=args.mesh_grading,surface_spacing_mm=args.surface_spacing,volumetric_discretization=args.volumetric_discretization,volume_stabilization_shear=args.volume_stabilization_shear)
    if root.exists() and any(root.iterdir()):raise ValueError('Use a new empty validation directory')
    root.mkdir(parents=True,exist_ok=True)
    base=FEMConfig.from_json(Path(__file__).parent/'initiation_config.json')
    if args.tolerance is not None:
        if not np.isfinite(args.tolerance) or args.tolerance<=0:parser.error('Tolerance must be finite and positive')
        base=replace(base,newton_tolerance_n=args.tolerance,contact_multiplier_tolerance_n_mm2=args.tolerance)
    if args.loading_end is not None and not 0<args.loading_end<=base.max_indentation_mm:
        parser.error('Loading end must lie in the prescribed experiment')
    report=dict(scope='Bounded intact-body verification; full experiment target remains 3 mm',
        source_sha256=source_hashes(),cases=[],all_passed=False)
    for name,angle,friction in [('normal_frictionless',0.,0.),('oblique_frictionless',60.,0.),('oblique_frictional',60.,.35)]:
        if args.case and args.case!=name:continue
        out=root/name;out.mkdir();config=replace(base,inclination_deg=angle,friction_coefficient=friction)
        if args.loading_end is not None:
            p=make_problem(config,options)
            targets=[depth for depth,_ in loading_depths(config,args.max_increment or p.options.surface_spacing_mm/2,stop_mm=args.loading_end)]
            if not targets or abs(targets[-1]-args.loading_end)>1e-12:
                parser.error('Loading end must be a scheduled target; changing the last step would change the diagnostic path')
            result=bounded_path(p,np.zeros(p.mesh.dof_count),p.initial_contact_state(),0.,targets,out,adaptive=True)
            result['name']=name;report['cases'].append(result);write_json(root/'validation.json',report)
            continue
        result=dict(name=name,config=config.to_dict(),material=from_config(config).to_dict(),completed=False,steps=[])
        report['cases'].append(result);write_json(root/'validation.json',report)
        started=time.monotonic()
        with (out/'run.log').open('w') as log,contextlib.redirect_stdout(log):
            try:
                p=make_problem(config,options);u=np.zeros(p.mesh.dof_count);state=p.initial_contact_state();old=0.;seconds=0.
                for depth in (.005,.01,.02,.025,.025,.015,.005,0.):
                    dt=max(abs(depth-old)/config.speed_mm_s,.02)
                    for refinement in range(65):
                        previous=u.copy();old_state=state.copy()
                        try:
                            u,state,diagnostics=p.step(previous,old,depth,dt,old_state)
                            diagnostics['spatial_refinements']=refinement
                            break
                        except ContactResolutionError as exc:
                            if refinement==64:raise
                            u,state=refine_resolution(p,previous,old_state,exc)
                            print('REFINE',refinement+1,len(exc.edges),flush=True)
                    r,_,contact,_=p.assemble(u,previous,depth,old,dt,state[:,0],state[:,1],roundoff_guard=False)
                    projection=np.max(np.abs(np.column_stack((contact['pressure'],contact['shear']))-state))
                    residual=np.linalg.norm(r[p.free]);penetration=p.edge_penetration(u,depth)
                    if residual>config.newton_tolerance_n or projection>config.contact_multiplier_tolerance_n_mm2 or penetration>config.contact_penetration_tolerance_mm:
                        raise RuntimeError('Original equilibrium/contact verification failed')
                    if np.min(state[:,0])< -1e-10 or np.any(np.abs(state[:,1])>friction*np.maximum(0,state[:,0])+1e-8):
                        raise RuntimeError('Coulomb admissibility failed')
                    if diagnostics['friction_dissipation_n_mm'] < -diagnostics['friction_dissipation_tolerance_n_mm']:
                        raise RuntimeError('Negative contact dissipation')
                    seconds+=dt;old=depth
                    result['steps'].append(dict(depth_mm=depth,time_s=seconds,**diagnostics,
                        checked_residual_n=float(residual),checked_projection_mpa=float(projection),checked_edge_penetration_mm=float(penetration)))
                    snapshot(out/'last_state.npz',p,u,state,p.fields(u),depth,seconds)
                    write_json(root/'validation.json',report)
                    print('ACCEPT',depth,'Fz',diagnostics['reaction_fz_n'],'Fx',diagnostics['reaction_fx_n'],flush=True)
                result['completed']=True
            except Exception as error:
                result['error']=repr(error);print('FAILED',repr(error),flush=True)
            result['wall_seconds']=time.monotonic()-started
        print(name,'passed' if result['completed'] else result['error'],flush=True)
    report['all_passed']=all(case['completed'] for case in report['cases']);write_json(root/'validation.json',report)
    if not report['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
