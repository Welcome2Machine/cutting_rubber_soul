"""Run full uncracked loading cycle with auditable initiation candidate events."""
from __future__ import annotations
import argparse
import csv
import json
import time
import traceback
from pathlib import Path
import numpy as np
from run_support import write_json,save_npz,source_hashes
from .model_config import FEMConfig
from .loading import resolved_loading_steps,predictor_checkpoint
from .initiation import refine_resolution, make_problem, InitiationOptions, IntactProblem, ContactResolutionError, ContactSolveError
from .fracture_criteria import StrengthSurface
from .material import from_config
from .initiation_event import CandidateRefinement,CandidateReached,criterion_values,location_evidence,plot_location


def snapshot(path,problem,u,multipliers,fields,depth,seconds):
    from .macro_element import checkpoint
    save_npz(path,reference_xy=problem.mesh.reference_xy,
             local_sn=problem.mesh.local_sn,
             triangles=problem.mesh.triangles,displacement=u,
             boundary_edges=problem.edges,contact_multipliers=multipliers,
             contact_nodes=problem.contact_nodes,contact_state_version=2,
             contact_discretization=problem.options.contact_discretization,
             volumetric_discretization=problem.options.volumetric_discretization,
             contact_reference_area_mm2=problem.weights,
             indentation_mm=depth,time_s=seconds,**checkpoint(problem.mesh),**predictor_checkpoint(problem,depth),**fields)


def run(config,options,out,strength_surface=None,*,mode='cycle',candidate_criterion='local_nominal',candidate_tolerance_mm=.0001):
    if mode not in ('cycle','to_candidate'):raise ValueError('Unknown run mode')
    if candidate_criterion=='multiaxial' and strength_surface is None:raise ValueError('Multiaxial stopping requires explicit strength calibration')
    if candidate_criterion not in ('local_nominal','multiaxial') or not np.isfinite(candidate_tolerance_mm) or candidate_tolerance_mm<=0:
        raise ValueError('Invalid candidate criterion or bracket tolerance')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output directory must be empty')
    write_json(out/'provenance.json',dict(source_sha256=source_hashes(snapshot=out/'source_snapshot')))
    write_json(out/'config.json',dict(material_and_experiment=config.to_dict(),initiation=vars(options),
        run_mode=mode,candidate_criterion=candidate_criterion,candidate_bracket_tolerance_mm=candidate_tolerance_mm,
        bulk_material=from_config(config).to_dict(),kinematics='plane_strain: F33=1',
        bulk_parameter_status='Elastic limit only; CB245NE nonlinear Ogden and Maxwell parameters uncalibrated',
        contact_discretization=options.contact_discretization,
        volumetric_discretization=options.volumetric_discretization,
        contact_integration=(
            'P2 displacement, positive P1 traction on half edges, four Gauss points per half edge'
            if options.contact_discretization=='quadratic_integral' else
            'P2 displacement, collocated vertex/midpoint traction, positive Lobatto area'
            if options.contact_discretization=='quadratic_nodes' else
            'P1 traction, quadratic P2 displacement, four Gauss points per edge'
            if options.contact_discretization=='macro' and options.volumetric_discretization in ('taylor_hood','quadratic_full') else
            'coarse P1 traction, refined P1 displacement, four Gauss points per subedge'
            if options.contact_discretization=='macro' else
            'P1 weighted nodal gap/slip, four Gauss points per edge'
            if options.contact_discretization=='integral' else 'lumped reference nodal area'),
        friction_model='rate-independent Coulomb normal/shear projection equations',
        loading_schedule='global integer substep indices; common diagnostic/production generator',
        contact_roundoff_policy='zero-gap iteration branch within 32*eps*coordinate_scale; final acceptance uses raw gap',
        contact_branch_fallback=dict(enabled=options.contact_discretization=='nodal' or options.volumetric_discretization=='quadratic_full',
            enumerated_nodes=6 if options.volumetric_discretization=='quadratic_full' else 9,
            maximum_patch_nodes=None if options.volumetric_discretization=='quadratic_full' else 32,
            combination_budget=4096 if options.volumetric_discretization=='quadratic_full' else 262144,maximum_predictors=8,
            eligibility='mixed force residual <= 10 * configured force tolerance',
            acceptance='original nonlinear equations, raw gap, cone, edge and dissipation checks'),
        contact_state_columns=['normal_multiplier_mpa','convected_shear_mpa'],
        contact_traction_measure='force per reference boundary area',
        strength_surface=strength_surface.to_dict() if strength_surface else None))
    started = time.time()
    status = dict(completed=False,state='initializing',model='uncracked initiation screening',
                  run_mode=mode,candidate_found=False,physical_nucleation_validated=False,
                  post_initiation_behavior='intact counterfactual; no crack propagation',
                  multiaxial_strength_status=(strength_surface.parameter_status if strength_surface else 'unavailable: no calibration supplied'))
    write_json(out/'status.json',status)
    stream = None
    try:
        problem = make_problem(config,options,strength_surface)
        print(f"mesh nodes={problem.mesh.node_count} elements={len(problem.mesh.triangles)}; no internal seams",flush=True)
        u = np.zeros(problem.mesh.dof_count)
        multipliers = problem.initial_contact_state()
        old_depth,seconds = 0.,0.
        events,rows = {},[]
        attempt = {}
        stream = (out/'results.csv').open('w',newline='',encoding='utf-8')
        writer = None

        def accept(depth,dt,phase,requested_step,subdivision):
            nonlocal u,multipliers,old_depth,seconds,writer
            print(f"SOLVE step={requested_step} phase={phase} depth={depth:.8g} dt={dt:.8g} subdivision={subdivision}",flush=True)
            attempt.update(depth_mm=depth,dt_s=dt,phase=phase,requested_step=requested_step,subdivision=subdivision)
            previous_seconds,previous_depth = seconds,old_depth
            state,lam,diagnostics = problem.step(u,old_depth,depth,dt,multipliers)
            fields = problem.fields(state)
            if mode=='to_candidate':
                ratio=float(criterion_values(fields,candidate_criterion,config.tensile_strength_n_mm2).max())
                if ratio>=1 and abs(depth-old_depth)>candidate_tolerance_mm:
                    raise CandidateRefinement()
                if ratio>=1:
                    snapshot(out/'before_candidate.npz',problem,u,multipliers,problem.fields(u),old_depth,seconds)
            row = dict(accepted_step=len(rows),requested_step=requested_step,phase=phase,
                       time_s=seconds+dt,indentation_mm=depth,**diagnostics)
            measures = {'local_nominal':fields['nominal_tensile_mpa'],
                        'averaged_nominal':fields['averaged_nominal_tensile_mpa'],
                        'cauchy':fields['cauchy_principal_mpa'].max(axis=1),
                        'mean_cauchy':fields['cauchy_mean_tensile_stress_mpa'],
                        'stretch':fields['principal_stretches'].max(axis=1)}
            if strength_surface is not None:
                measures['multiaxial'] = fields['multiaxial_strength_index']
                measures['averaged_multiaxial'] = fields['averaged_multiaxial_strength_index']
            for name,values in measures.items():
                index = int(values.argmax())
                row[name+'_max'] = float(values[index])
                element=int(fields['field_element_ids'][index]) if 'field_element_ids' in fields else index
                row[name+'_element'] = element
                row[name+'_sample'] = index
                for coordinate,key in [('reference_xz','reference_centroid_xz_mm'),('current_xz','current_centroid_xz_mm'),('reference_sn','reference_centroid_sn_mm')]:
                    for axis,value in zip(coordinate.split('_')[-1],fields[key][index]):
                        row[f'{name}_{coordinate}_{axis}_mm'] = float(value)
                if name in ('local_nominal','averaged_nominal','multiaxial','averaged_multiaxial'):
                    threshold = 1. if 'multiaxial' in name else config.tensile_strength_n_mm2
                    ratio = float(values[index]/threshold)
                    row[name+'_strength_ratio'] = ratio
                    if ratio >= 1 and name not in events:
                        events[name] = dict(kind='strength-threshold candidate, not observed fracture',
                            time_bracket_s=[previous_seconds,seconds+dt],
                            indentation_bracket_mm=[previous_depth,depth],phase=phase,
                            element=element,sample=index,ratio=ratio,
                            criterion=name,
                            crack_direction_status='not predicted; tensile direction is a stress diagnostic only',
                            reference_sn_mm=fields['reference_centroid_sn_mm'][index].tolist(),
                            current_xz_mm=fields['current_centroid_xz_mm'][index].tolist(),
                            tensile_direction_xzy=fields['tensile_direction_xzy'][index].tolist())
                        snapshot(out/f'first_{name}_candidate.npz',problem,state,lam,fields,depth,seconds+dt)
                        print('INITIATION CANDIDATE '+name+' '+json.dumps(events[name]),flush=True)
            row['after_local_candidate'] = 'local_nominal' in events
            row['after_multiaxial_candidate'] = 'multiaxial' in events
            if writer is None:
                writer = csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); stream.flush()
            u,multipliers,old_depth,seconds = state,lam,depth,seconds+dt
            rows.append(row)
            snapshot(out/'last_state.npz',problem,u,multipliers,fields,depth,seconds)
            if depth >= config.max_indentation_mm:
                snapshot(out/'max_depth_state.npz',problem,u,multipliers,fields,depth,seconds)
            status.update(state='running',accepted_steps=len(rows),time_s=seconds,
                          indentation_mm=depth,wall_seconds=time.time()-started,events=events)
            write_json(out/'status.json',status)
            write_json(out/'initiation_candidates.json',events)
            print(f"ACCEPT t={seconds:.6g} depth={depth:.6g} Fz={row['reaction_fz_n']:.6g} local_ratio={row['local_nominal_strength_ratio']:.6g} average_ratio={row['averaged_nominal_strength_ratio']:.6g}",flush=True)
            if mode=='to_candidate' and candidate_criterion in events:
                evidence=location_evidence(problem,fields,depth,candidate_criterion)
                evidence.update(time_bracket_s=events[candidate_criterion]['time_bracket_s'],
                    indentation_bracket_mm=events[candidate_criterion]['indentation_bracket_mm'])
                write_json(out/'first_candidate_location.json',evidence)
                plot_location(out/'first_candidate_location.png',problem,state,fields,evidence)
                status.update(candidate_found=True,stop_reason='first_strength_threshold_candidate',
                              candidate_criterion=candidate_criterion)
                raise CandidateReached()

        subdivisions,refinements = 0,0
        def advance(depth,dt,phase,index,level=0):
            nonlocal subdivisions,refinements,u,multipliers
            try:
                accept(depth,dt,phase,index,level)
            except CandidateRefinement:
                middle=(depth+old_depth)/2
                print(f'BRACKET candidate trial rejected; bisect depth={middle:.9g}',flush=True)
                advance(middle,dt/2,phase,index,level+1)
                advance(depth,dt/2,phase,index,level+1)
            except ContactResolutionError as exc:
                if refinements >= config.max_adaptive_substeps:
                    raise
                refinements += 1
                u,multipliers = refine_resolution(problem,u,multipliers,exc)
                print(f"REFINE depth={old_depth:.8g} boundary_edges={len(exc.edges)} bulk_elements={len(getattr(exc,'elements',[]))} nodes={problem.mesh.node_count}; {exc}; retry same physical increment",flush=True)
                advance(depth,dt,phase,index,level)
            except (RuntimeError,FloatingPointError) as exc:
                if (phase == 'hold' or abs(depth-old_depth)/2 < config.min_adaptive_increment_mm
                        or subdivisions >= config.max_adaptive_substeps):
                    raise
                subdivisions += 1
                middle = (depth+old_depth)/2
                print(f"RETRY {exc}; bisect depth={middle:.8g}",flush=True)
                advance(middle,dt/2,phase,index,level+1)
                advance(depth,dt/2,phase,index,level+1)
        for phase,depth,index in resolved_loading_steps(config,options):
            if mode=='to_candidate' and phase!='approach':break
            dt=(config.hold_s/config.hold_steps if phase=='hold' else
                abs(depth-old_depth)/(config.speed_mm_s if phase=='approach' else config.pullback_speed_mm_s))
            advance(depth,max(dt,1e-9),phase,index)
        status.update(completed=True,state='completed',wall_seconds=time.time()-started)
        if mode=='to_candidate':status.update(stop_reason='loading_limit_without_candidate')
        write_json(out/'status.json',status)
        print('COMPLETE '+('loading limit reached without candidate' if mode=='to_candidate' else 'full prescribed loading/hold/withdrawal cycle (uncracked counterfactual)'),flush=True)
    except CandidateReached:
        status.update(completed=True,state='candidate_found',wall_seconds=time.time()-started)
        write_json(out/'status.json',status)
        print('STOP first strength-threshold candidate saved; no hold or withdrawal',flush=True)
    except Exception as exc:
        status.update(state='failed',error=str(exc),wall_seconds=time.time()-started)
        if isinstance(exc,ContactSolveError):
            from .macro_element import checkpoint
            # The accepted output may use the mesh before an adaptive refinement.
            # Keep the actual failing mesh and trial separately, never as success.
            np.savez_compressed(out/'failed_attempt.npz',reference_xy=problem.mesh.reference_xy,
                local_sn=problem.mesh.local_sn,triangles=problem.mesh.triangles,
                boundary_edges=problem.edges,contact_nodes=problem.contact_nodes,
                contact_reference_area_mm2=problem.weights,contact_state_version=2,
                contact_discretization=problem.options.contact_discretization,
                volumetric_discretization=problem.options.volumetric_discretization,
                accepted_displacement=u,accepted_contact_multipliers=multipliers,
                trial_displacement=exc.u,trial_contact_multipliers=exc.multipliers,
                last_accepted_depth_mm=old_depth,last_accepted_time_s=seconds,**checkpoint(problem.mesh),**predictor_checkpoint(problem,old_depth))
            status['failed_attempt'] = attempt
        write_json(out/'status.json',status)
        traceback.print_exc()
        raise
    finally:
        if stream is not None:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='prototype/initiation_config.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--tip-radius',type=float,default=.02)
    parser.add_argument('--refinement',type=float,default=.15)
    parser.add_argument('--surface-spacing',type=float,default=.0025)
    parser.add_argument('--contact-discretization',choices=('nodal','integral','macro','quadratic_nodes','quadratic_integral'),default='quadratic_integral')
    parser.add_argument('--mesh-grading',choices=('legacy','graded'),default='graded')
    parser.add_argument('--volumetric-discretization',choices=('element','nodal_volume','macro_mixed','taylor_hood','quadratic_full'),default='quadratic_full')
    parser.add_argument('--volume-stabilization-shear',type=float,default=0.,help='Experimental volume-variance penalty in multiples of G')
    parser.add_argument('--averaging-radius',type=float,default=.3)
    parser.add_argument('--tangential-stiffness',type=float,default=250.,help='Tangential augmentation in MPa/mm; does not set physical stick compliance')
    parser.add_argument('--coulomb-formulation',choices=('alart_curnier','jean_moreau'),default='alart_curnier',help='Equivalent contact projection systems; jean_moreau is an experimental solver comparison')
    parser.add_argument('--strength-calibration',help='Explicit second-Piola multiaxial strength JSON; omitted when unmeasured')
    parser.add_argument('--mode',choices=('cycle','to_candidate'),default='to_candidate')
    parser.add_argument('--candidate-criterion',choices=('local_nominal','multiaxial'),default='local_nominal')
    parser.add_argument('--candidate-tolerance',type=float,default=.0001,help='First threshold bracket width in mm, not spatial accuracy')
    args = parser.parse_args()
    surface = StrengthSurface.from_json(args.strength_calibration) if args.strength_calibration else None
    run(FEMConfig.from_json(args.config),InitiationOptions(args.tip_radius,args.refinement,args.surface_spacing,args.averaging_radius,args.tangential_stiffness,args.coulomb_formulation,args.contact_discretization,args.mesh_grading,args.volumetric_discretization,args.volume_stabilization_shear),Path(args.output),surface,
        mode=args.mode,candidate_criterion=args.candidate_criterion,candidate_tolerance_mm=args.candidate_tolerance)


if __name__ == '__main__':
    main()
