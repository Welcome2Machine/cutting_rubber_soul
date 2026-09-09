"""Shared loading targets computed from global integer indices.

Accepted adaptive substeps never change the next scheduled target. A bounded
diagnostic takes a prefix of this schedule without rescaling the experiment.
"""
import numpy as np


def predictor_checkpoint(problem,depth):
    """Save only a loading predictor belonging to this accepted depth."""
    velocity=getattr(problem,'_accepted_velocity',None)
    if velocity is None or getattr(problem,'_velocity_depth',None)!=depth:return {}
    return dict(solver_predictor_version=1,accepted_loading_derivative=velocity,
        predictor_depth_mm=depth,predictor_loading=problem._velocity_loading)


def restore_predictor(problem,archive,depth):
    if 'solver_predictor_version' not in archive:return
    velocity=archive['accepted_loading_derivative']
    if (int(archive['solver_predictor_version'])!=1 or float(archive['predictor_depth_mm'])!=depth
            or velocity.shape!=(problem.mesh.dof_count,) or not np.all(np.isfinite(velocity))):
        raise ValueError('Invalid accepted loading predictor checkpoint')
    problem._accepted_velocity=velocity.copy()
    problem._velocity_depth=depth;problem._velocity_loading=bool(archive['predictor_loading'])


def recovery_targets(config,max_increment_mm,start_mm,stop_mm,failed_mm):
    """Merge a saved failure into the integer schedule without ULP-size steps."""
    tolerance=8*np.finfo(float).eps*max(1.,abs(start_mm),abs(stop_mm),abs(failed_mm))
    targets=[d for d,_ in loading_depths(config,max_increment_mm,stop_mm=stop_mm) if d-start_mm>tolerance]
    if failed_mm-start_mm>tolerance and not any(abs(d-failed_mm)<=tolerance for d in targets):
        targets.append(failed_mm)
    return sorted(targets)


def loading_depths(config,max_increment_mm,*,stop_mm=None,reverse=False):
    if not np.isfinite(max_increment_mm) or max_increment_mm<=0:
        raise ValueError('Maximum loading increment must be positive and finite')
    planned=(config.pullback_steps if reverse else config.approach_steps)-1
    ratio=config.max_indentation_mm/planned/max_increment_mm
    parts=max(1,int(np.ceil(ratio-32*np.finfo(float).eps*max(1.,ratio))))
    total=planned*parts
    increment=config.max_indentation_mm/total
    for i in range(1,total+1):
        depth=0. if reverse and i==total else (
            config.max_indentation_mm if i==total else i*increment)
        if reverse and i!=total:depth=config.max_indentation_mm-depth
        # Keep the scheduled float unchanged, but do not lose its last point
        # because decimal stop_mm rounds below i*increment (e.g. 0.35 mm).
        if stop_mm is not None and depth-stop_mm>8*np.finfo(float).eps*max(1.,abs(depth),abs(stop_mm)):break
        yield depth,(i+parts-1)//parts


def resolved_loading_steps(config,options):
    yield 'approach',0.,0
    for depth,index in loading_depths(config,options.surface_spacing_mm/2):
        yield 'approach',depth,index
    for i in range(config.hold_steps if config.hold_s>0 else 0):
        yield 'hold',config.max_indentation_mm,config.approach_steps+i
    offset=config.approach_steps+(config.hold_steps if config.hold_s>0 else 0)-1
    for depth,index in loading_depths(config,options.surface_spacing_mm/2,reverse=True):
        yield 'pullback',depth,offset+index
