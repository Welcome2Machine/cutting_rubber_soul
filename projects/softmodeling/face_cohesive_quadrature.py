"""Independent P1 face discretization of the existing scalar extrinsic law.

This kernel is NOT the production nodal-release model. Histories live at fixed
face quadrature points; they cannot be transplanted from nodal checkpoints.
Full crack area is counted once. Returned histories remain uncommitted.
"""
import numpy as np
from prototype.zero_gap_cut import extrinsic_response


def face_rule(reference_y, width, order=2):
    y=np.asarray(reference_y,float)
    if y.ndim!=1 or len(y)<2 or not np.isfinite(y).all() or np.any(np.diff(y)<=0):
        raise ValueError('Strictly increasing finite reference seam coordinates required')
    if not np.isfinite(width) or width<=0 or type(order) is not int or not 2<=order<=32:
        raise ValueError('Positive width and fixed quadrature order 2..32 required')
    z,w=np.polynomial.legendre.leggauss(order)
    shape=np.column_stack(((1-z)/2,(1+z)/2))
    area=np.diff(y)*width;weights=area[:,None]*w/2
    tributary=np.zeros(len(y));tributary[:-1]+=area/2;tributary[1:]+=area/2
    return dict(shape=shape,weights_mm2=weights,face_area_mm2=area,
        nodal_tributary_area_mm2=tributary,history_shape=weights.shape)


def response(opening, history, rule, strength, toughness, released_faces=None):
    opening=np.asarray(opening,float);old=np.asarray(history,float)
    shape,weights=rule['shape'],rule['weights_mm2']
    if opening.shape!=(len(weights)+1,) or old.shape!=weights.shape:
        raise ValueError('Face quadrature history identity differs; nodal history cannot be reused')
    if not np.isfinite(opening).all() or np.any(opening < -1e-12):
        raise ValueError('Nonnegative finite physical opening required')
    enabled=np.ones(len(weights),bool) if released_faces is None else np.asarray(released_faces)
    if enabled.dtype!=bool or enabled.shape!=(len(weights),):raise ValueError('Explicit face mask required')
    delta=opening[:-1,None]*shape[None,:,0]+opening[1:,None]*shape[None,:,1]
    q=extrinsic_response(delta,old,strength,toughness)
    prior=extrinsic_response(np.zeros_like(old),old,strength,toughness)
    weighted=weights*enabled[:,None]
    local=np.einsum('eq,eq,qi->ei',weighted,q['traction'],shape)
    tangent=np.einsum('eq,eq,qi,qj->eij',weighted,q['tangent'],shape,shape)
    force=np.zeros_like(opening);force[:-1]+=local[:,0];force[1:]+=local[:,1]
    hessian=np.zeros((len(opening),len(opening)))
    for e,k in enumerate(tangent):hessian[e:e+2,e:e+2]+=k
    history_trial=np.where(enabled[:,None],q['history'],old)
    return dict(force_wrt_full_opening_n=force,tangent_wrt_full_opening_n_mm=hessian,
        incremental_potential_n_mm=float(np.sum(weighted*(q['stored']+q['dissipated']-prior['dissipated']))),
        stored_n_mm=float(np.sum(weighted*q['stored'])),dissipated_n_mm=float(np.sum(weighted*q['dissipated'])),
        candidate_history=history_trial,fully_failed_faces=enabled & np.all(q['history']>=2*toughness/strength,axis=1),
        finite_crack_area_mm2=float(rule['face_area_mm2'][enabled].sum()),
        required_fracture_work_n_mm=float(toughness*rule['face_area_mm2'][enabled].sum()),
        physical_history_committed=False)
