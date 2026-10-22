"""CB245NE plane-strain adapter. No unmeasured Maxwell branch is invented."""
import numpy as np
from mechanics.finite_strain import Material,elastic_response


def from_config(config):
    # The shared kernel uses G=sum(mu_i*alpha_i)/2. Preserve the measured
    # small-strain G and K when varying the large-strain shape, not mu_i.
    # alpha=2 exactly retains the previous neo-Hookean law.
    alpha = config.ogden_alpha
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Ogden exponent must be finite and positive')
    return Material(mu_eq=(2*config.shear_modulus_n_mm2/alpha,),alpha_eq=(alpha,),
                    poisson=config.poisson_ratio,volumetric='quadratic_j')


def response(F,config,*,tangent=True,diagnostics=True):
    return elastic_response(F,from_config(config),kinematics='plane_strain',tangent=tangent,
                            diagnostics=diagnostics)


def bulk(mesh,u,config,*,with_tangent=True):
    areas,gradients,dofs=mesh.bulk_data
    F=np.eye(2)+np.einsum('eai,eaJ->eiJ',u.reshape(-1,2)[mesh.triangles],gradients)
    # P1 assembly consumes only P, energy, J and (optionally) the tangent.
    # Keep public response() diagnostics available for stress/field outputs,
    # without constructing unused Cauchy tensors and eigensystems per trial.
    q=response(F,config,tangent=with_tangent,diagnostics=False);volumes=areas*config.width_mm
    r=np.einsum('e,eiJ,eaJ->eai',volumes,q['P'],gradients).reshape(-1,6)
    k=None
    if with_tangent:
        C=q['tangent'].reshape(-1,2,2,2,2)
        k=np.einsum('e,eaJ,eiJkL,ebL->eaibk',volumes,gradients,C,gradients,optimize=True).reshape(-1,6,6)
    return r,k,dofs,float(volumes@q['energy']),float(q['J'].min())
