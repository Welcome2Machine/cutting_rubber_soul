"""Endpoint traction guard for the separate extrinsic P1 face route.

A discrete-gradient average reaction is an impulse-consistent force, not the
instantaneous endpoint traction. An accepted connected state must not silently
cross the strength surface before the next physical increment starts.
"""
import numpy as np
from mechanics.discrete_gradient import frozen_response
from release_dynamics import contact_velocity, instantaneous_friction
from prototype.zero_gap_cut import EnergyResolutionRequired


def outgoing_force(p):
    weights = p.rule['weights_mm2'] * (~p.active_faces)[:, None]
    local = np.einsum('eq,qi->ei', weights, p.rule['shape']) * p.strength
    force = np.zeros(len(p.seam))
    force[:-1] += local[:, 0]
    force[1:] += local[:, 1]
    return force


def endpoint_forces(p, x):
    """Instantaneous relaxed-endpoint fields, without committing any history.

    The candidate accepted state contains the Lie-relaxed Maxwell metric. Use
    that metric and the endpoint velocity, retaining native friction geometry.
    Tied x DOFs have zero acceleration; their connection demand is minus the
    physical force gradient. Other DOFs still require the full momentum solve.
    """
    metric = p.material_response(x)['metric']
    q = frozen_response(p.deformation(x), metric, p.config.shear_modulus_n_mm2,
                        p.material.mu_ov[0], p.config.bulk_modulus_n_mm2, tangent=False)
    area, grad, dofs = p.mesh.bulk_data
    local = np.einsum('e,eiJ,eaJ->eai', area*p.config.width_mm, q['P'], grad)
    bulk = np.zeros(p.total)
    np.add.at(bulk, dofs.ravel(), local.ravel())
    cohesive = np.zeros(p.total)
    cohesive[2*p.seam] = p.face_force(p.cohesive_response(x))[0]
    normal = np.zeros(p.total)
    normal[p.surface_dofs] = p.barrier.gradient(p.collisions(x), p.cmesh, x[p.surface])
    friction = instantaneous_friction(p, x, contact_velocity(p, p.endpoint_velocity(x)))
    return dict(bulk=bulk, cohesive=cohesive, normal=normal, friction=friction)


def check_connected_endpoint(p, x):
    """Reject a crossing in THIS step; never fix it by shrinking its successor.

    Same original half-domain force tolerance as the activation handoff; this
    is event localization, not a change to strength or the dynamical residual.
    No artificial relaxation, mass, work or fracture-energy payment is added.
    """
    fields = endpoint_forces(p, x)
    excess = -sum(fields.values())[2*p.seam] - outgoing_force(p)
    tied = (~p.released) & (p.rest[p.seam, 1] > 0)
    maximum = float(np.max(excess[tied], initial=0.))
    if maximum > p.force_tolerance_n:
        raise EnergyResolutionRequired(
            f'Instantaneous endpoint traction event unresolved: excess {maximum:.17g} N; '
            'refine current physical step before committing history')
    return dict(maximum_connected_endpoint_excess_half_n=maximum,
                endpoint_strength_check='relaxed Maxwell metric and native endpoint velocity friction')
