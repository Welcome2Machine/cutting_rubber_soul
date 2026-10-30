"""Explicit endpoint-BE reference for the complete conditional contact momentum.

No friction splitting, added damping, changed mass, or new initiation rule.
Original active-set constraints, CCD, force/energy gates and Maxwell Lie update
remain in the host. Numerical endpoint-work and kinetic losses stay separate.
"""
import types
import numpy as np
from prototype.ipc_reference import IPCProblem
from prototype.energy_cut import EnergyCutProblem
from prototype.finite_increment_cut import FrozenVirtualCrack


def endpoint_normal(self, x, *, tangent=False, projected=False):
    collisions = self.collisions(x)
    projection = self.ipc.PSDProjectionMethod.CLAMP if projected else self.ipc.PSDProjectionMethod.NONE
    g = self.barrier.gradient(collisions, self.cmesh, x[self.surface])
    h = self.barrier.hessian(collisions, self.cmesh, x[self.surface], projection) if tangent else None
    return g, h


def endpoint_cohesive(self, x):
    return self.cohesive_response(x)


def endpoint_acceptance(self, x, depth, iterations, outer_iterations):
    rec = EnergyCutProblem.acceptance(self, x, depth, iterations, outer_iterations)
    rec['reaction_sampling'] = 'backward Euler endpoint force; impulse is h times this force'
    rec['time_integrator'] = 'endpoint_backward_Euler_Lie_Maxwell'
    rec['numerical_inertia_dissipation'] = 'sum(m*dv^2) full reflected body; numerical, not physical friction'
    return rec


def install(p, method):
    """Install only after verifying the incoming state with its original method."""
    if method not in ('midpoint', 'endpoint_be'):
        raise ValueError('Unknown conditional momentum integrator')
    if getattr(p, 'conditional_integrator', method) != method:
        raise ValueError('Reconstruct an independent object before changing integrator')
    p.conditional_integrator = method
    if method == 'endpoint_be':
        p.kinematic_theta = 1.
        # FrozenVirtualCrack supplies only the original instantaneous bulk law;
        # no virtual geometry, history, or equilibrium solver is instantiated.
        methods = dict(material_bulk=FrozenVirtualCrack.material_bulk,
            lag=IPCProblem.lag, coupled_friction_tangent=IPCProblem.coupled_friction_tangent,
            normal_response=endpoint_normal, cohesive_algorithmic=endpoint_cohesive,
            acceptance=endpoint_acceptance)
        for name, function in methods.items():
            setattr(p, name, types.MethodType(function, p))
    return p


def impulse_ledger(p, x):
    """Half-domain vector balance plus physical reflected scalar work.

Reaction impulses are retained at all exact constrained DOFs, including ties,
bottom supports, blade and active unilateral nodes. They are not external
material energy. Free-DOF defects are NOT absorbed into a reaction.
"""
    h = p.material_dt
    du = p.step_increment(x)
    fields = {}
    r, _, dofs, _, _ = p.material_bulk((x[:p.n]-p.mesh.reference_xy).ravel(), with_tangent=False)
    bulk = np.zeros(p.total); np.add.at(bulk, dofs.ravel(), r.ravel())
    cohesive = np.zeros(p.total)
    cohesive[2*p.seam] = p.seam_area*p.released*p.cohesive_algorithmic(x)['traction']
    normal = np.zeros(p.total); normal[p.surface_dofs] = p.normal_response(x)[0]
    friction = np.zeros(p.total)
    lag = p.lag(x)
    if lag is not None:
        friction[p.surface_dofs] = p.friction.gradient(lag, p.cmesh, du[p.surface])
    physical = bulk+cohesive+normal+friction
    dv = (p.endpoint_velocity(x)-p.previous_velocity).ravel()
    momentum = p.mass_dof*dv
    dynamic = momentum/h+physical
    constraint = np.zeros(p.total)
    constrained = np.setdiff1d(np.arange(p.total), p.free)
    constraint[constrained] = h*dynamic[constrained]
    for name, g in (('bulk', bulk), ('cohesive', cohesive), ('normal', normal), ('friction', friction)):
        fields['impulse_'+name] = (-h*g).reshape(-1, 2)
    fields['impulse_constraint'] = constraint.reshape(-1, 2)
    fields['momentum_change'] = momentum.reshape(-1, 2)
    defect = momentum+h*physical-constraint
    fields['impulse_balance_defect'] = defect.reshape(-1, 2)
    tol = 1e-4
    force_defect = float(np.sqrt(2)*np.linalg.norm(defect[p.free])/h)
    if force_defect > tol:
        raise RuntimeError('Independent momentum impulse balance exceeds original force gate')
    fixed_body = constrained[constrained < 2*p.n]
    work_constraint = 2*float(constraint[fixed_body]@du.ravel()[fixed_body])/h
    summary = dict(free_impulse_residual_n_s=float(np.sqrt(2)*np.linalg.norm(defect[p.free])),
        equivalent_force_residual_n=force_defect,
        blade_normal_impulse_z_n_s=float(-2*h*normal.reshape(-1,2)[p.n:,1].sum()),
        blade_friction_impulse_z_n_s=float(-2*h*friction.reshape(-1,2)[p.n:,1].sum()),
        blade_work_n_mm=2*float(physical.reshape(-1,2)[p.n:].ravel()@du[p.n:].ravel()),
        body_constraint_work_n_mm=work_constraint,
        numerical_kinetic_loss_n_mm=(2*p.kinematic_theta-1)*float(p.mass_dof@(dv*dv)),
        vector_scope='half domain; blade vertical scalars and work reflected to full body')
    return fields, summary
