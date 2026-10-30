"""Conditional real-time cohesive release, NOT a replacement initiation rule.

Reuse the original mass, momentum, constitutive evolution and acceptance gates.
The sole imposed event is activation of specified virgin cohesive nodes. No
auxiliary load, static re-equilibration, virtual chord or upfront Gc debit.
"""
import types
import numpy as np
from prototype.ipc_reference import IPCProblem
from prototype.zero_gap_cut import EnergyResolutionRequired

DEFECT_KEYS = ('backward_euler_inertia_loss_n_mm', 'bulk_endpoint_defect_n_mm',
    'cohesive_endpoint_defect_n_mm', 'barrier_endpoint_defect_n_mm',
    'contact_topology_defect_n_mm', 'hard_contact_endpoint_loss_n_mm')


def check_energy_gate(p, energy):
    """The production mechanical-reservoir gate, also applied to the input trial."""
    if p.energy_normalization != 'mechanical_reservoir' or p.energy_tolerance is None:
        raise ValueError('Conditional release requires an explicit reservoir energy gate')
    defect = sum(abs(energy[k]) for k in DEFECT_KEYS)+abs(energy.get('friction_time_defect_n_mm', 0.))
    allowance = p.energy_absolute_tolerance+p.energy_tolerance*energy['mechanical_reservoir_n_mm']
    if defect > allowance:
        raise EnergyResolutionRequired(f'Input trial energy defect {defect:g} exceeds {allowance:g}')
    return dict(defect_n_mm=defect, allowance_n_mm=allowance, passed=True)


def contact_velocity(p, body_velocity):
    """Stored blade velocity is zero for mass accounting; restore its prescribed motion."""
    v = body_velocity.copy()
    v[p.n:] = (0., -p.loading_rate)
    v[p.mesh.bottom_nodes] = 0.
    return v


def instantaneous_friction(p, x, velocity, scale=1.):
    """Native regularized velocity law at x, independent of any position anchor.

    Scaling velocity and smoothing together verifies the equivalent incremental
    law without subtracting nearly equal global position coordinates.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite friction scale required')
    lag = IPCProblem.lag(p, x)
    force = np.zeros(p.total)
    if lag is not None:
        potential = p.ipc.FrictionPotential(p.friction_velocity_epsilon*scale)
        force[p.surface_dofs] = potential.gradient(lag, p.cmesh, scale*velocity[p.surface])
    return force


def activate(p, x, candidates):
    """One conditional graph event with explicit topology energy/force accounting."""
    ids = np.asarray(candidates, dtype=int)
    if (ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or np.any(ids < 0)
            or np.any(ids >= len(p.seam)) or np.any(p.released[ids])):
        raise ValueError('Candidates must be distinct virgin seam indices')
    if np.any(p.history[ids] != 0) or np.any(x[p.seam[ids], 0] != p.config.drop_s_mm):
        raise ValueError('Activation must start at exact zero opening and virgin history')
    v = contact_velocity(p, p.velocity)
    before_force = instantaneous_friction(p, x, v)
    before_normal = np.zeros(p.total)
    before_normal[p.surface_dofs] = p.barrier.gradient(p.collisions(x), p.cmesh, x[p.surface])
    before_barrier = 2*float(p.barrier(p.collisions(x), p.cmesh, x[p.surface]))
    before_edges = p.surface[np.asarray(p.cmesh.edges)].copy()
    old_release, old_active = p.released.copy(), p.active_contact.copy()
    try:
        p.released[ids] = True
        p.rebuild_contact()
        p.set_free()
        after_force = instantaneous_friction(p, x, v)
        after_normal = np.zeros(p.total)
        after_normal[p.surface_dofs] = p.barrier.gradient(p.collisions(x), p.cmesh, x[p.surface])
        after_barrier = 2*float(p.barrier(p.collisions(x), p.cmesh, x[p.surface]))
        # This event has zero motion and no external/auxiliary work. Reject a
        # finite barrier insertion energy, rather than silently resetting it.
        jump = after_barrier-before_barrier
        if abs(jump) > 1e-10:
            raise ValueError(f'Unfunded contact-topology energy jump: {jump:g} N mm')
    except BaseException:
        p.released, p.active_contact = old_release, old_active
        p.rebuild_contact(); p.set_free()
        raise
    scale_checks = []
    for h in (1e-2, 1e-4, 1e-6, 1e-8):
        difference = instantaneous_friction(p, x, v, h)-after_force
        scale_checks.append(dict(scale_s=h, full_force_difference_n=float(np.sqrt(2)*np.linalg.norm(difference))))
    # Existing damage remains enabled; only additional initiation is excluded
    # from this fixed-candidate experiment. Each worker owns its problem.
    p.propose_insertion = types.MethodType(lambda self, *a: np.zeros_like(self.released), p)
    return dict(seam_indices=ids.tolist(), nodes=p.seam[ids].tolist(),
        candidate_area_mm2=float(p.seam_area[ids].sum()),
        candidate_gc_capacity_n_mm=float(p.toughness*p.seam_area[ids].sum()),
        upfront_fracture_debit_n_mm=0., auxiliary_work_n_mm=0.,
        barrier_topology_jump_n_mm=jump,
        normal_topology_force_jump_n=float(np.sqrt(2)*np.linalg.norm(after_normal-before_normal)),
        friction_topology_force_jump_n=float(np.sqrt(2)*np.linalg.norm(after_force-before_force)),
        instantaneous_friction_force_norm_n=float(np.sqrt(2)*np.linalg.norm(after_force)),
        friction_scale_checks=scale_checks,
        boundary_edges_added=len(p.cmesh.edges)-len(before_edges),
        friction_state='current contact geometry and inherited endpoint relative velocity; no static-memory state',
        inherited_velocities_unchanged=True, inherited_material_and_damage_unchanged=True,
        additional_initiation_disabled=True, cohesive_damage_enabled=p.damage_enabled)


def independent_friction_work(p, start, end, increment, deadline):
    """Integrate the accepted temporal chord with GL3/7/15, not midpoint force.

    This probes time-discretization error; it is NOT a new virtual crack path.
    The primary chord uses its constant segment velocity and original velocity
    smoothing. Endpoint-velocity time rules can differ; that is temporal error,
    not quadrature uncertainty. Extended-precision relative interpolation avoids cancellation.
    """
    import time
    a = start.astype(np.longdouble)
    d = increment.astype(np.longdouble)
    origin = a[p.surface[0]].copy()
    a -= origin
    values = []
    for order in (3, 7, 15):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        total = 0.
        for z, w in zip(nodes, weights):
            if time.monotonic() >= deadline:
                raise TimeoutError('Independent temporal friction quadrature budget')
            x = np.asarray(a+np.longdouble((1+z)/2)*d, float)
            lag = IPCProblem.lag(p, x)
            power = 0. if lag is None else 2*float(p.friction.gradient(lag, p.cmesh,
                increment[p.surface])@increment[p.surface].ravel())
            if not np.isfinite(power) or power < -1e-12:
                raise ValueError('Invalid temporal friction work')
            total += .5*w*power
        values.append(total)
    uncertainty = 2*max(abs(np.diff(values)))
    return dict(method='relative_extended_precision_GL3_7_15_temporal_chord',
        values_n_mm=values, work_n_mm=values[-1], empirical_uncertainty_n_mm=float(uncertainty),
        uncertainty_is_rigorous_bound=False,
        converged=bool(uncertainty <= 1e-9+1e-5*abs(values[-1])))
