"""Scalar inertial fracture verification, not the production cutting solver.

A mass is joined to a moving support by a linear spring and to ground by the
production cohesive law (pure mode I). An optional explicit physical dashpot
is separately accounted. An incremental discrete gradient closes the energy
identity including irreversible damage; no residual is assigned to dissipation.
All parameters, especially mass and damping, are demonstrator inputs.
"""
import numpy as np
from scipy.optimize import brentq


def advance(law, *, displacement, velocity, history, support_before, support_after,
            dt, mass, spring, damping=0.):
    values = [displacement, velocity, history, support_before, support_after,
              dt, mass, spring, damping]
    if not np.isfinite(values).all() or min(dt, mass, spring) <= 0 or min(history, damping) < 0:
        raise ValueError('Finite state, positive dt/mass/spring, nonnegative history/damping required')
    u, v = float(displacement), float(velocity)
    q0 = law.response(u, 0., history)
    if float(q0['history']) > history+1e-12:
        raise ValueError('Accepted history is smaller than accepted opening')
    ymid = .5*(support_before+support_after)

    def cohesive_force(z):
        q = law.response(z, 0., history)
        jump = z-u
        if abs(jump) < 1e-12:
            return float(q0['traction'][0])
        return float(q['incremental_potential']-q0['stored'])/jump

    def residual(z):
        average_velocity = (z-u)/dt
        return (2*mass*(z-u-dt*v)/dt**2 + damping*average_velocity
                +spring*(.5*(z+u)-ymid)+cohesive_force(z))

    # Strict monotonicity lower bound for this scalar bilinear law. Reject a
    # time step that may skip between multiple algebraic roots.
    softening = law.normal_strength_mpa/(law.final_opening_mm-law.peak_opening_mm)
    if 2*mass/dt**2+damping/dt+.5*spring <= softening:
        raise ValueError('Reduce dt to resolve the inertial softening branch')
    span = max(abs(dt*v), abs(support_after-support_before), law.final_opening_mm, .01)
    center = u+dt*v
    for _ in range(30):
        low, high = center-span, center+span
        if residual(low) <= 0 <= residual(high):
            break
        span *= 2.
    else:
        raise RuntimeError('Inertial update has no bracket')
    z = brentq(residual, low, high, xtol=2e-15, rtol=1e-14)
    v1 = 2*(z-u)/dt-v
    q1 = law.response(z, 0., history)
    kinetic_change = .5*mass*(v1*v1-v*v)
    spring_change = .5*spring*((z-support_after)**2-(u-support_before)**2)
    cohesive_change = float(q1['stored']-q0['stored'])
    fracture = float(q1['dissipated']-q0['dissipated'])
    damping_work = damping*((z-u)/dt)**2*dt
    work = spring*(ymid-.5*(u+z))*(support_after-support_before)
    remainder = work-(kinetic_change+spring_change+cohesive_change+fracture+damping_work)
    return dict(displacement=z, velocity=v1, history=float(q1['history']),
                damage=float(q1['damage']), kinetic=.5*mass*v1*v1,
                work=work, kinetic_change=kinetic_change, spring_change=spring_change,
                cohesive_change=cohesive_change, fracture=fracture, damping=damping_work,
                energy_remainder=remainder, force_residual=abs(residual(z)))
