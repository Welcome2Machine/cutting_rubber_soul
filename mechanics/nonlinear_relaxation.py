"""Isolated plane-strain, stress-accelerated isochoric Maxwell diagnostic.

Not a production default, a fitted rubber law, or a Bergstrom--Boyce replica.
See docs/history/topics/nonlinear_relaxation.md for the potential and qualification limits.
Stress is Kirchhoff (MPa), time is seconds. History is inverse viscous C.
"""
import numpy as np
from mechanics.finite_strain import elastic_response, principal


def response(F, previous_metric, dt, material, stress_scale, *, tangent=True):
    """Return a new history without modifying the supplied committed history.

    Mobility: dv = c*(1 + s:s / stress_scale**2)*s, c=1/(2*Gv*tau0).
    The infinite stress scale recovers the existing isochoric Maxwell law.
    Only positive single-term alpha=2 branches are qualified here.
    """
    F = np.asarray(F, dtype=float)
    A = np.asarray(previous_metric, dtype=float)
    if (F.ndim != 3 or F.shape[1:] != (2, 2) or A.shape != (len(F), 3, 3)
            or not np.isfinite(F).all() or not np.isfinite(A).all()
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Finite batched F, metric and positive dt required')
    if not stress_scale > 0 or np.isnan(stress_scale):
        raise ValueError('Positive stress scale required (infinity is the control)')
    if (material.viscous_flow != 'isochoric_maxwell' or material.alpha_ov != (2.,)
            or len(material.mu_ov) != 1 or material.mu_ov[0] <= 0):
        raise ValueError('Qualified positive alpha=2 isochoric Maxwell branch required')
    if (not np.allclose(A, A.transpose(0, 2, 1), rtol=0, atol=1e-12)
            or np.max(np.abs(A[:, :2, 2])) > 1e-12
            or np.any(np.linalg.eigvalsh(A) <= 0)):
        raise ValueError('Positive plane-compatible symmetric history required')
    J = np.linalg.det(F)
    if np.any(J <= 0):
        raise FloatingPointError('Inverted deformation')
    eq = elastic_response(F, material, tangent=tangent, diagnostics=False)
    inverse = np.linalg.inv(F)
    ev, V = np.linalg.eigh(F @ A[:, :2, :2] @ F.transpose(0, 2, 1))
    if np.any(ev <= 0):
        raise FloatingPointError('Nonpositive elastic trial metric')
    trial = np.column_stack((.5*np.log(ev), .5*np.log(A[:, 2, 2])))
    e = trial.copy()
    c = 1/(2*material.moduli(True)[0]*material.relaxation_s)
    inv_scale2 = (1/stress_scale)**2
    D = np.eye(3)-np.ones((3, 3))/3
    tolerance = 1e-11*np.maximum(1., np.max(np.abs(trial), axis=1))
    for iteration in range(80):
        w, s, H = principal(e, material, True)
        # Explicit projection removes only floating-point spherical stress.
        s = s-s.mean(axis=1)[:, None]
        norm2 = np.sum(s*s, axis=1)
        multiplier = 1+norm2*inv_scale2
        dv = c*multiplier[:, None]*s
        mobility = c*(multiplier[:, None, None]*D
                      + 2*inv_scale2*s[:, :, None]*s[:, None, :])
        residual = e-trial+dt*dv
        local = np.eye(3)+dt*mobility@H
        if np.all(np.max(np.abs(residual), axis=1) <= tolerance):
            break
        delta = np.linalg.solve(local, -residual[..., None])[..., 0]
        fraction = np.minimum(1., .4/np.maximum(np.max(np.abs(delta), axis=1), 1e-30))
        e += fraction[:, None]*delta
    else:
        raise RuntimeError(f'Nonlinear material return failed: {np.max(np.abs(residual)):.6g}')
    tau = (V*s[:, None, :2])@V.transpose(0, 2, 1)
    Pv = tau@inverse.transpose(0, 2, 1)
    be = (V*np.exp(2*e[:, :2])[:, None, :])@V.transpose(0, 2, 1)
    metric = np.zeros_like(A)
    metric[:, :2, :2] = inverse@be@inverse.transpose(0, 2, 1)
    metric[:, 2, 2] = np.exp(2*e[:, 2])
    # Legendre transform of c*(|s|^2/2 + |s|^4/(4*s0^2)).
    # The quadratic Maxwell expression .5*dt*dissipation is WRONG here.
    primal = c*(.5*norm2+.75*norm2**2*inv_scale2)
    result = dict(P=eq['P']+Pv, overstress_P=Pv, metric=metric, J=J,
                  energy=eq['energy']+w, equilibrium_energy=eq['energy'],
                  overstress_energy=w, viscous_rate=np.sum(s*dv, axis=1),
                  incremental_potential=eq['energy']+w+dt*primal,
                  stress_norm=np.sqrt(norm2), mobility_multiplier=multiplier,
                  local_residual=float(np.max(np.abs(residual))),
                  local_iterations=iteration+1)
    if not tangent:
        return result
    directions = np.eye(4).reshape(4, 2, 2)
    db = (directions[None]@A[:, None, :2, :2]@F[:, None].transpose(0, 1, 3, 2)
          + F[:, None]@A[:, None, :2, :2]@directions.transpose(0, 2, 1)[None])
    dbv = V[:, None].transpose(0, 1, 3, 2)@db@V[:, None]
    dtrial = np.zeros((len(F), 3, 4))
    dtrial[:, :2] = (np.diagonal(dbv, axis1=2, axis2=3)/(2*ev[:, None, :])).transpose(0, 2, 1)
    g = H@np.linalg.inv(local)
    ds = np.einsum('nij,njk->nki', g, dtrial)
    gap = ev[:, 0]-ev[:, 1]
    off = (g[:, 0, 0]-g[:, 0, 1])/(2*ev[:, 0])
    np.divide(s[:, 0]-s[:, 1], gap, out=off,
              where=np.abs(gap) > 1e-8*np.max(ev, axis=1))
    dstress = np.zeros((len(F), 4, 2, 2))
    dstress[:, :, 0, 0] = ds[:, :, 0]
    dstress[:, :, 1, 1] = ds[:, :, 1]
    dstress[:, :, 0, 1] = off[:, None]*dbv[:, :, 0, 1]
    dstress[:, :, 1, 0] = dstress[:, :, 0, 1]
    dtau = V[:, None]@dstress@V[:, None].transpose(0, 1, 3, 2)
    dP = (dtau@inverse[:, None].transpose(0, 1, 3, 2)
          - Pv[:, None]@directions.transpose(0, 2, 1)[None]@inverse[:, None].transpose(0, 1, 3, 2))
    result['tangent'] = eq['tangent']+dP.reshape(len(F), 4, 4).transpose(0, 2, 1)
    return result
