"""Diagnostic 3-D extension of the existing isochoric Maxwell return map.

No production runner imports this module. The material definition and principal
energy are shared with finite_strain; the full spectral derivative includes
rotations of all three principal directions. History is never committed here.
"""
import numpy as np
from .finite_strain import principal


def response_3d(F, previous_metric, dt, material, *, tangent=True):
    F = np.asarray(F, dtype=float)
    A = np.asarray(previous_metric, dtype=float)
    if (F.ndim != 3 or F.shape[1:] != (3, 3) or A.shape != F.shape
            or not np.isfinite(F).all() or not np.isfinite(A).all()
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Finite batched 3-D F/history and positive dt required')
    if material.viscous_flow != 'isochoric_maxwell' or not material.mu_ov:
        raise ValueError('Diagnostic currently requires isochoric Maxwell')
    J = np.linalg.det(F)
    if np.any(J <= 0) or np.any(np.linalg.eigvalsh(A) <= 0):
        raise FloatingPointError('Nonpositive J/history')
    if not np.allclose(A, A.transpose(0, 2, 1), rtol=0, atol=1e-10):
        raise ValueError('History must be symmetric')
    eig, Q = np.linalg.eigh(F @ F.transpose(0, 2, 1))
    ev, V = np.linalg.eigh(F @ A @ F.transpose(0, 2, 1))
    if np.any(eig <= 0) or np.any(ev <= 0):
        raise FloatingPointError('Nonpositive trial metric')
    trial = .5*np.log(ev)
    e = trial.copy()
    shear = material.moduli(True)[0]
    L = dt*(np.eye(3)-np.ones((3, 3))/3)/(2*shear*material.relaxation_s)
    weq, teq, heq = principal(.5*np.log(eig), material)
    for _ in range(35):
        wov, tov, hov = principal(e, material, True)
        r = e-trial+np.einsum('ij,nj->ni', L, tov)
        local = np.eye(3)+L @ hov
        if np.all(np.max(np.abs(r), axis=1) <= 1e-11*np.maximum(1., np.max(np.abs(trial), axis=1))):
            break
        delta = np.linalg.solve(local, -r[..., None])[..., 0]
        fraction = np.minimum(1., .4/np.maximum(np.max(np.abs(delta), axis=1), 1e-30))
        e += fraction[:, None]*delta
    else:
        raise RuntimeError('3-D local material solve failed')
    tau = (Q*teq[:, None, :]) @ Q.transpose(0, 2, 1) + (V*tov[:, None, :]) @ V.transpose(0, 2, 1)
    inverse = np.linalg.inv(F)
    P = tau @ inverse.transpose(0, 2, 1)
    metric = inverse @ ((V*np.exp(2*e)[:, None, :]) @ V.transpose(0, 2, 1)) @ inverse.transpose(0, 2, 1)
    rate = np.einsum('ni,ij,nj->n', tov, L/dt, tov)
    result = dict(P=P, J=J, metric=metric, energy=weq+wov,
                  incremental_potential=weq+wov+.5*dt*rate,
                  equilibrium_energy=weq, overstress_energy=wov, viscous_rate=rate)
    if not tangent:
        return result
    directions = np.eye(9).reshape(9, 3, 3)
    matrices = []
    gov = hov @ np.linalg.inv(local)
    for basis, values, stress, moduli, old in ((Q, eig, teq, heq, np.eye(3)), (V, ev, tov, gov, A)):
        old = np.broadcast_to(old, F.shape)
        db = directions[None] @ old[:, None] @ F[:, None].transpose(0, 1, 3, 2)
        db += F[:, None] @ old[:, None] @ directions.transpose(0, 2, 1)[None]
        local_db = basis[:, None].transpose(0, 1, 3, 2) @ db @ basis[:, None]
        de = np.diagonal(local_db, axis1=2, axis2=3)/(2*values[:, None, :])
        ds = np.einsum('nij,nkj->nki', moduli, de)
        dtau = np.zeros_like(local_db)
        for i in range(3):
            dtau[:, :, i, i] = ds[:, :, i]
            for j in range(i):
                gap = values[:, i]-values[:, j]
                coefficient = (moduli[:, i, i]-moduli[:, i, j])/(2*values[:, i])
                np.divide(stress[:, i]-stress[:, j], gap, out=coefficient,
                          where=np.abs(gap) > 1e-8*np.max(values, axis=1))
                dtau[:, :, i, j] = coefficient[:, None]*local_db[:, :, i, j]
                dtau[:, :, j, i] = dtau[:, :, i, j]
        matrices.append(basis[:, None] @ dtau @ basis[:, None].transpose(0, 1, 3, 2))
    dP = (matrices[0]+matrices[1]) @ inverse[:, None].transpose(0, 1, 3, 2)
    dP -= P[:, None] @ directions.transpose(0, 2, 1)[None] @ inverse[:, None].transpose(0, 1, 3, 2)
    result['tangent'] = dP.reshape(len(F), 9, 9).transpose(0, 2, 1)
    return result
