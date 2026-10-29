"""C-based discrete gradient for the reference neo-Hookean/Maxwell solid.

Frozen inverse viscous metric during mechanics; relaxation is a separate,
finite physical-time update. Gonzalez (2000), doi:10.1016/S0045-7825(00)00189-4,
motivates the C-space construction. This is not a contact/Maxwell theorem.
"""
import numpy as np


def _invariants(c, a, ge, gv, bulk):
    determinant = c[:, 0, 0]*c[:, 1, 1]-c[:, 0, 1]*c[:, 1, 0]
    if np.any(determinant <= 0):
        raise FloatingPointError('Nonpositive C determinant')
    j = np.sqrt(determinant)
    inv = np.stack((c[:, 1, 1], -c[:, 0, 1], -c[:, 1, 0], c[:, 0, 0]), axis=1).reshape(-1, 2, 2)/determinant[:, None, None]
    q = ge*np.eye(2)+gv*a[:, :2, :2]
    invariant = np.sum(q*c, axis=(1, 2))+ge+gv*a[:, 2, 2]
    factor = j**(-2/3)
    b = bulk*j*(j-1)-factor*invariant/3
    s = factor[:, None, None]*q+b[:, None, None]*inv
    w = .5*factor*invariant-1.5*(ge+gv)+.5*bulk*(j-1)**2
    return w, s, (j, inv, q, invariant, factor, b)


def frozen_energy(f, a, ge, gv, bulk):
    return _invariants(f.transpose(0, 2, 1)@f, a, ge, gv, bulk)[0]


def frozen_response(f, a, ge, gv, bulk, *, tangent=True):
    """Instantaneous hyperelastic response with viscous metric held fixed.

    Unlike discrete_response(F,F), this differentiates BOTH occurrences of F;
    its tangent is the full endpoint elastic tangent, not half of it.
    """
    if np.any(np.linalg.det(f) <= 0):
        raise FloatingPointError('Inverted frozen deformation')
    energy, s, (j, inv, q, invariant, factor, b) = _invariants(f.transpose(0, 2, 1)@f, a, ge, gv, bulk)
    out = dict(P=f@s, energy=energy, J=j)
    if tangent:
        derivatives = []
        for direction in np.eye(4).reshape(4, 2, 2):
            dc = direction.T@f+f.transpose(0, 2, 1)@direction
            dj = .5*j*np.sum(inv*dc, axis=(1, 2))
            dfactor = (-2/3)*factor*dj/j
            di = np.sum(q*dc, axis=(1, 2))
            db = bulk*(2*j-1)*dj-(dfactor*invariant+factor*di)/3
            ds = dfactor[:, None, None]*q+db[:, None, None]*inv-b[:, None, None]*(inv@dc@inv)
            derivatives.append(direction@s+f@ds)
        out['tangent'] = np.stack(derivatives, axis=-1).reshape(-1, 4, 4)
    return out


def discrete_response(f0, f1, a, ge, gv, bulk, *, tangent=True):
    """Pbar:(F1-F0)=W(F1,A0)-W(F0,A0); Fbar^-1 Pbar is symmetric.

    Stable energy differences avoid subtracting O(1) energies for small steps.
    Below ||Delta C||=1e-7 the midpoint limit is used (third-order defect).
    The tangent is an exact derivative of the chosen branch, not a PSD force.
    """
    f0, f1, a = np.asarray(f0), np.asarray(f1), np.asarray(a)
    if np.any(np.linalg.det(f0) <= 0) or np.any(np.linalg.det(f1) <= 0):
        raise FloatingPointError('Inverted endpoint deformation')
    c0, c1 = f0.transpose(0, 2, 1)@f0, f1.transpose(0, 2, 1)@f1
    dc = c1-c0
    cm, fm = .5*(c0+c1), .5*(f0+f1)
    w0, _, z0 = _invariants(c0, a, ge, gv, bulk)
    w1, s1, z1 = _invariants(c1, a, ge, gv, bulk)
    _, sm, zm = _invariants(cm, a, ge, gv, bulk)
    j0, _, q, i0, a0, _ = z0
    j1, _, _, i1, _, _ = z1
    # In 2D det(C1)-det(C0)=cof(Cmid):Delta C exactly.
    cof = np.stack((cm[:, 1, 1], -cm[:, 1, 0], -cm[:, 0, 1], cm[:, 0, 0]), axis=1).reshape(-1, 2, 2)
    dj = np.sum(cof*dc, axis=(1, 2))/(j1+j0)
    da = a0*np.expm1((-2/3)*np.log1p(dj/j0))
    dw = .5*(a0*np.sum(q*dc, axis=(1, 2))+da*i1)+.5*bulk*dj*(j1+j0-2)
    n = dw-.5*np.sum(sm*dc, axis=(1, 2))
    den = np.sum(dc*dc, axis=(1, 2))
    use = den > 1e-14
    correction = np.divide(2*n, den, out=np.zeros_like(n), where=use)
    salgo = sm+correction[:, None, None]*dc
    p = fm@salgo
    out = dict(P=p, energy=w1, previous_energy=w0, energy_change=dw,
               J=np.linalg.det(f1))
    if not tangent:
        return out
    jm, inv, q, im, am, bm = zm
    deriv = []
    for direction in np.eye(4).reshape(4, 2, 2):
        ddc = direction.T@f1+f1.transpose(0, 2, 1)@direction
        dcm = .5*ddc
        djm = .5*jm*np.sum(inv*dcm, axis=(1, 2))
        dam = (-2/3)*am*djm/jm
        dim = np.sum(q*dcm, axis=(1, 2))
        dbm = bulk*(2*jm-1)*djm-(dam*im+am*dim)/3
        dsm = (dam[:, None, None]*q+dbm[:, None, None]*inv
               -bm[:, None, None]*(inv@dcm@inv))
        dn = .5*np.sum((s1-sm)*ddc-dsm*dc, axis=(1, 2))
        dd = 2*np.sum(dc*ddc, axis=(1, 2))
        dcorr = np.divide(2*dn-correction*dd, den, out=np.zeros_like(n), where=use)
        dsalgo = dsm+dcorr[:, None, None]*dc+correction[:, None, None]*ddc
        deriv.append(.5*direction@salgo+fm@dsalgo)
    out['tangent'] = np.stack(deriv, axis=-1).reshape(-1, 4, 4)
    return out


def cohesive_discrete_response(u0, u1, history, strength, toughness):
    """Exact path-average traction of linear, secant-unloading extrinsic CZM.

    Integrates reload-to-history and subsequent envelope separately, including
    complete failure. No fracture work is lost by overstepping delta_f.
    """
    u0, u1, old = np.broadcast_arrays(u0, u1, history)
    u0, u1 = np.maximum(u0, 0.), np.maximum(u1, 0.)
    if np.any(old < u0-1e-10) or strength <= 0 or toughness <= 0:
        raise ValueError('Invalid cohesive history or physical parameters')
    end = 2*toughness/strength
    k = np.maximum(old, u0)
    sec = np.divide(strength*np.maximum(1-k/end, 0.), k, out=np.zeros_like(k), where=k > 0)
    advance = u1 > k
    top = np.minimum(u1, k)
    work = .5*sec*(top-u0)*(top+u0)
    left, right = np.minimum(k, end), np.minimum(np.maximum(u1, k), end)
    work += strength*(right-left)*(1-(right+left)/(2*end))
    endpoint = np.where(advance, strength*np.maximum(1-u1/end, 0.), sec*u1)
    endpoint = np.where((k == 0) & (u1 == 0), strength, endpoint)
    du = u1-u0
    use = np.abs(du) > 1e-10
    traction = np.divide(work, du, out=endpoint.copy(), where=use)
    limit = .5*np.where((u1 >= k) & (u1 < end), -strength/end, sec)
    derivative = np.divide(endpoint-traction, du, out=np.asarray(limit).copy(), where=use)
    return dict(traction=traction, tangent=derivative, work=work)
