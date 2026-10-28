"""Finite-face virtual-gap closure geometry, independent of damage history.

Symmetric two-point quadrature measures each face point's distance to the
opposite *finite* segment. The resulting scalar potential is an optional
numerical gap correction, not a replacement for full IPC/CCD or its friction.
Closest-feature switches are C1 (piecewise C2) at positive distance. Zero
distance is inadmissible and must be excluded by the host's collision checks.
"""
import numpy as np


def point_segment_distance_jet(points, starts, ends):
    """Euclidean distance, gradient and Hessian in [point, start, end] DOFs.

    Differentiate the closest-point geometry, including the moving segment
    director. Freezing the closest barycentric coordinate gives a wrong tangent.
    """
    p, a, b = (np.asarray(v, float) for v in (points, starts, ends))
    if p.ndim != 2 or p.shape[1] != 2 or a.shape != p.shape or b.shape != p.shape:
        raise ValueError('Equal (n,2) finite point and segment arrays required')
    if not all(np.isfinite(v).all() for v in (p, a, b)):
        raise ValueError('Nonfinite face geometry')
    v, w = b-a, p-a
    l2 = np.sum(v*v, axis=1)
    if np.any(l2 <= 0.):
        raise ValueError('Nondegenerate finite segments required')
    t = np.sum(w*v, axis=1)/l2
    end = t >= 1.
    interior = (t > 0.) & (t < 1.)
    r = p-np.where(end[:, None], b, a)
    distance = np.linalg.norm(r, axis=1)
    gradient = np.zeros((len(p), 6))
    hessian = np.zeros((len(p), 6, 6))
    eye = np.eye(2)
    for endpoint, mask in ((1, ~interior & ~end), (2, ~interior & end)):
        if not np.any(mask):
            continue
        d, rr = distance[mask], r[mask]
        if np.any(d <= 0.):
            raise ValueError('Zero face distance is inadmissible; retain full CCD')
        c = np.zeros((2, 6)); c[:, :2] = eye
        c[:, 2*endpoint:2*endpoint+2] = -eye
        n = rr/d[:, None]
        gradient[mask] = n@c
        hessian[mask] = np.einsum('ai,nab,bj->nij', c,
            (eye-n[:, :, None]*n[:, None, :])/d[:, None, None], c)
    if np.any(interior):
        vv, ww = v[interior], w[interior]
        length = np.sqrt(l2[interior])
        cr = np.hstack((eye, -eye, np.zeros((2, 2))))
        cv = np.hstack((np.zeros((2, 2)), -eye, eye))
        rotation = np.array([[0., 1.], [-1., 0.]])
        cross = np.einsum('ni,ij,nj->n', ww, rotation, vv)
        gc = (rotation@vv.T).T@cr+(rotation.T@ww.T).T@cv
        hc = cr.T@rotation@cv+cv.T@rotation.T@cr
        inv = 1./length
        gi = -(vv@cv)/length[:, None]**3
        hi = np.einsum('ai,nab,bj->nij', cv,
            3.*vv[:, :, None]*vv[:, None, :]/length[:, None, None]**5
            -eye/length[:, None, None]**3, cv)
        signed = cross*inv
        if np.any(signed == 0.):
            raise ValueError('Zero face distance is inadmissible; retain full CCD')
        sign = np.sign(signed)
        distance[interior] = np.abs(signed)
        gradient[interior] = sign[:, None]*(inv[:, None]*gc+cross[:, None]*gi)
        hessian[interior] = sign[:, None, None]*(inv[:, None, None]*hc
            +gc[:, :, None]*gi[:, None, :]+gi[:, :, None]*gc[:, None, :]
            +cross[:, None, None]*hi)
    return distance, gradient, hessian


def finite_face_geometry(element_positions):
    """Distance jets [segment, side, Gauss point, ...] in four-node DOFs.

    Node order [L0,R0,L1,R1]; sides have equal reference quadrature weight.
    No pairing by infinite-line projections and no history-dependent switch.
    """
    x = np.asarray(element_positions, float)
    if x.ndim != 3 or x.shape[1:] != (4, 2):
        raise ValueError('Expected (nseg,4,2) interface positions')
    d = np.empty((len(x), 2, 2))
    j = np.empty(d.shape+(8,)); h = np.empty(d.shape+(8, 8))
    for side, (source, target) in enumerate((((0, 2), (1, 3)), ((1, 3), (0, 2)))):
        for gp, xi in enumerate((-1./np.sqrt(3.), 1./np.sqrt(3.))):
            weights = ((1.-xi)/2., (1.+xi)/2.)
            mapping = np.zeros((6, 8))
            for node, weight in zip(source, weights):
                mapping[:2, 2*node:2*node+2] = weight*np.eye(2)
            for offset, node in ((2, target[0]), (4, target[1])):
                mapping[offset:offset+2, 2*node:2*node+2] = np.eye(2)
            y = (x.reshape(-1, 8)@mapping.T).reshape(-1, 3, 2)
            distance, gradient, hessian = point_segment_distance_jet(y[:, 0], y[:, 1], y[:, 2])
            d[:, side, gp] = distance
            j[:, side, gp] = gradient@mapping
            h[:, side, gp] = np.einsum('ai,nab,bj->nij', mapping, hessian, mapping)
    return d, j, h


def finite_face_closure(element_positions, reference_distances, law):
    """Return potential derivatives per unit reference interface area.

    The average over both sides prevents doubling the pressure for parallel
    faces. If finite faces are farther apart than *all* reference distances,
    energy and all nodal forces are identically zero, including after failure.
    """
    distance, jacobian, curvature = finite_face_geometry(element_positions)
    reference = np.asarray(reference_distances, float)
    if reference.shape != distance.shape or not np.isfinite(reference).all() or np.any(reference <= 0.):
        raise ValueError('Positive finite reference distance per side and quadrature point required')
    opening = distance-reference
    q = law.response(opening, np.zeros_like(opening), np.zeros_like(opening))
    traction = q['compression_traction']
    stiffness = q['tangent'][..., 0, 0]*(opening <= 0.)
    gradient = np.einsum('nsgi,nsg->ni', jacobian, traction)/4.
    hessian = (np.einsum('nsgi,nsg,nsgj->nij', jacobian, stiffness, jacobian)
               +np.einsum('nsg,nsgij->nij', traction, curvature))/4.
    return dict(distance=distance, opening=opening, traction=traction,
        stored=q['compression_stored'], gradient=gradient, hessian=hessian)
