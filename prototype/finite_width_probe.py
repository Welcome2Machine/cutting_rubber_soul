"""Conditional compatible transverse modes around an accepted plane-strain field.

This is NOT a 3-D cutting solver. In-plane displacement is uniform across width;
the extra displacement w(X,Y,Z) is continuous P1(X,Y) x P1(Z), odd about Z=0.
The original in-plane contact tangent is retained. Out-of-plane friction is not
modelled. The computed direction releases the fictitious transverse reactions
holding the plane-strain state; it is not an accepted equilibrium or Fz curve.
"""
import numpy as np
from scipy.sparse import bmat, coo_matrix, kron
from mechanics.finite_strain_3d import response_3d

TRI_Q = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])


def embed(F):
    out = np.tile(np.eye(3), (len(F), 1, 1)); out[:, :2, :2] = F
    return out


def deformation(p, x):
    return np.eye(2)+np.einsum('eai,eaJ->eiJ',
        (x[:p.n]-p.mesh.reference_xy)[p.mesh.triangles], p.grad)


def transverse_blocks(p, q, layers, *, clamp_bottom=True):
    """Exact quadratic expansion at a z-uniform, reflection-symmetric field."""
    n = p.n; tri = p.mesh.triangles
    area, grad, dofs = p.mesh.bulk_data
    width = p.config.width_mm
    # Grading resolves the free end without adding central layers unnecessarily.
    z = width/2*(1-(1-np.linspace(0, 1, layers+1))**2)
    C = q['tangent']; P = q['P']
    if np.max(np.abs(C[:, 6:8, 8])) > 1e-8 or np.max(np.abs(P[:, 2, :2])) > 1e-8:
        raise ValueError('Block separation requires reflection-symmetric base state')
    ij = np.repeat(tri, 3, axis=1).ravel(); ji = np.tile(tri, (1, 3)).ravel()
    mass = (np.ones((3, 3))+np.eye(3))/12
    def sparse(v): return coo_matrix((v.ravel(), (ij, ji)), shape=(n, n)).tocsr()
    M = sparse(area[:, None, None]*C[:, 8, 8, None, None]*mass)
    S = sparse(np.einsum('e,eaJ,eJL,ebL->eab', area, grad, C[:, 6:8, 6:8], grad))
    mz = np.zeros((layers+1, layers+1)); kz = mz.copy()
    for l, h in enumerate(np.diff(z)):
        mz[l:l+2, l:l+2] += h*np.array([[2, 1], [1, 2]])/6
        kz[l:l+2, l:l+2] += np.array([[1, -1], [-1, 1]])/h
    K = (2*kron(mz, S)+2*kron(kz, M)).tocsr()
    r = np.zeros((layers+1, n))
    local = 2*area[:, None]*P[:, 2, 2, None]*np.ones((1, 3))/3
    np.add.at(r[-1], tri.ravel(), local.ravel())
    pc = C[:, [0, 1, 3, 4], 8].reshape(-1, 2, 2)
    coupling = np.einsum('e,eaJ,eiJ,b->eaib', 2*area, grad, pc, np.ones(3)/3).reshape(-1, 6, 3)
    rows = np.repeat(dofs, 3, axis=1).ravel()
    cols = (np.tile(tri, (1, 6))+layers*n).ravel()
    B = coo_matrix((coupling.ravel(), (rows, cols)), shape=(p.total, (layers+1)*n)).tocsr()
    fixed = np.arange(n)
    if clamp_bottom:
        fixed = np.r_[fixed, (np.arange(1, layers+1)[:, None]*n+p.mesh.bottom_nodes).ravel()]
    free = np.setdiff1d(np.arange((layers+1)*n), fixed)
    return dict(K=K, B=B, r=r.ravel(), free=free, z=z)


def release_direction(p, A, blocks, solver):
    u = p.free; w = blocks['free']; B = blocks['B'][u][:, w]
    matrix = bmat([[A[u][:, u], B], [B.T, blocks['K'][w][:, w]]], format='csr')
    rhs = np.r_[np.zeros(len(u)), -blocks['r'][w]]
    d = solver.solve(matrix, rhs)
    relative = float(np.linalg.norm(matrix@d-rhs)/max(np.linalg.norm(rhs), 1e-30))
    if relative > 1e-8: raise RuntimeError('Coupled conditional linear residual failed')
    du = np.zeros(p.total); du[u] = d[:len(u)]
    dw = np.zeros(len(blocks['r'])); dw[w] = d[len(u):]
    force = A@du+blocks['B']@dw
    derivative = float(-force.reshape(-1, 2)[p.n:, 1].sum())
    reduction = float(-.5*blocks['r']@dw)
    return du.reshape(-1, 2), dw.reshape(-1, p.n), dict(
        linear_relative=relative, dofs=len(rhs), derivative_fz_n=derivative,
        quadratic_release_n_mm=reduction, max_inplane_direction_mm=float(np.max(np.abs(du))),
        max_transverse_direction_mm=float(np.max(np.abs(dw))))


def integrated_bulk(p, x, w, z, *, chunk=4096):
    """Fresh 3-D material residual over the same compatible displacement space.

Three-point triangle/two-point layer quadrature is exact for the baseline
quadratic expansion; it is only a diagnostic quadrature for finite changes.
"""
    tri = p.mesh.triangles; area, grad, dofs = p.mesh.bulk_data
    F2 = deformation(p, x); ru = np.zeros(p.total); rw = np.zeros_like(w)
    min_j = np.inf; potential = 0.; max_det_error = 0.
    for l, h in enumerate(np.diff(z)):
        for t in ((1-1/np.sqrt(3))/2, (1+1/np.sqrt(3))/2):
            value = (1-t)*w[l]+t*w[l+1]
            slope = (w[l+1]-w[l])/h
            dxy = np.einsum('ea,eaJ->eJ', value[tri], grad)
            for N in TRI_Q:
                dz = slope[tri]@N
                for start in range(0, len(tri), chunk):
                    sl = slice(start, start+chunk)
                    F = embed(F2[sl]); F[:, 2, :2] = dxy[sl]; F[:, 2, 2] += dz[sl]
                    q = response_3d(F, p.previous_material_metric[sl], p.material_dt, p.material, tangent=False)
                    weight = area[sl]*h/3  # factor two for symmetry, Gauss weight 1/2
                    grad0 = np.column_stack(((1-t)*grad[sl].reshape(-1, 2), np.tile(-N/h, (len(F), 1)).ravel()))
                    grad1 = np.column_stack((t*grad[sl].reshape(-1, 2), np.tile(N/h, (len(F), 1)).ravel()))
                    r0 = weight[:, None]*np.einsum('eJ,eaJ->ea', q['P'][:, 2], grad0.reshape(-1, 3, 3))
                    r1 = weight[:, None]*np.einsum('eJ,eaJ->ea', q['P'][:, 2], grad1.reshape(-1, 3, 3))
                    u = np.einsum('e,eiJ,eaJ->eai', weight, q['P'][:, :2, :2], grad[sl])
                    np.add.at(ru, dofs[sl].ravel(), u.ravel())
                    np.add.at(rw[l], tri[sl].ravel(), r0.ravel())
                    np.add.at(rw[l+1], tri[sl].ravel(), r1.ravel())
                    min_j = min(min_j, float(q['J'].min()))
                    max_det_error = max(max_det_error, float(np.max(np.abs(np.linalg.det(q['metric'])-1))))
                    potential += float(weight@q['incremental_potential'])
    return ru, rw.ravel(), dict(min_j=min_j, potential_n_mm=potential, max_history_det_error=max_det_error)
