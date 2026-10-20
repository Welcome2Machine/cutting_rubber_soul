"""Continuous P1 pressure for quadrature-based finite-strain elements.

Used by the bounded InteriorP2 diagnostic and the explicit MINI IPC candidate.
Both retain the original P1 boundary trace. The restricted P2 space must not be
called unrestricted Taylor--Hood. Eliminate pressure from
integral(p*(J-1) - p*p/(2*K)), retaining its sparse linear block, never dense Schur.
MINI uses consistent mass; the older diagnostic also permits row-sum lumping.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags, bmat
from scipy.sparse.linalg import splu, spsolve


class ContinuousP1Volume:
    def __init__(self, element, *, lumped=False):
        self.element = element
        self.mesh = element.mesh
        self.config = element.config
        self.lumped = bool(lumped)
        self.K = self.config.bulk_modulus_n_mm2
        self.projected_fraction = 1.
        self.volumes = element.area * self.config.width_mm
        self.qvolume = (self.volumes[:, None] * element.weights).ravel()
        nq = len(element.weights)
        corners = element.elements[:, :3]
        rows = np.broadcast_to(corners[:, None, :], (len(corners), nq, 3))
        cols = np.broadcast_to(np.arange(len(corners)*nq).reshape(-1, nq, 1), rows.shape)
        vals = self.qvolume.reshape(-1, nq, 1)*element.points[None, :, :]
        self.incidence = coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())),
            shape=(element.base_nodes, len(self.qvolume))).tocsr()
        mass_local = self.volumes[:, None, None]*(np.ones((3, 3))+np.eye(3))/12
        self.mass = coo_matrix((mass_local.ravel(), (
            np.broadcast_to(corners[:, :, None], mass_local.shape).ravel(),
            np.broadcast_to(corners[:, None, :], mass_local.shape).ravel())),
            shape=(element.base_nodes, element.base_nodes)).tocsc()
        if lumped:
            self.mass = diags(np.asarray(self.mass.sum(axis=1)).ravel()).tocsc()
        self.pressure_mass = self.mass/self.K
        self._mass_factor = splu(self.pressure_mass)

    def correct(self, u, r, k, energy, *, condensed=None):
        """Return corrected local pieces plus B; H = H_local + B.T M^-1 B."""
        el = self.element
        nd = el.dofs.shape[1]
        F = el.deformation(u)
        J = np.linalg.det(F).ravel()
        b = np.asarray(self.incidence@(J-1)).ravel()
        p = self._mass_factor.solve(b)
        pressure = np.asarray(self.incidence.T@p).ravel()/self.qvolume
        diff = pressure-self.K*(J-1)
        cof = np.empty_like(F)
        cof[:, :, 0, 0] = F[:, :, 1, 1]
        cof[:, :, 1, 1] = F[:, :, 0, 0]
        cof[:, :, 0, 1] = -F[:, :, 1, 0]
        cof[:, :, 1, 0] = -F[:, :, 0, 1]
        dj = np.einsum('eqiJ,eqaJ->eqai', cof, el.grad).reshape(len(F), -1, nd)
        wdiff = (self.qvolume*diff).reshape(len(F), -1)
        r = r+np.einsum('eq,eqd->ed', wdiff, dj)
        energy = energy+.5*(b@p-self.K*(self.qvolume@((J-1)**2)))
        coupling = None
        if k is not None:
            det_h = np.zeros((2, 2, 2, 2))
            det_h[0, 0, 1, 1] = det_h[1, 1, 0, 0] = 1.
            det_h[0, 1, 1, 0] = det_h[1, 0, 0, 1] = -1.
            hj = np.einsum('eqaJ,iJkL,eqbL->eqaibk', el.grad, det_h, el.grad,
                           optimize=True).reshape(len(F), -1, nd, nd)
            k = k+np.einsum('eq,eqab->eab', wdiff, hj)-self.K*np.einsum(
                'eq,eqa,eqb->eab', self.qvolume.reshape(len(F), -1), dj, dj)
            d = coo_matrix((dj.ravel(), (
                np.repeat(np.arange(len(J)), nd),
                np.broadcast_to(el.dofs[:, None, :], dj.shape).ravel())),
                shape=(len(J), self.mesh.dof_count)).tocsr()
            coupling = self.incidence@d
            self.pressure_coupling = coupling
            if condensed is True:
                raise ValueError('Consistent pressure requires sparse blocks, not dense Schur assembly')
        self.last = dict(J=J, pressure_nodes=p, pressure_quadrature=pressure,
                         volume_energy=.5*float(b@p))
        return r, k, coupling if condensed is None else None, float(energy)

    def schur_action(self, coupling, direction):
        return coupling.T@self._mass_factor.solve(coupling@direction)

    def bulk(self, u, *, with_tangent=True):
        """Fused mixed assembly, equivalent to bulk()+correct().

        Replace the volumetric stress/tangent BEFORE contraction to element
        DOFs. Block quadrature work, and integrate B locally before sparse
        scatter. Avoid two element contractions and a quadrature-by-DOF sparse
        intermediate. correct() remains an independent comparison path.
        """
        from .material import response
        el = self.element
        minimum = el.minimum_j(u)
        if not np.isfinite(minimum) or minimum <= 0:
            raise FloatingPointError('Mixed element Jacobian nonpositive inside reference triangle')
        F = el.deformation(u)
        J = F[..., 0, 0]*F[..., 1, 1]-F[..., 0, 1]*F[..., 1, 0]
        b = np.asarray(self.incidence@(J.ravel()-1)).ravel()
        p = self._mass_factor.solve(b)
        pq = np.einsum('qa,ea->eq', el.points, p[el.elements[:, :3]])
        nd = el.dofs.shape[1]; ne, nq = J.shape
        r = np.empty((ne, nd)); k = np.empty((ne, nd, nd)) if with_tangent else None
        coupling = np.empty((ne, 3, nd)) if with_tangent else None
        energy = .5*float(b@p)
        det_h = np.zeros((4, 4));det_h[0, 3] = det_h[3, 0] = 1.
        det_h[1, 2] = det_h[2, 1] = -1.
        for start in range(0, ne, 1024):
            stop = min(start+1024, ne); sl = slice(start, stop)
            ff = F[sl]; jj = J[sl]; grad = el.grad[sl]
            cof = np.stack((ff[..., 1, 1], -ff[..., 1, 0], -ff[..., 0, 1], ff[..., 0, 0]), axis=-1)
            if el.config.ogden_alpha == 2.:
                # Exact plane-strain neo-Hookean mixed stress. Avoid computing
                # the full volume law only to subtract it again. In 2D,
                # cross(H,H) = outer(H,H) - Hess(det F)/J.
                flat = ff.reshape(-1, nq, 4)
                I1 = np.sum(flat*flat, axis=-1)+1.
                ga = el.config.shear_modulus_n_mm2*jj**(-2/3)
                beta = pq[sl]-ga*I1/(3*jj)
                P = ga[..., None]*flat+beta[..., None]*cof
                iso_energy = .5*(ga*I1-3*el.config.shear_modulus_n_mm2)
                if with_tangent:
                    C = ga[..., None, None]*np.eye(4)
                    C -= (2*ga/(3*jj))[..., None, None]*(
                        flat[..., :, None]*cof[..., None, :]+cof[..., :, None]*flat[..., None, :])
                    C += (5*ga*I1/(9*jj*jj))[..., None, None]*cof[..., :, None]*cof[..., None, :]
                    C += beta[..., None, None]*det_h
            else:
                q = response(ff.reshape(-1, 2, 2), el.config, tangent=with_tangent, diagnostics=False)
                diff = pq[sl]-self.K*(jj-1)
                P = q['P'].reshape(-1, nq, 4)+diff[..., None]*cof
                iso_energy = q['energy'].reshape(-1, nq)-.5*self.K*(jj-1)**2
                if with_tangent:
                    C = q['tangent'].reshape(-1, nq, 4, 4)
                    C += diff[..., None, None]*det_h-self.K*cof[..., :, None]*cof[..., None, :]
            # D maps displacement coefficients to flattened F=(00,01,10,11).
            D = np.zeros((stop-start, nq, 4, nd))
            D[:, :, 0, 0::2] = D[:, :, 2, 1::2] = grad[..., 0]
            D[:, :, 1, 0::2] = D[:, :, 3, 1::2] = grad[..., 1]
            force = (P[..., None, :]@D)[..., 0, :]
            r[sl] = self.volumes[sl, None]*np.einsum('eqd,q->ed', force, el.weights)
            energy += float(np.sum(self.qvolume.reshape(ne, nq)[sl]*iso_energy))
            if with_tangent:
                integrand = D.transpose(0, 1, 3, 2)@(C@D)
                k[sl] = self.volumes[sl, None, None]*np.einsum('eqab,q->eab', integrand, el.weights)
                dj = (cof[..., None, :]@D)[..., 0, :]
                coupling[sl] = np.einsum('qa,eqd,q,e->ead', el.points, dj, el.weights,
                                         self.volumes[sl], optimize=True)
        if with_tangent:
            rows = np.broadcast_to(el.elements[:, :3, None], coupling.shape).ravel()
            cols = np.broadcast_to(el.dofs[:, None, :], coupling.shape).ravel()
            self.pressure_coupling = coo_matrix((coupling.ravel(), (rows, cols)),
                shape=(el.base_nodes, self.mesh.dof_count)).tocsc()
        self.last = dict(J=J.ravel(), pressure_nodes=p, pressure_quadrature=pq.ravel(),
                         volume_energy=.5*float(b@p),
                         pressure_constraint_residual_mm3=float(np.linalg.norm(self.pressure_mass@p-b)))
        return r, k, el.dofs, float(energy), minimum

    def solve_ptc(self, local, metric, pressure, shift, rhs, free, linear_solver=None):
        from .mixed_system import solve_mixed
        bubbles = getattr(self.element, 'bubble_dofs', None)
        if bubbles is not None:
            index = np.full(self.mesh.dof_count, -1, dtype=int)
            index[free] = np.arange(len(free))
            bubbles = index[bubbles]
            if np.any(bubbles < 0):
                raise ValueError('MINI bubble coefficients must not be Dirichlet constrained')
        return solve_mixed(local, metric, pressure[0], self.pressure_mass,
                           self._mass_factor.solve, shift, rhs, bubbles, linear_solver)

    def solve(self, local, coupling, free, rhs):
        """Exact sparse saddle solve, checked in the eliminated physical system."""
        B = coupling[:, free].tocsc()
        A = local[free][:, free].tocsc()
        if hasattr(self.element, 'bubble_dofs'):
            return self.solve_ptc(A, 0*A, (B, self.pressure_mass), 0., rhs, free)[0]
        saddle = bmat([[A, B.T], [B, -self.pressure_mass]], format='csc')
        sol = spsolve(saddle, np.r_[rhs, np.zeros(B.shape[0])])
        du = sol[:len(free)]
        check = A@du+B.T@self._mass_factor.solve(B@du)-rhs
        if not np.all(np.isfinite(sol)) or np.linalg.norm(check) > 1e-7*max(1., np.linalg.norm(rhs)):
            raise RuntimeError('P2/P1 eliminated linear residual failed')
        return du


# Public diagnostic name retained for existing P2 studies and their tests.
P2P1Volume = ContinuousP1Volume
