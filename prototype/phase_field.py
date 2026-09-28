"""Reference-mesh AT1 damage subproblem at fixed displacement, diagnostic only.

Energy change = integral[((1-k)*(1-d)^2+k-1)*psi
                       + 3*Gc/8*(d/ell + ell*|Grad d|^2)] dV0.
P1 consistent mass, no prescribed crack, natural damage boundary conditions.
Using total elastic psi includes compression: this is an energetic control,
NOT the strength-aware, unilateral, mechanically coupled cutting model.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class FixedDisplacementAT1:
    def __init__(self, reference, triangles, width_mm, gc_n_mm, ell_mm, residual=1e-8):
        if not np.isfinite([width_mm, gc_n_mm, ell_mm, residual]).all():
            raise ValueError('Finite phase-field parameters required')
        if min(width_mm, gc_n_mm, ell_mm) <= 0 or not 0 <= residual < 1:
            raise ValueError('Positive width, Gc, ell and residual in [0,1) required')
        self.tri = np.asarray(triangles, int); X = np.asarray(reference, float)
        self.count = len(X); self.gc, self.ell, self.residual = gc_n_mm, ell_mm, residual
        edges = np.stack((X[self.tri[:, 1]]-X[self.tri[:, 0]],
                          X[self.tri[:, 2]]-X[self.tri[:, 0]]), axis=-1)
        self.volumes = np.abs(np.linalg.det(edges))/2*width_mm
        self.grad = np.array([[-1., -1.], [1., 0.], [0., 1.]])[None] @ np.linalg.inv(edges)
        self.mass = self.volumes[:, None, None]*(np.ones((3, 3))+np.eye(3))/12
        self.stiffness = self.volumes[:, None, None]*(self.grad @ self.grad.transpose(0, 2, 1))
        self.rows = np.broadcast_to(self.tri[:, :, None], self.mass.shape).ravel()
        self.cols = np.broadcast_to(self.tri[:, None, :], self.mass.shape).ravel()
        self.lumped = np.bincount(self.tri.ravel(), weights=np.repeat(self.volumes/3, 3), minlength=len(X))
        if np.any(self.lumped <= 0): raise ValueError('Unused node or degenerate mesh')

    def system(self, psi):
        psi = np.asarray(psi, float)
        if psi.shape != self.volumes.shape or not np.isfinite(psi).all() or np.min(psi) < -1e-10:
            raise ValueError('Expected nonnegative finite element elastic energy density')
        psi = np.maximum(psi, 0.)
        elastic = 1-self.residual
        Ae = 2*elastic*psi[:, None, None]*self.mass + 3*self.gc*self.ell/4*self.stiffness
        A = coo_matrix((Ae.ravel(), (self.rows, self.cols)), shape=(self.count, self.count)).tocsc()
        b = np.bincount(self.tri.ravel(),
                       weights=np.repeat((2*elastic*psi-3*self.gc/(8*self.ell))*self.volumes/3, 3),
                       minlength=self.count)
        return A, b

    def energy(self, damage, psi):
        d = np.asarray(damage)[self.tri]
        square = np.einsum('ei,eij,ej->e', d, self.mass, d)
        integral = self.volumes*d.mean(axis=1)
        released = float(np.sum((1-self.residual)*psi*(2*integral-square)))
        fracture = float(3*self.gc/8*(np.sum(integral)/self.ell
                         + self.ell*np.einsum('ei,eij,ej->', d, self.stiffness, d)))
        return released, fracture

    def validate_damage(self, damage):
        d = np.asarray(damage, float)
        if d.shape != (self.count,) or not np.isfinite(d).all() or np.any((d < 0) | (d > 1)):
            raise ValueError('Damage history must lie in [0,1]')
        return d

    def degradation(self, damage):
        """Exact element average of quadratic g(d), NOT g(mean(d))."""
        d = self.validate_damage(damage)[self.tri]
        square_average = (np.sum(d, axis=1)**2+np.sum(d*d, axis=1))/12
        return self.residual+(1-self.residual)*(1-2*d.mean(axis=1)+square_average)

    def kkt_residual(self, damage, psi, old):
        d = self.validate_damage(damage); lower = self.validate_damage(old)
        if np.any(d < lower): raise ValueError('Damage healing at acceptance')
        A, b = self.system(psi); g = A@d-b
        inactive = ((d <= lower+1e-13) & (g >= 0)) | ((d >= 1-1e-13) & (g <= 0)) | (lower == 1)
        return float(np.max(np.abs(np.where(inactive, 0., g)/self.lumped)))

    def solve(self, psi, old=None, *, tolerance_mpa=1e-6, max_iterations=300, seconds=60):
        A, b = self.system(psi)
        lower = np.zeros(self.count) if old is None else self.validate_damage(old).copy()
        d = lower.copy(); diagonal = A.diagonal(); deadline = time.monotonic()+seconds
        for iteration in range(max_iterations+1):
            if time.monotonic() > deadline: raise TimeoutError('AT1 diagnostic time budget')
            g = A@d-b
            inactive = ((d <= lower+1e-13) & (g >= 0)) | ((d >= 1-1e-13) & (g <= 0)) | (lower == 1)
            residual = np.where(inactive, 0., g)/self.lumped
            norm = float(np.max(np.abs(residual)))
            if norm <= tolerance_mpa:
                released, fracture = self.energy(d, np.maximum(psi, 0.))
                return d, dict(kind='fixed-displacement total-energy AT1 control; not physical crack evolution',
                               converged=True, kkt_residual_mpa=norm, iterations=iteration,
                               gc_n_mm=self.gc, residual_stiffness=self.residual,
                               max_damage=float(d.max()), damage_volume_mm3=float(self.lumped@d),
                               released_elastic_energy_n_mm=released, fracture_energy_n_mm=fracture,
                               energy_change_n_mm=fracture-released,
                               homogeneous_threshold_mpa=3*self.gc/(16*self.ell*(1-self.residual)),
                               mechanics_recoupled=False, compression_drives_damage=True)
            if iteration == max_iterations: break
            free = np.flatnonzero(~inactive)
            direction = np.zeros(self.count)
            direction[free] = spsolve(A[free][:, free], -g[free])
            direction = np.clip(d+direction, lower, 1.)-d
            slope = float(g@direction)
            if not np.isfinite(direction).all() or slope >= 0:
                direction = np.clip(d-g/diagonal, lower, 1.)-d
                slope = float(g@direction)
            if slope >= 0: raise RuntimeError('AT1 projected direction stagnated')
            # Exact quadratic line search, bounded segment between feasible points.
            curvature = float(direction@(A@direction))
            alpha = min(1., -slope/curvature) if curvature > 0 else 1.
            trial = np.clip(d+alpha*direction, lower, 1.)
            if np.array_equal(d, trial): raise RuntimeError('AT1 representational stagnation')
            d = trial
        raise RuntimeError(f'AT1 KKT iteration budget: residual={norm:g} MPa')
