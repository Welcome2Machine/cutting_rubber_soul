"""Objective, reference-area mixed-mode cohesive interfaces.

The scalar damage potential uses one toughness for every separation direction.
Normal and shear strengths differ through an anisotropic elastic metric, with
Ks = Kn (Ts/Tn)**2.  This is an explicit material restriction, not a general
mode-dependent toughness law. An optional nondegrading, reversible compression
branch acts relative to the stress-free reference gap. This is an explicit
virtual-gap closure penalty; the host's physical IPC contact remains separate.
An optional length regularizes this branch to a globally bounded constitutive
traction. It is a numerical pressure cap, not a material or friction error bound.

The four-node interface rotates with its current mean tangent.  Its analytic
gradient and Hessian include derivatives of that director.  Trial responses do
not modify the accepted history supplied by the caller.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class MixedModeLaw:
    normal_strength_mpa: float = .2
    shear_strength_mpa: float = .2
    normal_stiffness_mpa_per_mm: float = 50.
    toughness_n_per_mm: float = .006
    damage_enabled: bool = True
    compression_stiffness_mpa_per_mm: float = 0.
    compression_regularization_mm: float | None = None

    def __post_init__(self):
        parameters = [self.normal_strength_mpa, self.shear_strength_mpa,
                      self.normal_stiffness_mpa_per_mm, self.toughness_n_per_mm]
        if not np.isfinite(parameters).all() or min(parameters) <= 0:
            raise ValueError('Finite positive mixed-mode cohesive parameters required')
        if type(self.damage_enabled) is not bool:
            raise ValueError('damage_enabled must be a Boolean')
        if self.final_opening_mm <= self.peak_opening_mm:
            raise ValueError('Require delta_f=2 Gc/Tn > delta_0=Tn/Kn')
        if not np.isfinite(self.shear_stiffness_mpa_per_mm):
            raise ValueError('Derived shear stiffness must be finite')
        if (isinstance(self.compression_stiffness_mpa_per_mm, (bool, np.bool_))
                or not np.isfinite(self.compression_stiffness_mpa_per_mm)
                or self.compression_stiffness_mpa_per_mm < 0.):
            raise ValueError('Finite nonnegative compression stiffness required')
        ell = self.compression_regularization_mm
        if ell is not None:
            if (isinstance(ell, (bool, np.bool_)) or not np.isfinite(ell) or ell <= 0.
                    or not np.isfinite(self.compression_stiffness_mpa_per_mm*ell)):
                raise ValueError('Positive finite compression regularization and finite pressure cap required')

    @property
    def shear_stiffness_mpa_per_mm(self):
        return self.normal_stiffness_mpa_per_mm * (
            self.shear_strength_mpa / self.normal_strength_mpa)**2

    @property
    def peak_opening_mm(self):
        """Damage threshold in the anisotropic effective-opening metric."""
        return self.normal_strength_mpa / self.normal_stiffness_mpa_per_mm

    @property
    def final_opening_mm(self):
        return 2 * self.toughness_n_per_mm / self.normal_strength_mpa

    def to_dict(self):
        return dict(asdict(self),
                    shear_stiffness_mpa_per_mm=self.shear_stiffness_mpa_per_mm,
                    peak_opening_mm=self.peak_opening_mm,
                    final_opening_mm=self.final_opening_mm,
                    mode_toughness='shared_reference_area_toughness',
                    shear_stiffness_rule='Kn*(Ts/Tn)^2',
                    compression_model=('linear' if self.compression_regularization_mm is None
                                       else 'bounded_reversible_pseudo_huber'),
                    compression_traction_cap_mpa=(None if self.compression_regularization_mm is None else
                        self.compression_stiffness_mpa_per_mm*self.compression_regularization_mm),
                    compression_role='nondegrading reversible virtual-gap closure; physical IPC remains separate')

    def _dissipation(self, maximum):
        """Exact cumulative damage dissipation per reference area, in N/mm."""
        if not self.damage_enabled:
            return np.zeros_like(maximum)
        d0, df = self.peak_opening_mm, self.final_opening_mm
        # Integrating Y dd = .5 Kn kappa**2 d'(kappa) d kappa gives
        # this linear expression.  It reaches Gc exactly at complete failure.
        return self.toughness_n_per_mm * np.clip((maximum-d0)/(df-d0), 0., 1.)

    def history_response(self, maximum):
        """Recover history-only damage/dissipation without selecting a loading path.

        The representative opening equals the effective maximum in pure mode I.
        Its stored energy is not the stored energy of the current interface;
        use only ``history``, ``damage`` and ``dissipated`` from this helper.
        """
        return self.response(maximum, np.zeros_like(maximum), maximum)

    def response(self, opening_n, slip_s, previous_maximum):
        """Return a pure trial response; traction ordering is (normal, shear).

        ``incremental_potential`` is stored energy plus the increment of damage
        dissipation from the accepted history.  Its opening gradient is traction
        also during damage growth; stored energy alone does not have that
        property.  The tangent is the exact loading/unloading branch derivative,
        with the closed-normal branch selected at zero normal opening.
        """
        normal, slip, old = np.broadcast_arrays(np.asarray(opening_n, float),
                                               np.asarray(slip_s, float),
                                               np.asarray(previous_maximum, float))
        if not all(np.isfinite(a).all() for a in (normal, slip, old)) or np.any(old < 0):
            raise ValueError('Finite separations and nonnegative accepted history required')
        kn, ks = self.normal_stiffness_mpa_per_mm, self.shear_stiffness_mpa_per_mm
        alpha = ks/kn
        positive_normal = np.maximum(normal, 0.)
        effective = np.sqrt(positive_normal**2 + alpha*slip**2)
        history = np.maximum(old, effective)
        d0, df = self.peak_opening_mm, self.final_opening_mm
        damage = np.zeros_like(history)
        soft = (history > d0) & (history < df)
        if self.damage_enabled:
            damage[soft] = df*(history[soft]-d0)/(history[soft]*(df-d0))
            damage[history >= df] = 1.
        factor = 1.-damage
        w = np.stack((positive_normal, alpha*slip), axis=-1)
        damageable_traction = factor[..., None]*kn*w
        compression_opening = np.minimum(normal, 0.)
        kc, ell = self.compression_stiffness_mpa_per_mm, self.compression_regularization_mm
        if ell is None:
            compression_traction = kc*compression_opening
            compression_tangent = np.full_like(normal, kc)
            compression_stored = .5*kc*compression_opening**2
        else:
            # hypot avoids (u/ell)**2 overflow. Rationalizing hypot-ell avoids
            # catastrophic cancellation of the quadratic near-zero energy.
            radius = np.hypot(ell, compression_opening)
            ratio = ell/radius
            cap = kc*ell
            compression_traction = cap*(compression_opening/radius)
            compression_tangent = kc*ratio**3
            absolute_opening = -compression_opening
            compression_stored = cap*(absolute_opening*(absolute_opening/radius)/(1.+ratio))
        traction = damageable_traction.copy()
        traction[..., 0] += compression_traction
        tangent = np.zeros(normal.shape+(2, 2))
        tangent[..., 0, 0] = factor*kn*(normal > 0.)
        tangent[..., 1, 1] = factor*ks
        loading = self.damage_enabled & soft & (effective >= old)
        coefficient = np.zeros_like(history)
        coefficient[loading] = kn*df*d0/((df-d0)*effective[loading]**3)
        tangent -= coefficient[..., None, None]*w[..., :, None]*w[..., None, :]
        tangent[..., 0, 0] += compression_tangent*(normal <= 0.)
        damageable_stored = .5*factor*kn*effective**2
        stored = damageable_stored+compression_stored
        dissipated = self._dissipation(history)
        increment = dissipated-self._dissipation(old)
        return dict(traction=traction, tangent=tangent, history=history,
                    damage=damage, stored=stored, dissipated=dissipated,
                    dissipation_increment=increment, opening=normal, slip=slip,
                    effective_opening=effective,
                    initiation_ratio=effective/d0,
                    damageable_traction=damageable_traction,
                    compression_traction=compression_traction,
                    damageable_stored=damageable_stored,
                    compression_stored=compression_stored,
                    incremental_potential=stored+increment)


def _projection_geometry(x, weights, rotation):
    """f = r.T rotation v / |v| and its exact first and second derivatives."""
    eye = np.eye(2)
    cr = np.hstack((-weights[0]*eye, weights[0]*eye,
                    -weights[1]*eye, weights[1]*eye))
    cv = .5*np.hstack((-eye, -eye, eye, eye))
    r, v = x@cr.T, x@cv.T
    length = np.linalg.norm(v, axis=-1)
    if np.any(length <= 0.) or not np.isfinite(length).all():
        raise ValueError('Positive finite current mean interface segment length required')
    c = np.einsum('ni,ij,nj->n', r, rotation, v)
    gc = (rotation@v.T).T@cr + (rotation.T@r.T).T@cv
    hc = cr.T@rotation@cv + cv.T@rotation.T@cr
    b = 1./length
    gb = -(v@cv)/length[:, None]**3
    hb = np.einsum('ai,nab,bj->nij', cv,
        3.*v[..., :, None]*v[..., None, :]/length[:, None, None]**5
        -eye/length[:, None, None]**3, cv)
    gradient = b[:, None]*gc+c[:, None]*gb
    hessian = (b[:, None, None]*hc + gc[..., :, None]*gb[..., None, :]
               +gb[..., :, None]*gc[..., None, :] + c[:, None, None]*hb)
    return c*b, gradient, hessian, r, v/length[:, None]


def interface_response(positions, segments, reference_gaps, reference_areas,
                       previous_maximum, law, *, tangent=True, matrix_size=None,
                       reference_slips=None):
    """Assemble two-point Gauss quadrature on [L0, R0, L1, R1] segments.

    Reference areas are segment length times reference specimen width.  Gaps
    are positive stress-free normal distances, either one per segment or one
    per Gauss point.  Optional stress-free tangential offsets have the same
    shapes.  Both offsets are constants in the co-rotating interface frame,
    preserving objectivity. Accepted and trial histories have shape (nseg, 2).
    Normals are the clockwise rotation of the L0->L1 mean tangent, so input
    ordering must place R on the positive-normal side of L.
    """
    x = np.asarray(positions, float)
    segments = np.asarray(segments)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError('Finite (nnode, 2) positions required')
    if segments.ndim != 2 or segments.shape[1] != 4 or segments.dtype.kind not in 'iu':
        raise ValueError('Integer (nseg, 4) interface segments required')
    if np.any(segments < 0) or np.any(segments >= len(x)):
        raise ValueError('Interface node index outside positions')
    count = len(segments)
    areas = np.asarray(reference_areas, float)
    if areas.shape != (count,) or not np.isfinite(areas).all() or np.any(areas <= 0.):
        raise ValueError('Positive finite reference area per segment required')
    gaps = np.asarray(reference_gaps, float)
    if gaps.shape == (count,):
        gaps = np.repeat(gaps[:, None], 2, axis=1)
    if gaps.shape != (count, 2) or not np.isfinite(gaps).all() or np.any(gaps <= 0.):
        raise ValueError('Positive finite reference gap per segment/Gauss point required')
    slips0 = np.zeros((count, 2)) if reference_slips is None else np.asarray(reference_slips, float)
    if slips0.shape == (count,):
        slips0 = np.repeat(slips0[:, None], 2, axis=1)
    if slips0.shape != (count, 2) or not np.isfinite(slips0).all():
        raise ValueError('Finite reference slip per segment/Gauss point required')
    old = np.asarray(previous_maximum, float)
    if old.shape != (count, 2):
        raise ValueError('Accepted interface history must have shape (nseg, 2)')
    n = x.size if matrix_size is None else matrix_size
    if not isinstance(n, (int, np.integer)) or n < x.size:
        raise ValueError('matrix_size must be an integer at least positions.size')
    element_x = x[segments].reshape(count, 8)
    opening = np.empty((count, 2))
    slip = np.empty_like(opening)
    length = np.empty_like(opening)
    jacobian = np.empty((count, 2, 2, 8))
    curvature = np.empty((count, 2, 2, 8, 8))
    directions = np.empty((count, 2, 2))
    normal_rotation = np.array([[0., 1.], [-1., 0.]])
    for g, xi in enumerate((-1./np.sqrt(3.), 1./np.sqrt(3.))):
        weights = np.array([(1.-xi)/2., (1.+xi)/2.])
        un, jun, hun, r, t = _projection_geometry(element_x, weights, normal_rotation)
        us, jus, hus, _, _ = _projection_geometry(element_x, weights, np.eye(2))
        opening[:, g] = un-gaps[:, g]
        slip[:, g] = us-slips0[:, g]
        length[:, g] = np.linalg.norm(r, axis=-1)
        jacobian[:, g, 0], jacobian[:, g, 1] = jun, jus
        curvature[:, g, 0], curvature[:, g, 1] = hun, hus
        directions[:, g] = t
    q = law.response(opening, slip, old)
    weights = areas[:, None]/2.
    local_gradient = np.einsum('ngai,nga,ng->ni', jacobian, q['traction'],
                                np.broadcast_to(weights, (count, 2)))
    dofs = (2*segments[..., None]+np.arange(2)).reshape(count, 8)
    gradient = np.zeros(n)
    np.add.at(gradient, dofs.ravel(), local_gradient.ravel())
    local_compression_gradient = np.einsum('ngi,ng,ng->ni', jacobian[:, :, 0],
        q['compression_traction'], np.broadcast_to(weights, (count, 2)))
    compression_gradient = np.zeros(n)
    np.add.at(compression_gradient, dofs.ravel(), local_compression_gradient.ravel())
    q.update(length=length, tangent_direction=directions,
             normal_direction=directions@normal_rotation.T,
             gradient=gradient,
             compression_gradient=compression_gradient,
             damageable_gradient=gradient-compression_gradient,
             stored_energy_n_mm=float(np.sum(weights*q['stored'])),
             compression_stored_energy_n_mm=float(np.sum(weights*q['compression_stored'])),
             damageable_stored_energy_n_mm=float(np.sum(weights*q['damageable_stored'])),
             fracture_dissipation_n_mm=float(np.sum(weights*q['dissipated'])),
             fracture_dissipation_increment_n_mm=float(np.sum(weights*q['dissipation_increment'])),
             incremental_potential_n_mm=float(np.sum(weights*q['incremental_potential'])))
    if tangent:
        local_hessian = np.einsum('ngai,ngab,ngbj->ngij', jacobian,
                                  q['tangent'], jacobian)
        local_hessian += np.einsum('nga,ngaij->ngij', q['traction'], curvature)
        local_hessian = np.sum(weights[..., None, None]*local_hessian, axis=1)
        # Roundoff symmetrization leaves the exact constitutive/geometric
        # tangent intact; only the separate globalization metric is projected.
        local_hessian = .5*(local_hessian+local_hessian.swapaxes(-1, -2))
        eigenvalues, eigenvectors = np.linalg.eigh(local_hessian)
        projected = (eigenvectors*np.maximum(eigenvalues, 0.)[:, None, :])@eigenvectors.swapaxes(-1, -2)
        rows = np.broadcast_to(dofs[:, :, None], local_hessian.shape).ravel()
        cols = np.broadcast_to(dofs[:, None, :], local_hessian.shape).ravel()
        q['hessian'] = coo_matrix((local_hessian.ravel(), (rows, cols)), shape=(n, n)).tocsc()
        q['metric'] = coo_matrix((projected.ravel(), (rows, cols)), shape=(n, n)).tocsc()
    return q
