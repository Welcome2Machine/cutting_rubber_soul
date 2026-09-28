"""Experimental finite-strain directional relaxation and rational AT1 kernel.

The intact energy is unchanged. QR mode relaxation follows Hakimzadeh et al.
(2022); rational degradation is cohesive-inspired, NOT a reproduction of their
vector phase field or a calibrated multiaxial strength law. Units: N, mm, MPa.
"""
import time
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from .material import from_config
from mechanics.finite_strain import principal
from .phase_field import FixedDisplacementAT1


def diagonal_energy(a, b, c, G, K):
    """Original 3D isochoric neo-Hookean/quadratic-J energy and a,b derivatives."""
    J = a*b*c; I = a*a+b*b+c*c; s = J**(-2/3)
    W = G/2*(s*I-3)+K/2*(J-1)**2
    wa = G*s*(a-I/(3*a))+K*(J-1)*J/a
    wb = G*s*(b-I/(3*b))+K*(J-1)*J/b
    aa = G*s*(-1/3+5*I/(9*a*a))+K*(b*c)**2
    bb = G*s*(-1/3+5*I/(9*b*b))+K*(a*c)**2
    ab = G*s*(-2/3*(a/b+b/a)+2*I/(9*a*b))+K*(2*J-1)*c
    return W, wa, wb, aa, ab, bb


def free_normal_stretch(a, c, G, K):
    """Unique positive traction-free normal stretch at fixed tangential stretches."""
    a, c = np.broadcast_arrays(np.asarray(a, float), np.asarray(c, float))
    if np.any(a <= 0) or np.any(c <= 0):
        raise ValueError('Positive tangential stretches required')
    b = 1/(a*c)
    for _ in range(60):
        _, _, wb, _, _, bb = diagonal_energy(a, b, c, G, K)
        if np.max(np.abs(wb), initial=0.) <= 1e-10:
            return b
        step = np.clip(wb/bb, -.5*b, .5*b)
        b -= step
    raise RuntimeError('Traction-free normal relaxation did not converge')


def relaxed_response(F, normals, config, *, tangent=True):
    """Crack-parallel stretch retained, shear relaxed, normal closure unilateral.

    N is a REFERENCE normal fixed during differentiation. In the crack frame
    a=|F m|, b=det(F)/a. Opening relaxes b down to its traction-free value;
    closure keeps b. Plane strain retains F33=1. Analytic chain-rule tangent
    includes the Schur complement of the local normal relaxation.
    """
    F = np.asarray(F, float); N = np.asarray(normals, float)
    if F.ndim != 3 or F.shape[1:] != (2, 2) or N.shape != (len(F), 2):
        raise ValueError('Expected F(e,2,2) and reference normals(e,2)')
    if not np.isfinite(F).all() or not np.isfinite(N).all() or not np.allclose(np.linalg.norm(N, axis=1), 1., atol=1e-10):
        raise ValueError('Finite F and unit reference normals required')
    J = np.linalg.det(F)
    if np.any(J <= 0): raise FloatingPointError('Inverted deformation')
    m = np.column_stack((-N[:, 1], N[:, 0]))
    v = np.einsum('eij,ej->ei', F, m)
    a = np.linalg.norm(v, axis=1); t = v/a[:, None]; b = J/a
    G, K = config.shear_modulus_n_mm2, config.bulk_modulus_n_mm2
    star = free_normal_stretch(a, 1., G, K)
    opened = b >= star
    W, wa, wb, aa, ab, bb = diagonal_energy(a, np.minimum(b, star), 1., G, K)
    aa = np.where(opened, aa-ab*ab/bb, aa)
    ab = np.where(opened, 0., ab); bb = np.where(opened, 0., bb)
    wb = np.where(opened, 0., wb)
    da = np.einsum('ei,ej->eij', t, m).reshape(-1, 4)
    co = np.stack((F[:, 1, 1], -F[:, 1, 0], -F[:, 0, 1], F[:, 0, 0]), axis=1)
    db = co/a[:, None] - (J/a**2)[:, None]*da
    P = (wa[:, None]*da+wb[:, None]*db).reshape(-1, 2, 2)
    result = dict(energy=W, P=P, J=J, opened=opened, normal_stretch=b,
                  free_normal_stretch=star, tangential_stretch=a)
    effective_b = np.minimum(b, star)
    effective_j = a*effective_b
    wc = G*effective_j**(-2/3)*(1-(a*a+effective_b**2+1)/3)+K*(effective_j-1)*effective_j
    cauchy = np.zeros((len(F), 3, 3))
    cauchy[:, :2, :2] = P@F.transpose(0, 2, 1)/J[:, None, None]
    cauchy[:, 2, 2] = wc/J
    result['cauchy'] = cauchy
    if tangent:
        Ha = np.einsum('eik,ej,el->eijkl', np.eye(2)[None]-t[:, :, None]*t[:, None, :], m, m).reshape(-1, 4, 4)/a[:, None, None]
        HJ = np.zeros((4, 4)); HJ[0, 3] = HJ[3, 0] = 1.; HJ[1, 2] = HJ[2, 1] = -1.
        outer = lambda u, v: u[:, :, None]*v[:, None, :]
        Hb = (HJ[None]/a[:, None, None]-(outer(co, da)+outer(da, co))/a[:, None, None]**2
              +(2*J/a**3)[:, None, None]*outer(da, da)-(J/a**2)[:, None, None]*Ha)
        result['tangent'] = (aa[:, None, None]*outer(da, da)+ab[:, None, None]*(outer(da, db)+outer(db, da))
                             +bb[:, None, None]*outer(db, db)+wa[:, None, None]*Ha+wb[:, None, None]*Hb)
    return result


CONTINUOUS_ORIENTATION_METHOD = 'analytic_plane_strain_qr_v1'


def _continuous_qr_normal(F, seed, config):
    """Continuous candidate for this plane-strain NH/quadratic-J QR law only.

    With c=1 and fixed J, orientation enters solely through a=|F m|, bounded
    by the two singular stretches. On the closed branch W'(a) has the sign
    of a-sqrt(J). On the open branch wb=0, so W'(a)=G J_eff**(-2/3)
    (a*a-b_star*b_star)/a has the sign of a-1. Normal traction is negative
    between sqrt(J) and 1 when J<1, and positive between 1 and sqrt(J) when
    J>1. The scalar minimum is therefore sqrt(J) for J<1, or the projection
    of 1 into the attainable interval for J>=1. This removes the angular
    grid for this particular kernel, not for arbitrary fracture energies.

    Usually two reference normals attain this energy. Choose the one closest
    to the retained seed, modulo sign. Exact ties and isotropic F do not
    determine a unique crack direction; neither does this local energy test.
    """
    if (not np.isfinite(config.shear_modulus_n_mm2)
            or not np.isfinite(config.bulk_modulus_n_mm2)
            or min(config.shear_modulus_n_mm2, config.bulk_modulus_n_mm2) <= 0):
        raise ValueError('Continuous QR orientation requires positive finite shear and bulk moduli')
    # SVD avoids cancellation of the smaller eigenvalue of F.T@F in strongly
    # compressed elements. The input has already passed the positive-J check.
    _, stretch, Vh = np.linalg.svd(F)
    hi, lo = stretch[:, 0], stretch[:, 1]
    if np.any(lo <= np.finfo(float).eps*hi):
        raise FloatingPointError('Numerically singular deformation in orientation refinement')
    J = np.linalg.det(F)
    target = np.where(J < 1., np.sqrt(J), np.clip(1., lo, hi))
    gap = hi*hi-lo*lo
    distinct = gap > 16*np.finfo(float).eps*hi*hi
    high_weight = np.zeros(len(F))
    np.divide(target*target-lo*lo, gap, out=high_weight, where=distinct)
    high_weight = np.clip(high_weight, 0., 1.)
    low = np.sqrt(1-high_weight)[:, None]*Vh[:, 1]
    high = np.sqrt(high_weight)[:, None]*Vh[:, 0]
    tangent_plus, tangent_minus = low+high, low-high
    plus = np.column_stack((tangent_plus[:, 1], -tangent_plus[:, 0]))
    minus = np.column_stack((tangent_minus[:, 1], -tangent_minus[:, 0]))
    plus_distance = abs(np.einsum('ei,ei->e', plus, seed))
    minus_distance = abs(np.einsum('ei,ei->e', minus, seed))
    candidate = np.where((plus_distance >= minus_distance)[:, None], plus, minus)
    candidate *= np.where(np.einsum('ei,ei->e', candidate, seed) < 0., -1., 1.)[:, None]
    candidate[~distinct] = seed[~distinct]
    return candidate


def best_orientation(F, cauchy, normals, effective, config, orientation_samples=16, *, refine=False):
    """Retain the seed and seek lower QR energy; return (energy, reference normal).

    Default False preserves the discrete grid/principal search. With refine=True,
    replace the grid by the continuous candidate of _continuous_qr_normal, and
    still retain the input and principal seeds. This bounded analytic search
    is specific to the current plane-strain constitutive kernel; it does not
    establish a unique physical crack direction or a global fracture solution.
    orientation_samples is retained/validated for API compatibility in this mode.
    Neither mode changes the input arrays. n and -n represent the same plane.
    """
    if int(orientation_samples) != orientation_samples or orientation_samples < 4:
        raise ValueError('At least four integer crack orientations required')
    F = np.asarray(F, float); cauchy = np.asarray(cauchy, float)
    selected = np.asarray(normals, float).copy(); best = np.asarray(effective, float).copy()
    if F.ndim != 3 or F.shape[1:] != (2, 2):
        raise ValueError('Expected F(e,2,2)')
    if best.shape != (len(F),) or cauchy.shape != (len(F), 3, 3):
        raise ValueError('Expected one effective energy and 3D Cauchy tensor per element')
    if selected.shape != (len(F), 2) or not np.isfinite(selected).all() or not np.allclose(
            np.linalg.norm(selected, axis=1), 1., atol=1e-10):
        raise ValueError('Finite unit reference normals required')
    if not np.isfinite(F).all() or not np.isfinite(cauchy).all() or not np.isfinite(best).all():
        raise ValueError('Finite F, Cauchy tensors and seed energies required')
    if np.any(np.linalg.det(F) <= 0):
        raise FloatingPointError('Inverted deformation')
    if not len(F):
        return best, selected
    if refine:
        candidates = (_continuous_qr_normal(F, selected, config),)
    else:
        candidates = (np.tile([np.cos(angle), np.sin(angle)], (len(F), 1))
                      for angle in np.arange(orientation_samples)*np.pi/orientation_samples)
    for N in candidates:
        value = relaxed_response(F, N, config, tangent=False)['energy']
        better = value < best if refine else value < best-1e-12
        best[better] = value[better]; selected[better] = N[better]
    _, vectors = np.linalg.eigh(cauchy[:, :2, :2])
    N = np.einsum('eji,ej->ei', F, vectors[:, :, -1])
    N /= np.linalg.norm(N, axis=1)[:, None]
    value = relaxed_response(F, N, config, tangent=False)['energy']
    better = value < best-1e-12
    best[better] = value[better]; selected[better] = N[better]
    return best, selected


def _free_uniaxial_state(log_stretch, material):
    """Traction-free lateral branch of the current 3D neo-Hookean material."""
    if log_stretch == 0.:
        transverse_log = 0.
    else:
        transverse_log = brentq(lambda q: principal(np.array([[log_stretch, q, q]]), material)[1][0, 1],
                                -.5*log_stretch, 0., xtol=1e-14)
    W, tau, H = principal(np.array([[log_stretch, transverse_log, transverse_log]]), material)
    lateral_derivative = -H[0, 1, 0]/(H[0, 1, 1]+H[0, 1, 2])
    return float(W[0]), tau[0], H[0], transverse_log, float(lateral_derivative)


def free_uniaxial_release(config, stretches, *, orientation_samples=None, loading_angle=0.):
    """Audit the *3D free uniaxial* path, not the plane-strain cutting path.

    QR relaxation keeps the out-of-plane stretch c=t. For a normal inclined
    by theta, a^2=lambda^2 sin(theta)^2+t^2 cos(theta)^2 and b=lambda*t/a.
    The continuous maximum release is attained when the relaxed in-plane
    stretches a=b=s are both traction-free. Isotropy forces a=b at an interior
    minimum; the unique positive scalar traction-free root supplies s. We
    explicitly check that this state is accessible under unilateral opening.

    Optional sampling uses the same fixed reference angular grid plus axial
    principal normal as best_orientation; loading_angle exposes its grid bias.
    Continuous release is independent of this coordinate choice. This audits
    the existing QR law, not an experimentally established mixed-mode strength.
    """
    stretches = np.atleast_1d(np.asarray(stretches, float))
    if stretches.ndim != 1 or not np.isfinite(stretches).all() or np.any(stretches < 1.):
        raise ValueError('Finite tensile axial stretches >= 1 required')
    if orientation_samples is not None and (int(orientation_samples) != orientation_samples or orientation_samples < 4):
        raise ValueError('At least four integer crack orientations required')
    if not np.isfinite(loading_angle): raise ValueError('Finite loading angle required')
    material = from_config(config); G, K = material.moduli()
    rows = []
    for stretch in stretches:
        W, tau, H, q, qp = _free_uniaxial_state(float(np.log(stretch)), material)
        t = float(np.exp(q))
        s_log = 0. if q == 0. else brentq(
            lambda r: principal(np.array([[r, r, q]]), material)[1][0, 0], q, -.5*q, xtol=1e-14)
        s = float(np.exp(s_log))
        if s < t-1e-10 or s > stretch+1e-10 or stretch*t/s < s-1e-10:
            raise RuntimeError('Continuous QR optimum is outside the unilateral accessible branch')
        wr, tr, _ = principal(np.array([[s_log, s_log, q]]), material)
        angle = 0. if stretch == 1. else np.arcsin(np.sqrt(np.clip((s*s-t*t)/(stretch*stretch-t*t), 0., 1.)))
        axial_free = float(free_normal_stretch(t, t, G, K))
        axial_release = W-float(diagonal_energy(t, min(stretch, axial_free), t, G, K)[0])
        release = max(0., W-float(wr[0]))
        sampled = release
        if orientation_samples is not None:
            angles = np.r_[np.arange(orientation_samples)*np.pi/orientation_samples-loading_angle, 0.]
            a = np.sqrt(stretch**2*np.sin(angles)**2+t*t*np.cos(angles)**2)
            b = stretch*t/a
            relaxed = diagonal_energy(a, np.minimum(b, free_normal_stretch(a, t, G, K)), t, G, K)[0]
            sampled = max(0., W-float(relaxed.min()))
        rows.append((t, tau[0]/stretch, release, axial_release, angle, sampled,
                     tau[0]-tr[0, 2]*qp, (H[0, 0]+2*H[0, 1]*qp-tau[0])/stretch))
    values = np.asarray(rows).reshape(-1, 8)
    keys = ('transverse_stretch', 'nominal_stress_mpa', 'maximum_release_mpa', 'axial_normal_release_mpa',
            'maximizing_normal_angle_rad', 'sampled_release_mpa', 'release_log_stretch_derivative_mpa',
            'nominal_log_stretch_derivative_mpa')
    return {'axial_stretch': stretches.copy(), **dict(zip(keys, values.T))}


def strength_calibration(config):
    """Conditional QR threshold using the *maximum* free-uniaxial release.

    The former axial-normal-only calibration was inconsistent with the host's
    search over crack normals. Continuous maximization removes that mismatch;
    host angular sampling remains a separately measurable discretization error.
    No material parameter is fitted to the cutting curve. The independent
    catalogue breaking elongation is reported, not forcibly matched.
    """
    material = from_config(config); strength = config.tensile_strength_n_mm2
    if not np.isfinite(strength) or strength <= 0: raise ValueError('Positive nominal strength required')
    def nominal(log_stretch):
        return _free_uniaxial_state(log_stretch, material)[1][0]/np.exp(log_stretch)
    upper = np.log(2.)
    while nominal(upper) < strength and upper < np.log(128.): upper = min(2*upper, np.log(128.))
    if nominal(upper) < strength: raise ValueError('Nominal strength exceeds bounded calibration path (lambda <= 128)')
    log_stretch = brentq(lambda q: nominal(q)-strength, 0., upper, xtol=1e-13)
    stretch = float(np.exp(log_stretch))
    state = free_uniaxial_release(config, [stretch], orientation_samples=16)
    threshold = float(state['maximum_release_mpa'][0]); legacy = float(state['axial_normal_release_mpa'][0])
    # A path diagnostic, not a proof for arbitrary constitutive models. The
    # module is restricted to the present isochoric NH / quadratic-J material.
    path = free_uniaxial_release(config, np.exp(np.linspace(0., log_stretch, 65)))
    if (np.any(path['release_log_stretch_derivative_mpa'] < -1e-9)
            or np.any(path['nominal_log_stretch_derivative_mpa'] <= 0.)):
        raise ValueError('Nonmonotone nominal/QR calibration path requires an explicit first-crossing audit')
    old_root = brentq(lambda q: free_uniaxial_release(config, [np.exp(q)])['maximum_release_mpa'][0]-legacy,
                      0., log_stretch, xtol=1e-12)
    catalogue = free_uniaxial_release(config, [4.2])
    return dict(nominal_strength_mpa=float(strength), critical_release_mpa=threshold,
                calibration_version='maximum_directional_release_v2',
                calibration_axial_stretch=stretch, calibration_transverse_stretch=float(state['transverse_stretch'][0]),
                calibration_normal_angle_deg=float(np.degrees(state['maximizing_normal_angle_rad'][0])),
                legacy_axial_normal_threshold_mpa=legacy, legacy_first_onset_stretch=float(np.exp(old_root)),
                legacy_first_onset_nominal_mpa=float(nominal(old_root)),
                grid16_aligned_release_mpa=float(state['sampled_release_mpa'][0]),
                grid16_aligned_release_relative_deficit=float(1-state['sampled_release_mpa'][0]/threshold),
                path_audit_points=65, catalogue_breaking_stretch=4.2,
                catalogue_breaking_nominal_mpa=float(catalogue['nominal_stress_mpa'][0]),
                catalogue_breaking_release_mpa=float(catalogue['maximum_release_mpa'][0]),
                assumption='continuous maximum QR release fitted to catalogue nominal strength on free 3D uniaxial path; plane-strain and multiaxial onset unvalidated; breaking elongation not fitted')


class RationalAT1(FixedDisplacementAT1):
    """P1 rational degradation, AT1 crack density, consistent positive quadrature.

    g=k+(1-k)(1-d)^2/[(1-d)^2+a*d], a=3Gc/[8ell(1-k)Yc].
    Therefore the homogeneous d=0 first-variation threshold is Yc, independent
    of ell. This does NOT assert ell-independent post-peak behaviour.
    """
    shape = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])

    def __init__(self, *args, critical_release_mpa, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(critical_release_mpa) or critical_release_mpa <= 0:
            raise ValueError('Positive release threshold required')
        self.critical_release = critical_release_mpa
        self.a = 3*self.gc/(8*self.ell*(1-self.residual)*critical_release_mpa)
        self.gradient_matrix = coo_matrix((self.stiffness.ravel(), (self.rows, self.cols)), shape=(self.count, self.count)).tocsc()

    def law(self, d):
        d = np.asarray(d); u = (1-d)**2; up = 2*(d-1); v = u+self.a*d; vp = up+self.a
        g = u/v
        first = up/v-u*vp/v**2
        second = 2/v-2*up*vp/v**2-2*u/v**2+2*u*vp**2/v**3
        return self.residual+(1-self.residual)*g, (1-self.residual)*first, (1-self.residual)*second

    def degradation(self, damage):
        d = self.validate_damage(damage)[self.tri]@self.shape.T
        return self.law(d)[0].mean(axis=1)

    def system(self, psi):
        raise NotImplementedError('Rational degradation is nonlinear; use terms(damage, release)')

    def terms(self, damage, release, *, tangent=True):
        d = self.validate_damage(damage); Y = np.asarray(release, float)
        if Y.shape != self.volumes.shape or not np.isfinite(Y).all() or Y.min() < -1e-8:
            raise ValueError('Nonnegative finite directional release required')
        Y = np.maximum(Y, 0.)
        g, gp, gpp = self.law(d[self.tri]@self.shape.T)
        scale = self.volumes*Y/3
        fracture = 3*self.gc/8*(self.lumped@d/self.ell+self.ell*d@(self.gradient_matrix@d))
        energy = float(np.sum(scale[:, None]*(g-1))+fracture)
        local_gradient = scale[:, None]*(gp@self.shape)
        gradient = np.bincount(self.tri.ravel(), weights=local_gradient.ravel(), minlength=self.count)
        gradient += 3*self.gc/(8*self.ell)*self.lumped+3*self.gc*self.ell/4*(self.gradient_matrix@d)
        H = None
        if tangent:
            local_H = scale[:, None, None]*np.einsum('eq,qi,qj->eij', gpp, self.shape, self.shape)
            H = coo_matrix((local_H.ravel(), (self.rows, self.cols)), shape=(self.count, self.count)).tocsc()
            H += 3*self.gc*self.ell/4*self.gradient_matrix
        return energy, gradient, H, float(fracture)

    def energy(self, damage, release):
        value, _, _, fracture = self.terms(damage, release, tangent=False)
        return fracture-value, fracture

    def intact_stability(self, release):
        """Exact admissible first variation at d=0 for this discrete phase field.

        The ratio uses volume-weighted nodal release, not the largest element
        release. A negative normalized gradient permits increasing one nodal
        damage DOF. This local stability test proves neither a physical crack
        nor global energy minimization, and assumes no prescribed phase DOFs.
        """
        gradient = self.terms(np.zeros(self.count), release, tangent=False)[1]/self.lumped
        ratio = 1-gradient/(3*self.gc/(8*self.ell))
        return dict(nodal_release_mpa=ratio*self.critical_release, ratio=ratio,
                    normalized_gradient_mpa=gradient,
                    kkt_residual_mpa=float(np.max(np.maximum(-gradient, 0.))))

    def kkt_residual(self, damage, psi, old):
        d = self.validate_damage(damage); lower = self.validate_damage(old)
        if np.any(d < lower): raise ValueError('Damage healing')
        gradient = self.terms(d, psi, tangent=False)[1]
        active = ((d <= lower+1e-13) & (gradient >= 0)) | ((d >= 1-1e-13) & (gradient <= 0)) | (lower == 1)
        return float(np.max(np.abs(np.where(active, 0., gradient)/self.lumped)))

    def solve(self, psi, old=None, *, initial=None, tolerance_mpa=1e-6, max_iterations=300, seconds=60):
        lower = np.zeros(self.count) if old is None else self.validate_damage(old).copy()
        d = lower.copy() if initial is None else np.maximum(lower, self.validate_damage(initial))
        deadline = time.monotonic()+seconds
        for iteration in range(max_iterations+1):
            if time.monotonic() > deadline: raise TimeoutError('Rational phase subproblem wall budget')
            E, g, H, fracture = self.terms(d, psi)
            active = ((d <= lower+1e-13) & (g >= 0)) | ((d >= 1-1e-13) & (g <= 0)) | (lower == 1)
            residual = float(np.max(np.abs(np.where(active, 0., g)/self.lumped)))
            if residual <= tolerance_mpa:
                return d, dict(kkt_residual_mpa=residual, iterations=iteration, energy_change_n_mm=E,
                               fracture_energy_n_mm=fracture, max_damage=float(d.max()))
            if iteration == max_iterations: break
            free = np.flatnonzero(~active); A = H[free][:, free]
            # A guaranteed positive Gershgorin shift modifies ONLY the search model.
            bound = A.diagonal()-np.asarray(abs(A).sum(axis=1)).ravel()+abs(A.diagonal())
            shift = max(0., 1e-12-float(bound.min()))
            direction = np.zeros(self.count)
            direction[free] = spsolve(A+diags(np.full(len(free), shift)), -g[free])
            direction = np.clip(d+direction, lower, 1.)-d
            slope = float(g@direction)
            if not np.isfinite(direction).all() or slope >= 0:
                direction = np.clip(d-g/np.maximum(abs(H.diagonal()), 1e-12), lower, 1.)-d
                slope = float(g@direction)
            if slope >= 0: raise RuntimeError('Rational phase direction stagnated')
            alpha = 1.
            for _ in range(40):
                trial = np.clip(d+alpha*direction, lower, 1.)
                Et = self.terms(trial, psi, tangent=False)[0]
                if Et <= E+1e-4*alpha*slope:
                    d = trial; break
                alpha *= .5
            else: raise RuntimeError('Rational phase line search failed')
        raise RuntimeError(f'Rational phase KKT budget: {residual:g} MPa')
