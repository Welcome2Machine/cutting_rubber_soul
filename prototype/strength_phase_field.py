"""Opt-in elastic limit of Kamarei et al. (2025), arXiv:2506.16479v1.

Equations (29)--(32), (35), with z=1-d. This module solves ONLY the
fixed-deformation phase subproblem; it is not a cutting/contact implementation.
The scalar potential below integrates this subproblem at fixed F. It is NOT
the physical stored-plus-fracture energy: the independent strength microforce
does work. Do not pass it to the directional model's energy-budget check.

The constant second-Piola strengths are an explicit elastic approximation.
Hydrostatic strength and its provenance have no defaults. Units: N, mm, MPa.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from mechanics.finite_strain import principal
from .material import from_config
from .phase_field import FixedDisplacementAT1


@dataclass(frozen=True)
class StrengthCalibration:
    """Explicit elastic second-Piola strengths and matching critical energies.

    Direct construction is for independently calibrated data or labelled
    synthetic benchmarks. Energies must belong to the SAME material response
    used to build ElasticPhaseState; this cannot be inferred from four numbers.
    """
    uniaxial_second_pk_mpa: float
    hydrostatic_second_pk_mpa: float
    uniaxial_energy_mpa: float
    hydrostatic_energy_mpa: float
    provenance: str
    parameter_status: str

    def __post_init__(self):
        values = [self.uniaxial_second_pk_mpa, self.hydrostatic_second_pk_mpa,
                  self.uniaxial_energy_mpa, self.hydrostatic_energy_mpa]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Positive finite strengths and critical energies required')
        if 3*self.hydrostatic_second_pk_mpa <= self.uniaxial_second_pk_mpa:
            raise ValueError('Require 3*hydrostatic strength > uniaxial strength')
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError('Explicit calibration provenance required')
        if self.parameter_status not in ('measured', 'assumed', 'synthetic'):
            raise ValueError('Specify measured, assumed, or synthetic calibration')

    def to_dict(self):
        return asdict(self)


def elastic_state(F, config):
    """Undegraded energy and S2 from the repository's SAME 3-D elastic law.

    Accept full 3-D F, or embed 2-D F with F33=1 for plane strain. The spectral
    evaluation reuses principal(), also used by the current material adapter.
    """
    F = np.asarray(F, float)
    if F.ndim != 3 or F.shape[1:] not in ((2, 2), (3, 3)) or not np.isfinite(F).all():
        raise ValueError('Expected finite batched F(e,2,2) or F(e,3,3)')
    if F.shape[1] == 2:
        full = np.tile(np.eye(3), (len(F), 1, 1)); full[:, :2, :2] = F
        F = full
    if np.any(np.linalg.det(F) <= 0):
        raise ValueError('Positive deformation determinant required')
    squared, basis = np.linalg.eigh(F.transpose(0, 2, 1)@F)
    if np.any(squared <= 0):
        raise ValueError('Positive principal stretches required')
    energy, kirchhoff, _ = principal(.5*np.log(squared), from_config(config))
    second = (basis*(kirchhoff/squared)[:, None, :])@basis.transpose(0, 2, 1)
    return ElasticPhaseState(energy, second)


@dataclass(frozen=True)
class ElasticPhaseState:
    energy_mpa: np.ndarray
    second_pk_mpa: np.ndarray

    def __post_init__(self):
        W = np.asarray(self.energy_mpa, float)
        S = np.asarray(self.second_pk_mpa, float)
        if W.ndim != 1 or S.shape != (len(W), 3, 3) or not np.isfinite(W).all() or not np.isfinite(S).all():
            raise ValueError('Expected finite element energy and full 3-D second-Piola stress')
        if W.min(initial=0.) < -1e-10 or not np.allclose(S, S.transpose(0, 2, 1), atol=1e-10, rtol=1e-10):
            raise ValueError('Nonnegative elastic energy and symmetric second-Piola stress required')
        W = np.maximum(W, 0.); S = S.copy()
        W.setflags(write=False); S.setflags(write=False)
        object.__setattr__(self, 'energy_mpa', W)
        object.__setattr__(self, 'second_pk_mpa', S)


def calibration_from_config(config, *, hydrostatic_second_pk_mpa, provenance,
                            parameter_status):
    """Critical states of the existing neo-Hookean/quadratic-J material.

    Uniaxial nominal strength comes from config, not a Cauchy/S2 reinterpretation.
    Hydrostatic S2 strength is a separate explicit input. The returned stretches
    expose the current model's inconsistency with catalogue breaking elongation.
    """
    hs = float(hydrostatic_second_pk_mpa)
    if not np.isfinite(hs) or hs <= 0:
        raise ValueError('Explicit positive hydrostatic second-Piola strength required')
    if not np.isfinite(config.tensile_strength_n_mm2) or config.tensile_strength_n_mm2 <= 0:
        raise ValueError('Positive finite nominal tensile calibration strength required')
    material = from_config(config)

    def uniaxial(log_axial):
        def lateral(log_lateral):
            return principal(np.array([[log_axial, log_lateral, log_lateral]]), material)[1][0, 1]
        transverse = brentq(lateral, -log_axial-1., 1., xtol=1e-15, rtol=1e-14)
        stretches = np.exp([log_axial, transverse, transverse])
        energy, tau, tangent = principal(np.log(stretches)[None], material)
        H = tangent[0]
        lateral_slope = -H[1, 0]/(H[1, 1]+H[1, 2])
        # Exact derivative along the traction-free lateral branch, NOT a
        # finite-difference strength scan. Used to locate a possible S2 peak.
        second_slope = np.exp(-2*log_axial)*(H[0, 0]+2*H[0, 1]*lateral_slope-2*tau[0, 0])
        return float(tau[0, 0]/stretches[0]), float(energy[0]), stretches, float(second_slope)

    def first_root(function, target):
        previous = 0.
        for upper in np.arange(.1, 8.01, .1):
            if function(upper) >= target:
                return brentq(lambda q: function(q)-target, previous, upper, xtol=1e-15, rtol=1e-14)
            previous = upper
        raise ValueError('No tensile calibration root found on the connected loading branch')

    log_axial = first_root(lambda q: uniaxial(q)[0], config.tensile_strength_n_mm2)
    nominal, Wts, stretches, final_slope = uniaxial(log_axial)
    ts = nominal/stretches[0]
    # A nominal breaking stress cannot always be converted into a CONSTANT
    # S2 strength. This material's S2 tension curve can peak and then fall:
    # a later target would silently predict an earlier crossing of that same
    # strength surface. Refuse that incompatible calibration rather than fit it.
    # For this particular alpha=2/quadratic-J material the free uniaxial S2
    # branch has a tensile maximum. Locate it from the analytic path derivative
    # when the requested endpoint lies on the descending branch, then locate
    # the FIRST threshold crossing on [identity, maximum]. This detects an
    # arbitrarily close post-maximum target without relying on sampled stresses.
    if final_slope < 0:
        peak = brentq(lambda q: uniaxial(q)[3], 0., log_axial, xtol=1e-14)
        first = brentq(lambda q: uniaxial(q)[0]*np.exp(-q)-ts, 0., peak, xtol=1e-14)
        raise ValueError('Nominal target has an earlier crossing of its constant second-Piola strength: '
                         f'first stretch={np.exp(first):.12g}, requested stretch={stretches[0]:.12g}, '
                         f'peak stretch={np.exp(peak):.12g}; '
                         'a deformation-dependent strength calibration is required')
    def hydro(log_stretch):
        return float(principal(np.full((1, 3), log_stretch), material)[1][0, 0]*np.exp(-2*log_stretch))
    log_hydro = first_root(hydro, hs)
    Whs = float(principal(np.full((1, 3), log_hydro), material)[0][0])
    calibration = StrengthCalibration(ts, hs, Wts, Whs,
                                      provenance, parameter_status)
    details = dict(uniaxial_stretches=stretches.tolist(), hydrostatic_stretch=float(np.exp(log_hydro)),
                   nominal_tensile_strength_mpa=nominal, material=material.to_dict(),
                   assumption='constant second-Piola strength surface; no viscoelastic strength function fitted')
    return calibration, details


class StrengthPhaseField(FixedDisplacementAT1):
    """P1 phase discretization of Eq. (35), with exact bound irreversibility.

    Reuses reference geometry/gradient assembly and damage validation. Unlike
    the paper's penalty enforcement, d_old <= d <= 1 is enforced exactly.
    mesh_size_mm is ONE explicit representative h in Remark 10, not an
    automatically invented element-dependent constitutive parameter; zero
    selects the continuum Eq. (30). No QR relaxation or rational degradation.
    """
    # Positive six-point degree-four Dunavant rule, weights normalized to one.
    shape = np.array([[.108103018168070, .445948490915965, .445948490915965],
                      [.445948490915965, .108103018168070, .445948490915965],
                      [.445948490915965, .445948490915965, .108103018168070],
                      [.816847572980459, .091576213509771, .091576213509771],
                      [.091576213509771, .816847572980459, .091576213509771],
                      [.091576213509771, .091576213509771, .816847572980459]])
    weights = np.array([.223381589678011]*3+[.109951743655322]*3)

    def __init__(self, reference, triangles, width_mm, gc_n_mm, ell_mm, *,
                 calibration, mesh_size_mm=0.):
        super().__init__(reference, triangles, width_mm, gc_n_mm, ell_mm, residual=0.)
        if not isinstance(calibration, StrengthCalibration):
            raise TypeError('Explicit StrengthCalibration required')
        if not np.isfinite(mesh_size_mm) or mesh_size_mm < 0:
            raise ValueError('Finite nonnegative representative mesh size required')
        self.calibration, self.mesh_size_mm = calibration, float(mesh_size_mm)
        ts, hs = calibration.uniaxial_second_pk_mpa, calibration.hydrostatic_second_pk_mpa
        Wts, Whs = calibration.uniaxial_energy_mpa, calibration.hydrostatic_energy_mpa
        correction = 1+3*self.mesh_size_mm/(8*self.ell)
        self.delta = ((ts+(1+2*np.sqrt(3))*hs)/((8+3*np.sqrt(3))*hs)
                      *3*self.gc/(16*Wts*self.ell)/correction**2 + .4/correction)
        self.beta1 = -self.delta*self.gc/(8*self.ell*hs)+2*Whs/(3*hs)
        self.beta2 = (-np.sqrt(3)*(3*hs-ts)*self.delta*self.gc/(8*self.ell*hs*ts)
                      -2*Whs/(np.sqrt(3)*hs)+2*np.sqrt(3)*Wts/ts)
        self.beta0 = (self.delta-1)*3*self.gc/(8*self.ell)
        self.B = 3*self.delta*self.gc/(8*self.ell)
        self.gradient_coefficient = 3*self.ell*self.delta*self.gc/4
        # At intact hydrostatic compression Eq.35 gives R=B+beta1*tr(S0).
        # beta1<=0 is a sufficient no-compressive-onset condition for arbitrary
        # negative trace. Large ell/mesh corrections can violate it. This is a
        # supported-parameter guard, not a claim of a universal length theorem.
        if self.beta1 > 0:
            raise ValueError('Chosen length/mesh correction/calibration permits hydrostatic-compression '
                             'damage (beta1 > 0); outside this core\'s supported compressive regime')
        self.diagnostics = dict(delta=self.delta, beta0=self.beta0, beta1=self.beta1, beta2=self.beta2,
                                representative_mesh_size_mm=self.mesh_size_mm,
                                uniaxial_energy_length_mm=3*self.gc/(16*Wts),
                                compression_guard='beta1 <= 0; sufficient for intact hydrostatic compression',
                                zero_trace_extension='sign(0)=0, using undegraded stress')
        self.gradient_matrix = coo_matrix((self.stiffness.ravel(), (self.rows, self.cols)),
                                         shape=(self.count, self.count)).tocsc()

    def coefficients(self, state):
        if not isinstance(state, ElasticPhaseState) or state.energy_mpa.shape != self.volumes.shape:
            raise ValueError('One ElasticPhaseState value per element required')
        stress = state.second_pk_mpa
        trace = np.trace(stress, axis1=1, axis2=2)
        dev = stress-trace[:, None, None]*np.eye(3)/3
        root_j2 = np.sqrt(.5*np.sum(dev*dev, axis=(1, 2)))
        # Eq.32 is undefined at trace=0. Use the symmetric sign(0)=0
        # extension, explicitly; use undegraded S0 to avoid 0/0 at z=0.
        A = (1+np.sign(trace))*state.energy_mpa
        C = self.beta2*root_j2+self.beta1*trace
        return A, C

    def driving_force(self, damage, state):
        """Eq.32 chat and local part of Eq.31 at the integration points.

        The divergence term of Eq.31 is represented in the gradient matrix
        with delta, not double-counted as a pointwise local source.
        """
        d = self.validate_damage(damage)
        A, C = self.coefficients(state)
        z = 1-d[self.tri]@self.shape.T
        chat = C[:, None]*z*z+(2*state.energy_mpa-A)[:, None]*z
        return dict(chat_mpa=chat, local_ce_mpa=chat+self.beta0,
                    divergence_in_gradient_operator=True)

    def terms(self, damage, state, *, tangent=True):
        """Fixed-F solver potential, residual, and consistent tangent.

        The potential is zero at intact d=0; it includes strength source terms.
        It must never be labelled elastic or physical fracture energy.
        """
        d = self.validate_damage(damage); A, C = self.coefficients(state)
        dq = d[self.tri]@self.shape.T; z = 1-dq
        scale = self.volumes[:, None]*self.weights
        local_potential = self.B*dq + .5*A[:, None]*(z*z-1)-C[:, None]/3*(z*z*z-1)
        potential = float(np.sum(scale*local_potential)
                          + .5*self.gradient_coefficient*d@(self.gradient_matrix@d))
        derivative = self.B-A[:, None]*z+C[:, None]*z*z
        local_gradient = (scale*derivative)@self.shape
        residual = np.bincount(self.tri.ravel(), weights=local_gradient.ravel(), minlength=self.count)
        residual += self.gradient_coefficient*(self.gradient_matrix@d)
        tangent_matrix = None
        if tangent:
            curvature = A[:, None]-2*C[:, None]*z
            local = np.einsum('eq,qi,qj->eij', scale*curvature, self.shape, self.shape)
            tangent_matrix = coo_matrix((local.ravel(), (self.rows, self.cols)),
                                        shape=(self.count, self.count)).tocsc()
            tangent_matrix += self.gradient_coefficient*self.gradient_matrix
        return potential, residual, tangent_matrix

    def system(self, psi):
        raise NotImplementedError('Use terms(damage, ElasticPhaseState); strength source is nonlinear')

    def energy(self, damage, psi):
        raise NotImplementedError('Fixed-F solver potential is not the total physical fracture energy')

    def _active(self, damage, lower, residual):
        return (((damage <= lower+1e-13) & (residual >= 0))
                | ((damage >= 1-1e-13) & (residual <= 0)) | (lower == 1))

    def kkt_residual(self, damage, state, old):
        d = self.validate_damage(damage); lower = self.validate_damage(old)
        if np.any(d < lower):
            raise ValueError('Damage healing')
        residual = self.terms(d, state, tangent=False)[1]
        return float(np.max(np.abs(np.where(self._active(d, lower, residual), 0., residual)/self.lumped)))

    def solve(self, state, old=None, *, initial=None, tolerance_mpa=1e-6,
              max_iterations=300, seconds=60.):
        if (not np.isfinite([tolerance_mpa, seconds]).all() or min(tolerance_mpa, seconds) <= 0
                or type(max_iterations) is not int or max_iterations < 0):
            raise ValueError('Positive finite tolerances/budget and nonnegative iteration limit required')
        lower = np.zeros(self.count) if old is None else self.validate_damage(old).copy()
        d = lower.copy() if initial is None else np.maximum(lower, self.validate_damage(initial))
        deadline = time.monotonic()+seconds
        for iteration in range(max_iterations+1):
            if time.monotonic() > deadline:
                raise TimeoutError('Strength phase subproblem wall budget')
            potential, residual, tangent = self.terms(d, state)
            active = self._active(d, lower, residual)
            norm = float(np.max(np.abs(np.where(active, 0., residual)/self.lumped)))
            if norm <= tolerance_mpa:
                return d, dict(kkt_residual_mpa=norm, iterations=iteration, max_damage=float(d.max()),
                               fixed_deformation_solver_potential_n_mm=potential,
                               physical_energy_audit=False, phase_model='kamarei2025-elastic-fixed-F')
            if iteration == max_iterations:
                break
            free = np.flatnonzero(~active); H = tangent[free][:, free]
            bound = 2*H.diagonal()-np.asarray(abs(H).sum(axis=1)).ravel()
            shift = max(0., 1e-12-float(bound.min()))
            direction = np.zeros(self.count)
            direction[free] = spsolve(H+diags(np.full(len(free), shift)), -residual[free])
            direction = np.clip(d+direction, lower, 1.)-d
            slope = float(residual@direction)
            if not np.isfinite(direction).all() or slope >= 0:
                direction = np.clip(d-residual/np.maximum(abs(tangent.diagonal()), 1e-12), lower, 1.)-d
                slope = float(residual@direction)
            if slope >= 0:
                raise RuntimeError('Strength phase direction stagnated')
            alpha = 1.
            for _ in range(40):
                trial = np.clip(d+alpha*direction, lower, 1.)
                value = self.terms(trial, state, tangent=False)[0]
                if value <= potential+1e-4*alpha*slope:
                    d = trial
                    break
                alpha *= .5
            else:
                raise RuntimeError('Strength phase line search failed')
        raise RuntimeError(f'Strength phase KKT budget: {norm:g} MPa')
