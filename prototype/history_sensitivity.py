"""Diagnostic tangent perturbations of the isochoric viscous metric.

These helpers do not modify the production constitutive update or commit any
material history. A conditional FE sensitivity must propagate the derivative
of the previous accepted metric; keeping that derivative zero asks a different
question. Retractions keep trial histories SPD and preserve the base determinant.
"""
import numpy as np
from dataclasses import replace

PARAMETERS = ('log_g0', 'logit_q', 'log_tau')


def common_material(material, coordinate, increment):
    """One shared parameter perturbation, with K and both Ogden exponents fixed."""
    if coordinate not in PARAMETERS or not np.isfinite(increment):
        raise ValueError('Unknown coordinate or nonfinite increment')
    if len(material.mu_eq) != 1 or len(material.mu_ov) != 1:
        raise ValueError('This diagnostic requires exactly two single-term branches')
    ge, bulk = material.moduli(); gv, _ = material.moduli(True)
    g0 = ge+gv; q = ge/g0; tau = material.relaxation_s
    if coordinate == 'log_g0': g0 *= np.exp(increment)
    elif coordinate == 'logit_q': q = 1/(1+np.exp(-(np.log(ge/gv)+increment)))
    else: tau *= np.exp(increment)
    ge = g0*q; gv = g0*(1-q)
    return replace(material, mu_eq=(2*ge/material.alpha_eq[0],),
                   mu_ov=(2*gv/material.alpha_ov[0],), relaxation_s=tau,
                   poisson=(3*bulk-2*ge)/(2*(3*bulk+ge)))


def previous_friction_action(problem, hessian, previous_direction):
    """Exact R_previous*x'_previous at fixed current contact geometry.

    The lag/normal load depend on current x only in the native host. The old
    position enters delta=x-previous. Use the UNPROJECTED velocity Hessian.
    Supports one flattened direction or a matrix of direction columns.
    """
    direction = np.asarray(previous_direction, dtype=float)
    if direction.ndim not in (1, 2) or direction.shape[0] != problem.rest.size:
        raise ValueError('Expected flattened global direction(s)')
    if not np.isfinite(direction).all(): raise ValueError('Nonfinite direction')
    result = np.zeros_like(direction)
    if hessian is not None:
        result[problem.surface_dofs] = -hessian@direction[problem.surface_dofs]
    return result


def assemble_piola(problem, stress):
    """P1 reference-volume assembly, excluding all contact-force cancellation."""
    if problem.enrichment is not None or problem.element != 'p1':
        raise ValueError('History diagnostic currently qualifies native P1 only')
    areas, gradients, dofs = problem.mesh.bulk_data
    local = np.einsum('e,eiJ,eaJ->eai',areas*problem.config.width_mm,stress,gradients)
    result = np.zeros(problem.rest.size)
    np.add.at(result,dofs.ravel(),local.ravel())
    return result


def _metric(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim < 2 or value.shape[-2:] != (3, 3) or not np.isfinite(value).all():
        raise ValueError(f'{name} must contain finite 3 by 3 metrics')
    scale = np.maximum(1., np.max(np.abs(value), axis=(-2, -1)))
    if np.any(np.max(np.abs(value-value.swapaxes(-2, -1)), axis=(-2, -1)) > 1e-10*scale):
        raise ValueError(f'{name} must be symmetric')
    symmetric = .5*(value+value.swapaxes(-2, -1))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.any(eigenvalues <= 0):
        raise ValueError(f'{name} must be positive definite')
    return value, symmetric, eigenvalues, eigenvectors


def _tangent(base, direction, inverse):
    direction = np.asarray(direction, dtype=float)
    if direction.shape != base.shape or not np.isfinite(direction).all():
        raise ValueError('Metric direction must be finite and match the base shape')
    direction = .5*(direction+direction.swapaxes(-2, -1))
    trace = np.einsum('...ij,...ji->...', inverse, direction)
    return direction-(trace/3)[..., None, None]*base


def metric_perturb(A, dA, eps):
    """Return an SPD fixed-determinant perturbation with tangent ``dA``.

    The symmetric part is projected onto tr(A^-1 dA)=0, then retracted by
    A^(1/2) exp(eps A^(-1/2) dA A^(-1/2)) A^(1/2). Supports shape (...,3,3).
    Zero directions and zero steps return an exact copy of the original input.
    Invalid histories are rejected; eigenvalues are never clipped or repaired.
    """
    original, base, values, vectors = _metric(A, 'A')
    if not np.isscalar(eps) or not np.isfinite(eps):
        raise ValueError('Metric perturbation step must be finite')
    direction = np.asarray(dA, dtype=float)
    if direction.shape != base.shape or not np.isfinite(direction).all():
        raise ValueError('Metric direction must be finite and match the base shape')
    if eps == 0 or not np.any(direction):
        return original.copy()
    zero = ~np.any(direction != 0., axis=(-2, -1))
    inverse = (vectors*(1/values)[..., None, :]) @ vectors.swapaxes(-2, -1)
    direction = _tangent(base, direction, inverse)
    square_root = (vectors*np.sqrt(values)[..., None, :]) @ vectors.swapaxes(-2, -1)
    inverse_root = (vectors*(1/np.sqrt(values))[..., None, :]) @ vectors.swapaxes(-2, -1)
    logarithmic = inverse_root @ direction @ inverse_root
    logarithmic = .5*(logarithmic+logarithmic.swapaxes(-2, -1))
    rates, basis = np.linalg.eigh(logarithmic)
    # Remove only eigensolver roundoff in the already projected trace.
    rates -= rates.mean(axis=-1, keepdims=True)
    with np.errstate(over='raise', invalid='raise', under='raise'):
        exponential = (basis*np.exp(eps*rates)[..., None, :]) @ basis.swapaxes(-2, -1)
    result = square_root @ exponential @ square_root
    result = .5*(result+result.swapaxes(-2, -1))
    # Remove determinant roundoff without changing the first derivative.
    sign, logdet = np.linalg.slogdet(result)
    if np.any(sign <= 0) or not np.isfinite(logdet).all():
        raise FloatingPointError('Metric retraction lost positive definiteness')
    result *= np.exp((np.log(values).sum(axis=-1)-logdet)/3)[..., None, None]
    if np.any(np.linalg.eigvalsh(result) <= 0):
        raise FloatingPointError('Metric retraction lost positive definiteness')
    return np.where(zero[..., None, None], original, result)


def metric_direction(A, plus, minus, eps):
    """Centered metric derivative, symmetrized and projected at base ``A``.

    ``plus`` and ``minus`` must be trial updates obtained from the positive and
    negative parameter/history perturbations, not two accepted physical times.
    """
    _, base, values, vectors = _metric(A, 'A')
    p, _, _, _ = _metric(plus, 'plus')
    m, _, _, _ = _metric(minus, 'minus')
    if p.shape != base.shape or m.shape != base.shape:
        raise ValueError('Perturbed metrics must match the base shape')
    if not np.isscalar(eps) or not np.isfinite(eps) or eps <= 0:
        raise ValueError('Metric difference step must be positive and finite')
    inverse = (vectors*(1/values)[..., None, :]) @ vectors.swapaxes(-2, -1)
    return _tangent(base, (p-m)/(2*eps), inverse)
