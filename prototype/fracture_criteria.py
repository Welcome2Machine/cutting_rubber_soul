"""Objective stress measures and explicitly calibrated strength screening.

The Drucker-Prager form follows the second-Piola stress representation in
Kamarei et al., arXiv:2506.16479v1, Eq. (3). This module is not their coupled
phase-field/viscoelastic model. Constant strengths are an elastic approximation.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np


def stress_measures(F, cauchy):
    """Convert Cauchy stress to P and S2, including full 3-D plane strain."""
    F,cauchy = np.asarray(F,dtype=float),np.asarray(cauchy,dtype=float)
    if F.shape != cauchy.shape or F.shape[-2:] != (3,3):
        raise ValueError('F and Cauchy stress must have matching (...,3,3) shapes')
    if not np.isfinite(F).all() or not np.isfinite(cauchy).all():
        raise ValueError('Stress conversion requires finite inputs')
    if not np.allclose(cauchy,np.swapaxes(cauchy,-1,-2),rtol=1e-10,atol=1e-10):
        raise ValueError('Cauchy stress must be symmetric')
    J = np.linalg.det(F)
    if np.any(J <= 0):
        raise ValueError('Stress conversion requires positive det(F)')
    inverse = np.linalg.inv(F)
    P = J[...,None,None]*(cauchy@np.swapaxes(inverse,-1,-2))
    second = inverse@P
    return P,(second+np.swapaxes(second,-1,-2))/2


def principal_tensile_measures(F, cauchy):
    """Nominal normal traction on Cauchy principal planes, not a strength law.

    Nanson: N=F.T@n/|F.T@n|, so force per reference area is
    J*sigma_principal/|F.T@n|. This equals engineering stress in uniaxial
    tension. A catalogue uniaxial strength is only a screening assumption in
    multiaxial loading. Directions are plane normals, not propagation vectors.
    """
    F, cauchy = np.asarray(F, float), np.asarray(cauchy, float)
    P, S = stress_measures(F, cauchy)
    stress, directions = np.linalg.eigh(cauchy)
    pulled = np.swapaxes(F, -1, -2) @ directions
    lengths = np.linalg.norm(pulled, axis=-2)
    nominal = np.maximum(0., np.linalg.det(F)[..., None]*stress/lengths)
    index = nominal.argmax(axis=-1)
    pick = index[..., None, None]
    current_normal = np.take_along_axis(directions, pick, axis=-1)[..., 0]
    reference_normal = np.take_along_axis(pulled/lengths[..., None, :], pick, axis=-1)[..., 0]
    ordered = np.sort(nominal, axis=-1)
    unique = ordered[..., -1]-ordered[..., -2] > 1e-8*np.maximum(1., ordered[..., -1])
    return dict(cauchy_principal_mpa=stress, principal_stretches=np.linalg.svd(F, compute_uv=False),
                principal_plane_area_factors=lengths,
                nominal_principal_mpa=nominal, nominal_tensile_mpa=nominal.max(axis=-1),
                current_normal=current_normal, reference_normal=reference_normal,
                direction_unique=unique, first_piola_mpa=P, second_piola_mpa=S)


@dataclass(frozen=True)
class StrengthSurface:
    uniaxial_second_pk_mpa: float
    hydrostatic_second_pk_mpa: float
    provenance: str
    parameter_status: str
    stress_measure: str = 'second_piola_kirchhoff'

    def __post_init__(self):
        ts,hs = self.uniaxial_second_pk_mpa,self.hydrostatic_second_pk_mpa
        if not np.isfinite([ts,hs]).all() or ts <= 0 or hs <= 0 or 3*hs <= ts:
            raise ValueError('Require positive second-Piola strengths and 3*hydrostatic > uniaxial')
        if self.stress_measure != 'second_piola_kirchhoff':
            raise ValueError('This surface requires second Piola-Kirchhoff strength data')
        if self.parameter_status not in ('measured','assumed') or not self.provenance.strip():
            raise ValueError('Specify measured/assumed status and nonempty calibration provenance')

    @classmethod
    def from_json(cls,path):
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        # Require the explicit measure in files: do not silently reinterpret
        # nominal catalogue tensile strength as a second-Piola strength.
        if 'stress_measure' not in data:
            raise ValueError('Calibration file must explicitly name stress_measure')
        return cls(**data)

    def to_dict(self):
        return asdict(self)

    def index(self,second_pk):
        S = np.asarray(second_pk,dtype=float)
        if S.shape[-2:] != (3,3) or not np.isfinite(S).all():
            raise ValueError('Expected finite (...,3,3) second Piola stress')
        if not np.allclose(S,np.swapaxes(S,-1,-2),rtol=1e-10,atol=1e-10):
            raise ValueError('Second Piola stress must be symmetric')
        ts,hs = self.uniaxial_second_pk_mpa,self.hydrostatic_second_pk_mpa
        trace = np.trace(S,axis1=-2,axis2=-1)
        deviator = S-trace[...,None,None]*np.eye(3)/3
        sqrt_j2 = np.sqrt(.5*(deviator*deviator).sum(axis=(-2,-1)))
        alpha = ts/(np.sqrt(3)*(3*hs-ts))
        threshold = np.sqrt(3)*hs*ts/(3*hs-ts)
        return (sqrt_j2+alpha*trace)/threshold
