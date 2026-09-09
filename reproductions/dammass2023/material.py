"""Paper parameter wrapper around the independently tested shared kernel."""
from dataclasses import dataclass
from mechanics.finite_strain import Material as BaseMaterial, principal
from mechanics.finite_strain import response as material_response

@dataclass(frozen=True)
class Material(BaseMaterial):
    mu_eq: tuple = (-1.103,.0105)
    alpha_eq: tuple = (-4.883,7.951)
    mu_ov: tuple = (-.385,1.e-6)
    alpha_ov: tuple = (-4.29,8.4)
    poisson: float = .48
    relaxation_s: float = 2.3
    volumetric: str = 'paper2023'


def response(F,previous_metric,dt,material=Material(),*,tangent=True):
    return material_response(F,previous_metric,dt,material,tangent=tangent,kinematics='plane_stress')
