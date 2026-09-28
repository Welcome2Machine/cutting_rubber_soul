"""Declared interface scenarios, without inventing a calibrated oil/thermal law.

Silicone oil is represented by an effective boundary-friction coefficient.
Normal contact and bulk material parameters are unchanged. Temperature is
recorded as an environmental condition, NOT used as an uncalibrated multiplier.
"""
import math

DRY_FRICTION = .35
OIL_FRICTION = .10


def interface_condition(kind='custom', friction=None, temperature_c=25., *,
                        oil_viscosity_cst=None, oil_grade=None, oil_wait_minutes=None):
    if kind not in ('custom', 'dry', 'silicone_oil'):
        raise ValueError('Unknown interface scenario')
    default = dict(custom=0., dry=DRY_FRICTION, silicone_oil=OIL_FRICTION)[kind]
    coefficient = default if friction is None else float(friction)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Friction must be finite and nonnegative')
    if not math.isfinite(temperature_c) or temperature_c <= -273.15:
        raise ValueError('Ambient temperature must be finite and above absolute zero')
    if oil_viscosity_cst is not None and (not math.isfinite(oil_viscosity_cst) or oil_viscosity_cst <= 0):
        raise ValueError('Declared oil viscosity must be positive and finite')
    if oil_wait_minutes is not None and (not math.isfinite(oil_wait_minutes) or oil_wait_minutes < 0):
        raise ValueError('Oil exposure time must be nonnegative and finite')
    if kind != 'silicone_oil' and any(v is not None for v in (oil_viscosity_cst, oil_grade, oil_wait_minutes)):
        raise ValueError('Oil descriptors require --interface silicone_oil')
    return dict(
        kind=kind, friction_coefficient=coefficient,
        friction_source=('explicit scenario input; independent calibration not verified' if friction is not None
                         else 'existing uncalibrated dry prior' if kind == 'dry'
                         else 'unmeasured nominal silicone-oil scenario, not a literature measurement' if kind == 'silicone_oil'
                         else 'legacy frictionless diagnostic default'),
        calibrated=False,
        contact_model=('effective boundary lubrication with regularized Coulomb friction'
                       if kind == 'silicone_oil' else 'regularized Coulomb contact'),
        normal_contact='unchanged IPC distance barrier',
        coverage='uniform coefficient on active knife-to-original-rubber-boundary contacts; no rubber self-contact',
        oil_grade=oil_grade, oil_kinematic_viscosity_cst=oil_viscosity_cst,
        oil_wait_minutes=oil_wait_minutes, viscosity_reference_temperature_c=None,
        ambient_temperature_c=float(temperature_c),
        temperature_role='declared environment only; no calibrated temperature law',
        thermal_coupling=False, film_pressure=False, film_shear=False,
        adhesion=False, swelling=False, material_parameters_changed=False,
        limitations=[
            'Unknown film thickness, replenishment and viscosity prevent a resolved fluid-film calculation.',
            'Friction is not multiplied by oil viscosity or absolute temperature.',
            'CR temperature, crystallization and oil-exposure effects are not established for this material.',
            'This is a conditional dry/oil contrast, not a calibrated lubrication prediction.',
        ],
    )
