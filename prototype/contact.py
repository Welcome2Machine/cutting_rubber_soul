"""Coulomb projection and its derivatives (N, mm, MPa).

In the mixed solver, old_shear is the current shear multiplier iterate. The
fixed-point equation returned_shear == old_shear enforces exact stick/slip;
stiffness is an augmentation parameter, not a physical elastic-slip stiffness.
Only converged physical steps may commit the normal/shear multiplier history.
"""
import numpy as np


def coulomb_return(slip, old_shear, pressure, coefficient, stiffness, adhesion=0.):
    if stiffness <= 0 or coefficient < 0 or adhesion < 0:
        raise ValueError('Invalid Coulomb contact parameters')
    trial = old_shear + stiffness*slip
    limit = np.where(pressure > 0,coefficient*pressure+adhesion,0.)
    shear = np.clip(trial,-limit,limit)
    sticking = (pressure > 0) & (np.abs(trial) < limit)
    sliding = (pressure > 0) & ~sticking
    ds = stiffness*sticking.astype(float)
    dp = coefficient*np.sign(trial)*sliding
    plastic_increment = slip-(shear-old_shear)/stiffness
    dissipation = shear*plastic_increment
    return shear,ds,dp,dissipation,sticking
