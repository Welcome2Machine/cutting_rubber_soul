"""Objective radial cohesive bonds with a finite, stress-free reference gap.

This deliberately restricted mode-I interface is not a general mixed-mode CZM.
Traction is per reference interface area. The reference gap is NOT opening or
damage; current bond length minus reference length is the constitutive opening.
Compression is handled by the host's IPC, never by a duplicated penalty here.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class CohesiveLaw:
    strength_mpa: float = .2
    stiffness_mpa_per_mm: float = 50.
    toughness_n_per_mm: float = .006
    damage_enabled: bool = True

    def __post_init__(self):
        if not np.isfinite([self.strength_mpa, self.stiffness_mpa_per_mm,
                            self.toughness_n_per_mm]).all():
            raise ValueError('Finite cohesive parameters required')
        if min(self.strength_mpa, self.stiffness_mpa_per_mm, self.toughness_n_per_mm) <= 0:
            raise ValueError('Positive cohesive parameters required')
        if self.final_opening_mm <= self.peak_opening_mm:
            raise ValueError('Require delta_f=2 Gc/Tc > delta_0=Tc/K')
        if type(self.damage_enabled) is not bool:
            raise ValueError('damage_enabled must be a Boolean')

    @property
    def peak_opening_mm(self):
        return self.strength_mpa/self.stiffness_mpa_per_mm

    @property
    def final_opening_mm(self):
        return 2*self.toughness_n_per_mm/self.strength_mpa

    def to_dict(self):
        return dict(asdict(self), peak_opening_mm=self.peak_opening_mm,
                    final_opening_mm=self.final_opening_mm)

    def response(self, opening, previous_maximum):
        opening, old = np.broadcast_arrays(np.asarray(opening, float),
                                            np.asarray(previous_maximum, float))
        if not np.isfinite([opening, old]).all() or np.any(old < 0):
            raise ValueError('Finite opening and nonnegative accepted history required')
        delta = np.maximum(opening, 0.)
        history = np.maximum(old, delta)
        d0, df, stiffness = self.peak_opening_mm, self.final_opening_mm, self.stiffness_mpa_per_mm
        damage = np.zeros_like(history)
        if self.damage_enabled:
            soft = (history > d0) & (history < df)
            damage[soft] = df*(history[soft]-d0)/(history[soft]*(df-d0))
            damage[history >= df] = 1.
        factor = 1-damage
        traction = factor*stiffness*delta
        tangent = factor*stiffness*np.ones_like(delta)
        loading = (delta >= old) & (delta > d0) & (delta < df) & self.damage_enabled
        tangent[loading] = -self.strength_mpa/(df-d0)
        tangent[opening <= 0] = 0.
        stored = .5*factor*stiffness*delta**2
        if self.damage_enabled:
            k = np.minimum(history, df)
            work = np.where(k <= d0, .5*stiffness*k*k,
                .5*self.strength_mpa*d0+self.strength_mpa*(k-d0)
                -.5*self.strength_mpa/(df-d0)*(k-d0)**2)
            dissipated = work-.5*factor*stiffness*history**2
        else:
            dissipated = np.zeros_like(history)
        return dict(traction=traction, tangent=tangent, history=history, damage=damage,
                    stored=stored, dissipated=np.maximum(dissipated, 0.), opening=opening)


def bonds_response(positions, pairs, reference_lengths, areas, previous_maximum, law,
                   *, tangent=True, matrix_size=None):
    """Assemble central-force bond gradients/Hessians, invariant under rigid motion."""
    x, pairs = np.asarray(positions, float), np.asarray(pairs, int)
    lengths0, areas = np.asarray(reference_lengths, float), np.asarray(areas, float)
    r = x[pairs[:, 1]]-x[pairs[:, 0]]
    length = np.linalg.norm(r, axis=1)
    if np.any(length <= 0) or np.any(lengths0 <= 0) or np.any(areas <= 0):
        raise ValueError('Positive current/reference bond lengths and areas required')
    normal = r/length[:, None]
    q = law.response(length-lengths0, previous_maximum)
    nodal = areas[:, None]*q['traction'][:, None]*normal
    gradient = np.zeros_like(x)
    np.add.at(gradient, pairs[:, 0], -nodal)
    np.add.at(gradient, pairs[:, 1], nodal)
    q.update(length=length, direction=normal, gradient=gradient.ravel(),
             stored_energy_n_mm=float(areas@q['stored']),
             fracture_dissipation_n_mm=float(areas@q['dissipated']))
    if tangent:
        radial = q['traction']/length
        tensor = radial[:, None, None]*np.eye(2)+(q['tangent']-radial)[:, None, None]*normal[:, :, None]*normal[:, None, :]
        tensor *= areas[:, None, None]
        hessian = np.concatenate((np.concatenate((tensor, -tensor), axis=2),
                                  np.concatenate((-tensor, tensor), axis=2)), axis=1)
        values, vectors = np.linalg.eigh(hessian)
        projected = (vectors*np.maximum(values, 0.)[:, None, :])@vectors.transpose(0, 2, 1)
        dofs = (2*pairs[:, :, None]+np.arange(2)).reshape(-1, 4)
        rows = np.broadcast_to(dofs[:, :, None], hessian.shape).ravel()
        cols = np.broadcast_to(dofs[:, None, :], hessian.shape).ravel()
        n = x.size if matrix_size is None else matrix_size
        q['hessian'] = coo_matrix((hessian.ravel(), (rows, cols)), shape=(n, n)).tocsc()
        q['metric'] = coo_matrix((projected.ravel(), (rows, cols)), shape=(n, n)).tocsc()
    return q
