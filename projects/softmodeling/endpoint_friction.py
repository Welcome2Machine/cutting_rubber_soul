"""Monolithic endpoint friction with the original conservative discrete forces.

Midpoint kinematics and C/cohesive discrete gradients are unchanged. Friction
uses current endpoint geometry and endpoint relative velocity in the same full
momentum solve. Its signed temporal work defect is explicit and gated. This is
a first-order friction time candidate, NOT a damping coefficient or exact flow.
"""
import types
import numpy as np
from scipy.sparse import coo_matrix, diags
from prototype.energy_cut import EnergyCutProblem
from prototype.ipc_reference import IPCProblem
from prototype.zero_gap_cut import EnergyResolutionRequired
from release_dynamics import DEFECT_KEYS


def velocity_increment(p, du):
    """h*v_endpoint, retaining the prescribed blade velocity for friction."""
    value = 2*du-p.material_dt*p.previous_velocity
    value[p.n:] = du[p.n:]
    return value


def evaluate(self, x, lag=None, tangent=True, projected=True, *, with_metric=False):
    out = list(EnergyCutProblem.evaluate(self, x, None, tangent, projected, with_metric=with_metric))
    if lag is None: return tuple(out)
    increment = velocity_increment(self, self.step_increment(x))[self.surface]
    out[1][self.surface_dofs] += self.friction.gradient(lag, self.cmesh, increment)
    if tangent:
        factors = np.where(self.surface_dofs < 2*self.n, 2., 1.)
        def matrix(clamp):
            projection = self.ipc.PSDProjectionMethod.CLAMP if clamp else self.ipc.PSDProjectionMethod.NONE
            H = (self.friction.hessian(lag, self.cmesh, increment, projection)@diags(factors)).tocoo()
            return coo_matrix((H.data, (self.surface_dofs[H.row], self.surface_dofs[H.col])),
                shape=(self.total, self.total)).tocsc()
        out[2] += matrix(projected)
        if with_metric: out[4] += matrix(True)
    return tuple(out)


def coupled_tangent(self, x, lag, H):
    if lag is None: return H
    increment = velocity_increment(self, self.step_increment(x))[self.surface]
    J = -self.friction.force_jacobian(lag, self.cmesh, self.rest[self.surface],
        (x-self.rest)[self.surface], increment, self.barrier,
        self.ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS).tocoo()
    return H+coo_matrix((J.data, (self.surface_dofs[J.row], self.surface_dofs[J.col])),shape=H.shape).tocsc()


def energy_ledger(self, x):
    e = EnergyCutProblem.energy_ledger(self, x)
    du = self.step_increment(x); dv = velocity_increment(self, du)
    lag = self.lag(x)
    g = np.zeros(2*len(self.surface)) if lag is None else self.friction.gradient(lag,self.cmesh,dv[self.surface])
    physical = 2*float(g@dv[self.surface].ravel())
    algorithmic = 2*float(g@du[self.surface].ravel())
    defect = algorithmic-physical
    e['physical_balance_defect_n_mm'] += e['friction_work_n_mm']-physical
    e['friction_work_n_mm'] = physical
    e['friction_time_defect_n_mm'] = defect
    e['friction_algorithmic_work_n_mm'] = algorithmic
    e['algebraic_balance_residual_n_mm'] = e['physical_balance_defect_n_mm']-sum(e[k] for k in DEFECT_KEYS)-defect
    return e


def acceptance(self, x, depth, iterations, outer_iterations):
    rec = EnergyCutProblem.acceptance(self, x, depth, iterations, outer_iterations)
    e = rec['energy']
    if e['friction_work_n_mm'] < -1e-12 or e['friction_algorithmic_work_n_mm'] < -1e-12:
        raise EnergyResolutionRequired('Negative endpoint-friction work; physical time resolution required')
    if self.energy_normalization != 'mechanical_reservoir' or self.energy_tolerance is None:
        raise ValueError('Endpoint friction requires the original explicit reservoir energy gate')
    total = sum(abs(e[k]) for k in DEFECT_KEYS)+abs(e['friction_time_defect_n_mm'])
    allowance = self.energy_absolute_tolerance+self.energy_tolerance*e['mechanical_reservoir_n_mm']
    rec['complete_temporal_energy_ratio'] = total/max(allowance,1e-30)
    if total > allowance:
        raise EnergyResolutionRequired(f'Full endpoint-friction temporal defect {total:g} exceeds {allowance:g}')
    rec['friction_work_n_mm'] = .5*e['friction_work_n_mm']
    rec['time_integrator'] = 'C_discrete_gradient_midpoint_kinematics_endpoint_friction_Lie_Maxwell'
    rec['reaction_sampling'] = 'algorithmic mixed time force; h*F gives the momentum impulse'
    rec['friction_time_work'] = 'endpoint velocity law; signed difference from displacement-conjugate work is numerical, not physical dissipation'
    return rec


def install(p):
    if p.kinematic_theta != .5:
        raise ValueError('Endpoint-friction candidate preserves theta=0.5 conservative kinematics')
    p.conditional_integrator = 'endpoint_friction'
    for name, function in dict(evaluate=evaluate, lag=IPCProblem.lag,
            coupled_friction_tangent=coupled_tangent, energy_ledger=energy_ledger,
            acceptance=acceptance).items():
        setattr(p, name, types.MethodType(function,p))
    return p


def impulse_ledger(p, x):
    """Reuse the full component/constraint ledger with the actual friction force."""
    from stiff_contact import impulse_ledger as original
    native = p.friction
    class VelocityLaw:
        def gradient(self, lag, mesh, surface_du):
            # The ledger supplies the actual primary displacement increment.
            du = p.step_increment(x)
            np.testing.assert_array_equal(surface_du, du[p.surface])
            return native.gradient(lag,mesh,velocity_increment(p,du)[p.surface])
    try:
        p.friction = VelocityLaw()
        return original(p,x)
    finally:
        p.friction = native
