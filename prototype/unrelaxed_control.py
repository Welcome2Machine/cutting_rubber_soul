"""Exact unrelaxed limit of the two alpha=2 shear branches, explicit diagnostic.

Keep G0 and K, native IPC and evolving friction history. This is not a fitted
rubber law. Identity material metrics only retain the audited checkpoint schema.
"""
from dataclasses import replace
import numpy as np
from mechanics.finite_strain import elastic_response
from .ipc_viscoelastic import ViscoelasticProblem
from .ipc_reference import IPCProblem
from .material import from_config


class UnrelaxedProblem(ViscoelasticProblem):
    def __init__(self, *, instantaneous_shear=1.38,
                 bulk_modulus=23.34333333333331, loading_rate=.2, **kwargs):
        if not all(np.isfinite(v) and v > 0 for v in
                   (instantaneous_shear, bulk_modulus, loading_rate)):
            raise ValueError('Positive G0, K and loading rate required')
        if kwargs.get('element', 'p1') != 'p1' or kwargs.get('ogden_alpha', 2.) != 2.:
            raise ValueError('The exact merged diagnostic requires alpha=2 P1')
        if kwargs.get('mu', .35) <= 0:
            raise ValueError('This diagnostic retains the native friction host')
        IPCProblem.__init__(self, shear_modulus=instantaneous_shear, **kwargs)
        self.config = replace(self.config, poisson_ratio=(3*bulk_modulus-2*instantaneous_shear)
                              / (2*(3*bulk_modulus+instantaneous_shear)))
        self.config.validate()
        self.material = from_config(self.config)
        if not np.isclose(self.material.moduli()[1], bulk_modulus, rtol=1e-12):
            raise ValueError('Requested K was not preserved')
        self.loading_rate = float(loading_rate)
        self.material_metric = np.tile(np.eye(3), (len(self.mesh.triangles), 1, 1))
        self.previous_material_metric = self.material_metric.copy()
        self.material_dt = None
        self.material_dissipation = 0.
        self.parallel_linear = None

    def constitutive_response(self, F, *, tangent):
        if self.material_dt is None or not np.isfinite(self.material_dt) or self.material_dt <= 0:
            raise ValueError('Positive physical increment required')
        q = elastic_response(F, self.material, tangent=tangent, kinematics='plane_strain')
        q.update(incremental_potential=q['energy'], equilibrium_energy=q['energy'],
                 overstress_energy=np.zeros(len(F)))
        return q

    def material_bulk(self, u, *, with_tangent=True):
        # The existing closed-form elastic assembly omits unused diagnostics.
        return IPCProblem.material_bulk(self, u, with_tangent=with_tangent)

    def configure_compute(self, linear_threads=8):
        from .pardiso import PardisoDirection
        self.parallel_linear = PardisoDirection(linear_threads)
        self.ipc.set_num_threads(4)

    def close_compute(self):
        if self.parallel_linear is not None:
            self.parallel_linear.close()
            self.parallel_linear = None

    def linear_direction(self, matrix, rhs):
        if self.parallel_linear is None:
            return super().linear_direction(matrix, rhs)
        answer = self.parallel_linear.solve(matrix, rhs)
        residual = np.linalg.norm(matrix @ answer-rhs)
        if not np.isfinite(residual) or residual > max(1e-8*np.linalg.norm(rhs), 1e-10):
            raise RuntimeError('Unrelaxed control physical linear residual failed')
        return answer

    def audit_state(self, state, record):
        for name in ('material_metric', 'previous_material_metric'):
            if not np.array_equal(state[name], np.broadcast_to(np.eye(3), state[name].shape)):
                raise ValueError('Unrelaxed control cannot inherit a relaxed material history')
        return super().audit_state(state, record)
