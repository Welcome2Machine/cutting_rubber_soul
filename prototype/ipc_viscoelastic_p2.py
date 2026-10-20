"""Opt-in local P2/plane-stress diagnostics with original native IPC surface.

The same linear exterior and fixed bottom are retained. Interior edge midpoint
DOFs are shared; omitted midpoint DOFs equal endpoint averages on BOTH sides.
Each enriched element has its own quadrature histories; non-enriched elements
keep one exact P1 history. This is conforming displacement enrichment, NOT a
locking-free mixed formulation or a finite-width 3-D model.
"""
import numpy as np
from mechanics.finite_strain import response
from .ipc_p2 import InteriorP2
from .ipc_viscoelastic import ViscoelasticProblem


class _HistoryElement(InteriorP2):
    def bulk(self, u, *, with_tangent=True):
        return self.owner.material_bulk(u, with_tangent=with_tangent)


class QuadratureViscoelasticProblem(ViscoelasticProblem):
    def __init__(self, *, kinematics='plane_strain', enrichment_radius=None,
                 quadrature_order=3, quadrature_refinement=0, **kwargs):
        if kinematics not in ('plane_strain', 'plane_stress'):
            raise ValueError('Explicit PE or total-stress PS required')
        if kwargs.get('element', 'p1') != 'p1':
            raise ValueError('Construct from the original P1 contact mesh')
        super().__init__(**kwargs)
        self.kinematics = kinematics
        self.enrichment_radius = enrichment_radius
        self.base_nodes = self.n
        self.active_elements = np.empty(0, dtype=int)
        self.linear_elements = np.arange(len(self.mesh.triangles))
        if enrichment_radius is not None:
            el = _HistoryElement(self.mesh, self.config, quadrature_order,
                                 enrichment_radius=enrichment_radius,
                                 quadrature_refinement=quadrature_refinement)
            el.owner = self
            self.enrichment = el; self.mesh = el.mesh
            old_n = self.n; blade = self.rest[old_n:].copy()
            self.n = self.mesh.node_count
            self.rest = np.vstack((self.mesh.reference_xy, blade)); self.total = self.rest.size
            self.surface = np.where(self.surface < old_n, self.surface, self.surface+self.n-old_n)
            self.surface_dofs = (2*self.surface[:, None]+np.arange(2)).ravel()
            fixed = (2*self.mesh.bottom_nodes[:, None]+np.arange(2)).ravel()
            if self.case == 'patch':
                fixed = np.unique(np.r_[fixed, 2*np.arange(self.n)])
            self.free = np.setdiff1d(np.arange(2*self.n), fixed)
            self.previous = self.rest.copy()
            self.element = 'p2-local-visco'
            self.active_elements = np.flatnonzero(np.any(~el.boundary_slots, axis=1))
            self.linear_elements = np.flatnonzero(np.all(el.boundary_slots, axis=1))
            assert len(self.active_elements)>0
        areas = self.mesh.bulk_data[0]*self.config.width_mm
        self.integration_weights = areas[self.linear_elements]
        self.point_elements = self.linear_elements.copy()
        if self.enrichment is not None:
            self.integration_weights = np.r_[self.integration_weights,
                (areas[self.active_elements, None]*self.enrichment.weights).ravel()]
            self.point_elements = np.r_[self.point_elements,
                np.repeat(self.active_elements, len(self.enrichment.weights))]
        self.material_metric = np.tile(np.eye(3), (len(self.integration_weights), 1, 1))
        self.previous_material_metric = self.material_metric.copy()
        np.testing.assert_allclose(self.integration_weights.sum(), areas.sum(), rtol=1e-13)

    def material_integration_weights(self):
        return self.integration_weights

    def _deformation(self, u):
        uv = np.asarray(u).reshape(-1, 2)
        tri = self.mesh.triangles[self.linear_elements]
        F = np.eye(2)+np.einsum('eai,eaJ->eiJ', uv[tri], self.grad[self.linear_elements])
        if self.enrichment is not None:
            el = self.enrichment; ids = self.active_elements
            high = np.eye(2)+np.einsum('eai,eqaJ->eqiJ', uv[el.elements[ids]], el.grad[ids])
            F = np.concatenate((F, high.reshape(-1, 2, 2)))
        return F

    def material_response(self, x, *, tangent=False):
        if self.material_dt is None or self.material_dt<=0:
            raise ValueError('Material increment not prepared')
        F = self._deformation(x[:self.n]-self.mesh.reference_xy)
        return self._constitutive_response(F,tangent=tangent)

    def _constitutive_response(self,F,*,tangent):
        """Default exact kernel; explicit compute backends may partition points."""
        return self._audit_response(F,tangent=tangent,kinematics=self.kinematics,
            compute=lambda:response(F,self.previous_material_metric,self.material_dt,self.material,
                                    tangent=tangent,kinematics=self.kinematics))

    def material_bulk(self, u, *, with_tangent=True):
        F = self._deformation(u)
        minimum = float(np.linalg.det(F).min())
        if self.enrichment is not None:
            minimum = self.enrichment.minimum_j(u)
        if minimum <= 0 or not np.isfinite(minimum):
            raise FloatingPointError('Nonpositive true in-plane Jacobian')
        q = self._constitutive_response(F,tangent=with_tangent)
        nt = len(self.mesh.triangles); nl = len(self.linear_elements)
        nd = 6 if self.enrichment is None else 12
        r = np.zeros((nt, nd)); k = np.zeros((nt, nd, nd)) if with_tangent else None
        ids = self.linear_elements; grad = self.grad[ids]; vol = self.integration_weights[:nl]
        r[ids, :6] = np.einsum('e,eiJ,eaJ->eai', vol, q['P'][:nl], grad).reshape(-1, 6)
        if with_tangent:
            C = q['tangent'][:nl].reshape(-1, 2, 2, 2, 2)
            k[ids, :6, :6] = np.einsum('e,eaJ,eiJkL,ebL->eaibk', vol, grad, C, grad,
                                       optimize=True).reshape(-1, 6, 6)
        if self.enrichment is not None:
            el = self.enrichment; ids = self.active_elements; nq = len(el.weights)
            grad = el.grad[ids]; vol = self.integration_weights[nl:].reshape(-1, nq)
            r[ids] = np.einsum('eq,eqiJ,eqaJ->eai', vol, q['P'][nl:].reshape(-1,nq,2,2),
                               grad, optimize=True).reshape(-1, 12)
            if with_tangent:
                C = q['tangent'][nl:].reshape(-1,nq,2,2,2,2)
                for start in range(0, len(ids), 2048):
                    stop = min(start+2048, len(ids)); g = grad[start:stop]
                    z = np.einsum('eqiJkL,eqbL->eqiJbk', C[start:stop], g, optimize=True)
                    k[ids[start:stop]] = np.einsum('eq,eqaJ,eqiJbk->eaibk', vol[start:stop], g, z,
                                                   optimize=True).reshape(-1,12,12)
            dofs = el.dofs
        else:
            dofs = self.dofs
        return r,k,dofs,float(self.integration_weights@q['incremental_potential']),minimum

    def material_acceptance(self, x, depth, accepted):
        accepted = super().material_acceptance(x, depth, accepted)
        q = self.material_response(x)
        if self.kinematics == 'plane_stress' and q['plane_stress_scaled_residual'].max()>1e-8:
            raise RuntimeError('Total plane-stress residual failed')
        if np.min(q['J']) <= 0:
            raise RuntimeError('Nonpositive volume Jacobian')
        accepted['material'].update(kinematics=self.kinematics,
            min_volume_j=float(q['J'].min()), out_stretch_min=float(q['lambda_out'].min()),
            out_stretch_max=float(q['lambda_out'].max()))
        return accepted

    def discretization_data(self):
        data = dict(material_point_elements=self.point_elements,
            material_integration_weights=self.integration_weights,
            kinematics=np.array(self.kinematics), base_body_nodes=self.base_nodes)
        if self.enrichment is not None:
            data.update(self.enrichment.checkpoint_data(),
                enriched_elements=self.active_elements, enrichment_radius_mm=self.enrichment_radius)
        return data

    def checkpoint_data(self):
        return dict(super().checkpoint_data(), **self.discretization_data())

    def audit_state(self, state, record):
        for name, value in self.discretization_data().items():
            if name not in state or not np.array_equal(state[name], value):
                raise RuntimeError('Checkpoint quadrature/discretization differs: '+name)
        return super().audit_state(state, record)
