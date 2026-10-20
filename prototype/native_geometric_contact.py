"""Native P1/blade IPC with independently oriented GCP BODY self-contact.

The combined mesh keeps native full CCD. A distinct collision mesh filters
only force/lag assembly to body--blade pairs. No Python broad-phase callbacks,
weak blade multipliers, P2 enrichment, or transferred old material histories.
"""
import json
import numpy as np
from scipy.sparse import coo_matrix
from .ipc_reference import BASELINE, determinant_step
from .ipc_viscoelastic import ViscoelasticProblem
from .geometric_self_contact import GeometricSelfContact, orient_boundary

POLICY='native-blade-ipc-body-gcp-p1-v1'


class NativeGeometricProblem(ViscoelasticProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.element!='p1' or self.contact_formulation!='improved-max':
            raise ValueError('Explicit native P1/improved-max blade candidate only')
        self.full_ccd_mesh=self.cmesh
        count=int(np.count_nonzero(self.surface<self.n))
        self.body_surface=self.surface[:count]
        edges=np.asarray(self.cmesh.edges)
        body_edges=self.surface[edges[np.all(edges<count,axis=1)]]
        oriented=orient_boundary(self.rest,self.mesh.triangles,body_edges)
        self.self_dofs=(2*self.body_surface[:,None]+np.arange(2)).ravel()
        self.geometric_self=GeometricSelfContact(self.ipc,self.rest[self.body_surface],
            np.searchsorted(self.body_surface,oriented),dhat=self.dhat,
            stiffness_mpa=BASELINE['barrier_stiffness_mpa'],width_mm=self.config.width_mm,
            mu=self.config.friction_coefficient,eps=BASELINE['friction_smoothing_mm'])
        self.cmesh=self.ipc.CollisionMesh(self.rest[self.surface],edges)
        # Keep connectivity/area weights identical; change interaction set ONLY.
        self.cmesh.can_collide=self.ipc.make_vertex_patches_filter(
            (self.surface>=self.n).astype(np.int32))
        self.reference_contact_check=self.check_reference_contact()
        self.parallel_material=None;self.parallel_linear=None
        self.compute_settings=dict(material_workers=1,linear_backend='superlu',linear_threads=1)

    def configure_compute(self, material_workers=2, linear_threads=8):
        from .al_performance import ParallelMaterial,PardisoDirection
        self.parallel_material=ParallelMaterial(material_workers)
        self.parallel_linear=PardisoDirection(linear_threads)
        self.compute_settings=dict(material_workers=material_workers,linear_backend='pardiso',linear_threads=linear_threads)
        return self

    def close_compute(self):
        if self.parallel_material:self.parallel_material.close();self.parallel_material=None
        if self.parallel_linear:self.parallel_linear.close();self.parallel_linear=None

    def constitutive_response(self,F,*,tangent):
        if self.parallel_material:
            return self.parallel_material.evaluate(F,self.previous_material_metric,self.material_dt,
                self.material,tangent=tangent,kinematics='plane_strain')
        return super().constitutive_response(F,tangent=tangent)

    def linear_direction(self,matrix,rhs):
        if not self.parallel_linear:return super().linear_direction(matrix,rhs)
        answer=self.parallel_linear.solve(matrix,rhs)
        residual=np.linalg.norm(matrix@answer-rhs)
        if not np.isfinite(residual) or residual>max(1e-8*np.linalg.norm(rhs),1e-10):
            raise RuntimeError('Native/GCP physical linear solve failed')
        return answer

    def _scatter(self,matrix):
        m=matrix.tocoo()
        return coo_matrix((m.data,(self.self_dofs[m.row],self.self_dofs[m.col])),
                          shape=(self.total,self.total)).tocsc()

    def evaluate(self,x,lag=None,tangent=True,projected=True,*,with_metric=False,with_pressure_blocks=False):
        if with_pressure_blocks:raise ValueError('No mixed/P2 blocks in native/GCP candidate')
        result=list(super().evaluate(x,lag,tangent,projected,with_metric=with_metric))
        metric=bool(tangent and (projected or with_metric))
        terms=self.geometric_self.terms(x[self.body_surface],self.previous[self.body_surface],
                                        tangent=tangent,with_metric=metric)
        result[0]+=terms[0];result[1][self.self_dofs]+=terms[1]
        if tangent:
            result[2]+=self._scatter(terms[4] if projected else terms[2])
            if with_metric:result[4]+=self._scatter(terms[4])
        return tuple(result)

    def safe_step(self,x,direction):
        # Full original body/body AND blade/body CCD, irrespective of force filter.
        F=np.eye(2)+np.einsum('eai,eaJ->eiJ',
            (x[:self.n]-self.mesh.reference_xy)[self.mesh.triangles],self.grad)
        D=np.einsum('eai,eaJ->eiJ',direction[:self.n][self.mesh.triangles],self.grad)
        return min(determinant_step(F,D),self.ipc.compute_collision_free_stepsize(
            self.full_ccd_mesh,x[self.surface],(x+direction)[self.surface]))

    def acceptance(self,x,depth,iterations,outer_iterations):
        result=super().acceptance(x,depth,iterations,outer_iterations)
        if self.ipc.has_intersections(self.full_ccd_mesh,x[self.surface]):
            raise RuntimeError('Native/GCP full boundary intersection at acceptance')
        result['geometric_self_contact']=dict(self.geometric_self.last)
        result['body_self_contact_policy']=POLICY
        result['blade_friction_work_n_mm']=result['friction_work_n_mm']
        result['friction_work_n_mm']+=self.geometric_self.last['friction_work_n_mm']
        return result

    def checkpoint_data(self):
        return dict(super().checkpoint_data(),body_self_contact_policy=np.array(POLICY),
            geometric_self_spec=np.array(json.dumps(self.geometric_self.spec,sort_keys=True)))

    def audit_state(self,state,record):
        if str(state.get('body_self_contact_policy',''))!=POLICY:raise ValueError('Native/GCP policy mismatch')
        if str(state.get('geometric_self_spec',''))!=json.dumps(self.geometric_self.spec,sort_keys=True):
            raise ValueError('Native/GCP specification mismatch')
        return super().audit_state(state,record)

    def _audit_extra_acceptance(self,fresh,record):
        for key,value in fresh['geometric_self_contact'].items():
            saved=record['geometric_self_contact'][key]
            if value is None or saved is None:
                if value!=saved:raise ValueError('GCP missing-value mismatch: '+key)
            else:np.testing.assert_allclose(value,saved,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(fresh['friction_work_n_mm'],record['friction_work_n_mm'],rtol=1e-9,atol=1e-10)
        return dict(geometric_self_contact=fresh['geometric_self_contact'])
