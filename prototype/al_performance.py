"""Explicit AL compute backend: partitioned materials and multicore PARDISO.

No model, contact equation, time schedule, precision or acceptance changes.
The original class remains the default and independent audit implementation.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from mechanics.finite_strain import response
from .augmented_blade import AugmentedBladeProblem
from .quadratic import quadratic_minimum
from .pardiso import PardisoDirection


class ActiveJacobianMixin:
    """Same whole-triangle/time Bernstein test; linear cells need one F only."""
    def _jacobian_fields(self,u):
        uv=np.asarray(u).reshape(-1,2);p=self.owner
        linear=np.eye(2)+np.einsum('eai,eaJ->eiJ',uv[p.mesh.triangles[p.linear_elements]],
                                  p.grad[p.linear_elements])
        ids=p.active_elements
        curved=np.eye(2)+np.einsum('eai,eqaJ->eqiJ',uv[self.elements[ids]],self.vertex_grad[ids])
        return linear,curved

    def minimum_j(self,u):
        linear,curved=self._jacobian_fields(u)
        return float(min(np.linalg.det(linear).min(initial=np.inf),
                         quadratic_minimum(np.linalg.det(curved)).min(initial=np.inf)))

    def safe_step(self,u,direction):
        fl,fc=self._jacobian_fields(u);dl,dc=self._jacobian_fields(direction)
        dl-=np.eye(2);dc-=np.eye(2)
        def coeff(F,D):
            return (np.linalg.det(F),F[...,0,0]*D[...,1,1]+D[...,0,0]*F[...,1,1]
                    -F[...,0,1]*D[...,1,0]-D[...,0,1]*F[...,1,0],np.linalg.det(D))
        cl,bl,al=coeff(fl,dl);cc,bc,ac=coeff(fc,dc)
        def minimum(l,c):return min(l.min(initial=np.inf),quadratic_minimum(c).min(initial=np.inf))
        if minimum(cl,cc)<=0:raise FloatingPointError('Invalid initial P2 map')
        alpha=1.
        for _ in range(60):
            if min(minimum(cl+.5*alpha*bl,cc+.5*alpha*bc),
                   minimum(cl+alpha*bl+alpha*alpha*al,cc+alpha*bc+alpha*alpha*ac))>0:
                return alpha
            alpha*=.5
        raise FloatingPointError('No certified positive P2 step')


class ParallelMaterial:
    def __init__(self,workers=4,batch_size=4096):
        if workers<1 or batch_size<1:raise ValueError('Positive workers and batch size required')
        self.workers=int(workers);self.batch_size=int(batch_size)
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='material') if workers>1 else None

    def evaluate(self,F,A,dt,material,*,tangent,kinematics):
        spans=[(i,min(i+self.batch_size,len(F))) for i in range(0,len(F),self.batch_size)]
        def evaluate(span):
            i,j=span
            return response(F[i:j],A[i:j],dt,material,tangent=tangent,kinematics=kinematics)
        parts=list(self.pool.map(evaluate,spans)) if self.pool else [evaluate(s) for s in spans]
        return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}

    def close(self):
        if self.pool:self.pool.shutdown(wait=True);self.pool=None


        # The package owns a process-global MKL handle; retain the DLL search
        # registration until process exit so later library calls remain valid.


class FastAugmentedBladeProblem(AugmentedBladeProblem):
    def configure_compute(self,*,material_workers=2,batch_size=4096,linear_threads=8,
                          linear_backend='pardiso',runtime=None,active_jacobian=True):
        if linear_backend not in ('superlu','pardiso'):raise ValueError('Unknown linear backend')
        self.parallel_material=ParallelMaterial(material_workers,batch_size)
        self.parallel_linear=PardisoDirection(linear_threads,runtime) if linear_backend=='pardiso' else None
        self.compute_settings=dict(material_workers=material_workers,batch_size=batch_size,
            linear_threads=linear_threads,linear_backend=linear_backend,active_jacobian=active_jacobian)
        if active_jacobian:
            self.enrichment.__class__=type('ActiveHistoryElement',(ActiveJacobianMixin,type(self.enrichment)),{})
        return self

    def _constitutive_response(self,F,*,tangent):
        return self.parallel_material.evaluate(F,self.previous_material_metric,self.material_dt,
            self.material,tangent=tangent,kinematics=self.kinematics)

    def linear_direction(self,matrix,rhs):
        return self.parallel_linear.solve(matrix,rhs) if self.parallel_linear else super().linear_direction(matrix,rhs)

    def close_compute(self):
        self.parallel_material.close()
        if self.parallel_linear:self.parallel_linear.close()
