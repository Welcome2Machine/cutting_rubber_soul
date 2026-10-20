"""Explicit intact P1 candidate: equilibrium rubber plus isochoric Maxwell.

No damage, no bulk relaxation, no EPDM constants, no change to default IPC.
Parameters are hypotheses until independent material/rate data identify them.
Every nonlinear trial integrates from the last accepted material metric; only
a physically accepted load step commits it. Geometric cutbacks reduce real dt.
"""
from dataclasses import replace
import numpy as np
from mechanics.finite_strain import response
from .material import from_config
from .ipc_reference import IPCProblem


class ViscoelasticProblem(IPCProblem):
    def __init__(self, *, maxwell_shear, relaxation_s, loading_rate,
                 bulk_modulus=None, **kwargs):
        if not all(np.isfinite(x) and x>0 for x in
                   (maxwell_shear,relaxation_s,loading_rate)):
            raise ValueError('Explicit positive Maxwell shear, time and loading rate required')
        if kwargs.get('element','p1')!='p1' or kwargs.get('mu',.35)<=0:
            raise ValueError('This candidate requires P1 with the existing friction PTC host')
        if bulk_modulus is not None and (not np.isfinite(bulk_modulus) or bulk_modulus<=0):
            raise ValueError('Explicit bulk modulus must be finite and positive')
        super().__init__(**kwargs)
        if bulk_modulus is not None:
            # P1 has no cached pressure/enrichment operator. Geometry/contact
            # initialization is independent of Poisson ratio. Keep K fixed when
            # repartitioning equilibrium and relaxing shear stiffness.
            g=self.config.shear_modulus_n_mm2;k=float(bulk_modulus)
            self.config=replace(self.config,poisson_ratio=(3*k-2*g)/(2*(3*k+g)))
            self.config.validate()
            if not np.isclose(self.config.bulk_modulus_n_mm2,k,rtol=1e-12,atol=0):
                raise ValueError('Bulk modulus is not representable at this shear/Poisson ratio')
        if self.config.friction_coefficient<=0:
            raise ValueError('Use the friction PTC host for this material candidate')
        self.material=replace(from_config(self.config),mu_ov=(float(maxwell_shear),),
                              alpha_ov=(2.,),relaxation_s=float(relaxation_s),
                              viscous_flow='isochoric_maxwell')
        self.loading_rate=float(loading_rate)
        self.material_metric=np.tile(np.eye(3),(len(self.mesh.triangles),1,1))
        self.previous_material_metric=self.material_metric.copy()
        self.material_dt=None
        self.material_dissipation=0.

    def material_response(self, x, *, tangent=False):
        if self.material_dt is None or self.material_dt<=0:
            raise ValueError('Material increment not prepared')
        u=x[:self.n]-self.mesh.reference_xy
        F=np.eye(2)+np.einsum('eai,eaJ->eiJ',u[self.mesh.triangles],self.grad)
        return self.constitutive_response(F,tangent=tangent)

    def constitutive_response(self,F,*,tangent):
        """Optional bounded batches; every point keeps its OWN committed history.

        The default is the original single batch. Smaller independent batches
        reduce temporary tensor storage and avoid iterating far-field points
        until the most difficult point has converged. Local tolerances, material
        law and tangent are unchanged; no results are cached between evaluations.
        """
        size=getattr(self,'material_batch_size',0)
        if not isinstance(size,int) or size<0:
            raise ValueError('Material batch size must be a nonnegative integer')
        if not size or size>=len(F):
            return response(F,self.previous_material_metric,self.material_dt,self.material,
                            tangent=tangent,kinematics='plane_strain')
        batches=[response(F[i:i+size],self.previous_material_metric[i:i+size],
                    self.material_dt,self.material,tangent=tangent,kinematics='plane_strain')
                 for i in range(0,len(F),size)]
        return {key:np.concatenate([b[key] for b in batches],axis=0) for key in batches[0]}

    def material_bulk(self,u,*,with_tangent=True):
        areas,gradients,dofs=self.mesh.bulk_data
        F=np.eye(2)+np.einsum('eai,eaJ->eiJ',u.reshape(-1,2)[self.mesh.triangles],gradients)
        q=self.constitutive_response(F,tangent=with_tangent)
        volumes=areas*self.config.width_mm
        r=np.einsum('e,eiJ,eaJ->eai',volumes,q['P'],gradients).reshape(-1,6)
        k=None
        if with_tangent:
            C=q['tangent'].reshape(-1,2,2,2,2)
            k=np.einsum('e,eaJ,eiJkL,ebL->eaibk',volumes,gradients,C,gradients,
                        optimize=True).reshape(-1,6,6)
        return r,k,dofs,float(volumes@q['incremental_potential']),float(q['J'].min())

    def acceptance(self,x,depth,iterations,outer_iterations):
        accepted=super().acceptance(x,depth,iterations,outer_iterations)
        return self.material_acceptance(x,depth,accepted)

    def material_acceptance(self,x,depth,accepted):
        """Attach the same material audit to an independently accepted contact state."""
        q=self.material_response(x)
        metric=q['metric']
        if (not np.isfinite(metric).all() or np.min(np.linalg.eigvalsh(metric))<=0
                or np.max(np.abs(np.linalg.det(metric)-1))>1e-8):
            raise RuntimeError('Invalid isochoric viscous material history')
        if not np.isfinite(q['viscous_rate']).all() or np.min(q['viscous_rate']) < -1e-10:
            raise RuntimeError('Negative/nonfinite material dissipation')
        volumes=self.material_integration_weights()
        dissipation=self.material_dt*float(volumes@q['viscous_rate'])
        accepted['material']=dict(
            dt_s=self.material_dt,time_s=max(0.,float(depth))/self.loading_rate,
            stored_energy_n_mm=float(volumes@q['energy']),
            equilibrium_energy_n_mm=float(volumes@q['equilibrium_energy']),
            overstress_energy_n_mm=float(volumes@q['overstress_energy']),
            incremental_potential_n_mm=float(volumes@q['incremental_potential']),
            viscous_dissipation_increment_n_mm=dissipation,
            max_viscous_det_error=float(np.max(np.abs(np.linalg.det(metric)-1))),
            dissipation_quadrature='backward Euler endpoint; not exact integrated work')
        return accepted

    def material_integration_weights(self):
        """Reference volumes; explicit quadrature candidates override this."""
        return self.mesh.bulk_data[0]*self.config.width_mm

    def step(self,x,old,depth,deadline):
        dt=(float(depth)-float(old))/self.loading_rate
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive physical increment required')
        self.previous_material_metric=self.material_metric.copy()
        self.material_dt=dt
        # No update of committed material history occurs inside super.step.
        # If it rejects/times out, the next attempt starts from the same metric.
        value,record=super().step(x,old,depth,deadline)
        trial=self.material_response(value)
        self.material_metric=trial['metric'].copy()
        self.material_dissipation+=record['material']['viscous_dissipation_increment_n_mm']
        record['material']['cumulative_viscous_dissipation_n_mm']=self.material_dissipation
        return value,record

    def checkpoint_data(self):
        return dict(material_metric=self.material_metric,
                    previous_material_metric=self.previous_material_metric,
                    material_dt_s=self.material_dt,
                    cumulative_viscous_dissipation_n_mm=self.material_dissipation)

    def _audit_extra_acceptance(self,fresh,record):
        return {}

    def _audit_response(self,F,*,tangent,kinematics,compute):
        """Reuse only identical constitutive work WITHIN one fresh audit.

        Never retain a response across snapshots or physical Newton steps.
        Exact deformation/history comparisons also protect callers that mutate
        arrays in place. Geometry, force assembly and all acceptance gates still
        execute independently of this pure material evaluation.
        """
        if not hasattr(self,'_fresh_material_cache'):return compute()
        key=(self.material_dt,self.material,kinematics,tangent)
        saved=self._fresh_material_cache
        if (saved is not None and saved[0]==key and np.array_equal(saved[1],F)
                and np.array_equal(saved[2],self.previous_material_metric)):
            return saved[3]
        value=compute()
        self._fresh_material_cache=(key,F.copy(),self.previous_material_metric.copy(),value)
        return value

    def audit_state(self,state,record):
        """Fresh original force balance using BOTH exact pre-step histories."""
        before=(self.previous.copy(),self.previous_material_metric.copy(),self.material_dt)
        if hasattr(self,'_fresh_material_cache'):raise RuntimeError('Nested material audit')
        self._fresh_material_cache=None
        try:
            self.previous=state['previous_positions'].copy()
            self.previous_material_metric=state['previous_material_metric'].copy()
            self.material_dt=float(state['material_dt_s'])
            positions=state['positions']
            depth=float(state['depth_mm'])
            expected_blade=self.prescribed_blade_positions(depth)
            if (not np.array_equal(state['reference_xy'],self.rest)
                    or not np.array_equal(state['triangles'],self.mesh.triangles)
                    or not np.array_equal(positions[self.mesh.bottom_nodes],self.rest[self.mesh.bottom_nodes])
                    or not np.allclose(positions[self.n:],expected_blade,rtol=0,
                                       atol=self.blade_boundary_roundoff_tolerance(depth))):
                raise RuntimeError('Checkpoint geometry or prescribed boundaries differ')
            fresh=self.acceptance(positions,float(state['depth_mm']),0,0)
            trial=self.material_response(positions)
            if not np.allclose(trial['metric'],state['material_metric'],rtol=1e-10,atol=1e-11):
                raise RuntimeError('Material history does not reproduce saved accepted metric')
            if not np.isclose(fresh['reaction_fz_n'],record['reaction_fz_n'],rtol=1e-9,atol=1e-8):
                raise RuntimeError('Fresh force differs from saved force')
            return dict(depth_mm=record['depth_mm'],residual_n=fresh['residual_n'],
                        min_j=fresh['min_j'],reaction_fz_n=fresh['reaction_fz_n'],passed=True,
                        **self._audit_extra_acceptance(fresh,record))
        finally:
            del self._fresh_material_cache
            self.previous,self.previous_material_metric,self.material_dt=before
