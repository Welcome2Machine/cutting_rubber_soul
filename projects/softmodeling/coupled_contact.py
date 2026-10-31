"""Fully coupled backward-Euler IPC reference on a pure prescribed precrack.

x1=x0+h*v1; M(v1-v0)/h + g_bulk(x1,A0,h) + g_IPC(x1)
+ g_friction(x1,h*v1) + g_face(x1) = constraint reaction.
No hard blade impulse is added to IPC. The mirror constraint remains exact.
Frozen-metric finite-strain elasticity and implicit Maxwell are distinct layers.
BE numerical work is measured and gated, never called physical damping.
"""
import numpy as np
from scipy.sparse import coo_matrix
from mechanics.discrete_gradient import frozen_response
from prototype.ipc_reference import IPCProblem,LoadStepFailure
from tools.run_precrack_entry import PurePrecrackProblem,problem as original_problem
from face_cut import FaceCutProblem,eligible_nodes
from release_dynamics import check_energy_gate,DEFECT_KEYS


class CoupledContactProblem(PurePrecrackProblem):
    def material_response(self,x,*,tangent=False):
        F=self.deformation(x)
        if self.material_layer=='maxwell':
            return self.constitutive_response(F,tangent=tangent)
        q=frozen_response(F,self.previous_material_metric,self.config.shear_modulus_n_mm2,
                          self.material.mu_ov[0],self.config.bulk_modulus_n_mm2,tangent=tangent)
        q['metric']=self.previous_material_metric.copy()
        q['viscous_rate']=np.zeros(len(F))
        return q

    def material_bulk(self,u,*,with_tangent=True):
        # Re-evaluated at EVERY nonlinear iterate, including the full endpoint
        # tangent (both occurrences of F), not a frozen nodal force predictor.
        x=self.rest.copy();x[:self.n]+=u.reshape(-1,2)
        q=self.material_response(x,tangent=with_tangent)
        area,grad,dofs=self.mesh.bulk_data;volume=area*self.config.width_mm
        r=np.einsum('e,eiJ,eaJ->eai',volume,q['P'],grad).reshape(-1,6)
        k=None
        if with_tangent:
            k=np.einsum('e,eaJ,eiJkL,ebL->eaibk',volume,grad,q['tangent'].reshape(-1,2,2,2,2),grad,optimize=True).reshape(-1,6,6)
        return r,k,dofs,float(volume@q['energy']),float(q['J'].min())

    def lag(self,x):return IPCProblem.lag(self,x)

    def normal_response(self,x,*,tangent=False,projected=False):
        c=self.collisions(x);z=x[self.surface]
        projection=self.ipc.PSDProjectionMethod.CLAMP if projected else self.ipc.PSDProjectionMethod.NONE
        return self.barrier.gradient(c,self.cmesh,z),self.barrier.hessian(c,self.cmesh,z,projection) if tangent else None

    def cohesive_algorithmic(self,x):
        # This candidate is restricted to fully failed prescribed faces and
        # intact exact ties. No partially damaged face can be smuggled into it.
        return self.cohesive_response(x)

    def coupled_friction_tangent(self,x,lag,H):
        if lag is None:return H
        J=-self.friction.force_jacobian(lag,self.cmesh,self.rest[self.surface],
            (x-self.rest)[self.surface],self.step_increment(x)[self.surface],self.barrier,
            self.ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS).tocoo()
        return H+coo_matrix((J.data,(self.surface_dofs[J.row],self.surface_dofs[J.col])),shape=H.shape).tocsc()

    def energy_ledger(self,x):
        # Reuse area-consistent face accounting, replace the material dissipation
        # and inertia of the midpoint/Lie control by THIS discrete equation.
        e=FaceCutProblem.energy_ledger(self,x)
        du=self.step_increment(x);v=self.endpoint_velocity(x)
        inertia=float(self.mass@np.sum((v[:self.n]-self.previous_velocity[:self.n])**2,axis=1))
        D=2*self.material_dt*float(self.material_integration_weights()@self.material_response(x)['viscous_rate'])
        oldD=e['viscous_endpoint_dissipation_n_mm']
        e['viscous_endpoint_dissipation_n_mm']=D
        e['bulk_endpoint_defect_n_mm']+=oldD-D
        e['physical_balance_defect_n_mm']+=oldD-D
        e['backward_euler_inertia_loss_n_mm']=inertia
        e['algebraic_balance_residual_n_mm']=e['physical_balance_defect_n_mm']-sum(e[k] for k in DEFECT_KEYS)-e['friction_time_defect_n_mm']
        return e

    def acceptance(self,x,depth,iterations,outer_iterations):
        rec=IPCProblem.acceptance(self,x,depth,iterations,outer_iterations)
        rec['residual_n']*=np.sqrt(2)
        if rec['residual_n']>1e-4:raise LoadStepFailure('Coupled full-domain residual exceeds 1e-4 N')
        for k in ('reaction_fz_n','reaction_normal_fz_n','reaction_friction_fz_n'):rec[k]*=2
        rec['reaction_x_n']=0.
        g=self.evaluate(x,self.lag(x),tangent=False)[1].reshape(-1,2)
        active=self.active_contact.copy();active[self.seam[~self.released]]=False
        violation=max(0.,float(-g[:self.n,0][active].min(initial=0.)))
        if violation>1e-5 or x[:self.n,0].min()<self.config.drop_s_mm-1e-12:raise LoadStepFailure('Unilateral contact violation')
        if np.any(x[self.seam[~self.released],0]!=self.config.drop_s_mm):raise RuntimeError('Exact intact tie lost')
        np.testing.assert_array_equal(self.released,eligible_nodes(self.active_faces))
        np.testing.assert_array_equal(self.active_faces,self.precracked_faces)
        q=self.material_response(x);a=q['metric']
        if np.min(np.linalg.eigvalsh(a))<=0 or np.max(abs(np.linalg.det(a)-1))>1e-8:raise RuntimeError('Invalid Maxwell metric')
        e=self.energy_ledger(x)
        if min(e['friction_work_n_mm'],e['viscous_endpoint_dissipation_n_mm']) < -1e-12:raise RuntimeError('Negative physical dissipation')
        gate=check_energy_gate(self,e)
        rec.update(energy=e,energy_gate=gate,unilateral_force_violation_n=violation,
            material_layer=self.material_layer,time_integrator='fully_coupled_backward_euler_IPC',
            reaction_sampling='endpoint backward-Euler force; contact impulse = h times force',
            material=dict(dt_s=self.material_dt,stored_energy_n_mm=float(self.material_integration_weights()@q['energy']),
                viscous_dissipation_increment_n_mm=.5*e['viscous_endpoint_dissipation_n_mm'],
                max_viscous_det_error=float(np.max(abs(np.linalg.det(a)-1)))),
            released_nodes=int(self.released.sum()),face_events=[],
            surface_connected_failed_length_mm=self.connected_length(self.precracked_faces))
        return rec


def problem(config,layer='frozen_metric'):
    if layer not in ('frozen_metric','maxwell'):raise ValueError('Explicit material layer required')
    p=original_problem(config)
    p.__class__=CoupledContactProblem
    p.material_layer=layer;p.kinematic_theta=1.
    # Face assembler uses du for friction; lag() above uses endpoint geometry.
    # Thus du=h*v1 including prescribed blade motion, unlike old midpoint law.
    p.friction_time='midpoint'
    return p
