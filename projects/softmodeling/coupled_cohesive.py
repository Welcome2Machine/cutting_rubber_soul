"""Endpoint softening on the coupled BE/IPC equation, from its own initial state.

P1 full opening, full reference face area and the original secant unloading law
share one quadrature history. Gc is paid by irreversible area-integrated D only;
activation has no upfront Gc debit. Endpoint cohesive work error is kept in the
absolute numerical gate. This is the continuous face route, not an FFM decision.
"""
import time
import numpy as np
from coupled_contact import CoupledContactProblem
from face_cut import eligible_nodes
from tools.run_precrack_entry import reference_problem
from tools.face_events import check_connected_endpoint
from prototype.ipc_reference import LoadStepFailure
from step_control import PhysicalSolveAccuracyIncomplete,GlobalComputeBudgetExhausted
from prototype.zero_gap_cut import EnergyResolutionRequired


class CoupledCohesiveProblem(CoupledContactProblem):
    def affine_loading_candidate(self,x,old,depth):
        candidate=super().affine_loading_candidate(x,old,depth)
        # Enforce the prescribed analytical boundary at the predictor, before
        # all nonlinear/trajectory checks; never repair accepted states later.
        candidate[self.n:]=self.prescribed_blade_positions(depth)
        return candidate

    def step(self,x,old,depth,deadline):
        self.nonlinear_accuracy_certificate=None
        try:return super().step(x,old,depth,deadline)
        except (LoadStepFailure,TimeoutError) as exc:
            if isinstance(exc,(EnergyResolutionRequired,GlobalComputeBudgetExhausted)) or time.monotonic()>=getattr(self,'global_compute_deadline',float('inf')):raise
            # Rollback is already complete. Inspect the actual failed graph,
            # then restore the committed graph before propagating any failure.
            if getattr(self,'failed_stage',None)!='force_solve':raise
            saved={k:getattr(self,k).copy() for k in ('active_faces','released','active_contact')}
            residual=None
            try:
                for k in saved:setattr(self,k,getattr(self,'failed_'+k).copy())
                self.rebuild_contact();self.set_free()
                y=self.failed_positions;self.set_primary_increment(y,self.failed_increment)
                g=self.evaluate(y,self.lag(y),tangent=False)[1]
                residual=float(np.sqrt(2)*np.linalg.norm(g[self.free]))
            except (RuntimeError,ValueError,FloatingPointError):pass
            finally:
                for k,v in saved.items():setattr(self,k,v)
                self.rebuild_contact();self.set_free()
            # An original physical force failure may still use physical
            # cutback. A stricter solver-goal failure cannot certify a new dt.
            target=self.force_tolerance_n*np.sqrt(2)
            if residual is not None and target<residual<=1e-4:
                error=PhysicalSolveAccuracyIncomplete(
                    f'Numerical force accuracy unresolved: {residual:.17g} N > goal {target:.17g} N; original physical gate 1e-4 N met; no physical cutback')
                error.failure_measurement=dict(quantity='numerical_full_force_residual_n',value=residual,limit=target,units='N',comparison='upper')
                raise error from exc
            raise

    def check_face_topology(self,x):
        if self.pure_precrack:raise RuntimeError('Growth model must be explicitly selected')
        if np.any(self.previous_active_faces&~self.active_faces):raise RuntimeError('Activated face healed')
        np.testing.assert_array_equal(self.released,eligible_nodes(self.active_faces))
        q=self.cohesive_response(x);k=q['history'];old=self.previous_history
        if np.any(k<old) or np.any(k[~self.active_faces]!=0):raise RuntimeError('Invalid face-point history')
        if np.any(k[self.precracked_faces]<2*self.toughness/self.strength):raise RuntimeError('Initial crack healed')
        weights=self.rule['weights_mm2']*self.active_faces[:,None]
        if np.any(q['dissipated']>self.toughness+1e-14):raise RuntimeError('Gc charged more than once per full area')
        if np.sum(weights*q['dissipated'])>self.toughness*np.sum(weights)+1e-12:raise RuntimeError('Incorrect fracture area')

    def acceptance(self,x,depth,iterations,outer_iterations):
        endpoint=check_connected_endpoint(self,x)
        rec=super().acceptance(x,depth,iterations,outer_iterations)
        if getattr(self,'nonlinear_error_policy',None):
            from nonlinear_accuracy import binding,residual_budget
            g=self.evaluate(x,self.lag(x),tangent=False)[1]
            certificate=getattr(self,'nonlinear_accuracy_certificate',None)
            if rec['residual_n']>self.force_tolerance_n*np.sqrt(2):
                if not certificate or not certificate['passed'] or certificate['binding_sha256']!=binding(x,self.step_increment(x)):
                    # Restored-state audits must rebuild the certificate, not
                    # trust a saved producer Boolean or require live metadata.
                    from nonlinear_accuracy import inspect as inspect_accuracy
                    certificate=inspect_accuracy(self,x)
                    if not certificate['passed']:
                        raise PhysicalSolveAccuracyIncomplete('Nonlinear error certificate failed: '+certificate.get('reason',''))
            rec['nonlinear_accuracy']=dict(preferred_goal_met=rec['residual_n']<=self.force_tolerance_n*np.sqrt(2),certificate=certificate,**residual_budget(self,x,g))
        q=self.cohesive_response(x);failed=self.active_faces&np.all(q['history']>=2*self.toughness/self.strength,axis=1)
        rec.update(endpoint,fracture_route='coupled_backward_euler_continuous_face',
            active_faces=int(self.active_faces.sum()),fully_failed_faces=int(failed.sum()),
            fully_failed_nodes=int(eligible_nodes(failed).sum()),precracked_faces=int(self.precracked_faces.sum()),
            face_events=self.face_events,surface_connected_failed_length_mm=self.connected_length(failed),
            quadrature_order=self.rule['shape'].shape[0],initial_fracture_energy_excluded_n_mm=self.initial_fracture_energy_n_mm,
            root_tolerance_full_n=self.force_tolerance_n*np.sqrt(2),handoff_tolerance_full_n=self.activation_tolerance_n*np.sqrt(2))
        return rec


def problem(config,layer='maxwell',solve_tolerance_n=1e-9):
    if config.get('pure_precrack') is not False:raise ValueError('Explicit continuous growth configuration required')
    if layer not in ('maxwell','frozen_metric') or not 0<solve_tolerance_n<=1e-4:raise ValueError('Invalid material/root control')
    p=reference_problem(config,require_pure=False)
    p.__class__=CoupledCohesiveProblem;p.material_layer=layer;p.kinematic_theta=1.;p.friction_time='midpoint'
    p.force_tolerance_n=solve_tolerance_n/np.sqrt(2)
    p.activation_tolerance_n=1e-4/np.sqrt(2)
    p.history=p.cohesive_response(p.rest,p.history)['history'].copy();p.previous_history=p.history.copy()
    p.check_face_topology(p.rest)
    return p
