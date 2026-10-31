"""Endpoint softening on the coupled BE/IPC equation, from its own initial state.

P1 full opening, full reference face area and the original secant unloading law
share one quadrature history. Gc is paid by irreversible area-integrated D only;
activation has no upfront Gc debit. Endpoint cohesive work error is kept in the
absolute numerical gate. This is the continuous face route, not an FFM decision.
"""
import numpy as np
from coupled_contact import CoupledContactProblem
from face_cut import eligible_nodes
from tools.run_precrack_entry import reference_problem
from tools.face_events import check_connected_endpoint


class CoupledCohesiveProblem(CoupledContactProblem):
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
