"""Face-consistent extrinsic mode-I route on the existing symmetric geometry.

An intact face ties BOTH P1 endpoints. An activated face carries the original
extrinsic law at fixed Gauss points and is a real IPC boundary. A node becomes
eligible to open only when all its incident seam faces have activated. Thus an
interior opening needs a two-face support; a surface endpoint needs one face.
This is a separate continuous cohesive route, not the finite-increment rule.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix, diags
from prototype.energy_cut import EnergyCutProblem
from prototype.ipc_reference import IPCProblem, LoadStepFailure
from prototype.zero_gap_cut import extrinsic_response, EnergyResolutionRequired
from mechanics.discrete_gradient import cohesive_discrete_response
from face_cohesive_quadrature import face_rule
from release_dynamics import DEFECT_KEYS, check_energy_gate
from endpoint_friction import velocity_increment, coupled_tangent


def eligible_nodes(faces):
    """Exact union of constraints of every remaining intact P1 face."""
    faces = np.asarray(faces, bool)
    tied = np.zeros(len(faces)+1, bool)
    tied[:-1] |= ~faces
    tied[1:] |= ~faces
    return ~tied


class FaceCutProblem(EnergyCutProblem):
    def initialize_faces(self, order=4, precrack_mm=0., friction_time='midpoint'):
        if self.history.any() or self.released.any() or np.any(self.previous != self.rest):
            raise ValueError('Face route requires its own virgin initial state; no nodal history transfer')
        if friction_time not in ('midpoint', 'endpoint_friction') or self.kinematic_theta != .5:
            raise ValueError('Face route requires theta=.5 and an explicit supported friction time rule')
        self.friction_time = friction_time
        self.rule = face_rule(self.rest[self.seam, 1], self.config.width_mm, order)
        y = self.rest[self.seam, 1]
        if not np.isfinite(precrack_mm) or not 0 <= precrack_mm < y[-1]-y[0]:
            raise ValueError('Precrack must be nonnegative and shorter than specimen')
        tip = y[-1]-precrack_mm
        if np.min(np.abs(y-tip)) > 1e-10:
            raise ValueError('Precrack tip must coincide with a reference face endpoint; no rounded length')
        self.precracked_faces = y[:-1] >= tip-1e-10 if precrack_mm else np.zeros(len(y)-1, bool)
        self.active_faces = self.precracked_faces.copy()
        self.previous_active_faces = self.active_faces.copy()
        self.history = np.zeros(self.rule['history_shape'])
        self.history[self.precracked_faces] = 2*self.toughness/self.strength
        self.previous_history = self.history.copy()
        self.released = eligible_nodes(self.active_faces)
        self.rebuild_contact(); self.set_free()
        self.initial_fracture_energy_n_mm = float(self.toughness*self.rule['face_area_mm2'][self.precracked_faces].sum())
        return self

    def rebuild_contact(self):
        if not hasattr(self, 'active_faces'):
            return super().rebuild_contact()
        event = None
        if hasattr(self, 'pending_faces'):
            x = self.activation_positions
            before = 2*float(self.barrier(self.collisions(x),self.cmesh,x[self.surface]))
            event = self.pending_handoff.copy()
            self.active_faces = self.pending_faces.copy()
            self.released = eligible_nodes(self.active_faces)
            del self.pending_faces
        edges = self.all_boundary_edges
        seam_edges = np.all(np.isin(edges, self.seam), axis=1)
        pairs = {tuple(sorted((int(a),int(b)))) for a,b in zip(self.seam[:-1][self.active_faces],self.seam[1:][self.active_faces])}
        keep = ~seam_edges | np.array([tuple(sorted(map(int,e))) in pairs for e in edges])
        edges = edges[keep]
        self.surface = np.unique(edges)
        self.surface_dofs = (2*self.surface[:,None]+np.arange(2)).ravel()
        self.cmesh = self.ipc.CollisionMesh(self.rest[self.surface],np.searchsorted(self.surface,edges))
        self.cmesh.can_collide = self.ipc.make_static_obstacle_filter(int(np.count_nonzero(self.surface < self.n)))
        self._connected_boundary_pairs = {tuple(sorted(map(int,e))) for e in self.cmesh.edges}
        if event is not None:
            after = 2*float(self.barrier(self.collisions(x),self.cmesh,x[self.surface]))
            event['contact_topology_energy_jump_n_mm'] = after-before
            self.face_events.append(event)
            self.evidence('activation_after',x,event)
            if abs(after-before)>1e-10:
                raise RuntimeError('Unfunded face-contact topology energy jump; model handoff unresolved')

    def opening_at_points(self, x):
        u = 2*(x[self.seam,0]-self.config.drop_s_mm)
        return u[:-1,None]*self.rule['shape'][None,:,0]+u[1:,None]*self.rule['shape'][None,:,1]

    def cohesive_response(self, x, history=None):
        old=self.previous_history if history is None else history
        q=extrinsic_response(self.opening_at_points(x),old,self.strength,self.toughness)
        if getattr(self,'pure_precrack',False):q['history']=old.copy()
        return q

    def cohesive_algorithmic(self, x):
        return cohesive_discrete_response(self.opening_at_points(self.previous),self.opening_at_points(x),
                                           self.previous_history,self.strength,self.toughness)

    def face_force(self, q, projected=False):
        weights = self.rule['weights_mm2']*self.active_faces[:,None]
        shape = self.rule['shape']
        local = np.einsum('eq,eq,qi->ei',weights,q['traction'],shape)
        force = np.zeros(len(self.seam)); force[:-1] += local[:,0]; force[1:] += local[:,1]
        tangent = np.maximum(q['tangent'],0.) if projected else q['tangent']
        # Full opening = 2*x_right; half-body gradient has no factor two,
        # its derivative with respect to x_right does have factor two.
        local_h = 2*np.einsum('eq,eq,qi,qj->eij',weights,tangent,shape,shape)
        ids = np.column_stack((2*self.seam[:-1],2*self.seam[1:]))
        H = coo_matrix((local_h.ravel(),(np.broadcast_to(ids[:,:,None],local_h.shape).ravel(),
                 np.broadcast_to(ids[:,None,:],local_h.shape).ravel())),shape=(self.total,self.total)).tocsc()
        return force,H

    def lag(self, x):
        return IPCProblem.lag(self,x) if self.friction_time=='endpoint_friction' else super().lag(x)

    def evaluate(self, x, lag=None, tangent=True, projected=True, *, with_metric=False):
        if with_metric and not tangent: raise ValueError('Paired metric requires tangent')
        r,k,dofs,energy,minj = self.material_bulk((x[:self.n]-self.mesh.reference_xy).ravel(),with_tangent=tangent)
        g = np.zeros(self.total); np.add.at(g,dofs.ravel(),r.ravel())
        normal,nh = self.normal_response(x,tangent=tangent,projected=projected)
        du = self.step_increment(x)
        fu = velocity_increment(self,du) if self.friction_time=='endpoint_friction' else du
        fg = np.zeros_like(normal) if lag is None else self.friction.gradient(lag,self.cmesh,fu[self.surface])
        g[self.surface_dofs] += normal+fg
        q = self.cohesive_algorithmic(x); cf,ch = self.face_force(q,projected)
        g[2*self.seam] += cf
        inertia = self.mass_dof/(self.kinematic_theta*self.material_dt**2)
        g += inertia*(du-self.material_dt*self.previous_velocity).ravel()
        h=metric=None
        if tangent:
            if projected or with_metric:
                w,V=np.linalg.eigh(.5*(k+k.transpose(0,2,1)))
                kp=np.einsum('eik,ek,ejk->eij',V,np.maximum(w,0.),V)
            rows=np.broadcast_to(dofs[:,:,None],k.shape).ravel();cols=np.broadcast_to(dofs[:,None,:],k.shape).ravel()
            def assemble(local,clamp,contact):
                mat=coo_matrix((local.ravel(),(rows,cols)),shape=(self.total,self.total)).tocsc()
                if lag is not None:
                    proj=self.ipc.PSDProjectionMethod.CLAMP if clamp else self.ipc.PSDProjectionMethod.NONE
                    fh=self.friction.hessian(lag,self.cmesh,fu[self.surface],proj)
                    if self.friction_time=='endpoint_friction':fh=fh@diags(np.where(self.surface_dofs<2*self.n,2.,1.))
                    contact=contact+fh
                c=contact.tocoo()
                mat+=coo_matrix((c.data,(self.surface_dofs[c.row],self.surface_dofs[c.col])),shape=mat.shape).tocsc()
                return mat+diags(inertia)+self.face_force(q,clamp)[1]
            h=assemble(kp if projected else k,projected,nh)
            if with_metric:metric=h if projected else assemble(kp,True,self.normal_response(x,tangent=True,projected=True)[1])
        return (energy,g,h,minj,metric) if with_metric else (energy,g,h,minj)

    def coupled_friction_tangent(self,x,lag,H):
        return coupled_tangent(self,x,lag,H) if self.friction_time=='endpoint_friction' else super().coupled_friction_tangent(x,lag,H)

    def evidence(self,kind,x,data):
        hook=getattr(self,"evidence_hook",None)
        if hook is not None:hook(kind,self,x,data)

    def propose_insertion(self,x,traction,depth,deadline):
        """Current constraint resultant versus outgoing force of virgin faces.

        Candidate patches are the incident faces of a tied node, selected in
        reference index order among the largest excess. No depth or surface
        preference and no second virtual Gc debit. Every newly free tensile DOF
        is checked against the original force tolerance before the graph event.
        """
        if getattr(self,"pure_precrack",False):return np.zeros_like(self.released)
        g=self.evaluate(x,self.lag(x),tangent=False)[1]
        weights=self.rule['weights_mm2']*(~self.active_faces)[:,None]
        local=np.einsum('eq,qi->ei',weights,self.rule['shape'])*self.strength
        outgoing=np.zeros(len(self.seam));outgoing[:-1]+=local[:,0];outgoing[1:]+=local[:,1]
        excess=-g[2*self.seam]-outgoing
        eligible=(~self.released)&(self.rest[self.seam,1]>0)&(excess>0)
        mask=np.zeros_like(self.released)
        if not eligible.any():return mask
        candidate=int(np.flatnonzero(eligible)[np.argmax(excess[eligible])])
        faces=self.active_faces.copy();faces[max(0,candidate-1):min(len(faces),candidate+1)]=True
        newly_free=eligible_nodes(faces)&~self.released
        if np.max(excess[newly_free],initial=0.)>self.force_tolerance_n:
            raise EnergyResolutionRequired('Constraint-to-cohesive traction event unresolved; refine physical time before activation')
        if getattr(self,"evidence_hook",None) is not None:
            from tools.face_events import endpoint_forces
            endpoint=endpoint_forces(self,x)
            ranking=sorted(np.flatnonzero(eligible).tolist(),key=lambda i:(-excess[i],i))
            self.evidence("activation",x,dict(candidate_seam_index=candidate,candidate_ranking=ranking,
                algorithmic_gradient=g,algorithmic_excess_half_n=excess,outgoing_half_n=outgoing,
                endpoint_force_components=endpoint,endpoint_excess_half_n=-sum(endpoint.values())[2*self.seam]-outgoing,
                new_faces=np.flatnonzero(faces&~self.active_faces),new_free_nodes=self.seam[newly_free],
                force_tolerance_half_n=self.force_tolerance_n,target_depth_mm=depth))
        self.pending_faces=faces;mask[newly_free]=True
        self.activation_positions=x.copy()
        self.pending_handoff=dict(candidate_seam_index=candidate,new_faces=np.flatnonzero(faces&~self.active_faces).tolist(),
            new_free_nodes=self.seam[newly_free].tolist(),maximum_outgoing_force_excess_half_n=float(np.max(excess[newly_free],initial=0.)),
            new_area_mm2=float(self.rule['face_area_mm2'][faces&~self.active_faces].sum()),upfront_gc_debit_n_mm=0.)
        return mask

    def solve_active(self,x,depth,deadline):
        # Reuse the established nonlinear solvers and exact active-set algorithm.
        # Its graph update sets released nodes; rebuild_contact applies pending
        # face activation atomically before the next assembly.
        return super().solve_active(x,depth,deadline)

    def step(self,x,old,depth,deadline):
        before=self.active_faces.copy();self.previous_active_faces=before.copy()
        self.face_events=[]
        try:return super().step(x,old,depth,deadline)
        except BaseException:
            self.failed_active_faces=self.active_faces.copy()
            self.evidence("failed",self.failed_positions,dict(stage=self.failed_stage))
            self.active_faces=before;self.released=eligible_nodes(before)
            if hasattr(self,'pending_faces'):del self.pending_faces
            self.rebuild_contact();self.set_free();raise

    def acceptance(self,x,depth,iterations,outer_iterations):
        if getattr(self,'pure_precrack',False) and not np.array_equal(self.active_faces,self.precracked_faces):
            raise RuntimeError('Pure precrack gained an activated face')
        rec=super().acceptance(x,depth,iterations,outer_iterations)
        if not np.array_equal(self.released,eligible_nodes(self.active_faces)):
            raise RuntimeError('Face ties and free-node mask disagree')
        if np.any(x[self.seam[~self.released],0]!=self.config.drop_s_mm):
            raise RuntimeError('An intact face lost its exact endpoint connection')
        if np.any(self.previous_history[~self.active_faces]!=0):raise RuntimeError('Inactive face has damage history')
        q=self.cohesive_response(x)
        failed=self.active_faces&np.all(q['history']>=2*self.toughness/self.strength,axis=1)
        rec.update(fracture_route='face_extrinsic_continuous',active_faces=int(self.active_faces.sum()),
            fully_failed_faces=int(failed.sum()),fully_failed_nodes=int(eligible_nodes(failed).sum()),
            precracked_faces=int(self.precracked_faces.sum()),face_events=self.face_events,
            surface_connected_failed_length_mm=self.connected_length(failed),
            quadrature_order=self.rule['shape'].shape[0],initial_fracture_energy_excluded_n_mm=self.initial_fracture_energy_n_mm)
        if self.friction_time=='endpoint_friction':
            e=rec['energy']
            if min(e['friction_work_n_mm'],e['friction_algorithmic_work_n_mm']) < -1e-12:
                raise EnergyResolutionRequired('Negative endpoint friction work')
            check_energy_gate(self,e)
            rec['friction_work_n_mm']=.5*e['friction_work_n_mm']
            rec['time_integrator']='face_C_discrete_gradient_endpoint_friction_Lie_Maxwell'
        return rec

    def connected_length(self,failed):
        length=0.
        for index in range(len(failed)-1,-1,-1):
            if not failed[index]:break
            length+=self.rule['face_area_mm2'][index]/self.config.width_mm
        return float(length)

    def energy_ledger(self,x):
        du=self.step_increment(x);v=self.endpoint_velocity(x);volume=self.material_integration_weights()
        old,frozen=self.frozen(self.previous),self.frozen(x);relaxed=self.material_response(x)['energy']
        viscous=2*float(volume@(frozen-relaxed))
        if viscous < -1e-9:raise RuntimeError('Maxwell split increased fixed-F energy')
        stored=2*float(volume@(relaxed-old))
        r,_,dofs,_,_=self.material_bulk((x[:self.n]-self.mesh.reference_xy).ravel(),with_tangent=False)
        bw=2*float(np.sum(r*du.ravel()[dofs]))
        q,oldq=self.cohesive_response(x),self.cohesive_response(self.previous,self.previous_history)
        weights=self.rule['weights_mm2']*self.active_faces[:,None]
        cs=float(np.sum(weights*(q['stored']-oldq['stored'])));cd=float(np.sum(weights*(q['dissipated']-oldq['dissipated'])))
        cw=2*float(self.face_force(self.cohesive_algorithmic(x))[0]@du[self.seam,0])
        b1=float(self.barrier(self.collisions(x),self.cmesh,x[self.surface]));b0=float(self.barrier(self.collisions(self.previous),self.cmesh,self.previous[self.surface]))
        jump=2*(b0-self.previous_barrier_energy);bd=2*(b1-self.previous_barrier_energy)
        bg=self.normal_response(x)[0];bwork=2*float(bg@du[self.surface].ravel())
        fu=velocity_increment(self,du) if self.friction_time=='endpoint_friction' else du
        fg=self.friction.gradient(self.lag(x),self.cmesh,fu[self.surface])
        friction=2*float(fg@fu[self.surface].ravel());fw=2*float(fg@du[self.surface].ravel())
        kinetic=float(self.mass@np.sum(v[:self.n]**2,axis=1));oldk=float(self.mass@np.sum(self.previous_velocity[:self.n]**2,axis=1))
        g=self.evaluate(x,self.lag(x),tangent=False)[1].reshape(-1,2)
        external=2*float(np.sum(g[self.n:]*du[self.n:]))
        contact=self.active_contact.copy();contact[self.seam[~self.released]]=False
        contact_loss=-2*float(g[:self.n,0][contact]@du[:self.n,0][contact])
        defects=dict(backward_euler_inertia_loss_n_mm=0.,bulk_endpoint_defect_n_mm=bw-stored-viscous,
            cohesive_endpoint_defect_n_mm=cw-cs-cd,barrier_endpoint_defect_n_mm=bwork-bd+jump,
            contact_topology_defect_n_mm=-jump,hard_contact_endpoint_loss_n_mm=contact_loss,
            friction_time_defect_n_mm=fw-friction)
        physical=stored+cs+cd+viscous+bd+friction+kinetic-oldk
        reservoir=float(volume@(relaxed+old))+.5*float(np.sum(weights*(q['stored']+oldq['stored'])))+b1+self.previous_barrier_energy+.5*(kinetic+oldk)
        ledger=dict(external_work_n_mm=external,bulk_stored_delta_n_mm=stored,mechanical_reservoir_n_mm=reservoir,
            cohesive_stored_delta_n_mm=cs,fracture_dissipation_n_mm=cd,viscous_endpoint_dissipation_n_mm=viscous,
            friction_work_n_mm=friction,friction_algorithmic_work_n_mm=fw,barrier_delta_n_mm=bd,kinetic_delta_n_mm=kinetic-oldk,
            kinetic_energy_n_mm=kinetic,physical_balance_defect_n_mm=external-physical,
            algebraic_balance_residual_n_mm=external-physical-sum(defects.values()),**defects)

        self.evidence("energy",x,dict(ledger=ledger,friction_gradient_surface=fg,friction_velocity_increment_surface=fu[self.surface],friction_surface_nodes=self.surface))
        return ledger
