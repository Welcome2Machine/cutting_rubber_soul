"""Initially rigid, zero-gap mode-I seam in a symmetric 2D cutting control.

This is an explicit NEW geometry: 30 x 5 x 20 mm, centred blade (s=15),
not the off-centre s=7 baseline. Only the right half is discretized. Reflection
gives exact ties before insertion and frictionless unilateral crack-face contact
after insertion. Both in-plane momentum components are solved. It cannot model
asymmetric bifurcation, mixed-mode slip, oblique cutting or a freely chosen path.
All half-body and blade EV/VV/self-contact/CCD remain in native IPC.
"""
from dataclasses import replace
import time
import numpy as np
from scipy.sparse import diags
from scipy.spatial import Delaunay
from .geometry import intact_mesh
from .model_config import FEMMesh
from .ipc_reference import IPCProblem, LoadStepFailure, BASELINE
from .ipc_viscoelastic import ViscoelasticProblem


class EnergyResolutionRequired(LoadStepFailure):
    """Reject physical time step before committing any internal variables."""


def extrinsic_response(opening, history, strength, toughness, unloading='secant'):
    """Linear extrinsic envelope with an explicitly selected unloading law.

    At kappa=0, t=T is the outgoing branch of the initially rigid graph.
    Ties carry the set-valued traction until insertion. No initial spring.
    Per UNIT FULL crack reference area: secant D=T*kappa/2; the
    constant-traction variant has D=Gc*(kappa/delta_f)**2, both capped at Gc.
    """
    if unloading not in ('secant', 'constant_traction'):
        raise ValueError('Unknown cohesive unloading law')
    u, old = np.broadcast_arrays(np.asarray(opening, float), np.asarray(history, float))
    if (strength <= 0 or toughness <= 0 or not np.isfinite([strength, toughness]).all()
            or not np.isfinite(u).all() or not np.isfinite(old).all()
            or np.any(u < -1e-12) or np.any(old < 0)):
        raise ValueError('Invalid extrinsic opening/history/parameters')
    u = np.maximum(u, 0.)
    end = 2*toughness/strength
    k = np.maximum(old, u)
    envelope = strength*np.maximum(1-k/end, 0.)
    secant = np.divide(envelope, k, out=np.zeros_like(k), where=k > 0)
    traction = np.where(k > 0, secant*u, strength)
    tangent = np.where(u >= old, -strength/end, secant)
    tangent = np.where(k >= end, 0., tangent)
    stored = .5*secant*u*u
    dissipated = .5*strength*np.minimum(k, end)
    if unloading == 'constant_traction':
        # Initially rigid nonsmooth graph: t=T*beta even on unloading,
        # beta=max(1-kappa/delta_f,0). No secant unloading spring.
        # Own recoverable/irreversible split, not a copied LCP discretization:
        # psi=T*beta*u; D=Gc*(1-beta)^2. Their loading derivative is t.
        # See Collins-Craft et al. (2022), DOI 10.1016/j.cma.2022.115545,
        # and the author's spring_czm-dynamic_with_impacts.py.
        traction = envelope
        tangent = np.where((u >= old) & (k < end), -strength/end, 0.)
        stored = envelope*u
        dissipated = toughness*np.minimum(k/end, 1.)**2
    return dict(traction=traction, tangent=tangent, stored=stored,
                dissipated=dissipated, history=k)


class ZeroGapCutProblem(ViscoelasticProblem):
    def __init__(self, *, density_kg_m3, strength_mpa, toughness_n_mm,
                 seam_spacing_mm=.1, damage_enabled=True, nonlinear_solver='ptc',
                 energy_tolerance=None, friction_velocity_epsilon=.0002,
                 energy_normalization='increment_work', energy_absolute_tolerance=0.,
                 cohesive_unloading='secant', **kwargs):
        if kwargs.pop('inclination', 0.) != 0.:
            raise ValueError('Reflection control only supports 0 degrees')
        if not np.isfinite([density_kg_m3, strength_mpa, toughness_n_mm,
                            seam_spacing_mm]).all() or min(density_kg_m3,
                strength_mpa, toughness_n_mm, seam_spacing_mm) <= 0:
            raise ValueError('Positive finite explicit physical inputs required')
        super().__init__(inclination=0., **kwargs)
        if nonlinear_solver not in ('ptc', 'newton_lm'):
            raise ValueError('Unknown dynamic force solver')
        self.nonlinear_solver = nonlinear_solver
        if energy_tolerance is not None and not 0 < energy_tolerance < 1:
            raise ValueError('Energy tolerance must be in (0,1)')
        if not np.isfinite(friction_velocity_epsilon) or friction_velocity_epsilon <= 0:
            raise ValueError('Positive friction velocity smoothing required')
        self.energy_tolerance = energy_tolerance
        if energy_normalization not in ('increment_work', 'mechanical_reservoir'):
            raise ValueError('Unknown energy error normalization')
        if not np.isfinite(energy_absolute_tolerance) or energy_absolute_tolerance < 0:
            raise ValueError('Energy absolute tolerance must be finite and nonnegative')
        self.energy_normalization = energy_normalization
        self.energy_absolute_tolerance = energy_absolute_tolerance
        self.friction_velocity_epsilon = friction_velocity_epsilon
        self.strength, self.toughness = strength_mpa, toughness_n_mm
        if cohesive_unloading not in ('secant', 'constant_traction'):
            raise ValueError('Unknown cohesive unloading law')
        self.cohesive_unloading = cohesive_unloading
        self.density = density_kg_m3*1e-12  # kg/m3 -> N s2/mm4
        if type(damage_enabled) is not bool:
            raise ValueError('damage_enabled must be boolean')
        self.damage_enabled = damage_enabled
        self.force_tolerance_n = BASELINE['force_tolerance_n']/np.sqrt(2.)
        blade = self.rest[self.n:].copy()
        old_edges = self.surface[np.asarray(self.cmesh.edges)]
        knife_edges = old_edges[np.all(old_edges >= self.n, axis=1)]-self.n
        centre = self.config.length_mm/2
        blade[:, 0] += centre-self.config.drop_s_mm
        self.config = replace(self.config, drop_s_mm=centre)
        full, _ = intact_mesh(self.config, self.options, fine_halfwidth_mm=self.fine_halfwidth)
        levels = np.linspace(0, self.config.height_mm,
                             int(np.ceil(self.config.height_mm/seam_spacing_mm))+1)
        sn = np.unique(np.round(np.vstack((full.local_sn[full.local_sn[:, 0] > centre+1e-10],
            np.column_stack((np.full(len(levels), centre), levels)))), 10), axis=0)
        tri = Delaunay(sn).simplices.copy()
        det = np.linalg.det(np.stack((sn[tri[:, 1]]-sn[tri[:, 0]],
                                     sn[tri[:, 2]]-sn[tri[:, 0]]), axis=2))
        tri[det < 0] = tri[det < 0][:, [0, 2, 1]]
        seam = np.flatnonzero(sn[:, 0] == centre)
        seam = seam[np.argsort(sn[seam, 1])]
        self.mesh = FEMMesh(sn.copy(), sn, tri, np.empty((0, 2), int), np.empty(0),
            np.flatnonzero(sn[:, 1] == 0), seam, np.ones(len(sn), int))
        self.n = len(sn)
        self.start = -.02
        blade[:, 1] += (-2*self.dhat)-self.start
        self.rest = np.vstack((sn, blade))
        self.total = self.rest.size
        edges = np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
        edges, count = np.unique(edges, axis=0, return_counts=True)
        edges = np.vstack((edges[count == 1], knife_edges+self.n))
        self.all_boundary_edges = edges.copy()
        self.surface = np.unique(edges)
        self.surface_dofs = (2*self.surface[:, None]+np.arange(2)).ravel()
        self.cmesh = self.ipc.CollisionMesh(self.rest[self.surface], np.searchsorted(self.surface, edges))
        self.cmesh.can_collide = self.ipc.make_static_obstacle_filter(int(np.count_nonzero(self.surface < self.n)))
        self._connected_boundary_pairs = {tuple(sorted(map(int, e))) for e in self.cmesh.edges}
        if self.ipc.has_intersections(self.cmesh, self.rest[self.surface]):
            raise ValueError('Intersecting zero-gap half-domain reference')
        self.grad, self.dofs = self.mesh.bulk_data[1:]
        self.base_free = np.setdiff1d(np.arange(2*self.n),
                         (2*self.mesh.bottom_nodes[:, None]+np.arange(2)).ravel())
        lengths = np.diff(sn[seam, 1])
        self.seam = seam
        self.seam_area = self.config.width_mm*np.r_[lengths[0]/2,
                              (lengths[:-1]+lengths[1:])/2, lengths[-1]/2]
        self.released = np.zeros(len(seam), bool)
        self.history = np.zeros(len(seam))
        self.previous_history = self.history.copy()
        self.mass = np.zeros(self.n)
        np.add.at(self.mass, tri.ravel(), np.repeat(self.mesh.bulk_data[0]*self.config.width_mm*self.density/3, 3))
        self.mass_dof = np.r_[np.repeat(self.mass, 2), np.zeros(self.total-2*self.n)]
        self.velocity = np.zeros_like(self.rest)
        self.previous_velocity = self.velocity.copy()
        self.previous = self.rest.copy()
        self.material_metric = np.tile(np.eye(3), (len(tri), 1, 1))
        self.previous_material_metric = self.material_metric.copy()
        self.active_contact = np.zeros(self.n, bool)
        self.active_contact[seam] = True
        self.set_free()
        self.rebuild_contact()
        self.previous_barrier_energy = 0.

    def rebuild_contact(self):
        """Closed internal material is not a surface. Expose an edge on release.

        Mirror contact is covered by the exact half-space constraint. All true
        half-domain faces remain in native IPC, including newly exposed edges.
        """
        edges = self.all_boundary_edges
        seam_edge = np.all(np.isin(edges, self.seam), axis=1)
        released_node = np.zeros(len(self.rest), bool)
        released_node[self.seam[self.released]] = True
        edges = edges[~seam_edge | np.any(released_node[edges], axis=1)]
        self.surface = np.unique(edges)
        self.surface_dofs = (2*self.surface[:, None]+np.arange(2)).ravel()
        self.cmesh = self.ipc.CollisionMesh(self.rest[self.surface], np.searchsorted(self.surface, edges))
        self.cmesh.can_collide = self.ipc.make_static_obstacle_filter(int(np.count_nonzero(self.surface < self.n)))
        self._connected_boundary_pairs = {tuple(sorted(map(int, e))) for e in self.cmesh.edges}

    def set_free(self):
        tied = self.seam[~self.released]
        constrained = np.union1d(2*tied, 2*np.flatnonzero(self.active_contact))
        self.free = np.setdiff1d(self.base_free, constrained)

    def update_constraint_free(self, x, gradient):
        at_plane = x[:self.n, 0]-self.config.drop_s_mm <= 1e-12
        active = at_plane & (gradient.reshape(-1, 2)[:self.n, 0] >= 0)
        active[self.seam[~self.released]] = True
        self.active_contact = active
        self.set_free()

    def cohesive_response(self, x, history=None):
        return extrinsic_response(2*(x[self.seam, 0]-self.config.drop_s_mm),
            self.previous_history if history is None else history, self.strength, self.toughness,
            self.cohesive_unloading)

    def endpoint_velocity(self, x):
        velocity = self.step_increment(x)/self.material_dt
        velocity[self.n:] = 0.
        return velocity

    def insertion_mask(self, traction):
        return ((~self.released) & (traction > self.strength)
                & (self.mesh.local_sn[self.seam, 1] > 0))

    def set_primary_increment(self, x, increment):
        if not np.allclose(x, self.previous+increment, rtol=0, atol=1e-13):
            raise ValueError('Primary increment and positions differ beyond roundoff')
        self._primary_positions = x.copy()
        self._primary_increment = increment.copy()

    def step_increment(self, x):
        if (hasattr(self, '_primary_positions') and np.array_equal(x, self._primary_positions)):
            return self._primary_increment.copy()
        return x-self.previous

    def evaluate(self, x, lag=None, tangent=True, projected=True, *, with_metric=False):
        out = list(IPCProblem.evaluate(self, x, lag, tangent, projected, with_metric=with_metric))
        q = self.cohesive_response(x)
        mask = self.released.astype(float)
        ids = 2*self.seam
        out[1][ids] += self.seam_area*mask*q['traction']
        old_d = self.cohesive_response(self.previous, self.previous_history)['dissipated']
        out[0] += .5*float(self.seam_area@(mask*(q['stored']+q['dissipated']-old_d)))
        inertia = self.mass_dof/(self.material_dt**2)
        delta = (self.step_increment(x)-self.material_dt*self.previous_velocity).ravel()
        out[0] += .5*float(inertia@(delta*delta))
        out[1] += inertia*delta
        if tangent:
            extra = inertia.copy()
            extra[ids] += 2*self.seam_area*mask*q['tangent']
            out[2] += diags(np.maximum(extra, inertia) if projected else extra)
            if with_metric:
                out[4] += diags(np.maximum(extra, inertia))
        return tuple(out)

    def safe_step(self, x, direction):
        safe = super().safe_step(x, direction)
        moving = direction[:self.n, 0] < 0
        if np.any(moving):
            limit = (x[:self.n, 0][moving]-self.config.drop_s_mm)/(-direction[:self.n, 0][moving])
            safe = min(safe, float(limit.min()))
        return max(0., safe)

    def _ptc_trial(self, x, direction, maximum):
        # Permit landing exactly on the affine unilateral boundary. No gap offset.
        trial, alpha, geometric = super()._ptc_trial(x, direction, maximum)
        trial[:self.n, 0] = np.maximum(trial[:self.n, 0], self.config.drop_s_mm)
        return trial, alpha, geometric

    def solve_active(self, x, depth, deadline):
        total_iterations = 0
        for outer in range(50):
            if time.monotonic() >= deadline:
                raise TimeoutError('Zero-gap active-set budget')
            # Closed-face complementarity uses the PHYSICAL full dynamic residual.
            g = self.evaluate(x, self.lag(x), tangent=False)[1].reshape(-1, 2)
            at_plane = x[:self.n, 0]-self.config.drop_s_mm <= 1e-12
            active = at_plane & (g[:self.n, 0] >= 0)
            active[self.seam[~self.released]] = True
            self.active_contact = active
            self.set_free()
            if self.nonlinear_solver == 'newton_lm':
                from .force_globalization import solve
                x, iterations = solve(self, x, depth, deadline)
            else:
                x, iterations = self.solve_friction(x, depth, deadline)
            total_iterations += iterations+1
            g = self.evaluate(x, self.lag(x), tangent=False)[1].reshape(-1, 2)
            traction = -g[self.seam, 0]/self.seam_area
            insert = self.insertion_mask(traction)
            if self.damage_enabled and insert.any():
                self.insertion_events.append(dict(nodes=self.seam[insert].tolist(),
                    reference_y_mm=self.rest[self.seam[insert], 1].tolist(),
                    outgoing_traction_mpa=traction[insert].tolist(),
                    overshoot_mpa=(traction[insert]-self.strength).tolist()))
                self.released |= insert
                self.rebuild_contact()
                continue
            at_plane = x[:self.n, 0]-self.config.drop_s_mm <= 1e-12
            next_active = at_plane & (g[:self.n, 0] >= 0)
            next_active[self.seam[~self.released]] = True
            if np.array_equal(next_active, self.active_contact):
                return x, total_iterations, outer+1
        raise LoadStepFailure('Zero-gap active set did not settle')

    def step(self, x, old, depth, deadline):
        before_release, before_active = self.released.copy(), self.active_contact.copy()
        self.insertion_events = []
        if hasattr(self, '_primary_positions'):
            del self._primary_positions, self._primary_increment
        self.previous = x.copy()
        self.previous_velocity = self.velocity.copy()
        self.previous_history = self.history.copy()
        self.previous_material_metric = self.material_metric.copy()
        self.previous_barrier_energy = float(self.barrier(self.collisions(x), self.cmesh, x[self.surface]))
        self.material_dt = (depth-old)/self.loading_rate
        self.friction = self.ipc.FrictionPotential(self.friction_velocity_epsilon*self.material_dt)
        self.trial = x.copy()
        stage = 'loading_predictor'
        try:
            self.set_free()
            candidate = self._loading_predictor(x, old, depth)
            # A timeout before the first Newton iteration must not save a stale
            # trial from a previous attempt with this attempt's dt/history.
            self.trial = candidate.copy()
            self.set_primary_increment(candidate, candidate-self.previous)
            stage = 'force_solve'
            value, iterations, outer = self.solve_active(candidate, depth, deadline)
            stage = 'accepted_motion'
            if not self.accepted_chord_is_safe(value):
                raise LoadStepFailure('Accepted-state motion violates full CCD/J or unilateral bound')
            stage = 'acceptance'
            rec = self.acceptance(value, depth, iterations, outer)
            rec['accepted_chord_ccd_safe'] = True
            energy = rec['energy']
            defect_keys = ('backward_euler_inertia_loss_n_mm', 'bulk_endpoint_defect_n_mm',
                'cohesive_endpoint_defect_n_mm', 'barrier_endpoint_defect_n_mm',
                'contact_topology_defect_n_mm', 'hard_contact_endpoint_loss_n_mm')
            scale_keys = ('external_work_n_mm', 'bulk_stored_delta_n_mm', 'cohesive_stored_delta_n_mm',
                'fracture_dissipation_n_mm', 'viscous_endpoint_dissipation_n_mm', 'friction_work_n_mm',
                'barrier_delta_n_mm', 'kinetic_delta_n_mm')
            error = sum(abs(energy[k]) for k in defect_keys)/max(1e-9, sum(abs(energy[k]) for k in scale_keys))
            rec['energy_resolution_ratio'] = error
            defect = sum(abs(energy[k]) for k in defect_keys)
            if self.energy_normalization == 'mechanical_reservoir':
                allowed = self.energy_absolute_tolerance+(self.energy_tolerance or 0.)*energy['mechanical_reservoir_n_mm']
                rec['energy_reservoir_error_ratio'] = defect/max(allowed, 1e-30)
                if self.energy_tolerance is not None and defect > allowed:
                    raise EnergyResolutionRequired(f'Numerical energy defect {defect:g} exceeds reservoir allowance {allowed:g}')
            elif self.energy_tolerance is not None and error > self.energy_tolerance:
                raise EnergyResolutionRequired(f'Numerical energy ratio {error:g} exceeds {self.energy_tolerance:g}')
            velocity = self.endpoint_velocity(value)
            history = self.cohesive_response(value)['history'].copy()
            metric = self.material_response(value)['metric'].copy()
            self._remember_predictor_acceptance(value, depth, old)
            self.velocity, self.history, self.material_metric = velocity, history, metric
            return value, rec
        except BaseException:
            self.failed_stage = stage
            self.failed_positions = getattr(self, '_primary_positions', getattr(self, 'trial', x)).copy()
            self.failed_increment = self.step_increment(self.failed_positions)
            self.failed_released = self.released.copy()
            self.failed_active_contact = self.active_contact.copy()
            self.released, self.active_contact = before_release, before_active
            self.rebuild_contact()
            self.set_free()
            raise

    def accepted_chord_is_safe(self, x):
        """Audit the temporal chord, in addition to every nonlinear CCD update."""
        return bool(self.safe_step(self.previous, x-self.previous) >= 1-1e-10)

    def acceptance(self, x, depth, iterations, outer_iterations):
        rec = super().acceptance(x, depth, iterations, outer_iterations)
        rec['reaction_fz_n'] *= 2
        rec['reaction_x_n'] = 0.
        rec['residual_n'] *= np.sqrt(2.)
        if rec['residual_n'] > BASELINE['force_tolerance_n']:
            raise LoadStepFailure('Reflected full-domain residual exceeds 1e-4 N')
        g = self.evaluate(x, self.lag(x), tangent=False)[1].reshape(-1, 2)
        contact = self.active_contact.copy()
        contact[self.seam[~self.released]] = False
        violation = max(0., float(-g[:self.n, 0][contact].min(initial=0.)))
        if violation > 1e-5 or x[:self.n, 0].min() < self.config.drop_s_mm-1e-12:
            raise LoadStepFailure('Unilateral crack contact violation')
        if np.any(x[self.seam[~self.released], 0] != self.config.drop_s_mm):
            raise RuntimeError('Intact zero-gap constraint lost')
        rec.update(released_nodes=int(self.released.sum()),
            fully_failed_nodes=int(np.sum(self.cohesive_response(x)['history'] >= 2*self.toughness/self.strength)),
            unilateral_force_violation_n=violation, energy=self.energy_ledger(x),
            insertion_events=getattr(self, 'insertion_events', []))
        return rec

    def energy_ledger(self, x):
        """Full reflected domain endpoint-work identity with separate defects.

        BE inertia has known numerical loss. Constitutive/contact/cohesive endpoint
        defects are computed separately; their sum is NOT physical dissipation.
        No unaccounted energy is relabelled as kinetic energy or material damping.
        """
        from mechanics.finite_strain import principal
        du = self.step_increment(x)
        v = du/self.material_dt
        volumes = self.material_integration_weights()
        q = self.material_response(x)
        oldF = np.eye(2)+np.einsum('eai,eaJ->eiJ',
            (self.previous[:self.n]-self.mesh.reference_xy)[self.mesh.triangles], self.grad)
        fullF = np.tile(np.eye(3), (len(oldF), 1, 1))
        fullF[:, :2, :2] = oldF
        eq_log = .5*np.log(np.linalg.eigvalsh(fullF@fullF.transpose(0, 2, 1)))
        ov_log = .5*np.log(np.linalg.eigvalsh(fullF@self.previous_material_metric@fullF.transpose(0, 2, 1)))
        old_energy = principal(eq_log, self.material)[0]+principal(ov_log, self.material, True)[0]
        bulk_r, _, dofs, _, _ = self.material_bulk((x[:self.n]-self.mesh.reference_xy).ravel(), with_tangent=False)
        bulk_work = 2*float(np.sum(bulk_r*du.ravel()[dofs]))
        stored_delta = 2*float(volumes@(q['energy']-old_energy))
        viscous = 2*self.material_dt*float(volumes@q['viscous_rate'])
        coh = self.cohesive_response(x)
        old_coh = self.cohesive_response(self.previous, self.previous_history)
        mask = self.released.astype(float)
        cstored = float(self.seam_area@(mask*(coh['stored']-old_coh['stored'])))
        cd = float(self.seam_area@(mask*(coh['dissipated']-old_coh['dissipated'])))
        cw = 2*float((self.seam_area*mask*coh['traction'])@du[self.seam, 0])
        collisions = self.collisions(x)
        old_col = self.collisions(self.previous)
        new_topology_old_energy = float(self.barrier(old_col, self.cmesh, self.previous[self.surface]))
        topology_jump = 2*(new_topology_old_energy-self.previous_barrier_energy)
        barrier_delta = 2*(float(self.barrier(collisions, self.cmesh, x[self.surface]))-self.previous_barrier_energy)
        barrier_work = 2*float(self.barrier.gradient(collisions, self.cmesh, x[self.surface])@du[self.surface].ravel())
        lag = self.lag(x)
        friction = 2*float(self.friction.gradient(lag, self.cmesh, du[self.surface])@du[self.surface].ravel())
        kinetic_delta = float(self.mass@(np.sum(v[:self.n]**2, axis=1)-np.sum(self.previous_velocity[:self.n]**2, axis=1)))
        reservoir = (float(volumes@(q['energy']+old_energy))+
            .5*float(self.seam_area@(mask*(coh['stored']+old_coh['stored'])))+
            float(self.barrier(collisions, self.cmesh, x[self.surface]))+self.previous_barrier_energy+
            .5*float(self.mass@(np.sum(v[:self.n]**2, axis=1)+np.sum(self.previous_velocity[:self.n]**2, axis=1))))
        inertia_loss = float(self.mass@np.sum((v[:self.n]-self.previous_velocity[:self.n])**2, axis=1))
        physical_g = IPCProblem.evaluate(self, x, lag, tangent=False)[1].reshape(-1, 2)
        dynamic_g = self.evaluate(x, lag, tangent=False)[1].reshape(-1, 2)
        contact = self.active_contact.copy()
        contact[self.seam[~self.released]] = False
        contact_loss = -2*float(dynamic_g[:self.n, 0][contact]@du[:self.n, 0][contact])
        external = 2*float(np.sum(physical_g[self.n:]*du[self.n:]))
        # Reaction sign follows gradient: positive actuator work is g_blade.du_blade.
        physical = stored_delta+cstored+barrier_delta+cd+viscous+friction+kinetic_delta
        defects = dict(backward_euler_inertia_loss_n_mm=inertia_loss,
            bulk_endpoint_defect_n_mm=bulk_work-stored_delta-viscous,
            cohesive_endpoint_defect_n_mm=cw-cstored-cd,
            barrier_endpoint_defect_n_mm=barrier_work-barrier_delta+topology_jump,
            contact_topology_defect_n_mm=-topology_jump,
            hard_contact_endpoint_loss_n_mm=contact_loss)
        return dict(external_work_n_mm=external, bulk_stored_delta_n_mm=stored_delta,
            mechanical_reservoir_n_mm=reservoir,
            cohesive_stored_delta_n_mm=cstored, fracture_dissipation_n_mm=cd,
            viscous_endpoint_dissipation_n_mm=viscous, friction_work_n_mm=friction,
            barrier_delta_n_mm=barrier_delta, kinetic_delta_n_mm=kinetic_delta,
            kinetic_energy_n_mm=float(self.mass@np.sum(v[:self.n]**2, axis=1)),
            physical_balance_defect_n_mm=external-physical,
            algebraic_balance_residual_n_mm=external-physical-sum(defects.values()), **defects)
