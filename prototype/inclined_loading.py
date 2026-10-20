"""Explicit matched-normal-history paths; original IPC defaults stay vertical.

Scalar depth is nominal NORMAL depth q, never actual contact penetration/slip.
Both members of a pair approach from q=-2*dhat along the specimen normal.
Their trajectories diverge only after q=0. Frame rotation is a covariance check,
not a change of blade orientation relative to the specimen.
"""
import numpy as np
from .ipc_reference import IPCProblem
from .ipc_viscoelastic_p2 import QuadratureViscoelasticProblem
from .ipc_viscoelastic import ViscoelasticProblem
from .model_config import _rotation_vectors, local_to_lab


class InclinedPathMixin:
    def __init__(self, *, specimen_angle, path_mode, normal_speed=.1,
                 frame_rotation=0., **kwargs):
        if path_mode not in ('normal', 'oblique'):
            raise ValueError('Require normal or oblique path')
        if (not np.isfinite(specimen_angle) or not 0 <= specimen_angle < 80
                or not np.isfinite(frame_rotation) or not 0 <= frame_rotation <= specimen_angle):
            raise ValueError('Require finite 0 <= frame rotation <= specimen angle < 80')
        self.specimen_angle = float(specimen_angle)
        self.path_mode = path_mode
        self.frame_rotation = float(frame_rotation)
        phi = np.deg2rad(frame_rotation)
        self.frame_matrix = np.array([[np.cos(phi),-np.sin(phi)], [np.sin(phi),np.cos(phi)]])
        if any(k in kwargs for k in ('inclination', 'loading_rate')):
            raise ValueError('Use specimen_angle and normal_speed for this explicit path')
        super().__init__(inclination=specimen_angle-frame_rotation,
                         loading_rate=normal_speed, **kwargs)
        if self.case != 'oblique':
            raise ValueError('Matched paths require the actual rounded blade geometry')
        self.es, self.en = _rotation_vectors(self.config.inclination_deg)
        self.tangent_ratio = (float(np.tan(np.deg2rad(specimen_angle)))
                              if path_mode == 'oblique' else 0.)
        self.loading_direction = -self.en+self.tangent_ratio*self.es
        # Rotate the blade about the same physical circle centre. The specimen
        # and its bottom have already been constructed in the rotated frame.
        zero = IPCProblem.prescribed_blade_positions(self, 0.)
        center = local_to_lab(np.array([[self.config.drop_s_mm,self.config.height_mm]]),
                              self.config)[0]+self.options.tip_radius_mm*self.en
        self.blade_at_zero = (zero-center)@self.frame_matrix.T+center
        self.start = -2*self.dhat
        self.rest[self.n:] = self.prescribed_blade_positions(self.start)
        edges = np.asarray(self.cmesh.edges).copy()
        self.cmesh = self.ipc.CollisionMesh(self.rest[self.surface], edges)
        self.cmesh.can_collide = self.ipc.make_static_obstacle_filter(int(np.count_nonzero(self.surface<self.n)))
        self.previous = self.rest.copy()
        self._accepted_predictor_history = []
        if self.ipc.has_intersections(self.cmesh,self.rest[self.surface]):
            raise ValueError('Initial matched-path geometry intersects')
        self.reference_contact_check = self.check_reference_contact()

    def motion_signature(self):
        return dict(version=1,depth_coordinate='nominal_reference_normal_mm',
            specimen_angle_deg=self.specimen_angle,path_mode=self.path_mode,
            frame_rotation_deg=self.frame_rotation,normal_speed_mm_s=self.loading_rate,
            tangent_over_normal_speed=self.tangent_ratio,
            prezero='common pure-normal approach from q=-2*dhat',
            initial_normal_depth_mm=self.start)

    def prescribed_blade_positions(self, depth):
        q = float(depth)
        if not np.isfinite(q):raise ValueError('Nonfinite normal depth')
        direction = -self.en if q <= 0 else self.loading_direction
        return self.blade_at_zero+q*direction

    def affine_loading_candidate(self, x, old, depth):
        delta = self.prescribed_blade_positions(depth)[0]-self.prescribed_blade_positions(old)[0]
        value = x.copy()
        value[:self.n] += (self.mesh.local_sn[:,1]/self.config.height_mm)[:,None]*delta
        value[self.mesh.bottom_nodes] = self.rest[self.mesh.bottom_nodes]
        value[self.n:] = self.prescribed_blade_positions(depth)
        return value

    def acceptance(self, x, depth, iterations, outer_iterations):
        record = super().acceptance(x,depth,iterations,outer_iterations)
        reaction = np.array([record['reaction_x_n'],record['reaction_fz_n']])
        normal, tangent = float(reaction@self.en), float(-reaction@self.es)
        lab = reaction@self.frame_matrix
        direction = -self.en if depth <= 0 else self.loading_direction
        power = float(-reaction@direction*self.loading_rate)
        ratio = 0. if depth <= 0 else self.tangent_ratio
        np.testing.assert_allclose(power,(normal+ratio*tangent)*self.loading_rate,
                                   rtol=1e-12,atol=1e-12)
        record['motion'] = dict(normal_depth_mm=float(depth),
            tangent_travel_mm=max(0.,float(depth))*self.tangent_ratio,
            elapsed_from_zero_s=max(0.,float(depth))/self.loading_rate,
            normal_speed_mm_s=self.loading_rate,tangent_speed_mm_s=ratio*self.loading_rate,
            laboratory_translation_mm=((self.prescribed_blade_positions(depth)[0]-self.blade_at_zero[0])@self.frame_matrix).tolist(),
            laboratory_reaction_xy_n=lab.tolist(),normal_reaction_n=normal,
            tangent_resistance_n=tangent,actuator_power_nmm_s=power,
            generalized_resistance_n=power/self.loading_rate)
        return record

    def checkpoint_data(self):
        return dict(super().checkpoint_data(),
            path_parameters=np.array([self.specimen_angle,self.frame_rotation,self.loading_rate,self.tangent_ratio,self.start]))

    def audit_state(self, state, record):
        expected=np.array([self.specimen_angle,self.frame_rotation,self.loading_rate,self.tangent_ratio,self.start])
        if 'path_parameters' not in state or not np.array_equal(state['path_parameters'],expected):
            raise RuntimeError('Checkpoint loading path differs')
        return super().audit_state(state,record)

    def _audit_extra_acceptance(self,fresh,record):
        # Check horizontal reaction/power from the SAME fresh assembly with
        # both exact histories; a second identical acceptance adds no evidence.
        result=super()._audit_extra_acceptance(fresh,record)
        for key in ('normal_reaction_n','tangent_resistance_n','actuator_power_nmm_s'):
            np.testing.assert_allclose(fresh['motion'][key],record['motion'][key],rtol=1e-9,atol=1e-8)
        return dict(result,motion=fresh['motion'])


class InclinedPathProblem(InclinedPathMixin, QuadratureViscoelasticProblem):
    """Original local-P2 path; its mechanics and defaults are unchanged."""


class NativeInclinedPathProblem(InclinedPathMixin, ViscoelasticProblem):
    """Same prescribed motion on the original native P1 material assembly.

    One history per triangle, no hidden P2 points/DOFs and no state transfer
    from enriched paths. Only metadata needed for surface comparisons is added.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.enrichment is not None or self.nodal_volume is not None:
            raise ValueError('Native matched path requires unmodified P1')
        self.base_nodes = self.n
        self.kinematics = 'plane_strain'

    def checkpoint_data(self):
        return dict(super().checkpoint_data(), base_body_nodes=self.n)
