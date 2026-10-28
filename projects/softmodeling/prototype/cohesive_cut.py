"""Prescribed-seam cutting extensions using real separated IPC boundaries.

A finite numerical reference gap avoids zero-distance IPC collision pairs.
Both faces are included in the original complete collision/CCD machinery, with
no additional collision filtering. Radial bonds transmit opening traction;
the legacy branch remains a mode-I mechanism demonstrator.  The explicit
mixed-mode branch uses objective four-node, two-point interface elements.
"""
from dataclasses import replace
import json
import numpy as np
from scipy.sparse import diags
from scipy.spatial import Delaunay
from .ipc_viscoelastic import ViscoelasticProblem
from .model_config import FEMMesh, local_to_lab
from .cohesive import CohesiveLaw, bonds_response
from .cohesive_mixed import MixedModeLaw, interface_response
from .cohesive_network import network_mesh


def split_mesh(mesh, config, gap_mm, seam_spacing_mm):
    """Two independently triangulated material halves, with an explicit void strip."""
    if not np.isfinite([gap_mm, seam_spacing_mm]).all() or min(gap_mm, seam_spacing_mm) <= 0:
        raise ValueError('Positive finite seam gap/spacing required')
    middle = config.drop_s_mm
    levels = np.linspace(0., config.height_mm,
                         int(np.ceil(config.height_mm/seam_spacing_mm))+1)
    parts, triangles, seam_ids, offset = [], [], [], 0
    for side in (-1, 1):
        sx = middle+side*gap_mm/2
        keep = side*(mesh.local_sn[:, 0]-sx) > 1e-10
        sn = np.unique(np.round(np.vstack((mesh.local_sn[keep],
                      np.column_stack((np.full(len(levels), sx), levels)))), 10), axis=0)
        tri = Delaunay(sn).simplices.copy()
        xy = local_to_lab(sn, config)
        det = np.linalg.det(np.stack((xy[tri[:, 1]]-xy[tri[:, 0]],
                                     xy[tri[:, 2]]-xy[tri[:, 0]]), axis=2))
        tri[det < 0] = tri[det < 0][:, [0, 2, 1]]
        seam = np.flatnonzero(np.isclose(sn[:, 0], sx, atol=1e-10, rtol=0))
        seam = seam[np.argsort(sn[seam, 1])]
        if len(seam) != len(levels):
            raise ValueError('Inconsistent seam node sampling')
        parts.append(sn);triangles.append(tri+offset);seam_ids.append(seam+offset)
        offset += len(sn)
    sn = np.vstack(parts)
    pairs = np.column_stack(seam_ids)
    lengths = np.diff(levels)
    tributary = np.r_[lengths[0]/2, (lengths[:-1]+lengths[1:])/2, lengths[-1]/2]
    sides = np.r_[np.full(len(parts[0]), -1), np.full(len(parts[1]), 1)]
    result = FEMMesh(local_to_lab(sn, config), sn, np.vstack(triangles), pairs,
                     tributary, np.flatnonzero(np.isclose(sn[:, 1], 0.)),
                     pairs.ravel(), sides)
    return result


class CohesiveCutProblem(ViscoelasticProblem):
    def __init__(self, *, cohesive_law=None, cohesive_kind='radial_mode_i',
                 fracture_topology='prescribed_seam', gap_mm=.006,
                 seam_spacing_mm=.1, network_metric_floor_n_per_mm=1.,
                 network_predictor='body_stationary_if_safe', **kwargs):
        if cohesive_kind not in ('radial_mode_i', 'mixed_mode_energy'):
            raise ValueError('Unknown cohesive_kind')
        self.cohesive_kind = cohesive_kind
        self.is_mixed_mode = cohesive_kind == 'mixed_mode_energy'
        if fracture_topology not in ('prescribed_seam', 'edge_network'):
            raise ValueError('Unknown fracture_topology')
        if fracture_topology == 'edge_network' and not self.is_mixed_mode:
            raise ValueError('The edge network requires the objective mixed-mode law')
        self.fracture_topology = fracture_topology
        self.network_topology = None
        if (isinstance(network_metric_floor_n_per_mm, bool)
                or not np.isfinite(network_metric_floor_n_per_mm)
                or network_metric_floor_n_per_mm <= 0):
            raise ValueError('Positive finite network numerical metric floor required')
        self.network_metric_floor_n_per_mm = (float(network_metric_floor_n_per_mm)
            if fracture_topology == 'edge_network' else 0.)
        if network_predictor not in ('body_stationary_if_safe', 'baseline'):
            raise ValueError('Unknown network numerical predictor policy')
        self.network_predictor = network_predictor
        self.network_predictor_trace = None
        angle = float(kwargs.get('inclination', 0.))
        if not np.isfinite(angle) or not 0 <= angle < 90:
            raise ValueError('Inclination must be finite and within [0, 90) degrees')
        if not self.is_mixed_mode and angle != 0:
            raise ValueError('Only 0-degree prescribed-seam model is implemented')
        kwargs['inclination'] = angle
        super().__init__(**kwargs)
        if gap_mm <= 2*self.dhat:
            raise ValueError('Reference seam gap must exceed 2*dhat; no zero-gap IPC switch')
        law_class = MixedModeLaw if self.is_mixed_mode else CohesiveLaw
        self.cohesive_law = cohesive_law or law_class()
        if not isinstance(self.cohesive_law, law_class):
            raise ValueError('Cohesive law and cohesive_kind differ')
        self.gap_mm, self.seam_spacing_mm = float(gap_mm), float(seam_spacing_mm)
        blade = self.rest[self.n:].copy()
        old_edges = np.asarray(self.cmesh.edges)
        global_edges = self.surface[old_edges]
        knife_edges = global_edges[np.all(global_edges >= self.n, axis=1)]-self.n
        if self.fracture_topology == 'edge_network':
            self.mesh, self.network_topology = network_mesh(self.mesh, self.config, self.gap_mm)
        else:
            self.mesh = split_mesh(self.mesh, self.config, self.gap_mm, self.seam_spacing_mm)
        self.n = self.mesh.node_count
        self.rest = np.vstack((self.mesh.reference_xy, blade))
        self.total = self.rest.size
        tri = self.mesh.triangles
        edges = np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
        unique, count = np.unique(edges, axis=0, return_counts=True)
        edges = np.vstack((unique[count == 1], knife_edges+self.n))
        self.surface = np.unique(edges)
        self.surface_dofs = (2*self.surface[:, None]+np.arange(2)).ravel()
        self.cmesh = self.ipc.CollisionMesh(self.rest[self.surface],
                                            np.searchsorted(self.surface, edges))
        self._connected_boundary_pairs = {tuple(sorted(map(int, e))) for e in self.cmesh.edges}
        self.cmesh.can_collide = self.ipc.make_static_obstacle_filter(int(np.count_nonzero(self.surface < self.n)))
        if self.ipc.has_intersections(self.cmesh, self.rest[self.surface]):
            raise ValueError('Intersecting finite-gap reference mesh')
        fixed = (2*self.mesh.bottom_nodes[:, None]+np.arange(2)).ravel()
        self.free = np.setdiff1d(np.arange(2*self.n), fixed)
        self.grad, self.dofs = self.mesh.bulk_data[1:]
        self.previous = self.rest.copy()
        self.material_metric = np.tile(np.eye(3), (len(tri), 1, 1))
        self.previous_material_metric = self.material_metric.copy()
        self.bond_lengths = np.linalg.norm(self.rest[self.mesh.seam_pairs[:, 1]]-
                                           self.rest[self.mesh.seam_pairs[:, 0]], axis=1)
        self.bond_areas = self.mesh.seam_segment_lengths_mm*self.config.width_mm
        if self.network_topology is not None:
            topology = self.network_topology
            self.interface_segments = topology.segments
            self.interface_lengths = topology.reference_lengths_mm
            self.interface_areas = topology.reference_areas
            self.interface_gaps = topology.reference_gaps
            self.interface_slips = topology.reference_slips
            self.fracture_topology_metadata = dict(topology.to_dict(),
                fracture_topology=self.fracture_topology,
                numerical_metric_floor_n_per_mm=self.network_metric_floor_n_per_mm,
                numerical_metric_role='positive PTC direction metric only; no physical energy, force, true tangent or dissipation',
                numerical_predictor_policy=self.network_predictor,
                crack_length_definition='sum of original reference lengths of exterior-connected fully failed interface edges; may branch')
        else:
            pairs = self.mesh.seam_pairs
            self.interface_segments = np.column_stack((pairs[:-1], pairs[1:]))
            self.interface_lengths = np.diff(self.mesh.local_sn[pairs[:, 0], 1])
            self.interface_areas = self.interface_lengths*self.config.width_mm
            self.interface_gaps = .5*(self.bond_lengths[:-1]+self.bond_lengths[1:])
            self.interface_slips = np.zeros((len(self.interface_segments), 2))
            self.fracture_topology_metadata = dict(fracture_topology=self.fracture_topology,
                candidate_interfaces=len(self.interface_segments),
                path_restriction='one_prescribed_reference_seam',
                reference_gap_mm=self.gap_mm,
                reference_interface_area_mm2=float(self.interface_areas.sum()),
                intact_equivalence=False,
                crack_length_definition='sum of consecutive top-surface-connected fully failed reference segments')
        self.cohesive_maximum = np.zeros((len(self.interface_segments), 2)
                                        if self.is_mixed_mode else len(self.bond_lengths))
        self.previous_cohesive_maximum = self.cohesive_maximum.copy()
        self.reference_contact_check = self.check_reference_contact()
        # The intrinsic network can spend many valid PTC iterations closing its
        # finite gaps. The old 40-step secant plateau cutoff does not characterize
        # this new stationary-body baseline. Keep the hard iteration, residual,
        # geometry and wall-clock gates; do not change the original branches.
        self.stagnation_window = 0 if self.network_topology is not None else 40
        if self.network_topology is not None:
            self.fracture_topology_metadata.update(
                numerical_stagnation_window=self.stagnation_window,
                numerical_max_iterations=self.max_iterations,
                numerical_stopping_policy='no early plateau cutoff; unchanged hard iteration, physical residual, CCD/J and wall-clock gates')

    def affine_loading_candidate(self, x, old, depth):
        """Choose a safe base for the existing predictor without remote compression.

        The intrinsic network has no cohesive compressive stiffness before IPC
        closes its finite gaps. Artificially compressing every triangle can
        leave slowly relaxing remote force modes. A stationary-body base avoids
        that perturbation when moving only the blade is collision-free. The
        original affine retry remains available and equations/history are shared.
        """
        affine = super().affine_loading_candidate(x, old, depth)
        if self.network_topology is None:
            return affine
        trace = dict(policy=self.network_predictor, selected_base='baseline_affine',
                     from_depth_mm=float(old), target_depth_mm=float(depth),
                     physical_history_substeps_added=False)
        if self.network_predictor == 'baseline' or self.predictor == 'affine':
            trace['reason'] = ('explicit_baseline_policy' if self.network_predictor == 'baseline'
                               else 'conservative_affine_retry')
            self.network_predictor_trace = trace
            print('NETWORK_PREDICTOR_BASE '+json.dumps(trace, sort_keys=True), flush=True)
            return affine
        candidate = x.copy()
        candidate[self.n:] = self.prescribed_blade_positions(depth)
        fixed = np.setdiff1d(np.arange(2*self.n), self.free)
        try:
            if (not np.isfinite(candidate).all()
                    or not np.array_equal(candidate.ravel()[fixed], self.rest.ravel()[fixed])):
                raise ValueError('Stationary-body candidate violates fixed boundaries')
            fraction = float(self.safe_step(x, candidate-x))
            trace['stationary_safe_fraction'] = fraction
            if not np.isfinite(fraction) or fraction < 1.-1e-10:
                raise ValueError('Stationary-body candidate path fails complete CCD/J')
            if self.ipc.has_intersections(self.cmesh, candidate[self.surface]):
                raise ValueError('Stationary-body candidate endpoint intersects')
            _, gradient, _, minj = self.evaluate(candidate, self.lag(candidate), tangent=False)
            residual = float(np.linalg.norm(gradient[self.free]))
            if (not np.isfinite(gradient).all() or not np.isfinite(residual)
                    or not np.isfinite(minj) or minj <= 0.):
                raise ValueError('Stationary-body candidate endpoint has invalid mechanics')
            trace.update(selected_base='body_stationary', stationary_fresh_residual_n=residual,
                         stationary_min_j=float(minj), reason='safe_stationary_body_preserves_remote_initial_state')
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            trace['reason'] = 'stationary_fallback: '+type(exc).__name__+': '+str(exc)
            candidate = affine
        self.network_predictor_trace = trace
        print('NETWORK_PREDICTOR_BASE '+json.dumps(trace, sort_keys=True), flush=True)
        return candidate

    def cohesive_response(self, x, *, tangent=False):
        if self.is_mixed_mode:
            return interface_response(x, self.interface_segments, self.interface_gaps,
                                      self.interface_areas, self.previous_cohesive_maximum,
                                      self.cohesive_law, tangent=tangent,
                                      matrix_size=self.total,
                                      reference_slips=self.interface_slips)
        return bonds_response(x, self.mesh.seam_pairs, self.bond_lengths, self.bond_areas,
                              self.previous_cohesive_maximum, self.cohesive_law,
                              tangent=tangent, matrix_size=self.total)

    def cohesive_history_response(self, history=None):
        """Damage/dissipation from accepted maxima, without inventing separations."""
        history = self.cohesive_maximum if history is None else np.asarray(history, float)
        if history.shape != self.cohesive_maximum.shape:
            raise ValueError('Cohesive history shape differs from interface discretization')
        if self.is_mixed_mode:
            return self.cohesive_law.history_response(history)
        return self.cohesive_law.response(history, history)

    def cohesive_history_dissipation(self, history=None):
        q = self.cohesive_history_response(history)
        if self.is_mixed_mode:
            return float(self.interface_areas @ np.mean(q['dissipated'], axis=1))
        return float(self.bond_areas @ q['dissipated'])

    def cohesive_broken_segments(self, history=None):
        d = self.cohesive_history_response(history)['damage']
        if self.is_mixed_mode:
            return np.all(d >= 1-1e-10, axis=1)
        return (d[:-1] >= 1-1e-10) & (d[1:] >= 1-1e-10)

    def evaluate(self, x, lag=None, tangent=True, projected=True, *, with_metric=False):
        base = super().evaluate(x, lag, tangent, projected, with_metric=with_metric)
        q = self.cohesive_response(x, tangent=tangent)
        energy, gradient, hessian, minj = base[:4]
        energy += q['stored_energy_n_mm']+q['fracture_dissipation_n_mm']
        gradient = gradient+q['gradient']
        if tangent:
            hessian = hessian+(q['metric'] if projected else q['hessian'])
        if with_metric:
            metric = base[4]+q['metric']
            if self.network_topology is not None:
                # Intrinsic separated triangles have nearly unconstrained
                # relative normal motions while cohesive compression is inactive
                # and IPC has not yet engaged.  This positive algorithmic metric
                # regularizes only PTC directions; physical equations stay intact.
                metric += diags(np.r_[np.full(2*self.n, self.network_metric_floor_n_per_mm),
                                      np.zeros(self.total-2*self.n)], format='csc')
            return energy, gradient, hessian, minj, metric
        return energy, gradient, hessian, minj

    def material_acceptance(self, x, depth, accepted):
        accepted = super().material_acceptance(x, depth, accepted)
        q = self.cohesive_response(x)
        dissipated_before = self.cohesive_history_dissipation(self.previous_cohesive_maximum)
        if (np.any(q['history'] < self.previous_cohesive_maximum)
                or q['fracture_dissipation_n_mm'] < dissipated_before-1e-10):
            raise RuntimeError('Cohesive irreversibility/dissipation failure')
        if self.is_mixed_mode:
            accepted['fracture'] = self._mixed_fracture_record(x, q, dissipated_before)
            if self.network_topology is not None:
                accepted['network_numerics'] = dict(
                    metric_floor_n_per_mm=self.network_metric_floor_n_per_mm,
                    predictor_policy=self.network_predictor,
                    stagnation_window=self.stagnation_window,
                    max_iterations=self.max_iterations,
                    predictor_base=self.network_predictor_trace)
            return accepted
        d = q['damage']
        contact = self.collisions(x)
        seam_nodes = set(map(int, self.mesh.seam_pairs.ravel()))
        seam_edges = {tuple(sorted((int(a), int(b))))
                      for side in (0, 1) for a, b in zip(self.mesh.seam_pairs[:-1, side], self.mesh.seam_pairs[1:, side])}
        node_bonds = {int(node): int(i) for i, pair in enumerate(self.mesh.seam_pairs) for node in pair}
        broken = d >= 1-1e-10
        connected = []
        for i in range(len(broken)-1, -1, -1):
            if not broken[i]:
                break
            connected.append(i)
        contact_stencils = []
        new_face_ev = new_face_vv = 0
        for stencil in contact.ev_collisions:
            edge = tuple(map(int, self.surface[np.asarray(self.cmesh.edges)[stencil.edge_id]]))
            vertex = int(self.surface[stencil.vertex_id])
            if (vertex >= self.n and tuple(sorted(edge)) in seam_edges
                    or all(v >= self.n for v in edge) and vertex in seam_nodes):
                new_face_ev += 1
                bond_ids = [node_bonds[v] for v in edge if v in node_bonds]
                if vertex in node_bonds:
                    bond_ids.append(node_bonds[vertex])
                separated = bool(all(broken[i] for i in bond_ids))
                if len(bond_ids) == 1:
                    i = bond_ids[0]
                    separated = separated and any(broken[j] for j in (i-1, i+1) if 0 <= j < len(broken))
                contact_stencils.append(dict(kind='EV', global_edge_nodes=list(edge), global_vertex=vertex,
                    bond_ids=bond_ids, bond_damage=d[bond_ids].tolist(),
                    touches_fully_separated_face=separated,
                    touches_surface_connected_crack=separated and all(i in connected for i in bond_ids)))
        for stencil in contact.vv_collisions:
            a, b = map(int, self.surface[[stencil.vertex0_id, stencil.vertex1_id]])
            if a >= self.n and b in seam_nodes or b >= self.n and a in seam_nodes:
                new_face_vv += 1
                i = node_bonds[b if b in seam_nodes else a]
                separated = bool(broken[i] and any(broken[j] for j in (i-1, i+1) if 0 <= j < len(broken)))
                contact_stencils.append(dict(kind='VV', global_nodes=[a,b], bond_ids=[i],
                    bond_damage=[float(d[i])], touches_fully_separated_face=separated,
                    touches_surface_connected_crack=separated and i in connected))
        connected_length = (float(self.mesh.local_sn[self.mesh.seam_pairs[-1,0],1]
                            -self.mesh.local_sn[self.mesh.seam_pairs[min(connected),0],1])
                            if len(connected) >= 2 else 0.)
        accepted['fracture'] = dict(max_damage=float(d.max()),
            damaged_bonds=int(np.count_nonzero(d > 1e-8)),
            separated_bonds=int(np.count_nonzero(d >= 1-1e-10)),
            separated_reference_length_mm=float(self.mesh.seam_segment_lengths_mm@(d >= 1-1e-10)),
            cohesive_stored_energy_n_mm=q['stored_energy_n_mm'],
            fracture_dissipation_n_mm=q['fracture_dissipation_n_mm'],
            fracture_dissipation_increment_n_mm=q['fracture_dissipation_n_mm']-dissipated_before,
            maximum_constitutive_opening_mm=float(q['opening'].max()),
            minimum_paired_face_distance_mm=float(q['length'].min()),
            blade_new_face_ev_contacts=new_face_ev, blade_new_face_vv_contacts=new_face_vv,
            blade_fully_separated_face_contacts=sum(s['touches_fully_separated_face'] for s in contact_stencils),
            blade_surface_connected_crack_contacts=sum(s['touches_surface_connected_crack'] for s in contact_stencils),
            surface_connected_failed_bonds=connected, surface_connected_crack_length_mm=connected_length,
            blade_lowest_z_mm=float(x[self.n:,1].min()), contact_stencils=contact_stencils,
            ipc_barrier_stored_energy_n_mm=float(self.barrier(contact, self.cmesh, x[self.surface])),
            max_body_step_motion_mm=float(np.max(np.linalg.norm(x[:self.n]-self.previous[:self.n], axis=1))),
            status='uncalibrated finite-gap radial interface mechanism test')
        return accepted

    def _mixed_fracture_record(self, x, q, dissipated_before):
        """Classify crack faces by complete segments, never one failed Gauss point."""
        d = q['damage']
        broken = np.all(d >= 1-1e-10, axis=1)
        if self.network_topology is None:
            connected = []
            for i in range(len(broken)-1, -1, -1):
                if not broken[i]:
                    break
                connected.append(i)
        else:
            connected = self._network_connected_segments(broken)
        connected_set = set(connected)
        seam_nodes = set(map(int, self.interface_segments.ravel()))
        edge_segments = {}
        node_segments = {node: [] for node in seam_nodes}
        for i, (l0, r0, l1, r1) in enumerate(self.interface_segments):
            for a, b in ((l0, l1), (r0, r1)):
                edge_segments[tuple(sorted((int(a), int(b))))] = i
            for node in (l0, r0, l1, r1):
                node_segments[int(node)].append(i)
        boundary_ambiguous_nodes = set()
        for edge in self.surface[np.asarray(self.cmesh.edges)]:
            a, b = map(int, edge)
            if a < self.n and b < self.n and tuple(sorted((a, b))) not in edge_segments:
                boundary_ambiguous_nodes.update((a, b))
        contact = self.collisions(x)
        contact_stencils = []
        new_face_ev = new_face_vv = 0

        def flags(ids, node=None):
            # A seam vertex can belong to both a failed and an intact segment.
            # Requiring all incident segments prevents promoting that ambiguity.
            ambiguous = node is not None and node in boundary_ambiguous_nodes
            separated = bool(not ambiguous and ids and all(broken[i] for i in ids))
            return dict(segment_ids=ids, gauss_damage=d[ids].tolist(),
                        attribution_ambiguous=ambiguous,
                        ambiguity_reason=('node also belongs to an original noncohesive body boundary edge'
                                          if ambiguous else None),
                        touches_fully_separated_face=separated,
                        touches_surface_connected_crack=separated and
                        all(i in connected_set for i in ids))

        for stencil in contact.ev_collisions:
            edge = tuple(map(int, self.surface[np.asarray(self.cmesh.edges)[stencil.edge_id]]))
            vertex = int(self.surface[stencil.vertex_id])
            key = tuple(sorted(edge))
            if vertex >= self.n and key in edge_segments:
                ids = [edge_segments[key]]
                node = None
            elif all(v >= self.n for v in edge) and vertex in seam_nodes:
                ids = node_segments[vertex]
                node = vertex
            else:
                continue
            new_face_ev += 1
            contact_stencils.append(dict(kind='EV', global_edge_nodes=list(edge),
                                         global_vertex=vertex, **flags(ids, node)))
        for stencil in contact.vv_collisions:
            a, b = map(int, self.surface[[stencil.vertex0_id, stencil.vertex1_id]])
            if a >= self.n and b in seam_nodes:
                node = b
            elif b >= self.n and a in seam_nodes:
                node = a
            else:
                continue
            new_face_vv += 1
            contact_stencils.append(dict(kind='VV', global_nodes=[a, b],
                                         **flags(node_segments[node], node)))
        return dict(cohesive_kind=self.cohesive_kind,
            fracture_topology=self.fracture_topology,
            damage_sampling='two_Gauss_points_per_reference_segment',
            compatibility_bond_counts='mixed branch bond counts denote segments; full failure requires both Gauss points',
            max_damage=float(d.max()), gauss_damage=d.tolist(),
            damaged_gauss_points=int(np.count_nonzero(d > 1e-8)),
            separated_gauss_points=int(np.count_nonzero(d >= 1-1e-10)),
            damaged_segments=int(np.count_nonzero(np.any(d > 1e-8, axis=1))),
            separated_segments=int(np.count_nonzero(broken)),
            damaged_bonds=int(np.count_nonzero(np.any(d > 1e-8, axis=1))),
            separated_bonds=int(np.count_nonzero(broken)),
            separated_reference_length_mm=float(self.interface_lengths @ broken),
            cohesive_stored_energy_n_mm=q['stored_energy_n_mm'],
            fracture_dissipation_n_mm=q['fracture_dissipation_n_mm'],
            fracture_dissipation_increment_n_mm=q['fracture_dissipation_n_mm']-dissipated_before,
            maximum_constitutive_opening_mm=float(q['opening'].max()),
            maximum_absolute_slip_mm=float(np.abs(q['slip']).max()),
            maximum_effective_opening_mm=float(q['effective_opening'].max()),
            maximum_strength_utilization=float(q['initiation_ratio'].max()),
            strength_utilization_definition='undamaged energy norm / damage-onset threshold; not current degraded traction ratio',
            minimum_paired_face_distance_mm=float(q['length'].min()),
            blade_new_face_ev_contacts=new_face_ev, blade_new_face_vv_contacts=new_face_vv,
            blade_fully_separated_face_contacts=sum(s['touches_fully_separated_face'] for s in contact_stencils),
            blade_surface_connected_crack_contacts=sum(s['touches_surface_connected_crack'] for s in contact_stencils),
            surface_connected_failed_segments=connected,
            surface_connected_failed_bonds=connected,
            surface_connected_crack_length_mm=float(self.interface_lengths[connected].sum()),
            surface_connected_failed_interface_reference_length_mm=float(self.interface_lengths[connected].sum()),
            connected_length_definition=self.fracture_topology_metadata['crack_length_definition'],
            blade_lowest_z_mm=float(x[self.n:, 1].min()), contact_stencils=contact_stencils,
            ipc_barrier_stored_energy_n_mm=float(self.barrier(contact, self.cmesh, x[self.surface])),
            max_body_step_motion_mm=float(np.max(np.linalg.norm(x[:self.n]-self.previous[:self.n], axis=1))),
            status=('uncalibrated mesh-edge-selecting finite-gap objective mixed-mode network'
                    if self.network_topology is not None else
                    'uncalibrated prescribed-path finite-gap objective mixed-mode interface'))

    def _network_connected_segments(self, broken):
        """Connect fully failed original edges to any original exterior surface."""
        topology = self.network_topology
        incident = {}
        for i, vertices in enumerate(topology.original_edges):
            for vertex in vertices:
                incident.setdefault(int(vertex), []).append(i)
        visited = set(map(int, np.flatnonzero(broken & topology.exterior_seed_segments)))
        pending = list(visited)
        while pending:
            i = pending.pop()
            for vertex in topology.original_edges[i]:
                for neighbour in incident[int(vertex)]:
                    if broken[neighbour] and neighbour not in visited:
                        visited.add(neighbour)
                        pending.append(neighbour)
        return sorted(visited)

    def step(self, x, old, depth, deadline):
        self.previous_cohesive_maximum = self.cohesive_maximum.copy()
        value, record = super().step(x, old, depth, deadline)
        # No trial damage was committed by the nonlinear residual evaluations.
        self.cohesive_maximum = self.cohesive_response(value)['history'].copy()
        return value, record

    def checkpoint_data(self):
        result = dict(super().checkpoint_data(), cohesive_maximum=self.cohesive_maximum,
                    previous_cohesive_maximum=self.previous_cohesive_maximum,
                    seam_pairs=self.mesh.seam_pairs, reference_bond_lengths_mm=self.bond_lengths,
                    reference_bond_areas_mm2=self.bond_areas)
        if self.is_mixed_mode:
            result.update(cohesive_kind=self.cohesive_kind,
                fracture_topology=self.fracture_topology,
                interface_segments=self.interface_segments,
                reference_interface_areas_mm2=self.interface_areas,
                reference_interface_gaps_mm=self.interface_gaps,
                reference_interface_slips_mm=self.interface_slips,
                reference_interface_lengths_mm=self.interface_lengths)
            if self.network_topology is not None:
                result.update(original_interface_edges=self.network_topology.original_edges,
                    interface_exterior_seed=self.network_topology.exterior_seed_segments,
                    original_vertex_ids=self.network_topology.original_vertex_ids,
                    network_metric_floor_n_per_mm=self.network_metric_floor_n_per_mm,
                    network_predictor_policy=self.network_predictor,
                    network_stagnation_window=self.stagnation_window,
                    network_max_iterations=self.max_iterations)
        return result

    def audit_state(self, state, record):
        if self.is_mixed_mode:
            if str(state.get('cohesive_kind', '')) != self.cohesive_kind:
                raise RuntimeError('Mixed-mode checkpoint kind differs or is missing')
            if str(state.get('fracture_topology', '')) != self.fracture_topology:
                raise RuntimeError('Mixed-mode checkpoint fracture topology differs or is missing')
            expected_fields = [
                    ('interface_segments', self.interface_segments),
                    ('reference_interface_areas_mm2', self.interface_areas),
                    ('reference_interface_gaps_mm', self.interface_gaps),
                    ('reference_interface_slips_mm', self.interface_slips),
                    ('reference_interface_lengths_mm', self.interface_lengths)]
            if self.network_topology is not None:
                expected_fields.extend([
                    ('original_interface_edges', self.network_topology.original_edges),
                    ('interface_exterior_seed', self.network_topology.exterior_seed_segments),
                    ('original_vertex_ids', self.network_topology.original_vertex_ids),
                    ('network_metric_floor_n_per_mm', self.network_metric_floor_n_per_mm),
                    ('network_predictor_policy', self.network_predictor),
                    ('network_stagnation_window', self.stagnation_window),
                    ('network_max_iterations', self.max_iterations)])
            for key, expected in expected_fields:
                if not np.array_equal(state.get(key), expected):
                    raise RuntimeError('Mixed-mode checkpoint interface topology/reference differs: '+key)
            self.cohesive_history_response(state['cohesive_maximum'])
            self.cohesive_history_response(state['previous_cohesive_maximum'])
        elif str(state.get('cohesive_kind', 'radial_mode_i')) != 'radial_mode_i':
            raise RuntimeError('Legacy checkpoint cannot restore mixed-mode history')
        before = self.previous_cohesive_maximum
        self.previous_cohesive_maximum = state['previous_cohesive_maximum'].copy()
        try:
            result = super().audit_state(state, record)
            q = self.cohesive_response(state['positions'])
            np.testing.assert_allclose(q['history'], state['cohesive_maximum'], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(q['fracture_dissipation_n_mm'],
                record['fracture']['fracture_dissipation_n_mm'], rtol=1e-10, atol=1e-10)
            return dict(result, cohesive_history_passed=True,
                        max_damage=float(q['damage'].max()))
        finally:
            self.previous_cohesive_maximum = before


def make_cohesive_problem(config):
    kind = config.get('cohesive_kind', 'radial_mode_i')
    law_class = MixedModeLaw if kind == 'mixed_mode_energy' else CohesiveLaw
    law = law_class(**config['cohesive'])
    material, mesh = config['material'], config['mesh']
    return CohesiveCutProblem(cohesive_law=law, cohesive_kind=kind,
        fracture_topology=config.get('fracture_topology', 'prescribed_seam'),
        network_metric_floor_n_per_mm=config.get('network_metric_floor_n_per_mm', 1.),
        network_predictor=config.get('network_predictor', 'body_stationary_if_safe'),
        inclination=config.get('angle_deg', 0.), gap_mm=config['reference_gap_mm'],
        seam_spacing_mm=mesh['seam_spacing_mm'], maxwell_shear=material['maxwell_shear_mpa'],
        relaxation_s=material['relaxation_s'], loading_rate=.2,
        bulk_modulus=material['bulk_mpa'], shear_modulus=material['equilibrium_shear_mpa'],
        mu=config['friction_coefficient'], dhat=config['dhat_mm'],
        local_spacing=mesh['local_spacing_mm'], surface_spacing=mesh['surface_spacing_mm'],
        coarse_spacing=mesh['coarse_spacing_mm'], nose_facets=mesh['nose_facets'],
        tip_radius=config['tip_radius_mm'], fine_halfwidth=3., max_iterations=500,
        predictor='secant', hessian_mode='regularized')
