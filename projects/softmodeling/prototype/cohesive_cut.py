"""Restricted 0-degree cutting extension using real separated IPC boundaries.

A finite numerical reference gap avoids zero-distance IPC collision pairs.
Both faces are included in the original complete collision/CCD machinery, with
no additional collision filtering. Radial bonds transmit opening traction;
this is a mode-I mechanism demonstrator, not general shear/interface fracture.
"""
from dataclasses import replace
import numpy as np
from scipy.spatial import Delaunay
from .ipc_viscoelastic import ViscoelasticProblem
from .model_config import FEMMesh, local_to_lab
from .cohesive import CohesiveLaw, bonds_response


def split_mesh(mesh, config, gap_mm, seam_spacing_mm):
    """Two independently triangulated material halves, with an explicit void strip."""
    if config.inclination_deg != 0:
        raise ValueError('First cohesive demonstrator supports only 0 degrees; path is prescribed')
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
    def __init__(self, *, cohesive_law=None, gap_mm=.006, seam_spacing_mm=.1, **kwargs):
        if kwargs.get('inclination', 0.) != 0:
            raise ValueError('Only 0-degree prescribed-seam model is implemented')
        kwargs['inclination'] = 0.
        super().__init__(**kwargs)
        if gap_mm <= 2*self.dhat:
            raise ValueError('Reference seam gap must exceed 2*dhat; no zero-gap IPC switch')
        self.cohesive_law = cohesive_law or CohesiveLaw()
        self.gap_mm, self.seam_spacing_mm = float(gap_mm), float(seam_spacing_mm)
        blade = self.rest[self.n:].copy()
        old_edges = np.asarray(self.cmesh.edges)
        global_edges = self.surface[old_edges]
        knife_edges = global_edges[np.all(global_edges >= self.n, axis=1)]-self.n
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
        self.cohesive_maximum = np.zeros(len(self.bond_lengths))
        self.previous_cohesive_maximum = self.cohesive_maximum.copy()
        self.reference_contact_check = self.check_reference_contact()
        self.stagnation_window = 40

    def cohesive_response(self, x, *, tangent=False):
        return bonds_response(x, self.mesh.seam_pairs, self.bond_lengths, self.bond_areas,
                              self.previous_cohesive_maximum, self.cohesive_law,
                              tangent=tangent, matrix_size=self.total)

    def evaluate(self, x, lag=None, tangent=True, projected=True, *, with_metric=False):
        base = super().evaluate(x, lag, tangent, projected, with_metric=with_metric)
        q = self.cohesive_response(x, tangent=tangent)
        energy, gradient, hessian, minj = base[:4]
        energy += q['stored_energy_n_mm']+q['fracture_dissipation_n_mm']
        gradient = gradient+q['gradient']
        if tangent:
            hessian = hessian+(q['metric'] if projected else q['hessian'])
        if with_metric:
            return energy, gradient, hessian, minj, base[4]+q['metric']
        return energy, gradient, hessian, minj

    def material_acceptance(self, x, depth, accepted):
        accepted = super().material_acceptance(x, depth, accepted)
        q = self.cohesive_response(x)
        old = self.cohesive_law.response(self.previous_cohesive_maximum, self.previous_cohesive_maximum)
        dissipated_before = float(self.bond_areas@old['dissipated'])
        if (np.any(q['history'] < self.previous_cohesive_maximum)
                or q['fracture_dissipation_n_mm'] < dissipated_before-1e-10):
            raise RuntimeError('Cohesive irreversibility/dissipation failure')
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

    def step(self, x, old, depth, deadline):
        self.previous_cohesive_maximum = self.cohesive_maximum.copy()
        value, record = super().step(x, old, depth, deadline)
        # No trial damage was committed by the nonlinear residual evaluations.
        self.cohesive_maximum = self.cohesive_response(value)['history'].copy()
        return value, record

    def checkpoint_data(self):
        return dict(super().checkpoint_data(), cohesive_maximum=self.cohesive_maximum,
                    previous_cohesive_maximum=self.previous_cohesive_maximum,
                    seam_pairs=self.mesh.seam_pairs, reference_bond_lengths_mm=self.bond_lengths,
                    reference_bond_areas_mm2=self.bond_areas)

    def audit_state(self, state, record):
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
    law = CohesiveLaw(**config['cohesive'])
    material, mesh = config['material'], config['mesh']
    return CohesiveCutProblem(cohesive_law=law, gap_mm=config['reference_gap_mm'],
        seam_spacing_mm=mesh['seam_spacing_mm'], maxwell_shear=material['maxwell_shear_mpa'],
        relaxation_s=material['relaxation_s'], loading_rate=.2,
        bulk_modulus=material['bulk_mpa'], shear_modulus=material['equilibrium_shear_mpa'],
        mu=config['friction_coefficient'], dhat=config['dhat_mm'],
        local_spacing=mesh['local_spacing_mm'], surface_spacing=mesh['surface_spacing_mm'],
        coarse_spacing=mesh['coarse_spacing_mm'], nose_facets=mesh['nose_facets'],
        tip_radius=config['tip_radius_mm'], fine_halfwidth=3., max_iterations=500,
        predictor='secant', hessian_mode='regularized')
