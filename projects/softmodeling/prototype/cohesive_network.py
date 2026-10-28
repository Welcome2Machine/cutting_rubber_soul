"""Intrinsic cohesive interfaces on every internal edge of a P1 mesh.

Every original triangle receives independent nodes. Internal sides move inward
by half the numerical gap; exterior sides stay on their original supporting
lines. This creates real separated boundaries for full IPC/CCD, at the cost of
material area loss and small openings at original boundary vertices. It is not
an intact-equivalent discretization. Cracks can select among all original
internal edges, but cannot travel through triangle interiors.
"""
from dataclasses import dataclass

import numpy as np

from .model_config import FEMMesh, local_to_lab


@dataclass
class NetworkTopology:
    segments: np.ndarray
    reference_areas: np.ndarray
    reference_gaps: np.ndarray
    reference_slips: np.ndarray
    reference_lengths_mm: np.ndarray
    original_vertex_ids: np.ndarray
    original_edge_ids: np.ndarray
    original_edges: np.ndarray
    all_original_edges: np.ndarray
    original_triangles: np.ndarray
    exterior_edges: np.ndarray
    exterior_nodes: np.ndarray
    exterior_seed_segments: np.ndarray
    top_surface_seed_segments: np.ndarray
    edge_to_segment: dict
    original_edge_to_segment: dict
    original_area_mm2: float
    retained_area_mm2: float
    bulk_area_ratio: float
    triangle_area_ratios: np.ndarray
    original_bottom_copies: np.ndarray
    bottom_support_nodes: np.ndarray
    excluded_displaced_bottom_copies: np.ndarray
    bottom_support_tolerance_mm: float
    gap_mm: float

    def to_dict(self):
        """Compact provenance; node and interface arrays belong in saved fields."""
        return dict(kind='intrinsic_all_internal_edges',
                    path_restriction='original_mesh_edges',
                    reference_gap_mm=self.gap_mm,
                    original_triangles=len(self.original_triangles),
                    material_nodes=len(self.original_vertex_ids),
                    candidate_interfaces=len(self.segments),
                    reference_interface_area_mm2=float(self.reference_areas.sum()),
                    original_area_mm2=self.original_area_mm2,
                    retained_area_mm2=self.retained_area_mm2,
                    removed_area_mm2=self.original_area_mm2-self.retained_area_mm2,
                    bulk_area_ratio=self.bulk_area_ratio,
                    minimum_triangle_area_ratio=float(self.triangle_area_ratios.min()),
                    minimum_reference_normal_gap_mm=float(self.reference_gaps.min())
                        if self.reference_gaps.size else None,
                    maximum_reference_slip_offset_mm=float(np.abs(self.reference_slips).max())
                        if self.reference_slips.size else None,
                    exterior_seed_interfaces=int(np.count_nonzero(self.exterior_seed_segments)),
                    top_surface_seed_interfaces=int(np.count_nonzero(self.top_surface_seed_segments)),
                    bottom_boundary_rule='original_bottom_copy_on_actual_inset_support_line_n_eq_0',
                    original_bottom_copy_count=len(self.original_bottom_copies),
                    fixed_bottom_node_count=len(self.bottom_support_nodes),
                    excluded_displaced_bottom_copy_count=len(self.excluded_displaced_bottom_copies),
                    bottom_support_tolerance_mm=self.bottom_support_tolerance_mm,
                    intact_equivalence=False,
                    boundary_geometry='original_exterior_lines_with_vertex_gaps')


def network_mesh(mesh, config, gap_mm):
    """Create independent, inset triangles and all internal-edge interfaces.

    Input must be a conforming manifold P1 triangulation with no intersections.
    Each triangle stays inside its original triangle, so valid input cannot
    gain overlapping material. Excessive gaps are rejected instead of clipping
    collapsed triangles. Input arrays and original vertex identities are kept.
    """
    if not np.isfinite(gap_mm) or gap_mm <= 0.:
        raise ValueError('Positive finite network gap required')
    if not np.isfinite(config.width_mm) or config.width_mm <= 0.:
        raise ValueError('Positive finite reference specimen width required')
    sn = np.asarray(mesh.local_sn, float)
    triangles = np.asarray(mesh.triangles)
    if sn.ndim != 2 or sn.shape[1] != 2 or not np.isfinite(sn).all():
        raise ValueError('Finite original (nnode, 2) coordinates required')
    if (triangles.ndim != 2 or triangles.shape[1] != 3
            or triangles.dtype.kind not in 'iu' or not len(triangles)):
        raise ValueError('Nonempty integer P1 triangle connectivity required')
    if np.any(triangles < 0) or np.any(triangles >= len(sn)):
        raise ValueError('Original triangle index outside nodes')
    triangles = triangles.copy()
    xyz = sn[triangles]
    signed = np.linalg.det(np.stack((xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=2))
    if np.any(signed == 0.) or not np.isfinite(signed).all():
        raise ValueError('Nondegenerate finite original triangles required')
    triangles[signed < 0.] = triangles[signed < 0.][:, [0, 2, 1]]
    xyz = sn[triangles]
    original_areas = np.abs(signed)/2.
    if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
        raise ValueError('Duplicate original triangle')
    owners = {}
    for triangle_id, nodes in enumerate(triangles):
        for k in range(3):
            a, b = int(nodes[k]), int(nodes[(k+1) % 3])
            owners.setdefault(tuple(sorted((a, b))), []).append((triangle_id, k, a, b))
    if any(len(value) > 2 for value in owners.values()):
        raise ValueError('Nonmanifold original edge')
    all_edges = np.array(sorted(owners), dtype=int).reshape(-1, 2)
    boundary = np.array([edge for edge in all_edges if len(owners[tuple(edge)]) == 1], dtype=int).reshape(-1, 2)
    external_nodes = np.unique(boundary)
    internal_edge_ids = np.array([i for i, edge in enumerate(all_edges)
                                  if len(owners[tuple(edge)]) == 2], dtype=int)
    internal_edges = all_edges[internal_edge_ids]

    # The clockwise unit normal points outside each CCW triangle. Moving an
    # internal line inward decreases its support value by gap/2.
    edges = np.roll(xyz, -1, axis=1)-xyz
    lengths = np.linalg.norm(edges, axis=2)
    outward = np.stack((edges[..., 1], -edges[..., 0]), axis=2)/lengths[..., None]
    offsets = np.einsum('tki,tki->tk', outward, xyz)
    insets = np.zeros_like(offsets)
    for edge in internal_edges:
        entries = owners[tuple(edge)]
        if entries[0][2:] != entries[1][2:][::-1]:
            raise ValueError('Adjacent original triangles must lie on opposite edge sides')
        for triangle_id, k, _, _ in entries:
            insets[triangle_id, k] = gap_mm/2.
    support = offsets-insets
    new_xyz = np.empty_like(xyz)
    for k in range(3):
        adjacent = [(k-1) % 3, k]
        new_xyz[:, k] = np.linalg.solve(outward[:, adjacent], support[:, adjacent, None])[..., 0]
    tolerance = 128*np.finfo(float).eps*max(1., float(np.abs(sn).max()))
    bottom_original = np.asarray(mesh.bottom_nodes, dtype=int)
    original_bottom_mask = np.isin(triangles, bottom_original)
    support_mask = original_bottom_mask & (np.abs(new_xyz[..., 1]) <= tolerance)
    # A copy shifted into the bulk no longer belongs to the support boundary.
    # Only roundoff-sized departures of actual support nodes are snapped to the
    # exact n=0 line, retaining the intact affine predictor's fixed-node value.
    new_xyz[..., 1][support_mask] = 0.
    signed_new = np.linalg.det(np.stack((new_xyz[:, 1]-new_xyz[:, 0],
                                         new_xyz[:, 2]-new_xyz[:, 0]), axis=2))
    residual = np.einsum('tki,tji->tjk', outward, new_xyz)-support[:, None, :]
    if (not np.isfinite(new_xyz).all() or np.any(signed_new <= 0.)
            or np.any(residual > tolerance)):
        raise ValueError('Network gap collapses a triangle; refine gap/mesh explicitly')
    retained_areas = signed_new/2.
    if np.any(retained_areas > original_areas*(1.+1e-10)):
        raise ValueError('Inset construction unexpectedly increased material area')

    new_sn = new_xyz.reshape(-1, 2)
    new_triangles = np.arange(new_sn.shape[0]).reshape(-1, 3)
    original_vertex_ids = triangles.ravel().copy()
    segments = np.empty((len(internal_edges), 4), dtype=int)
    physical_edge_map = {}
    original_edge_map = {}
    for interface_id, edge in enumerate(internal_edges):
        (left_id, k, a, b), (right_id, _, _, _) = owners[tuple(edge)]
        left0, left1 = 3*left_id+k, 3*left_id+(k+1) % 3
        right0 = 3*right_id+int(np.flatnonzero(triangles[right_id] == a)[0])
        right1 = 3*right_id+int(np.flatnonzero(triangles[right_id] == b)[0])
        segments[interface_id] = [left0, right0, left1, right1]
        physical_edge_map[tuple(sorted((left0, left1)))] = interface_id
        physical_edge_map[tuple(sorted((right0, right1)))] = interface_id
        original_edge_map[tuple(map(int, edge))] = interface_id
    new_xy = local_to_lab(new_sn, config)
    segment_points = new_xy[segments]
    v = .5*(segment_points[:, 2]+segment_points[:, 3]
            -segment_points[:, 0]-segment_points[:, 1])
    t = v/np.linalg.norm(v, axis=1)[:, None]
    n = np.column_stack((t[:, 1], -t[:, 0]))
    gaps = np.empty((len(segments), 2))
    slips = np.empty_like(gaps)
    for g, xi in enumerate((-1./np.sqrt(3.), 1./np.sqrt(3.))):
        r = ((1.-xi)/2.*(segment_points[:, 1]-segment_points[:, 0])
             +(1.+xi)/2.*(segment_points[:, 3]-segment_points[:, 2]))
        gaps[:, g] = np.einsum('ij,ij->i', r, n)
        slips[:, g] = np.einsum('ij,ij->i', r, t)
    if np.any(gaps < gap_mm-tolerance):
        raise ValueError('Inset interface reference normal gap below requested gap')
    reference_lengths = np.linalg.norm(sn[internal_edges[:, 1]]-sn[internal_edges[:, 0]], axis=1)
    reference_areas = reference_lengths*config.width_mm
    original_bottom_copies = np.flatnonzero(original_bottom_mask.ravel())
    bottom_nodes = np.flatnonzero(support_mask.ravel())
    displaced_bottom_copies = np.flatnonzero((original_bottom_mask & ~support_mask).ravel())
    seam_nodes = np.unique(segments)
    result = FEMMesh(new_xy, new_sn, new_triangles, np.empty((0, 2), dtype=int),
                     np.empty(0), bottom_nodes, seam_nodes, np.zeros(len(new_sn), dtype=int))
    top_original = np.flatnonzero(np.isclose(sn[:, 1], config.height_mm, atol=tolerance, rtol=0.))
    topology = NetworkTopology(
        segments=segments, reference_areas=reference_areas, reference_gaps=gaps,
        reference_slips=slips, reference_lengths_mm=reference_lengths,
        original_vertex_ids=original_vertex_ids, original_edge_ids=internal_edge_ids,
        original_edges=internal_edges, all_original_edges=all_edges,
        original_triangles=triangles, exterior_edges=boundary, exterior_nodes=external_nodes,
        exterior_seed_segments=np.any(np.isin(internal_edges, external_nodes), axis=1),
        top_surface_seed_segments=np.any(np.isin(internal_edges, top_original), axis=1),
        edge_to_segment=physical_edge_map, original_edge_to_segment=original_edge_map,
        original_area_mm2=float(original_areas.sum()), retained_area_mm2=float(retained_areas.sum()),
        bulk_area_ratio=float(retained_areas.sum()/original_areas.sum()),
        triangle_area_ratios=retained_areas/original_areas,
        original_bottom_copies=original_bottom_copies, bottom_support_nodes=bottom_nodes.copy(),
        excluded_displaced_bottom_copies=displaced_bottom_copies,
        bottom_support_tolerance_mm=tolerance, gap_mm=float(gap_mm))
    return result, topology
