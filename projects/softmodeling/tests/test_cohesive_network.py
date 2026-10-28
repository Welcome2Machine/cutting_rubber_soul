"""All-edge network topology, gap geometry and objective reference checks."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from prototype.cohesive_mixed import MixedModeLaw, interface_response
from prototype.cohesive_network import network_mesh
from prototype.model_config import FEMConfig, FEMMesh, local_to_lab


def original_mesh(points, triangles, angle=0.):
    points = np.asarray(points, float)
    triangles = np.asarray(triangles, int)
    config = FEMConfig(length_mm=1., height_mm=1., width_mm=2., inclination_deg=angle)
    mesh = FEMMesh(local_to_lab(points, config), points, triangles,
                   np.empty((0, 2), int), np.empty(0),
                   np.flatnonzero(np.isclose(points[:, 1], 0.)),
                   np.empty(0, int), np.zeros(len(points), int))
    return mesh, config


def grid_mesh():
    points = np.array([[i/3., j/3.] for j in range(4) for i in range(4)])
    triangles = []
    for j in range(3):
        for i in range(3):
            a = 4*j+i
            triangles.extend(([a, a+1, a+5], [a, a+5, a+4]))
    return original_mesh(points, triangles)


class CohesiveNetworkTests(unittest.TestCase):
    def setUp(self):
        self.original, self.config = original_mesh(
            [[0., 0.], [1., 0.], [1., 1.], [0., 1.]], [[0, 1, 2], [0, 2, 3]])
        self.gap = .02
        self.law = MixedModeLaw(shear_strength_mpa=.3)

    def response(self, mesh, topology, positions=None, old=None):
        if old is None:
            old = np.zeros((len(topology.segments), 2))
        return interface_response(mesh.reference_xy if positions is None else positions,
            topology.segments, topology.reference_gaps, topology.reference_areas,
            old, self.law, reference_slips=topology.reference_slips)

    def test_two_triangle_geometry_and_exact_area_loss(self):
        mesh, topology = network_mesh(self.original, self.config, self.gap)
        self.assertEqual(mesh.node_count, 6)
        self.assertEqual(topology.segments.shape, (1, 4))
        self.assertEqual(mesh.seam_pairs.shape, (0, 2))
        expected_area = (1.-self.gap/np.sqrt(2.))**2
        self.assertAlmostEqual(topology.original_area_mm2, 1.)
        self.assertAlmostEqual(topology.retained_area_mm2, expected_area)
        self.assertAlmostEqual(topology.bulk_area_ratio, expected_area)
        self.assertTrue(np.all(mesh.bulk_data[0] > 0.))
        np.testing.assert_allclose(topology.reference_gaps, self.gap, atol=1e-15)
        np.testing.assert_allclose(topology.reference_lengths_mm, np.sqrt(2.))
        np.testing.assert_allclose(topology.reference_areas, 2*np.sqrt(2.))
        self.assertFalse(topology.to_dict()['intact_equivalence'])

    def test_every_internal_edge_is_candidate_without_a_selected_path(self):
        original, config = grid_mesh()
        mesh, topology = network_mesh(original, config, .01)
        edges = np.sort(np.vstack((original.triangles[:, [0, 1]],
                                   original.triangles[:, [1, 2]],
                                   original.triangles[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(topology.original_edges, unique[counts == 2])
        self.assertEqual(len(topology.segments), np.count_nonzero(counts == 2))
        self.assertEqual(len(topology.edge_to_segment), 2*len(topology.segments))
        self.assertEqual(len(np.unique(mesh.triangles)), mesh.triangles.size)
        self.assertEqual(mesh.node_count, 3*len(original.triangles))
        self.assertEqual(len(topology.original_edge_to_segment), len(topology.segments))
        # The regular grid supplies horizontal, vertical and diagonal candidates.
        vectors = original.local_sn[topology.original_edges[:, 1]]-original.local_sn[topology.original_edges[:, 0]]
        self.assertTrue(np.any(np.isclose(vectors[:, 0], 0.)))
        self.assertTrue(np.any(np.isclose(vectors[:, 1], 0.)))
        self.assertTrue(np.any(np.all(np.abs(vectors) > .1, axis=1)))

    def test_inset_triangles_stay_inside_original_and_do_not_overlap(self):
        original, config = grid_mesh()
        mesh, topology = network_mesh(original, config, .02)
        inset = mesh.local_sn[mesh.triangles]
        for tri, points in zip(topology.original_triangles, inset):
            reference = original.local_sn[tri]
            edges = np.roll(reference, -1, axis=0)-reference
            outward = np.column_stack((edges[:, 1], -edges[:, 0]))
            supports = np.einsum('ij,ij->i', outward, reference)
            self.assertTrue(np.all(points@outward.T <= supports+1e-13))
        # Separating-axis theorem independently checks convex triangle overlap.
        for i, first in enumerate(inset):
            for second in inset[i+1:]:
                axes = []
                for points in (first, second):
                    e = np.roll(points, -1, axis=0)-points
                    axes.extend(np.column_stack((e[:, 1], -e[:, 0])))
                projected_first = first@np.asarray(axes).T
                projected_second = second@np.asarray(axes).T
                separated = ((projected_first.max(axis=0) <= projected_second.min(axis=0)+1e-13)
                             | (projected_second.max(axis=0) <= projected_first.min(axis=0)+1e-13))
                self.assertTrue(np.any(separated))

    def test_face_normal_gap_and_stress_free_reference_with_slip_offsets(self):
        # Unequal adjacent triangles produce nonzero reference tangential offsets.
        original, config = original_mesh([[0., 0.], [1., 0.], [.7, 1.], [0., .6]],
                                        [[0, 1, 2], [0, 2, 3]])
        mesh, topology = network_mesh(original, config, self.gap)
        self.assertGreater(np.max(np.abs(topology.reference_slips)), 1e-5)
        self.assertGreaterEqual(topology.reference_gaps.min(), self.gap-1e-13)
        q = self.response(mesh, topology)
        np.testing.assert_allclose(q['traction'], 0., atol=2e-13)
        np.testing.assert_allclose(q['gradient'], 0., atol=2e-13)
        self.assertLess(q['stored_energy_n_mm'], 1e-25)
        self.assertEqual(q['fracture_dissipation_n_mm'], 0.)
        preloaded = interface_response(mesh.reference_xy, topology.segments,
            topology.reference_gaps, topology.reference_areas, np.zeros((1, 2)), self.law)
        self.assertGreater(preloaded['stored_energy_n_mm'], 1e-8)

    def test_rigid_rotation_preserves_network_law_and_geometry(self):
        original, config = grid_mesh()
        mesh, topology = network_mesh(original, config, .01)
        moved = mesh.reference_xy.copy()
        moved[:, 0] *= 1.025
        moved[:, 1] += .005*moved[:, 0]
        q = self.response(mesh, topology, moved)
        angle = .71
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = self.response(mesh, topology, moved@rotation.T+[-2., 3.])
        self.assertAlmostEqual(q['incremental_potential_n_mm'], rotated['incremental_potential_n_mm'], places=12)
        np.testing.assert_allclose(rotated['gradient'].reshape(-1, 2),
                                   q['gradient'].reshape(-1, 2)@rotation.T, atol=2e-11)
        tilted, tilted_topology = network_mesh(original, replace(config, inclination_deg=57.), .01)
        self.assertAlmostEqual(tilted_topology.bulk_area_ratio, topology.bulk_area_ratio)
        np.testing.assert_allclose(tilted_topology.reference_gaps, topology.reference_gaps, atol=1e-14)
        np.testing.assert_allclose(self.response(tilted, tilted_topology)['gradient'], 0., atol=3e-12)

    def test_reference_slip_offsets_preserve_consistent_energy_gradient_and_tangent(self):
        original, config = original_mesh([[0., 0.], [1., 0.], [.7, 1.], [0., .6]],
                                        [[0, 1, 2], [0, 2, 3]])
        mesh, topology = network_mesh(original, config, self.gap)
        moved = 1.7*mesh.reference_xy
        q = self.response(mesh, topology, moved)
        h = 1e-7
        gradient = np.empty(moved.size)
        hessian = np.empty((moved.size, moved.size))
        for j in range(moved.size):
            shift = (h*np.eye(moved.size)[j]).reshape(moved.shape)
            plus = self.response(mesh, topology, moved+shift)
            minus = self.response(mesh, topology, moved-shift)
            gradient[j] = (plus['incremental_potential_n_mm']-minus['incremental_potential_n_mm'])/(2*h)
            hessian[:, j] = (plus['gradient']-minus['gradient'])/(2*h)
        np.testing.assert_allclose(q['gradient'], gradient, rtol=1e-6, atol=2e-9)
        np.testing.assert_allclose(q['hessian'].toarray(), hessian, rtol=1e-6, atol=2e-8)

    def test_surface_seeds_and_fixed_original_vertex_identity(self):
        original, config = grid_mesh()
        mesh, topology = network_mesh(original, config, .01)
        boundary_ids = set(topology.exterior_nodes.tolist())
        seeds = [a in boundary_ids or b in boundary_ids for a, b in topology.original_edges]
        np.testing.assert_array_equal(topology.exterior_seed_segments, seeds)
        self.assertTrue(np.any(topology.exterior_seed_segments))
        self.assertTrue(np.any(~topology.exterior_seed_segments))
        expected_top = np.any(np.isclose(original.local_sn[topology.original_edges, 1], config.height_mm), axis=1)
        np.testing.assert_array_equal(topology.top_surface_seed_segments, expected_top)
        np.testing.assert_array_equal(mesh.bottom_nodes,
            np.flatnonzero(np.isin(topology.original_vertex_ids, original.bottom_nodes)
                           & np.isclose(mesh.local_sn[:, 1], 0., atol=topology.bottom_support_tolerance_mm, rtol=0.)))
        self.assertTrue(np.all(np.isin(topology.original_vertex_ids[mesh.bottom_nodes], original.bottom_nodes)))

    def assert_actual_bottom_support(self, original, config, gap):
        from prototype.ipc_reference import IPCProblem
        mesh, topology = network_mesh(original, config, gap)
        np.testing.assert_array_equal(mesh.local_sn[mesh.bottom_nodes, 1], 0.)
        actual_bottom = np.flatnonzero(mesh.local_sn[:, 1] == 0.)
        np.testing.assert_array_equal(mesh.bottom_nodes, actual_bottom)
        excluded = topology.excluded_displaced_bottom_copies
        self.assertGreater(len(excluded), 0)
        self.assertTrue(np.all(mesh.local_sn[excluded, 1] > topology.bottom_support_tolerance_mm))
        self.assertFalse(np.any(np.isin(excluded, mesh.bottom_nodes)))
        self.assertTrue(np.all(np.isin(topology.original_vertex_ids[excluded], original.bottom_nodes)))
        summary = topology.to_dict()
        self.assertEqual(summary['original_bottom_copy_count'], len(mesh.bottom_nodes)+len(excluded))
        self.assertEqual(summary['excluded_displaced_bottom_copy_count'], len(excluded))
        self.assertEqual(summary['fixed_bottom_node_count'], len(mesh.bottom_nodes))
        # Exercise the unchanged production affine arithmetic without a solve.
        positions = np.vstack((mesh.reference_xy, [[0., 2.], [1., 2.]]))
        problem = SimpleNamespace(mesh=mesh, config=config, n=mesh.node_count)
        candidate = IPCProblem.affine_loading_candidate(problem, positions, 0., .01)
        np.testing.assert_array_equal(candidate[mesh.bottom_nodes], positions[mesh.bottom_nodes])
        self.assertTrue(np.all(candidate[excluded, 1] < positions[excluded, 1]))
        return mesh, topology

    def test_fan_fixes_actual_bottom_faces_and_releases_inset_vertex_copies(self):
        original, config = original_mesh(
            [[0., 0.], [1., 0.], [1., 1.], [0., 1.], [.5, .5]],
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        for angle in (0., 57.):
            with self.subTest(angle=angle):
                mesh, topology = self.assert_actual_bottom_support(
                    original, replace(config, inclination_deg=angle), .02)
                self.assertEqual(len(mesh.bottom_nodes), 2)
                self.assertEqual(len(topology.excluded_displaced_bottom_copies), 2)

    def test_real_coarse_mesh_bottom_predictor_remains_exact(self):
        from prototype.geometry import GeometryOptions, intact_mesh
        config = FEMConfig(mesh_size_mm=1.5, inclination_deg=0.)
        options = GeometryOptions(refinement_mm=.75, surface_spacing_mm=.2)
        original, _ = intact_mesh(config, options)
        mesh, topology = self.assert_actual_bottom_support(original, config, .006)
        self.assertGreater(len(mesh.triangles), 100)
        self.assertLess(len(mesh.bottom_nodes), len(topology.original_bottom_copies))

    def test_exterior_faces_remain_on_original_lines_with_vertex_gaps(self):
        mesh, topology = network_mesh(self.original, self.config, self.gap)
        for tri_id, ids in enumerate(topology.original_triangles):
            for k in range(3):
                a, b = ids[k], ids[(k+1) % 3]
                if tuple(sorted((a, b))) in topology.original_edge_to_segment:
                    continue
                edge = self.original.local_sn[b]-self.original.local_sn[a]
                normal = np.array([edge[1], -edge[0]])
                points = mesh.local_sn[[3*tri_id+k, 3*tri_id+(k+1) % 3]]
                np.testing.assert_allclose((points-self.original.local_sn[a])@normal, 0., atol=1e-13)
        self.assertGreater(np.max(np.linalg.norm(mesh.local_sn-self.original.local_sn[topology.original_vertex_ids], axis=1)), 0.)

    def test_gap_refinement_reduces_area_loss_without_claiming_elastic_equivalence(self):
        ratios = [network_mesh(self.original, self.config, gap)[1].bulk_area_ratio
                  for gap in (.04, .02, .01)]
        self.assertTrue(ratios[0] < ratios[1] < ratios[2] < 1.)

    def test_input_is_unchanged_and_clockwise_input_is_normalized(self):
        before_points = self.original.local_sn.copy()
        before_triangles = self.original.triangles.copy()
        network_mesh(self.original, self.config, self.gap)
        np.testing.assert_array_equal(before_points, self.original.local_sn)
        np.testing.assert_array_equal(before_triangles, self.original.triangles)
        clockwise = replace(self.original, triangles=before_triangles[:, [0, 2, 1]])
        mesh, topology = network_mesh(clockwise, self.config, self.gap)
        self.assertTrue(np.all(mesh.bulk_data[0] > 0.))
        self.assertEqual(len(topology.segments), 1)

    def test_invalid_gap_collapse_and_nonmanifold_are_rejected(self):
        for gap in (0., -.001, np.nan, 2.):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                network_mesh(self.original, self.config, gap)
        duplicate = replace(self.original, triangles=np.vstack((self.original.triangles,
                                                               self.original.triangles[0])))
        with self.assertRaises(ValueError):
            network_mesh(duplicate, self.config, self.gap)
        original, config = original_mesh(
            [[0., 0.], [1., 0.], [0., 1.], [1., -1.], [.5, .4]],
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]])
        with self.assertRaises(ValueError):
            network_mesh(original, config, self.gap)

    def test_single_triangle_has_no_internal_network(self):
        original, config = original_mesh([[0., 0.], [1., 0.], [0., 1.]], [[0, 1, 2]])
        mesh, topology = network_mesh(original, config, self.gap)
        self.assertEqual(topology.segments.shape, (0, 4))
        self.assertEqual(topology.reference_gaps.shape, (0, 2))
        self.assertEqual(topology.bulk_area_ratio, 1.)
        self.assertIsNone(topology.to_dict()['minimum_reference_normal_gap_mm'])
        q = self.response(mesh, topology)
        np.testing.assert_array_equal(q['gradient'], np.zeros(6))
        self.assertEqual(q['stored_energy_n_mm'], 0.)

    def test_failed_network_faces_reclose_with_real_ipc_friction_and_ccd(self):
        """Prescribed rigid closure/contact test, not a free equilibrium solve."""
        from prototype.ipc_reference import toolkit
        ipc = toolkit()
        gap = .006
        mesh, topology = network_mesh(self.original, self.config, gap)
        reference = mesh.reference_xy.copy()
        triangles = mesh.triangles
        edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        cmesh = ipc.CollisionMesh(reference, edges)
        cmesh.can_collide = ipc.make_static_obstacle_filter(len(reference))
        dhat = .0025
        barrier = ipc.BarrierPotential(dhat, 6.6, True)

        def contacts(positions):
            collision = ipc.NormalCollisions()
            collision.use_area_weighting = True
            collision.collision_set_type = ipc.NormalCollisions.IMPROVED_MAX_APPROX
            collision.build(cmesh, positions, dhat)
            return collision

        history = np.full((1, 2), self.law.final_opening_mm)
        ref_response = self.response(mesh, topology, old=history)
        normal = ref_response['normal_direction'][0, 0]
        tangent = ref_response['tangent_direction'][0, 0]
        right_triangle = int(topology.segments[0, 1]//3)
        moving = triangles[right_triangle]
        closed = reference.copy()
        closed[moving] -= .005*normal
        self.assertGreaterEqual(ipc.compute_collision_free_stepsize(cmesh, reference, closed), 1.-1e-12)
        self.assertFalse(ipc.has_intersections(cmesh, closed))
        collision = contacts(closed)
        self.assertGreater(len(collision.ev_collisions)+len(collision.vv_collisions), 0)
        self.assertGreater(float(barrier(collision, cmesh, closed)), 0.)
        barrier_gradient = barrier.gradient(collision, cmesh, closed).reshape(-1, 2)
        self.assertGreater(float(-barrier_gradient[moving].sum(axis=0)@normal), 0.)
        response = self.response(mesh, topology, closed, old=history)
        np.testing.assert_array_equal(response['traction'], 0.)
        np.testing.assert_array_equal(response['gradient'], 0.)
        np.testing.assert_array_equal(response['history'], history)
        self.assertEqual(response['stored_energy_n_mm'], 0.)
        self.assertEqual(response['fracture_dissipation_increment_n_mm'], 0.)
        # Each triangle undergoes only translation, so neither closure nor
        # contact detection relies on inverted or deformed bulk elements.
        deformation = np.eye(2)+np.einsum('eai,eaJ->eiJ',
            (closed-reference)[triangles], mesh.bulk_data[1])
        np.testing.assert_allclose(np.linalg.det(deformation), 1., atol=1e-14)
        crossed = closed.copy()
        crossed[moving] -= .004*normal
        fraction = float(ipc.compute_collision_free_stepsize(cmesh, closed, crossed))
        self.assertGreater(fraction, 0.)
        self.assertLess(fraction, 1.)
        self.assertFalse(ipc.has_intersections(cmesh, closed+fraction*(crossed-closed)))

        # Actual native Coulomb stencils also remain after complete cohesive
        # failure. This verifies their activation under prescribed sliding;
        # it is not a dynamic dissipation or a cutting-equilibrium certificate.
        tangential = ipc.TangentialCollisions()
        tangential.build(cmesh, closed, collision, barrier, .35)
        friction = ipc.FrictionPotential(1e-5)
        sliding = np.zeros_like(closed)
        sliding[moving] = .0005*tangent
        self.assertFalse(ipc.has_intersections(cmesh, closed+sliding))
        self.assertGreater(float(friction(tangential, cmesh, sliding)), 0.)
        friction_gradient = friction.gradient(tangential, cmesh, sliding).ravel()
        self.assertGreater(float(friction_gradient@sliding.ravel()), 0.)
        self.assertEqual(float(barrier(contacts(reference), cmesh, reference)), 0.)
        unloaded = self.response(mesh, topology, reference, old=history)
        np.testing.assert_array_equal(unloaded['damage'], 1.)
        np.testing.assert_array_equal(unloaded['traction'], 0.)


if __name__ == '__main__':
    unittest.main()
