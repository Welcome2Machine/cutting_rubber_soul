"""Mixed-interface host integration gates; no long cutting/loading calculation."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from prototype.cohesive_cut import CohesiveCutProblem, make_cohesive_problem
from prototype.ipc_reference import LoadStepFailure
from prototype.ipc_viscoelastic import ViscoelasticProblem


def configuration(angle=0.):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/fracture_demo.json').read_text())
    config.update(cohesive_kind='mixed_mode_energy', angle_deg=angle)
    config['cohesive'] = dict(normal_strength_mpa=.2, shear_strength_mpa=.3,
        normal_stiffness_mpa_per_mm=50., toughness_n_per_mm=.006,
        damage_enabled=True)
    config['mesh'].update(local_spacing_mm=.8, surface_spacing_mm=.2,
                          coarse_spacing_mm=1., seam_spacing_mm=.25,
                          nose_facets=16)
    return config


class MixedCutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = make_cohesive_problem(configuration())

    def test_rotated_reference_has_no_false_damage_and_keeps_full_collision_faces(self):
        p = make_cohesive_problem(configuration(57.))
        self.assertAlmostEqual(p.config.inclination_deg, 57.)
        self.assertEqual(p.cohesive_maximum.shape, (len(p.interface_segments), 2))
        self.assertAlmostEqual(p.interface_areas.sum(), 100.)
        np.testing.assert_allclose(p.interface_gaps, .006, atol=2e-14)
        q = p.cohesive_response(p.rest)
        np.testing.assert_allclose(q['gradient'], 0., atol=2e-10)
        self.assertEqual(q['fracture_dissipation_n_mm'], 0.)
        self.assertTrue(np.isin(p.mesh.seam_pairs, p.surface).all())
        self.assertFalse(p.ipc.has_intersections(p.cmesh, p.rest[p.surface]))
        # The existing complete CCD must still stop a path crossing both faces.
        direction = np.zeros_like(p.rest)
        angle = np.deg2rad(57.)
        direction[np.flatnonzero(p.mesh.side_of_node > 0)] = -.012*np.array([np.cos(angle), -np.sin(angle)])
        self.assertLess(p.safe_step(p.rest, direction), 1.)

    def test_legacy_angle_is_still_rejected(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'configs/fracture_demo.json').read_text())
        config['angle_deg'] = 33.
        with self.assertRaisesRegex(ValueError, 'Only 0-degree'):
            make_cohesive_problem(config)

    def test_host_history_helpers_use_gauss_area_and_never_promote_partial_segment(self):
        p = self.problem
        history = np.zeros_like(p.cohesive_maximum)
        history[-1, 0] = p.cohesive_law.final_opening_mm
        self.assertFalse(p.cohesive_broken_segments(history).any())
        self.assertAlmostEqual(p.cohesive_history_dissipation(history),
            .5*p.interface_areas[-1]*p.cohesive_law.toughness_n_per_mm)
        history[-1, 1] = p.cohesive_law.final_opening_mm
        self.assertEqual(np.flatnonzero(p.cohesive_broken_segments(history)).tolist(),
                         [len(p.interface_segments)-1])
        self.assertAlmostEqual(p.cohesive_history_dissipation(history),
            p.interface_areas[-1]*p.cohesive_law.toughness_n_per_mm)
        with self.assertRaisesRegex(ValueError, 'shape'):
            p.cohesive_history_response(np.zeros(len(p.bond_lengths)))

    def test_failed_attempt_does_not_commit_bulk_or_interface_history(self):
        p = make_cohesive_problem(configuration(33.))
        before_metric = p.material_metric.copy()
        before_damage = p.cohesive_maximum.copy()
        def reject(x, depth, deadline):
            y = x.copy()
            # Large outward jump along the reference material-s coordinate.
            angle = np.deg2rad(33.)
            y[p.mesh.seam_pairs[-3:, 1]] += .08*np.array([np.cos(angle), -np.sin(angle)])
            self.assertGreater(p.cohesive_response(y)['damage'].max(), .99)
            raise LoadStepFailure('Injected rejected mixed trial')
        with patch.object(p, 'solve_friction', side_effect=reject):
            with self.assertRaises(LoadStepFailure):
                p.step(p.rest.copy(), p.start, 0., float('inf'))
        np.testing.assert_array_equal(p.material_metric, before_metric)
        np.testing.assert_array_equal(p.cohesive_maximum, before_damage)

    def test_crack_contact_and_connected_length_require_two_failed_gauss_points(self):
        p = self.problem
        left_edge = tuple(sorted(map(int, p.interface_segments[-1, [0, 2]])))
        all_edges = p.surface[np.asarray(p.cmesh.edges)]
        edge_id = next(i for i, edge in enumerate(all_edges)
                       if tuple(sorted(map(int, edge))) == left_edge)
        knife_surface_id = int(np.flatnonzero(p.surface >= p.n)[0])
        partial_vertex = int(p.mesh.seam_pairs[-2, 0])
        vertex_surface_id = int(np.flatnonzero(p.surface == partial_vertex)[0])
        contacts = SimpleNamespace(
            ev_collisions=[SimpleNamespace(edge_id=edge_id, vertex_id=knife_surface_id)],
            vv_collisions=[SimpleNamespace(vertex0_id=knife_surface_id,
                                          vertex1_id=vertex_surface_id)])
        q = p.cohesive_response(p.rest)
        q['damage'][-1, 0] = 1.
        with patch.object(p, 'collisions', return_value=contacts), patch.object(p, 'barrier', return_value=0.):
            record = p._mixed_fracture_record(p.rest, q, 0.)
            self.assertEqual(record['separated_segments'], 0)
            self.assertEqual(record['surface_connected_crack_length_mm'], 0.)
            self.assertEqual(record['blade_fully_separated_face_contacts'], 0)
            q['damage'][-1, 1] = 1.
            record = p._mixed_fracture_record(p.rest, q, 0.)
        self.assertEqual(record['separated_segments'], 1)
        self.assertAlmostEqual(record['surface_connected_crack_length_mm'], p.interface_lengths[-1])
        self.assertTrue(record['contact_stencils'][0]['touches_fully_separated_face'])
        self.assertFalse(record['contact_stencils'][1]['touches_fully_separated_face'])

    def test_checkpoint_rejects_different_quadrature_and_reproduces_local_history(self):
        p = self.problem
        positions = p.rest.copy()
        positions[p.mesh.seam_pairs[-3:, 1], 0] += .015
        q = p.cohesive_response(positions)
        state = dict(p.checkpoint_data(), positions=positions,
                     cohesive_maximum=q['history'].copy())
        record = {'fracture': {'fracture_dissipation_n_mm': q['fracture_dissipation_n_mm']}}
        # This isolates the extra interface audit.  Full equilibrium remains
        # the separate superclass audit, tested by the solver regressions.
        with patch.object(ViscoelasticProblem, 'audit_state', return_value={'passed': True}):
            self.assertTrue(p.audit_state(state, record)['cohesive_history_passed'])
            broken = dict(state, cohesive_maximum=np.zeros(len(p.bond_lengths)))
            with self.assertRaisesRegex(ValueError, 'shape'):
                p.audit_state(broken, record)
            broken = dict(state, reference_interface_areas_mm2=p.interface_areas*1.01)
            with self.assertRaisesRegex(RuntimeError, 'topology/reference'):
                p.audit_state(broken, record)
            broken = dict(state, cohesive_kind='radial_mode_i')
            with self.assertRaisesRegex(RuntimeError, 'kind'):
                p.audit_state(broken, record)

    def test_original_surface_corner_is_not_promoted_to_crack_face_contact(self):
        p = self.problem
        body_corner = int(p.mesh.seam_pairs[-1, 0])
        corner_id = int(np.flatnonzero(p.surface == body_corner)[0])
        all_edges = p.surface[np.asarray(p.cmesh.edges)]
        blade_edge = next(i for i, edge in enumerate(all_edges) if np.all(edge >= p.n))
        blade_vertex = int(np.flatnonzero(p.surface >= p.n)[0])
        actual_interface_edge = tuple(sorted(map(int, p.interface_segments[-1, [0, 2]])))
        interface_edge = next(i for i, edge in enumerate(all_edges)
                              if tuple(sorted(map(int, edge))) == actual_interface_edge)
        contacts = SimpleNamespace(ev_collisions=[
            SimpleNamespace(edge_id=blade_edge, vertex_id=corner_id),
            SimpleNamespace(edge_id=interface_edge, vertex_id=blade_vertex)],
            vv_collisions=[SimpleNamespace(vertex0_id=blade_vertex, vertex1_id=corner_id)])
        q = p.cohesive_response(p.rest)
        q['damage'][-1, :] = 1.
        with patch.object(p, 'collisions', return_value=contacts), patch.object(p, 'barrier', return_value=0.):
            record = p._mixed_fracture_record(p.rest, q, 0.)
        node_ev, edge_ev, node_vv = record['contact_stencils']
        for stencil in (node_ev, node_vv):
            self.assertTrue(stencil['attribution_ambiguous'])
            self.assertIn('original noncohesive', stencil['ambiguity_reason'])
            self.assertFalse(stencil['touches_fully_separated_face'])
            self.assertFalse(stencil['touches_surface_connected_crack'])
        self.assertFalse(edge_ev['attribution_ambiguous'])
        self.assertTrue(edge_ev['touches_fully_separated_face'])
        self.assertTrue(edge_ev['touches_surface_connected_crack'])

    def test_network_uses_all_internal_edges_and_real_ipc_faces_at_nonzero_angle(self):
        config = configuration(57.)
        config['fracture_topology'] = 'edge_network'
        config['mesh'].update(local_spacing_mm=1., surface_spacing_mm=.3,
                              coarse_spacing_mm=1.5)
        p = make_cohesive_problem(config)
        topology = p.network_topology
        self.assertEqual(p.fracture_topology, 'edge_network')
        self.assertGreater(len(p.interface_segments), 2)
        self.assertEqual(len(p.interface_segments), len(topology.original_edge_ids))
        self.assertEqual(len(p.mesh.triangles)*3, p.n)
        self.assertEqual(len(p.mesh.seam_pairs), 0)
        self.assertGreater(p.fracture_topology_metadata['bulk_area_ratio'], 0.)
        self.assertLess(p.fracture_topology_metadata['bulk_area_ratio'], 1.)
        self.assertTrue(np.isin(p.interface_segments, p.surface).all())
        self.assertFalse(p.ipc.has_intersections(p.cmesh, p.rest[p.surface]))
        self.assertLess(p.reference_contact_check['body_contact_force_norm_n'], 1e-4)
        candidate = p.affine_loading_candidate(p.rest.copy(), p.start, 0.)
        fixed = (2*p.mesh.bottom_nodes[:, None]+np.arange(2)).ravel()
        np.testing.assert_array_equal(candidate.ravel()[fixed], p.rest.ravel()[fixed])
        q = p.cohesive_response(p.rest)
        self.assertEqual(q['fracture_dissipation_n_mm'], 0.)
        np.testing.assert_allclose(q['gradient'], 0., atol=2e-10)
        state = p.checkpoint_data()
        np.testing.assert_array_equal(state['reference_interface_slips_mm'], topology.reference_slips)
        np.testing.assert_array_equal(state['original_interface_edges'], topology.original_edges)
        np.testing.assert_array_equal(state['interface_exterior_seed'], topology.exterior_seed_segments)

    def test_network_connectivity_does_not_promote_an_isolated_failed_cluster(self):
        # Edges 0--2 form one surface-rooted chain; edge 3 is an internal island.
        holder = SimpleNamespace(network_topology=SimpleNamespace(
            original_edges=np.array([[0, 1], [1, 2], [2, 3], [4, 5]]),
            exterior_seed_segments=np.array([True, False, False, False])))
        connected = CohesiveCutProblem._network_connected_segments(
            holder, np.array([True, True, False, True]))
        self.assertEqual(connected, [0, 1])
        connected = CohesiveCutProblem._network_connected_segments(
            holder, np.ones(4, dtype=bool))
        self.assertEqual(connected, [0, 1, 2])

    def test_network_numerical_floor_changes_only_paired_direction_metric(self):
        config = configuration()
        config['fracture_topology'] = 'edge_network'
        config['network_metric_floor_n_per_mm'] = 1.
        config['mesh'].update(local_spacing_mm=1., surface_spacing_mm=.3,
                              coarse_spacing_mm=1.5)
        p = make_cohesive_problem(config)
        p.material_dt = .025
        x = p.rest.copy()
        first = p.evaluate(x, p.lag(x), tangent=True, projected=False, with_metric=True)
        p.network_metric_floor_n_per_mm = 2.
        second = p.evaluate(x, p.lag(x), tangent=True, projected=False, with_metric=True)
        ordinary = p.evaluate(x, p.lag(x), tangent=True, projected=False)
        p.network_metric_floor_n_per_mm = 1.
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[3], second[3])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual((first[2]-second[2]).nnz, 0)
        np.testing.assert_array_equal(ordinary[1], first[1])
        self.assertEqual((ordinary[2]-first[2]).nnz, 0)
        difference = second[4]-first[4]
        expected = np.r_[np.ones(2*p.n), np.zeros(p.total-2*p.n)]
        np.testing.assert_allclose(difference.diagonal(), expected, rtol=0., atol=1e-12)
        off_diagonal = difference.copy()
        off_diagonal.setdiag(0.)
        off_diagonal.eliminate_zeros()
        self.assertEqual(off_diagonal.nnz, 0)
        self.assertEqual(p.checkpoint_data()['network_metric_floor_n_per_mm'], 1.)
        self.assertEqual(p.fracture_topology_metadata['numerical_metric_floor_n_per_mm'], 1.)

    def test_network_stationary_base_preserves_body_and_physical_histories(self):
        config = configuration(57.)
        config['fracture_topology'] = 'edge_network'
        p = make_cohesive_problem(config)
        p.material_dt = -p.start/p.loading_rate
        bulk = p.material_metric.copy()
        cohesive = p.cohesive_maximum.copy()
        candidate = p.affine_loading_candidate(p.rest.copy(), p.start, 0.)
        np.testing.assert_array_equal(candidate[:p.n], p.rest[:p.n])
        np.testing.assert_array_equal(candidate[p.n:], p.prescribed_blade_positions(0.))
        np.testing.assert_array_equal(p.material_metric, bulk)
        np.testing.assert_array_equal(p.cohesive_maximum, cohesive)
        self.assertEqual(p.network_predictor_trace['selected_base'], 'body_stationary')
        self.assertFalse(p.network_predictor_trace['physical_history_substeps_added'])
        self.assertEqual(p.checkpoint_data()['network_predictor_policy'], 'body_stationary_if_safe')

    def test_network_stationary_base_falls_back_on_unsafe_path_and_honors_affine_retry(self):
        config = configuration()
        config['fracture_topology'] = 'edge_network'
        p = make_cohesive_problem(config)
        expected = ViscoelasticProblem.affine_loading_candidate(p, p.rest.copy(), p.start, 0.)
        with patch.object(p, 'safe_step', return_value=.5):
            candidate = p.affine_loading_candidate(p.rest.copy(), p.start, 0.)
        np.testing.assert_array_equal(candidate, expected)
        self.assertIn('fallback', p.network_predictor_trace['reason'])
        p.predictor = 'affine'
        with patch.object(p, 'safe_step', side_effect=AssertionError('Affine retry must use baseline')):
            candidate = p.affine_loading_candidate(p.rest.copy(), p.start, 0.)
        np.testing.assert_array_equal(candidate, expected)
        self.assertEqual(p.network_predictor_trace['reason'], 'conservative_affine_retry')
        p.predictor = 'secant'
        p.network_predictor = 'baseline'
        candidate = p.affine_loading_candidate(p.rest.copy(), p.start, 0.)
        np.testing.assert_array_equal(candidate, expected)
        self.assertEqual(p.network_predictor_trace['reason'], 'explicit_baseline_policy')

    def test_only_network_disables_early_plateau_cutoff_and_keeps_hard_budget(self):
        config = configuration()
        config['fracture_topology'] = 'edge_network'
        p = make_cohesive_problem(config)
        self.assertEqual(p.stagnation_window, 0)
        self.assertEqual(p.max_iterations, 500)
        self.assertEqual(p.fracture_topology_metadata['numerical_stagnation_window'], 0)
        self.assertEqual(p.fracture_topology_metadata['numerical_max_iterations'], 500)
        self.assertEqual(p.checkpoint_data()['network_stagnation_window'], 0)
        self.assertEqual(p.checkpoint_data()['network_max_iterations'], 500)
        self.assertEqual(self.problem.stagnation_window, 40)
        root = Path(__file__).resolve().parents[1]
        legacy = make_cohesive_problem(json.loads((root/'configs/fracture_demo.json').read_text()))
        self.assertEqual(legacy.stagnation_window, 40)
        self.assertEqual(legacy.max_iterations, 500)


if __name__ == '__main__':
    unittest.main()
