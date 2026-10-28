"""Mixed-interface host integration gates; no long cutting/loading calculation."""
import json
import copy
import hashlib
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
        before_energy = p.cohesive_energy_ledger.copy()
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
        self.assertEqual(p.cohesive_energy_ledger, before_energy)

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
                     cohesive_maximum=q['history'].copy(), **p._mixed_energy_record(q))
        record = {'fracture': dict(fracture_dissipation_n_mm=q['fracture_dissipation_n_mm'],
                                   **p._mixed_energy_record(q))}
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

    def test_compression_energy_is_recoverable_separate_and_counted_once_in_host(self):
        config = configuration()
        config['cohesive']['compression_stiffness_mpa_per_mm'] = 120.
        p = make_cohesive_problem(config)
        p.material_dt = .1
        x = p.rest.copy()
        x[p.mesh.seam_pairs[:, 1], 0] -= .001
        q = p.cohesive_response(x)
        self.assertGreater(q['compression_stored_energy_n_mm'], 0.)
        self.assertLess(q['damageable_stored_energy_n_mm'], 1e-20)
        self.assertEqual(q['fracture_dissipation_n_mm'], 0.)
        self.assertEqual(float(q['damage'].max()), 0.)
        lag = p.lag(x)
        base = ViscoelasticProblem.evaluate(p, x, lag, tangent=False)
        total = p.evaluate(x, lag, tangent=False)
        self.assertAlmostEqual(total[0]-base[0], q['stored_energy_n_mm'], places=12)
        np.testing.assert_allclose(total[1]-base[1], q['gradient'], atol=1e-12)
        record = p.material_acceptance(x, 0., {})
        self.assertEqual(record['fracture']['compression_stiffness_mpa_per_mm'], 120.)
        self.assertEqual(record['fracture']['compression_stored_energy_n_mm'],
                         q['compression_stored_energy_n_mm'])
        self.assertEqual(record['fracture']['fracture_dissipation_increment_n_mm'], 0.)
        self.assertGreater(record['fracture']['compression_internal_force_norm_n'], 0.)
        # Full damage removes the damageable law, not recoverable closure.
        p.previous_cohesive_maximum[:] = p.cohesive_law.final_opening_mm
        failed = p.cohesive_response(x)
        self.assertAlmostEqual(failed['compression_stored_energy_n_mm'],
                               q['compression_stored_energy_n_mm'])
        self.assertAlmostEqual(failed['fracture_dissipation_n_mm'],
                               p.interface_areas.sum()*p.cohesive_law.toughness_n_per_mm)
        self.assertEqual(failed['fracture_dissipation_increment_n_mm'], 0.)

    def test_mixed_checkpoint_tracks_accepted_energy_and_rejects_changed_compression(self):
        config = configuration()
        config['cohesive']['compression_stiffness_mpa_per_mm'] = 120.
        p = make_cohesive_problem(config)
        x = p.rest.copy()
        x[p.mesh.seam_pairs[:, 1], 0] -= .001
        # Isolate host commit bookkeeping; this is not an equilibrium certificate.
        with patch.object(ViscoelasticProblem, 'step', return_value=(x, {})):
            p.step(p.rest.copy(), p.start, 0., float('inf'))
        q = p.cohesive_response(x)
        state = dict(p.checkpoint_data(), positions=x)
        record = {'fracture': dict(fracture_dissipation_n_mm=q['fracture_dissipation_n_mm'],
                                   **p._mixed_energy_record(q))}
        self.assertGreater(state['compression_stored_energy_n_mm'], 0.)
        self.assertEqual(json.loads(state['cohesive_law_parameters_json'])
                         ['compression_stiffness_mpa_per_mm'], 120.)
        with patch.object(ViscoelasticProblem, 'audit_state', return_value={'passed': True}):
            self.assertTrue(p.audit_state(state, record)['cohesive_history_passed'])
            broken = dict(state, compression_stored_energy_n_mm=0.)
            with self.assertRaises(AssertionError):
                p.audit_state(broken, record)
            changed = configuration()
            changed['cohesive']['compression_stiffness_mpa_per_mm'] = 121.
            other = make_cohesive_problem(changed)
            with self.assertRaisesRegex(RuntimeError, 'cohesive_law_parameters_json'):
                other.audit_state(state, record)

    def test_fixed_preparation_clearance_preserves_zero_depth_geometry_and_physical_dt(self):
        for angle in (0., 57.):
            with self.subTest(angle=angle):
                default = make_cohesive_problem(configuration(angle))
                self.assertAlmostEqual(default.start, -2*default.dhat/np.cos(np.deg2rad(angle)))
                config = configuration(angle)
                config.update(fracture_topology='edge_network', preparation_clearance_mm=.02)
                first = make_cohesive_problem(config)
                config.update(reference_gap_mm=.0006, dhat_mm=.00025)
                second = make_cohesive_problem(config)
                for p in (first, second):
                    self.assertEqual(p.start, -.02)
                    self.assertEqual(p.preparation_metadata['mode'], 'explicit_vertical_clearance')
                    self.assertEqual(p.checkpoint_data()['initial_depth_mm'], -.02)
                    np.testing.assert_allclose(p.prescribed_blade_positions(0.),
                                               default.prescribed_blade_positions(0.), atol=2e-14, rtol=0.)
                    np.testing.assert_array_equal(p.prescribed_blade_positions(p.start), p.rest[p.n:])
                    self.assertFalse(p.ipc.has_intersections(p.cmesh, p.rest[p.surface]))
                    self.assertLessEqual(p.reference_contact_check['body_contact_force_norm_n'], 1e-4)
                    before = p.material_metric.copy()
                    with patch.object(p, 'solve_friction', side_effect=LoadStepFailure('dt probe')):
                        with self.assertRaises(LoadStepFailure):
                            p.step(p.rest.copy(), p.start, .005, float('inf'))
                    self.assertAlmostEqual(p.material_dt, (.005+.02)/p.loading_rate)
                    np.testing.assert_array_equal(p.material_metric, before)

    def test_invalid_or_prestressed_preparation_is_rejected_and_audit_identifies_start(self):
        for value in (0., -.02, float('nan'), float('inf'), True):
            with self.subTest(clearance=value):
                config = configuration()
                config['preparation_clearance_mm'] = value
                with self.assertRaisesRegex(ValueError, 'preparation_clearance_mm'):
                    make_cohesive_problem(config)
        config = configuration()
        config['preparation_clearance_mm'] = .000001
        with self.assertRaisesRegex(ValueError, 'Reference contact force'):
            make_cohesive_problem(config)
        p = self.problem
        state = dict(p.checkpoint_data(), initial_depth_mm=-.02)
        with self.assertRaisesRegex(RuntimeError, 'initial_depth_mm'):
            p.audit_state(state, {})

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

    def test_explicit_ccd_only_tightens_network_and_checkpoint_identifies_policy(self):
        config = configuration()
        config.update(fracture_topology='edge_network', reference_gap_mm=6e-7,
                      dhat_mm=2.5e-7, ccd_tolerance_mm=2.5e-9,
                      preparation_clearance_mm=.02)
        p = make_cohesive_problem(config)
        ccd = p.network_narrow_phase_ccd
        self.assertEqual(ccd.tolerance, 2.5e-9)
        self.assertEqual(ccd.max_iterations, 10000000)
        self.assertEqual(ccd.conservative_rescaling, .8)
        self.assertEqual(p.safe_step(p.rest, np.zeros_like(p.rest)), 1.)
        self.assertFalse(p.ipc.has_intersections(p.cmesh, p.rest[p.surface]))
        direction = np.zeros_like(p.rest)
        direction[p.n:] = p.prescribed_blade_positions(.01)-p.rest[p.n:]
        fraction = p.safe_step(p.rest, direction)
        self.assertGreater(fraction, 0.)
        self.assertLess(fraction, 1.)
        self.assertFalse(p.ipc.has_intersections(p.cmesh, (p.rest+fraction*direction)[p.surface]))
        for halvings in range(1, 21):
            prefix = p.rest+(fraction*2.**(-halvings))*direction
            self.assertFalse(p.ipc.has_intersections(p.cmesh, prefix[p.surface]))
            np.testing.assert_array_equal(prefix[:p.n], p.rest[:p.n])
        saved = p.checkpoint_data()['network_ccd_policy_json']
        self.assertEqual(json.loads(saved)['tolerance_mm'], 2.5e-9)
        changed = dict(p.checkpoint_data(), network_ccd_policy_json='{}')
        with self.assertRaisesRegex(RuntimeError, 'network_ccd_policy_json'):
            p.audit_state(changed, {})
        # Omission delegates byte-for-byte to the original safe_step behavior.
        with patch.object(ViscoelasticProblem, 'safe_step', return_value=.375) as original:
            self.assertEqual(self.problem.safe_step(self.problem.rest, np.zeros_like(self.problem.rest)), .375)
            original.assert_called_once()
        for value in (0., -1., True, float('nan'), 3e-8, 2e-6):
            with self.subTest(ccd_tolerance=value):
                invalid = dict(config, ccd_tolerance_mm=value)
                with self.assertRaisesRegex(ValueError, 'ccd_tolerance_mm'):
                    make_cohesive_problem(invalid)
        config['fracture_topology'] = 'prescribed_seam'
        with self.assertRaisesRegex(ValueError, 'only by the edge network'):
            make_cohesive_problem(config)

    def test_microgap_safe_blend_keeps_exact_loading_and_histories(self):
        for angle in (0., 57.):
            with self.subTest(angle=angle):
                config = configuration(angle)
                config.update(fracture_topology='edge_network', reference_gap_mm=6e-7,
                              dhat_mm=2.5e-7, ccd_tolerance_mm=2.5e-9,
                              preparation_clearance_mm=.02, network_predictor='safe_affine_blend')
                config['cohesive'].update(normal_stiffness_mpa_per_mm=5000.,
                                          compression_stiffness_mpa_per_mm=5000.)
                p = make_cohesive_problem(config)
                p.material_dt = -p.start/p.loading_rate
                material_before = p.material_metric.copy()
                cohesive_before = p.cohesive_maximum.copy()
                energy_before = p.cohesive_energy_ledger.copy()
                candidate = p.affine_loading_candidate(p.rest.copy(), p.start, 0.)
                trace = p.network_predictor_trace
                self.assertEqual(trace['selected_base'], 'safe_affine_blend')
                self.assertGreater(trace['body_affine_weight'], 0.)
                self.assertLess(trace['body_affine_weight'], 1.)
                self.assertEqual(len(trace['blend_candidates']), 8)
                baseline = next(q for q in trace['blend_candidates'] if q['body_affine_weight'] == 1.)
                self.assertLess(trace['blend_fresh_residual_n'], baseline['fresh_residual_n'])
                self.assertGreaterEqual(p.safe_step(p.rest, candidate-p.rest), 1.-1e-10)
                self.assertFalse(p.ipc.has_intersections(p.cmesh, candidate[p.surface]))
                np.testing.assert_array_equal(candidate[p.n:], p.prescribed_blade_positions(0.))
                fixed = np.setdiff1d(np.arange(2*p.n), p.free)
                np.testing.assert_array_equal(candidate.ravel()[fixed], p.rest.ravel()[fixed])
                np.testing.assert_array_equal(p.material_metric, material_before)
                np.testing.assert_array_equal(p.cohesive_maximum, cohesive_before)
                self.assertEqual(p.cohesive_energy_ledger, energy_before)
                self.assertFalse(trace['physical_history_substeps_added'])
                # A conservative affine retry still bypasses the blend search.
                p.predictor = 'affine'
                expected = ViscoelasticProblem.affine_loading_candidate(p, p.rest.copy(), p.start, 0.)
                with patch.object(p, 'safe_step', side_effect=AssertionError('Retry must use original affine')):
                    actual = p.affine_loading_candidate(p.rest.copy(), p.start, 0.)
                np.testing.assert_array_equal(actual, expected)

    def test_model_backtracking_is_explicit_network_only_and_checkpoint_bound(self):
        config = configuration()
        config.update(fracture_topology='edge_network', ptc_model_backtracks=20)
        p = make_cohesive_problem(config)
        self.assertEqual(p.ptc_model_backtracks, 20)
        self.assertEqual(p.fracture_topology_metadata['numerical_ptc_model_backtracks'], 20)
        self.assertEqual(p.checkpoint_data()['network_ptc_model_backtracks'], 20)
        state = dict(p.checkpoint_data(), network_ptc_model_backtracks=12)
        with self.assertRaisesRegex(RuntimeError, 'network_ptc_model_backtracks'):
            p.audit_state(state, {})
        self.assertEqual(self.problem.ptc_model_backtracks, 0)
        for value in (-1, True, 1.5, 31):
            with self.subTest(backtracks=value), self.assertRaisesRegex(ValueError, 'ptc_model_backtracks'):
                make_cohesive_problem(dict(config, ptc_model_backtracks=value))
        with self.assertRaisesRegex(ValueError, 'ptc_model_backtracks'):
            make_cohesive_problem(dict(config, fracture_topology='prescribed_seam'))

    def test_fresh_event_and_energy_audit_rejects_record_tampering(self):
        p = make_cohesive_problem(configuration())
        p.material_dt = .1
        fresh = p.material_acceptance(p.rest.copy(), 0., {'friction_work_n_mm': 0.})
        saved = copy.deepcopy(fresh)
        checked = p._audit_extra_acceptance(fresh, saved)
        self.assertTrue(checked['cohesive_events_passed'])
        digest = hashlib.sha256(json.dumps(saved, sort_keys=True, separators=(',', ':'),
            ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
        self.assertEqual(checked['accepted_record_sha256'], digest)
        self.assertIn('fracture', checked)
        self.assertIn('material', checked)
        for group, key, value in (
                ('fracture', 'max_damage', 1.),
                ('fracture', 'separated_segments', 1),
                ('fracture', 'separated_bonds', 1),
                ('fracture', 'surface_connected_crack_length_mm', .1),
                ('fracture', 'blade_surface_connected_crack_contacts', 1),
                ('fracture', 'ipc_barrier_stored_energy_n_mm', .1),
                ('material', 'stored_energy_n_mm', .1),
                ('material', 'viscous_dissipation_increment_n_mm', .1)):
            broken = copy.deepcopy(saved)
            broken[group][key] = value
            with self.subTest(field=key), self.assertRaisesRegex(RuntimeError, key):
                p._audit_extra_acceptance(fresh, broken)
        broken = copy.deepcopy(saved)
        broken['friction_work_n_mm'] = .1
        with self.assertRaisesRegex(RuntimeError, 'friction_work_n_mm'):
            p._audit_extra_acceptance(fresh, broken)

    def test_fresh_stencil_audit_is_order_independent_but_checks_attribution(self):
        p = make_cohesive_problem(configuration())
        p.material_dt = .1
        fresh = p.material_acceptance(p.rest.copy(), 0., {'friction_work_n_mm': 0.})
        # Isolate native enumeration order versus actual event attribution.
        stencil = dict(kind='EV', global_edge_nodes=[0, 1], global_vertex=p.n,
                       segment_ids=[0], gauss_damage=[[1., 1.]],
                       attribution_ambiguous=False, ambiguity_reason=None,
                       touches_fully_separated_face=True,
                       touches_surface_connected_crack=True)
        other = dict(stencil, global_edge_nodes=[2, 3], segment_ids=[1])
        vv = dict(stencil, kind='VV', global_nodes=[p.n, 4])
        del vv['global_edge_nodes'], vv['global_vertex']
        fresh['fracture']['contact_stencils'] = [stencil, other, vv]
        saved = copy.deepcopy(fresh)
        saved['fracture']['contact_stencils'].reverse()
        for contact in saved['fracture']['contact_stencils']:
            contact['global_nodes' if contact['kind'] == 'VV' else 'global_edge_nodes'].reverse()
        self.assertTrue(p._audit_extra_acceptance(fresh, saved)['cohesive_events_passed'])
        before = copy.deepcopy(saved)
        checked = p._audit_extra_acceptance(fresh, saved)
        self.assertEqual(saved, before)  # Canonicalization must not change the record hash.
        self.assertEqual(checked['fracture']['contact_stencils'][-1]['global_nodes'], [4, p.n])
        for index, field, value in ((0, 'global_nodes', [p.n, 5]),
                                    (1, 'global_vertex', p.n+1),
                                    (1, 'global_edge_nodes', [2, 5])):
            broken = copy.deepcopy(saved)
            broken['fracture']['contact_stencils'][index][field] = value
            with self.subTest(identity=field), self.assertRaises(RuntimeError):
                p._audit_extra_acceptance(fresh, broken)
        saved['fracture']['contact_stencils'][0]['touches_surface_connected_crack'] = False
        with self.assertRaisesRegex(RuntimeError, 'touches_surface_connected_crack'):
            p._audit_extra_acceptance(fresh, saved)

    def test_native_reference_audit_exports_fresh_values_and_record_integrity(self):
        # A stress-free reference balance exercises the full audit dispatch,
        # without taking a load increment or claiming a cutting solution.
        p = make_cohesive_problem(configuration())
        p.material_dt = .1
        record = p.acceptance(p.rest.copy(), p.start, 0, 0)
        state = dict(p.checkpoint_data(), positions=p.rest.copy(),
                     previous_positions=p.rest.copy(), reference_xy=p.rest.copy(),
                     triangles=p.mesh.triangles, depth_mm=p.start)
        checked = p.audit_state(state, record)
        self.assertTrue(checked['passed'])
        self.assertTrue(checked['cohesive_events_passed'])
        self.assertTrue(checked['cohesive_history_passed'])
        self.assertEqual(checked['fracture']['max_damage'], 0.)
        self.assertEqual(checked['friction_work_n_mm'], 0.)
        self.assertEqual(len(checked['accepted_record_sha256']), 64)
        broken = copy.deepcopy(record)
        broken['fracture']['max_damage'] = 1.
        with self.assertRaisesRegex(RuntimeError, 'max_damage'):
            p.audit_state(state, broken)

    def test_native_contact_stencil_endpoints_are_undirected_but_roles_are_not(self):
        p = make_cohesive_problem(configuration())
        p.material_dt = .1
        x = p.rest.copy()
        x[p.n:] = p.prescribed_blade_positions(0.)
        self.assertFalse(p.ipc.has_intersections(p.cmesh, x[p.surface]))
        # Native contact enumeration at first touch, without solving or claiming
        # this prescribed diagnostic geometry is a mechanical equilibrium.
        fresh = p.material_acceptance(x, 0., {'friction_work_n_mm': 0.})
        self.assertGreater(len(fresh['fracture']['contact_stencils']), 0)
        saved = copy.deepcopy(fresh)
        saved['fracture']['contact_stencils'].reverse()
        for contact in saved['fracture']['contact_stencils']:
            contact['global_edge_nodes' if contact['kind'] == 'EV' else 'global_nodes'].reverse()
        self.assertTrue(p._audit_extra_acceptance(fresh, saved)['cohesive_events_passed'])
        for index, contact in enumerate(saved['fracture']['contact_stencils']):
            if contact['kind'] == 'EV':
                broken = copy.deepcopy(saved)
                changed = broken['fracture']['contact_stencils'][index]
                changed['global_edge_nodes'][0], changed['global_vertex'] = (
                    changed['global_vertex'], changed['global_edge_nodes'][0])
                with self.assertRaises(RuntimeError):
                    p._audit_extra_acceptance(fresh, broken)
                break
        else:
            self.fail('Expected a native EV stencil to exercise vertex/edge role preservation')

    def test_residual_guard_policy_is_explicit_network_only_and_checkpoint_bound(self):
        config = configuration()
        config.update(fracture_topology='edge_network', ptc_model_backtracks=12, ptc_residual_guard=True)
        p = make_cohesive_problem(config)
        self.assertTrue(p.ptc_residual_guard)
        self.assertTrue(p.fracture_topology_metadata['numerical_ptc_residual_guard'])
        p.material_dt = .1
        record = p.material_acceptance(p.rest.copy(), 0., {'friction_work_n_mm': 0.})
        self.assertTrue(record['network_numerics']['ptc_residual_guard'])
        q = p.cohesive_response(p.rest)
        state = dict(p.checkpoint_data(), positions=p.rest.copy())
        self.assertTrue(state['network_ptc_residual_guard'])
        with patch.object(ViscoelasticProblem, 'audit_state', return_value={'passed': True}):
            self.assertTrue(p.audit_state(state, record)['cohesive_history_passed'])
            with self.assertRaisesRegex(RuntimeError, 'network_ptc_residual_guard'):
                p.audit_state(dict(state, network_ptc_residual_guard=False), record)
        for changes in ({'ptc_residual_guard': 1}, {'ptc_residual_guard': 'true'},
                        {'ptc_model_backtracks': 0}, {'fracture_topology': 'prescribed_seam'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_cohesive_problem(dict(config, **changes))
        default = make_cohesive_problem(configuration())
        self.assertFalse(default.ptc_residual_guard)

    def test_backtrack_aware_shift_is_explicit_and_checkpoint_bound(self):
        config = configuration()
        config.update(fracture_topology='edge_network', ptc_model_backtracks=12,
                      ptc_residual_guard=False, ptc_shift_policy='backtrack_aware')
        p = make_cohesive_problem(config)
        p.material_dt = .1
        record = p.material_acceptance(p.rest.copy(), 0., {'friction_work_n_mm': 0.})
        self.assertEqual(record['network_numerics']['ptc_shift_policy'], 'backtrack_aware')
        self.assertFalse(record['network_numerics']['ptc_residual_guard'])
        state = dict(p.checkpoint_data(), positions=p.rest.copy())
        self.assertEqual(state['network_ptc_shift_policy'], 'backtrack_aware')
        with patch.object(ViscoelasticProblem, 'audit_state', return_value={'passed': True}):
            self.assertTrue(p.audit_state(state, record)['cohesive_history_passed'])
            with self.assertRaisesRegex(RuntimeError, 'network_ptc_shift_policy'):
                p.audit_state(dict(state, network_ptc_shift_policy='residual_ratio'), record)
        for changes in ({'ptc_shift_policy': 'unknown'}, {'ptc_model_backtracks': 0},
                        {'fracture_topology': 'prescribed_seam'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_cohesive_problem(dict(config, **changes))
        self.assertEqual(make_cohesive_problem(configuration()).ptc_shift_policy, 'residual_ratio')


if __name__ == '__main__':
    unittest.main()
