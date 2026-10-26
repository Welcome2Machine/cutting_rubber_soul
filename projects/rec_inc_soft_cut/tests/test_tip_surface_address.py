import unittest

import numpy as np

from src.analysis.tip_surface_address import (TipSurfaceAddressEstimator, analyze_tip_surface_series,
                                              extend_initial_surface_reference)


def synthetic_estimator():
    s = np.arange(-60., 61., 6.)
    boundary = np.column_stack([400 + s, np.full(len(s), 200.)])
    station, layer = np.meshgrid(np.arange(len(s)), np.arange(9), indexing='ij')
    station, layer = station.ravel(), layer.ravel()
    seeds = boundary[station] + np.column_stack([np.zeros(len(station)), 1.5 + 6 * layer])
    surface = dict(boundary_xy=boundary, surface_s_px=s, tangent=np.array([1., 0.]),
                   inward_normal=np.array([0., 1.]), contact_s_px=0.)
    return TipSurfaceAddressEstimator(seeds, np.arange(len(seeds)) + 1000, layer, station, surface, 5.)


class TipAddressTests(unittest.TestCase):
    def test_inverse_large_affine_deformation_marks_tip_and_original_address_separately(self):
        estimator = synthetic_estimator()
        a = np.array([[1.6, .35], [.2, .8]])
        offset = np.array([-200., -30.])
        current = estimator.reference @ a.T + offset
        expected = np.array([412., 200.])
        tip = a @ expected + offset
        result = estimator.estimate_frame(current, np.ones(len(current), bool), tip,
                                          branch_labels=np.ones(len(current), int), blade_mask=np.zeros((768, 1024), bool))
        self.assertEqual(result['label'], 'direct_supported')
        self.assertAlmostEqual(result['s_px'], 12., places=6)
        self.assertAlmostEqual(result['s_um'], 60., places=6)
        np.testing.assert_array_equal(result['current_marker_xy'], tip)
        np.testing.assert_allclose(result['reference_marker_xy'], expected, atol=1e-6)
        self.assertTrue(set(result['support_ids']).issubset(set(estimator.ids)))

    def test_missing_surface_rows_can_use_bracketing_subsurface_blocks_conditionally(self):
        estimator = synthetic_estimator()
        valid = estimator.layers > 0
        current = estimator.reference + [3., 7.]
        result = estimator.estimate_frame(current, valid, [409., 207.],
                                          branch_labels=np.ones(len(current), int))
        self.assertEqual(result['state'], 1)
        self.assertAlmostEqual(result['s_px'], 6., places=6)
        self.assertGreater(result['reference_extrapolation_px'], 0.)
        self.assertLessEqual(result['reference_extrapolation_px'], 12.)

    def test_new_internal_face_is_not_projected_to_claim_original_surface_contact(self):
        estimator = synthetic_estimator()
        result = estimator.estimate_frame(estimator.reference, np.ones(len(estimator.ids), bool), [400., 225.])
        self.assertEqual(result['state'], 0)
        self.assertTrue(np.isnan(result['s_px']))
        self.assertIn('new_crack_face', result['reason'])
        np.testing.assert_allclose(result['raw_inverse_material_xy'], [400., 225.], atol=1e-6)
        self.assertAlmostEqual(result['raw_surface_normal_offset_px'], 25.)
        np.testing.assert_array_equal(result['current_marker_xy'], [400., 225.])

    def test_far_nearest_point_and_blocked_points_never_become_contact(self):
        estimator = synthetic_estimator()
        supported = np.ones(len(estimator.ids), bool)
        for tip, blocked in (([600., 200.], None), ([400., 200.], supported)):
            result = estimator.estimate_frame(estimator.reference, supported, tip, blocked=blocked)
            self.assertEqual(result['state'], 0)
            self.assertTrue(np.isnan(result['reference_marker_xy']).all())
            np.testing.assert_array_equal(result['current_marker_xy'], tip)

    def test_unpartitioned_patch_is_conditional_even_with_surface_points(self):
        estimator = synthetic_estimator()
        result = estimator.estimate_frame(estimator.reference, np.ones(len(estimator.ids), bool), [400., 200.])
        self.assertEqual(result['state'], 1)

    def test_material_velocity_and_convective_address_rate_differ_under_stretch_and_gaps(self):
        estimator = synthetic_estimator()
        count = 5
        positions = np.tile(estimator.reference, (count, 1, 1))
        positions[:, :, 0] = 400 + 2 * (positions[:, :, 0] - 400) + 2 * np.arange(count)[:, None]
        support = np.ones(positions.shape[:2], bool)
        support[2] = False
        result = analyze_tip_surface_series(estimator, positions, support,
            np.tile([400., 200.], (count, 1)), np.arange(count, dtype=float),
            branch_labels=np.ones(support.shape, int))
        self.assertAlmostEqual(result['address_rate_um_s'][1], -5., places=6)
        self.assertAlmostEqual(result['relative_tangent_velocity_um_s'][1], 10., places=6)
        self.assertTrue(np.isnan(result['address_rate_um_s'][2:4]).all())
        self.assertTrue(np.isnan(result['relative_tangent_step_um'][2:4]).all())
        self.assertGreater(result['motion_step_state'][4], 0)

    def test_curved_surface_normal_columns_survive_large_bending(self):
        estimator = synthetic_estimator()
        s = estimator.reference[:, 0] - 400.
        depth = estimator.reference[:, 1] - 200.
        angle = s / 20.
        current = np.column_stack([400 + 20 * np.sin(angle) - depth * np.sin(angle),
                                   200 + 20 * (1 - np.cos(angle)) + depth * np.cos(angle)])
        endpoints = np.array([6., 12.]) / 20.
        tip = np.mean(np.column_stack([400 + 20 * np.sin(endpoints), 200 + 20 * (1 - np.cos(endpoints))]), axis=0)
        result = estimator.estimate_frame(current, np.ones(len(current), bool), tip,
            branch_labels=np.ones(len(current), int), blade_mask=np.zeros((768, 1024), bool))
        self.assertEqual(result['state'], 2)
        self.assertEqual(result['route'], 'original_surface_normal_columns')
        self.assertAlmostEqual(result['s_px'], 9., places=6)
        np.testing.assert_array_equal(result['current_marker_xy'], tip)

    def test_surface_curve_cannot_claim_direct_contact_across_blade_interior(self):
        estimator = synthetic_estimator()
        blade = np.zeros((768, 1024), bool)
        blade[199:202, 399:401] = True
        result = estimator.estimate_frame(estimator.reference, np.ones(len(estimator.ids), bool), [403., 200.],
            branch_labels=np.ones(len(estimator.ids), int), blade_mask=blade)
        self.assertNotEqual(result['state'], 2)
        self.assertNotEqual(result['route'], 'original_surface_normal_columns')

    def test_full_initial_surface_extension_preserves_stations_and_uses_sparse_outer_ids(self):
        base = synthetic_estimator()
        surface = dict(boundary_xy=base.boundary.copy(), surface_s_px=base.addresses.copy(),
            tangent=base.tangent.copy(), inward_normal=base.normal.copy(), contact_s_px=0.)
        mask = np.zeros((768, 1024), np.uint8); mask[201:] = 1
        expanded = extend_initial_surface_reference(surface, mask)
        np.testing.assert_array_equal(expanded['boundary_xy'][expanded['saved_station_indices']], surface['boundary_xy'])
        np.testing.assert_array_equal(expanded['surface_s_px'][expanded['saved_station_indices']], surface['surface_s_px'])
        xx, yy = np.meshgrid(np.arange(40., 1001., 24.), [212., 236., 260.])
        extra = np.column_stack([xx.ravel(), yy.ravel()])
        estimator = TipSurfaceAddressEstimator(np.vstack([base.reference, extra]),
            np.r_[base.ids, 2000 + np.arange(len(extra))], np.r_[base.layers, np.full(len(extra), -1)],
            np.r_[base.stations, np.full(len(extra), -1)], surface, 5., initial_material_mask=mask)
        result = estimator.estimate_frame(estimator.reference, np.ones(len(estimator.ids), bool), [800., 200.75],
            branch_labels=np.ones(len(estimator.ids), int), blade_mask=np.zeros((768, 1024), bool))
        self.assertEqual(result['state'], 1)
        self.assertAlmostEqual(result['s_px'], 400., places=4)
        self.assertTrue(np.all(result['support_ids'] >= 2000))

    def test_competing_material_branches_are_not_averaged_to_a_false_contact(self):
        estimator = synthetic_estimator()
        branches = np.where(estimator.layers % 2 == 0, 1, 2)
        current = estimator.reference.copy()
        current[branches == 2, 0] += 20.
        result = estimator.estimate_frame(current, np.ones(len(current), bool), [400., 200.],
            branch_labels=branches, blade_mask=np.zeros((768, 1024), bool))
        self.assertEqual(result['state'], 0)
        self.assertIn('ambiguous', result['reason'])
        self.assertTrue(np.isnan(result['s_um']))


if __name__ == '__main__':
    unittest.main()
