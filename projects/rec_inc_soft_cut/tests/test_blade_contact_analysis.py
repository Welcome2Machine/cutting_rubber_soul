"""Small data-only checks of the measurement runner's coordinate contracts."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from src.workflows.measurement import BladeFrames, pack_address, read_case, surface_depths, restrict_contact_phase


class BladeContactAnalysisTests(unittest.TestCase):
    def test_contact_phase_clears_incoming_edge_but_preserves_contact_address_and_withdrawal(self):
        count, contact = 6, 2
        names = ('s_px', 's_um', 'reference_s_px', 'low_px', 'high_px', 'low_um', 'high_um',
                 'surface_turn_deg', 'migration_from_initial_contact_um', 'address_rate_um_s',
                 'relative_tangent_step_um', 'relative_tangent_velocity_um_s')
        address = {key: np.arange(count, dtype=float) - contact for key in names}
        address.update(state=np.ones(count, np.uint8), label=np.full(count, 'conditional_inference', dtype='U32'),
            reason=np.full(count, 'ok', dtype='U64'), motion_step_state=np.ones(count, np.uint8),
            reference_marker_xy=np.arange(count * 2, dtype=float).reshape(count, 2),
            current_marker_xy=np.arange(count * 2, dtype=float).reshape(count, 2) + 100.,
            raw_reference_s_um=np.arange(count, dtype=float) + 500.)
        current_before = address['current_marker_xy'].copy()
        raw_before = address['raw_reference_s_um'].copy()
        final = restrict_contact_phase(address, np.arange(count), contact)
        np.testing.assert_array_equal(final['state'], [0, 0, 1, 1, 1, 1])
        for name in ('address_rate_um_s', 'relative_tangent_step_um', 'relative_tangent_velocity_um_s'):
            self.assertTrue(np.isnan(final[name][:contact + 1]).all())
            np.testing.assert_array_equal(final[name][contact + 1:], [1., 2., 3.])
        np.testing.assert_array_equal(final['motion_step_state'], [0, 0, 0, 1, 1, 1])
        self.assertEqual(final['s_um'][contact], 0.)
        self.assertEqual(final['migration_from_initial_contact_um'][contact], 0.)
        self.assertTrue(np.isnan(final['reference_marker_xy'][:contact]).all())
        np.testing.assert_array_equal(final['reference_marker_xy'][contact], [4., 5.])
        np.testing.assert_array_equal(final['current_marker_xy'], current_before)
        np.testing.assert_array_equal(final['raw_reference_s_um'], raw_before)

    def test_ragged_support_ids_roundtrip_without_pickle_or_id_renumbering(self):
        rows = [np.array([900, 5], np.int64), np.empty(0, np.int64), np.array([42], np.int64)]
        source = dict(state=np.array([2, 0, 1], np.uint8), support_ids=rows,
                      reason=np.array(['ok', 'unsupported', 'conditional']), metadata={'not_an_array': True})
        packed = pack_address(source)
        np.testing.assert_array_equal(packed['support_id_offsets'], [0, 2, 2, 3])
        self.assertNotIn('metadata', packed)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'address.npz'
            np.savez_compressed(path, **packed)
            with np.load(path, allow_pickle=False) as stored:
                offsets, flat = stored['support_id_offsets'], stored['support_ids_flat']
                for index, expected in enumerate(rows):
                    np.testing.assert_array_equal(flat[offsets[index]:offsets[index + 1]], expected)
                for key in stored.files:
                    self.assertNotEqual(stored[key].dtype, object)
        np.testing.assert_array_equal(rows[0], [900, 5])

    def test_initial_normal_depth_uses_sloping_material_coordinates_and_no_extent_clamp(self):
        angle = np.radians(57.)
        tangent = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-tangent[1], tangent[0]])
        center = np.array([300., 180.])
        addresses = np.arange(-60., 61., 6.)
        boundary = center + addresses[:, None] * tangent

        def surface_point(address):
            return center + address * tangent if addresses[0] <= address <= addresses[-1] else np.full(2, np.nan)

        estimator = SimpleNamespace(boundary=boundary, tangent=tangent, normal=normal,
                                    addresses=addresses, surface_point=surface_point)
        along = np.array([-42., 0., 36., 70.])
        normal_depth = np.array([1.5, 7.5, 25.5, 1.5])
        seed = center + along[:, None] * tangent + normal_depth[:, None] * normal
        before = seed.copy()
        result = surface_depths(seed, estimator)
        np.testing.assert_allclose(result[:3], normal_depth[:3], atol=1e-10)
        self.assertTrue(np.isnan(result[-1]))
        np.testing.assert_array_equal(seed, before)

    def test_blade_frame_reader_respects_imported_frame_mapping(self):
        visited = []

        def reader(frame):
            visited.append(frame)
            return np.array([[0, 1, 2], [255, 2, 1]], np.uint8)

        masks = BladeFrames(reader, np.array([3, 8]))
        np.testing.assert_array_equal(masks[1], [[False, False, True], [False, True, False]])
        self.assertEqual(visited, [8])

    def test_read_case_preserves_complete_withdrawal_and_actual_irregular_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / '0930l30a1'
            (case / 'tracking').mkdir(parents=True)
            count = 7
            mapping = np.arange(count)
            times = np.array([0., .015, .025, .04, .05, .065, .075])
            ids = np.array([18, 91])
            (case / 'plan.json').write_text(json.dumps(dict(full_video_frames=count, contact_frame=1, reversal_frame=3)))
            (case / 'geometry.json').write_text(json.dumps(dict(contact_to_reversal_mm=3., contact_to_reversal_px=400.)))
            positions = np.arange(count * 2 * 2).reshape(count, 2, 2).astype(float)
            np.savez(case / 'tracking/tracks.npz', positions=positions, supported=np.ones((count, 2), bool),
                     point_ids=ids, source_frame_indices=mapping, motion_step_valid=np.ones((count, 2), bool))
            np.savez(case / 'queries.npz', point_ids=ids)
            depth = np.array([-.1, 0., 1.5, 3., 1.2, -.2, np.nan])
            tip = np.column_stack([np.full(count, 400.), 200 + depth * 400 / 3.])
            np.savez(case / 'geometry.npz', tips_px=tip, supported=np.isfinite(depth), depth_mm=depth)
            np.savez(case / 'frame_mapping.npz', frame_indices=mapping, source_indices=[0, 3, 5, 8, 10, 13, 15],
                     physical_time_s=times)
            source, plan, _, tracks, _, geometry, timing, scale = read_case('0930l30a1', SimpleNamespace(source=root))
            self.assertEqual(source, case)
            self.assertEqual(plan['reversal_frame'], 3)
            np.testing.assert_array_equal(tracks['positions'], positions)
            np.testing.assert_array_equal(geometry['depth_mm'], depth)
            np.testing.assert_array_equal(timing['physical_time_s'], times)
            self.assertEqual(scale, 7.5)
            self.assertLess(geometry['depth_mm'][5], 0.)
            self.assertFalse(geometry['supported'][-1])

            # A loading preparation can contain context after reversal. Its
            # measurement scope must stop at reversal without truncating IDs
            # or losing irregular acquisition timestamps/scalar metadata.
            (case / 'plan.json').write_text(json.dumps(dict(full_video_frames=10,
                full_video=False, contact_frame=1, reversal_frame=3)))
            np.savez(case / 'frame_mapping.npz', frame_indices=mapping,
                     source_indices=[0, 3, 5, 8, 10, 13, 15], physical_time_s=times,
                     physical_acquisition_fps=np.array(200.))
            _, plan, _, tracks, _, geometry, timing, _ = read_case('0930l30a1', SimpleNamespace(source=root))
            self.assertFalse(plan['full_video'])
            np.testing.assert_array_equal(tracks['positions'], positions[:4])
            np.testing.assert_array_equal(tracks['point_ids'], ids)
            np.testing.assert_array_equal(tracks['motion_step_valid'].shape, (4, 2))
            np.testing.assert_array_equal(geometry['depth_mm'], depth[:4])
            np.testing.assert_array_equal(timing['physical_time_s'], times[:4])
            np.testing.assert_array_equal(timing['source_indices'], [0, 3, 5, 8])
            self.assertEqual(timing['physical_acquisition_fps'], 200.)

            broken_mapping = np.array([0, 1, 2, 4, 5, 6, 7])
            np.savez(case / 'tracking/tracks.npz', positions=positions, supported=np.ones((count, 2), bool),
                     point_ids=ids, source_frame_indices=broken_mapping, motion_step_valid=np.ones((count, 2), bool))
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                read_case('0930l30a1', SimpleNamespace(source=root))
            np.savez(case / 'tracking/tracks.npz', positions=positions[:3], supported=np.ones((3, 2), bool),
                     point_ids=ids, source_frame_indices=np.arange(3), motion_step_valid=np.ones((3, 2), bool))
            with self.assertRaisesRegex(ValueError, 'through reversal'):
                read_case('0930l30a1', SimpleNamespace(source=root))


if __name__ == '__main__':
    unittest.main()
