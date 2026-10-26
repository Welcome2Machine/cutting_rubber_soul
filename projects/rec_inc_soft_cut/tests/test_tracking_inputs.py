import unittest
import json
import tempfile
from pathlib import Path
import shutil

import cv2
import numpy as np

from src.workflows.tracking_inputs import (build_queries, contact_candidates, prefix_endpoint,
    sequence_endpoint, local_reversal_support, validated_reuse_inputs, add_full_material_queries, _normalize)


class ValidationInputTests(unittest.TestCase):
    def test_observed_endpoint_and_loading_only(self):
        tips = np.column_stack([np.full(120, 500.), 100 + np.arange(120) * 3.])
        tips[101:, 1] = tips[100, 1] - np.arange(1, 20) * 3
        support = np.ones(120, bool)
        support[40] = False
        depth, end, stroke = prefix_endpoint(tips, support, 10, 100, 1, 3)
        self.assertEqual(end, 41)
        self.assertEqual(stroke, 270.)
        self.assertAlmostEqual(depth[10], 0.)
        self.assertAlmostEqual(depth[100], 3.)

    def test_unsupported_calibration_is_rejected(self):
        tips = np.column_stack([np.zeros(100), np.arange(100) * 3.])
        supported = np.ones(100, bool)
        supported[10] = False
        with self.assertRaisesRegex(ValueError, "image supported"):
            prefix_endpoint(tips, supported, 10, 90)

    def test_contact_needs_sustained_proximity(self):
        indices = np.arange(8)
        gaps = np.array([20, 1, 20, 3, 1, 1, 1, 1.])
        found = contact_candidates(indices, np.ones(8, bool), gaps)
        self.assertEqual(found["2.0"], 4)
        self.assertEqual(found["4.0"], 3)

    def test_queries_keep_all_layers_without_global_coarse_grid(self):
        yy, xx = np.mgrid[:768, :1024]
        labels = np.where(yy > .6 * xx + 40, 1, 0).astype(np.uint8)
        labels[:40] = 255
        queries, cohort, _, _ = build_queries(labels, 500)
        self.assertEqual(len(np.unique(queries["depth_layer"])), 25)
        self.assertGreater(len(queries["seed_points"]), 650)
        self.assertLessEqual(len(queries["seed_points"]), 800)
        points = np.rint(queries["seed_points"]).astype(int)
        self.assertTrue(np.all(labels[points[:, 1], points[:, 0]] == 1))
        self.assertEqual(queries["coarse_count"], 0)
        np.testing.assert_array_equal(queries["point_ids"], np.arange(len(points)))

    def test_local_reference_requires_endpoint_and_band_agreement(self):
        registered = np.array([[400., 624.], [400., 590.], [400., 624.], [400., 624.], [400., 624.]])
        semantic = np.tile([399., 623.], (5, 1))
        raw = np.tile([400., 617.], (5, 1))
        scores = np.array([.98, .99, .99, .64, .99])
        bands = np.array([2., 1., 21., 2., 2.])
        raw[-1] = np.nan
        np.testing.assert_array_equal(local_reversal_support(registered, semantic, raw, scores, bands),
                                      [True, False, False, False, False])

    def test_complete_stroke_uses_calibrated_reversal_despite_roundoff(self):
        tips = np.column_stack([np.zeros(100), 123.456 + np.arange(100) * 4.321987])
        support = np.ones(100, bool)
        depth, endpoint, _ = prefix_endpoint(tips, support, 13, 93, 3., 3.)
        self.assertEqual(endpoint, 93)
        self.assertEqual(depth[93], 3.)
        self.assertEqual(depth[13], 0.)

    def test_complete_stroke_does_not_take_earlier_higher_noisy_tip(self):
        tips = np.column_stack([np.zeros(100), np.arange(100) * 4.])
        tips[70, 1] = 380.
        support = np.ones(100, bool)
        _, endpoint, _ = prefix_endpoint(tips, support, 10, 90, 3., 3.)
        self.assertEqual(endpoint, 90)

    def test_reuse_rejects_different_source_before_loading_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'plan.json').write_text(json.dumps(dict(name='0930l30a1', source_sha256='old', contact_to_reversal_mm=3)))
            (folder/'geometry.json').write_text(json.dumps(dict(contact_to_reversal_mm=3)))
            with self.assertRaisesRegex(ValueError, 'identity/checksum'):
                validated_reuse_inputs(folder, '0930l30a1', 'new', 3)

    def test_full_video_keeps_withdrawal_and_unsupported_last_frame(self):
        tips = np.column_stack([np.zeros(150), np.arange(150) * 4.])
        tips[91:, 1] = 360. - np.arange(1, 60) * 7.
        supported = np.ones(150, bool)
        supported[-1] = False
        depth, endpoint, _ = sequence_endpoint(tips, supported, 10, 90, 3., 3., full_video=True)
        self.assertEqual(endpoint, 149)
        self.assertEqual(depth[90], 3.)
        self.assertLess(depth[110], depth[100])
        self.assertLess(depth[148], 0.)
        self.assertTrue(np.isnan(depth[-1]))
        _, loading_end, _ = sequence_endpoint(tips, supported, 10, 90, 3., 3.)
        self.assertEqual(loading_end, 90)

    def test_full_material_preserves_core_and_covers_outer_material_tiles(self):
        from src.tracking.fixed_grid import FixedGridProcessor, grid_0915_config
        yy, xx = np.mgrid[:768, :1024]
        labels = np.where(yy > .6 * xx + 40, 1, 0).astype(np.uint8)
        labels[:40] = 255
        labels[480:560, 600:640] = 2
        core, _, _, _ = build_queries(labels, 500)
        combined = add_full_material_queries(core, labels, pitch=24)
        n = len(core['seed_points'])
        np.testing.assert_array_equal(combined['seed_points'][:n], core['seed_points'])
        np.testing.assert_array_equal(combined['point_ids'][:n], core['point_ids'])
        np.testing.assert_array_equal(combined['depth_layer'][:n], core['depth_layer'])
        self.assertTrue(combined['is_core_point'][:n].all())
        self.assertFalse(combined['is_core_point'][n:].any())
        extra = combined['seed_points'][n:].astype(int)
        self.assertGreater(len(extra), 400)
        self.assertTrue(np.all(labels[extra[:, 1], extra[:, 0]] == 1))
        grid = cv2.dilate(FixedGridProcessor(grid_0915_config()).mask().astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        blade = cv2.dilate((labels == 2).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        self.assertFalse(np.any(grid[extra[:, 1], extra[:, 0]] | blade[extra[:, 1], extra[:, 0]]))
        safe = (labels == 1) & ~grid & ~blade
        safe[:4] = safe[-4:] = False
        safe[:, :4] = safe[:, -4:] = False
        occupied = {(int(y) // 24, int(x) // 24) for x, y in combined['seed_points']}
        for top in range(0, 768, 24):
            for left in range(0, 1024, 24):
                if safe[top:top + 24, left:left + 24].any():
                    self.assertIn((top // 24, left // 24), occupied)
        self.assertEqual(len(np.unique(combined['point_ids'])), len(combined['point_ids']))
        # Extending a previously expanded set must not append duplicate points.
        again = add_full_material_queries(combined, labels, pitch=24)
        np.testing.assert_array_equal(again['seed_points'], combined['seed_points'])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is needed for lossless input verification')
    def test_normalization_extension_reuses_pixels_without_gain_state_reset(self):
        from src.tracking.shadow_normalization import MaterialShadowNormalizer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_dir = root / 'masks'; mask_dir.mkdir()
            source = root / 'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'FFV1'), 12., (1024, 768))
            if not writer.isOpened():
                self.skipTest('OpenCV FFV1 encoder unavailable')
            pattern = np.random.default_rng(5).integers(0, 16, (768, 1024, 1), dtype=np.uint8)
            images = [np.repeat(pattern + value, 3, axis=2) for value in (32, 45, 65, 36, 42)]
            labels = np.ones((768, 1024), np.uint8)
            for i, image in enumerate(images):
                writer.write(image)
                cv2.imwrite(str(mask_dir / f'f{i:06d}.png'), labels)
            writer.release()
            previous = root / 'previous.mkv'
            _normalize(source, mask_dir, 3, previous)
            full = root / 'full.mkv'
            metadata = _normalize(source, mask_dir, 5, full, reuse_video=previous, reuse_frames=3)
            expected = MaterialShadowNormalizer()
            reader = cv2.VideoCapture(str(full))
            try:
                for image in images:
                    ok, actual = reader.read()
                    self.assertTrue(ok)
                    reference, _ = expected.process(image, labels)
                    np.testing.assert_array_equal(actual, reference)
                self.assertFalse(reader.read()[0])
            finally:
                reader.release()
            self.assertEqual(metadata['reused_pixel_frames'], 3)


if __name__ == "__main__":
    unittest.main()
