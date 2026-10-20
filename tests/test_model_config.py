import unittest
from prototype.model_config import FEMConfig


class ModelConfigTests(unittest.TestCase):
    def test_current_moduli_and_explicit_legacy_input_are_distinct(self):
        import json
        from pathlib import Path

        current = FEMConfig()
        current.validate()
        self.assertEqual(current.shear_modulus_n_mm2, .47)
        self.assertEqual(current.poisson_ratio, .49)
        self.assertAlmostEqual(current.bulk_modulus_n_mm2, 23.343333333333333)

        # A new default must not silently revise old input/checkpoint physics.
        path = Path(__file__).resolve().parents[1] / 'prototype' / 'ipc_baseline.json'
        legacy = FEMConfig(**json.loads(path.read_text(encoding='utf-8'))['model'])
        self.assertEqual(legacy.shear_modulus_n_mm2, .66)
        self.assertAlmostEqual(legacy.bulk_modulus_n_mm2, 32.78)
        restored = FEMConfig(**legacy.to_dict())
        self.assertEqual(restored.to_dict(), legacy.to_dict())

    def test_target_loading_is_three_mm_and_fifteen_seconds(self):
        c=FEMConfig()
        c.validate()
        self.assertEqual(c.max_indentation_mm,3)
        self.assertAlmostEqual(c.max_indentation_mm/c.speed_mm_s,15.)


    def test_invalid_numerical_configuration_fails_closed(self):
        for kwargs in ({'mesh_size_mm':float('nan')},{'newton_tolerance_n':float('inf')},
                       {'hold_s':-1},{'blade_tip_angle_deg':0},{'approach_steps':3.5},
                       {'cohesive_mode_mix':0},{'drop_s_mm':30}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FEMConfig(**kwargs).validate()


    def test_json_loader_accepts_powershell_utf8_bom(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text('{"mesh_size_mm": 0.7}', encoding="utf-8-sig")
            self.assertAlmostEqual(FEMConfig.from_json(path).mesh_size_mm, 0.7)
