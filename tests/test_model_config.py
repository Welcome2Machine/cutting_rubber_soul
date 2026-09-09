import unittest
from prototype.model_config import FEMConfig


class ModelConfigTests(unittest.TestCase):
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
