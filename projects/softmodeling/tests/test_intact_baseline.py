"""A completed label must not certify a partial or different-material baseline."""
import copy
import json
import unittest

from tools.qualify_intact_baseline import SPEC, validate_report


class IntactBaselineTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(SPEC.read_text(encoding='utf-8'))
        self.report = dict(completed=True, status='completed', target_mm=2., elapsed_s=1.,
            rejected_load_attempts=[], signature=dict(
                material=dict(mu_eq=[.14], alpha_eq=[2.], mu_ov=[1.24], alpha_ov=[2.],
                    relaxation_s=1.7, volumetric='quadratic_j', viscous_flow='isochoric_maxwell'),
                config=dict(inclination_deg=57, friction_coefficient=.35), loading_rate=.2,
                physical_force_tolerance_n=1e-4, contact_formulation='improved-max', dhat=.0025),
            steps=[dict(depth_mm=i/100, reaction_fz_n=i/10, residual_n=1e-5, min_j=.5,
                load_step=dict(subdivision_level=0, predictor='secant')) for i in range(201)])
        self.audit = dict(passed=True, states=201)

    def test_incomplete_path_cannot_be_certified_by_completed_label(self):
        for change in ('missing', 'wrong_endpoint', 'failed'):
            with self.subTest(change=change):
                r = copy.deepcopy(self.report)
                if change == 'missing':del r['steps'][100]
                if change == 'wrong_endpoint':r['steps'][-1]['depth_mm'] = 1.99
                if change == 'failed':r['status'] = 'failed'
                with self.assertRaises(ValueError):validate_report(r, self.audit, self.spec, 57)

    def test_material_and_acceptance_changes_are_rejected(self):
        for change in ('material', 'residual', 'volume', 'substep'):
            with self.subTest(change=change):
                r = copy.deepcopy(self.report)
                if change == 'material':r['signature']['material']['mu_ov'] = [.345]
                if change == 'residual':r['steps'][20]['residual_n'] = 2e-4
                if change == 'volume':r['steps'][20]['min_j'] = 0
                if change == 'substep':r['steps'][20]['load_step']['subdivision_level'] = 1
                with self.assertRaises(ValueError):validate_report(r, self.audit, self.spec, 57)

    def test_complete_path_retains_recovery_information(self):
        self.report['steps'][50]['load_step']['predictor'] = 'affine'
        self.report['rejected_load_attempts'] = [dict(predictor='secant')]
        result = validate_report(self.report, self.audit, self.spec, 57)
        self.assertEqual(result['states'], 201)
        self.assertEqual(result['affine_accepts'], 1)
        self.assertEqual(result['rejected_attempts'], 1)
        self.assertEqual(result['cutbacks'], 0)



if __name__ == '__main__':unittest.main()
