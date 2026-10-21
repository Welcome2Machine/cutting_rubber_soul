"""Radius controls must not silently change material, zero, mesh or pairing."""
import unittest
import numpy as np
from tools.run_tip_radius_study import jobs,make,PARAMETERS,pair_verdict
from prototype.initiation import rounded_blade_distance,local_to_lab
from prototype.material_case import signature,resume_compatibility


class TipRadiusStudyTests(unittest.TestCase):
    def test_fixed_common_parameters_and_bounds(self):
        tasks=jobs();self.assertEqual(len(tasks),6)
        self.assertEqual({j['angle'] for j in tasks},{0,33})
        self.assertEqual({j['radius_mm'] for j in tasks},{.01,.02,.04})
        for j in tasks:
            self.assertEqual(tuple(j['parameters']),PARAMETERS)
            self.assertEqual((j['target_mm'],j['increment'],j['seconds']),(.5,.01,3600))

    def test_body_mesh_material_zero_and_signature_are_protected(self):
        ps=[make(33,PARAMETERS,radius=r) for r in (.01,.02,.04)]
        for p in ps:
            np.testing.assert_array_equal(p.mesh.reference_xy,ps[0].mesh.reference_xy)
            np.testing.assert_array_equal(p.mesh.triangles,ps[0].mesh.triangles)
            self.assertEqual(p.material,ps[0].material)
            self.assertEqual(p.config,ps[0].config)
            self.assertEqual(p.config.blade_tip_angle_deg,23.)
            self.assertEqual(p.config.blade_thickness_mm,.245)
            contact=local_to_lab(np.array([[p.config.drop_s_mm,p.config.height_mm]]),p.config)
            gap=-rounded_blade_distance(contact,p.config,p.options,0.)[0]
            self.assertLess(abs(float(gap[0])),1e-12)
        self.assertNotEqual(signature(ps[0])['mesh_sha256'],signature(ps[1])['mesh_sha256'])
        with self.assertRaisesRegex(ValueError,'identical'):
            resume_compatibility(dict(signature=signature(ps[0])),dict(signature=signature(ps[1])))

    def test_failed_short_curves_cannot_make_common_radius_verdict(self):
        records=[dict(radius_mm=.02,angle=a,complete=True,fz05_n=10.,rmse_01_05_n=3.) for a in (0,33)]
        records.append(dict(radius_mm=.01,angle=0,complete=True,fz05_n=9.,rmse_01_05_n=2.))
        records.append(dict(radius_mm=.01,angle=33,complete=False))
        r=pair_verdict(records)[0];self.assertFalse(r['paired_complete']);self.assertFalse(r['lowers_both'])
        records[-1]=dict(radius_mm=.01,angle=33,complete=True,fz05_n=11.,rmse_01_05_n=4.)
        r=pair_verdict(records)[0];self.assertTrue(r['paired_complete']);self.assertFalse(r['lowers_both']);self.assertFalse(r['promotion'])

    def test_unauthorized_factory_parameters_rejected(self):
        for a,r,p in ((57,.02,PARAMETERS),(0,.005,PARAMETERS),(0,.02,(.14,1.0,1.7))):
            with self.assertRaises(ValueError):make(a,p,radius=r)


if __name__=='__main__':unittest.main()
