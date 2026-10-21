"""Refinement must not change the physical problem or reuse coarse histories."""
import unittest
import numpy as np
from prototype.material_case import signature,resume_compatibility
from tools.run_tip_mesh_study import jobs,make,geometry_check,paired_summary,PARAMETERS
from tools.run_tip_radius_study import make as coarse_make


class TipMeshStudyTests(unittest.TestCase):
    def test_fine_factory_geometry_and_resume_isolation(self):
        coarse=coarse_make(33,PARAMETERS,radius=.02)
        fine=make(33,PARAMETERS,radius=.02)
        self.assertTrue(geometry_check(fine,coarse)['passed'])
        self.assertNotEqual(signature(fine),signature(coarse))
        with self.assertRaisesRegex(ValueError,'identical'):
            resume_compatibility(dict(signature=signature(coarse)),dict(signature=signature(fine)))
        sharper=make(33,PARAMETERS,radius=.01)
        np.testing.assert_array_equal(fine.mesh.reference_xy,sharper.mesh.reference_xy)
        np.testing.assert_array_equal(fine.mesh.triangles,sharper.mesh.triangles)
        self.assertEqual(fine.material,sharper.material)

    def test_only_authorized_four_independent_jobs(self):
        tasks=jobs();self.assertEqual(len(tasks),4)
        self.assertEqual({(j['angle'],j['radius_mm']) for j in tasks},{(a,r) for a in (0,33) for r in (.01,.02)})
        for j in tasks:
            self.assertNotIn('resume',j)
            self.assertEqual((j['surface_spacing_mm'],j['target_mm'],j['increment'],j['seconds']),(.0025,.5,.01,3600))
        with self.assertRaises(ValueError):make(0,PARAMETERS,radius=.04)
        with self.assertRaises(ValueError):coarse_make(0,PARAMETERS,radius=.02,surface_spacing=.001)

    def test_partial_cases_cannot_establish_mesh_radius_interaction(self):
        rows=[dict(angle=0,radius_mm=r,mesh=h,complete=True,fz05_n=f)
              for r,h,f in ((.02,'coarse',10),(.01,'coarse',9.8),(.02,'fine',9.5),(.01,'fine',9.4))]
        rows[-1]['complete']=False
        self.assertEqual(paired_summary(rows),[])
        rows[-1]['complete']=True;v=paired_summary(rows)[0]
        self.assertAlmostEqual(v['radius_signal_n']['coarse'],-.2)
        self.assertAlmostEqual(v['radius_signal_n']['fine'],-.1)
        self.assertTrue(v['mesh_change_exceeds_old_radius_signal'])
        self.assertFalse(v['converged']);self.assertFalse(v['promotion'])


if __name__=='__main__':unittest.main()
