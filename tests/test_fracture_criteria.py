import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
import csv
import numpy as np
from prototype.fracture_criteria import StrengthSurface,stress_measures
from prototype.initiation import IntactProblem,InitiationOptions
from prototype.model_config import FEMConfig
from prototype.run_initiation import run


class StrengthTests(unittest.TestCase):
    def setUp(self):
        # Synthetic strengths used only to check the mathematics.
        self.surface = StrengthSurface(2.,3.,'synthetic unit test','assumed')

    def test_uniaxial_and_hydrostatic_calibration_points(self):
        tensors = np.array([np.diag([2.,0,0]),np.eye(3)*3,np.zeros((3,3)),-np.eye(3)*3])
        np.testing.assert_allclose(self.surface.index(tensors),[1,1,0,-1],atol=1e-12)

    def test_reference_stress_objectivity(self):
        F = np.array([[1.2,.3,0],[0,.9,0],[0,0,1.]])
        sigma = np.array([[2.,.4,0],[.4,-.2,0],[0,0,.1]])
        angle = .8
        Q = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        P,S2 = stress_measures(F,sigma)
        Pr,S2r = stress_measures(Q@F,Q@sigma@Q.T)
        np.testing.assert_allclose(Pr,Q@P,atol=1e-12)
        np.testing.assert_allclose(S2r,S2,atol=1e-12)
        self.assertAlmostEqual(float(self.surface.index(S2r)),float(self.surface.index(S2)))

    def test_nominal_strength_is_not_second_piola_strength(self):
        stretch = 4.2
        F = np.diag([stretch,stretch**-.5,stretch**-.5])
        P,S2 = stress_measures(F,np.diag([4.4*stretch,0,0]))
        self.assertAlmostEqual(P[0,0],4.4)
        self.assertAlmostEqual(S2[0,0],4.4/stretch)

    def test_missing_or_incompatible_calibration_rejected(self):
        with self.assertRaises(ValueError):
            StrengthSurface(4.4,1.,'test','assumed')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.json'
            data = self.surface.to_dict(); data.pop('stress_measure')
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'explicitly'):
                StrengthSurface.from_json(path)
            data['stress_measure'] = 'cauchy'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'second Piola'):
                StrengthSurface.from_json(path)

    def test_criterion_is_wired_into_full_domain_fields(self):
        p = IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2),self.surface)
        fields = p.fields(np.zeros(p.mesh.dof_count))
        np.testing.assert_allclose(fields['multiaxial_strength_index'],0,atol=1e-12)
        self.assertEqual(fields['second_piola_stress_xzy_mpa'].shape,(len(p.mesh.triangles),3,3))

    def test_configured_candidate_is_exported_in_short_complete_cycle(self):
        config = FEMConfig(mesh_size_mm=1.5,max_indentation_mm=.01,approach_steps=2,
                           pullback_steps=2,hold_s=0)
        options = InitiationOptions(refinement_mm=.7,surface_spacing_mm=.0025)
        synthetic = StrengthSurface(1e-6,3e-6,'synthetic output-contract test only','assumed')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'run'
            with contextlib.redirect_stdout(io.StringIO()):
                run(config,options,out,synthetic)
            status = json.loads((out/'status.json').read_text())
            self.assertTrue(status['completed'])
            self.assertEqual(status['multiaxial_strength_status'],'assumed')
            events = json.loads((out/'initiation_candidates.json').read_text())
            self.assertIn('multiaxial',events)
            with (out/'results.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertIn('multiaxial_strength_ratio',rows[0])
            self.assertTrue((out/'first_multiaxial_candidate.npz').is_file())


if __name__ == '__main__':
    unittest.main()
