import contextlib
from dataclasses import replace
import io,json,tempfile,unittest
from pathlib import Path
import numpy as np
from prototype.model_config import FEMConfig
from prototype.initiation import InitiationOptions
from prototype.run_initiation import run


class FirstCandidateTests(unittest.TestCase):
    def test_quadratic_candidate_preserves_sample_and_element_location(self):
        # Synthetic threshold checks the production P2 event/output path only.
        c=replace(FEMConfig(),max_indentation_mm=.00625,approach_steps=2,tensile_strength_n_mm2=.005)
        options=InitiationOptions(surface_spacing_mm=.01,mesh_grading='graded',
            contact_discretization='quadratic_integral',volumetric_discretization='quadratic_full')
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            out=Path(directory)
            run(c,options,out,mode='to_candidate',candidate_tolerance_mm=.0001)
            status=json.loads((out/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['state'],'candidate_found')
            self.assertFalse(status['physical_nucleation_validated'])
            event=json.loads((out/'first_candidate_location.json').read_text(encoding='utf-8'))
            with np.load(out/'last_state.npz') as state:
                sample=event['peak_sample']
                self.assertEqual(event['peak_element'],int(state['field_element_ids'][sample]))
                np.testing.assert_allclose(event['peak_current_xz_mm'],state['current_centroid_xz_mm'][sample])
                self.assertEqual(len(state['J']),7*len(state['triangles']))
            self.assertTrue((out/'first_candidate_location.png').is_file())

    def test_real_loading_event_is_bracketed_saved_and_stops_before_hold(self):
        # Synthetic low strength exercises stopping, not CB245NE calibration.
        c=replace(FEMConfig(),max_indentation_mm=.00625,approach_steps=2,tensile_strength_n_mm2=.01)
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            out=Path(directory)
            run(c,InitiationOptions(surface_spacing_mm=.01),out,mode='to_candidate',candidate_tolerance_mm=.0001)
            status=json.loads((out/'status.json').read_text())
            self.assertEqual(status['state'],'candidate_found')
            self.assertFalse(status['physical_nucleation_validated'])
            event=json.loads((out/'first_candidate_location.json').read_text())
            a,b=event['indentation_bracket_mm'];self.assertGreater(b,a);self.assertLessEqual(b-a,.0001)
            with np.load(out/'before_candidate.npz') as before,np.load(out/'last_state.npz') as after:
                self.assertLess(before['nominal_tensile_mpa'].max(),.01)
                self.assertGreaterEqual(after['nominal_tensile_mpa'].max(),.01)
                self.assertEqual(float(after['indentation_mm']),b)
            rows=np.genfromtxt(out/'results.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
            self.assertTrue(np.all(np.atleast_1d(rows)['phase']=='approach'))

    def test_loading_limit_is_not_reported_as_a_candidate(self):
        c=replace(FEMConfig(),max_indentation_mm=.001,approach_steps=2,tensile_strength_n_mm2=1e6)
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            out=Path(directory);run(c,InitiationOptions(surface_spacing_mm=.01),out,mode='to_candidate')
            status=json.loads((out/'status.json').read_text())
            self.assertFalse(status['candidate_found']);self.assertEqual(status['stop_reason'],'loading_limit_without_candidate')
            self.assertEqual(status['indentation_mm'],.001)

    def test_no_implicit_multiaxial_calibration(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):
            run(FEMConfig(),InitiationOptions(),Path(d),mode='to_candidate',candidate_criterion='multiaxial')


if __name__=='__main__':unittest.main()
