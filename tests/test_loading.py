from dataclasses import replace
import unittest
import numpy as np
from prototype.model_config import FEMConfig
from prototype.initiation import InitiationOptions
from prototype.loading import loading_depths,resolved_loading_steps


class LoadingTests(unittest.TestCase):
    def test_production_and_bounded_prefix_have_identical_float_targets(self):
        c=FEMConfig(approach_steps=481,pullback_steps=481)
        o=InitiationOptions()
        prefix=[x for x,_ in loading_depths(c,o.surface_spacing_mm/2,stop_mm=.13)]
        full=[x for phase,x,_ in resolved_loading_steps(c,o) if phase=='approach'][1:]
        np.testing.assert_array_equal(prefix,full[:len(prefix)])
        np.testing.assert_array_equal(prefix,np.arange(1,105)*.00125)
        self.assertEqual(full[-1],3.)
        self.assertEqual(len(full),2400)
        decimal_prefix=[x for x,_ in loading_depths(c,o.surface_spacing_mm/2,stop_mm=.35)]
        self.assertEqual(len(decimal_prefix),280)
        np.testing.assert_array_equal(decimal_prefix,full[:280])

    def test_noncommensurate_spacing_preserves_endpoints_and_hold_time(self):
        c=replace(FEMConfig(),max_indentation_mm=.317,approach_steps=4,pullback_steps=6)
        o=InitiationOptions(surface_spacing_mm=.007)
        steps=list(resolved_loading_steps(c,o))
        forward=np.array([x for phase,x,_ in steps if phase=='approach'])
        reverse=np.array([c.max_indentation_mm]+[x for phase,x,_ in steps if phase=='pullback'])
        self.assertTrue(np.all(np.diff(forward)>0));self.assertTrue(np.all(np.diff(reverse)<0))
        self.assertLessEqual(np.diff(forward).max(),o.surface_spacing_mm/2+1e-15)
        self.assertLessEqual(-np.diff(reverse).min(),o.surface_spacing_mm/2+1e-15)
        self.assertEqual(forward[-1],c.max_indentation_mm);self.assertEqual(reverse[-1],0.)
        self.assertEqual(sum(phase=='hold' for phase,_,_ in steps),c.hold_steps)


if __name__=='__main__':unittest.main()
