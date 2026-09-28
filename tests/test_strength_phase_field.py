"""Independent, explicitly synthetic checks of the elastic strength phase core."""
import unittest
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from prototype.material import response, from_config
from mechanics.finite_strain import principal
from prototype.fracture_criteria import stress_measures
from prototype.model_config import FEMConfig
from prototype.strength_phase_field import (StrengthCalibration, ElasticPhaseState,
                                           StrengthPhaseField, calibration_from_config, elastic_state)


class StrengthPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Synthetic strength pair for algorithm verification, NOT CB245NE.
        # At the real catalogue 4.4 MPa the current material is incompatible
        # with a constant-S2 surface (there is an earlier strength crossing).
        cls.config = FEMConfig(tensile_strength_n_mm2=.4)
        cls.calibration, cls.details = calibration_from_config(
            cls.config, hydrostatic_second_pk_mpa=2.,
            provenance='synthetic nominal .4 MPa / hydrostatic S2 2 MPa; not CB245NE data',
            parameter_status='synthetic')

    def make(self, ell=.04, h=0.):
        return StrengthPhaseField([[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
                                  [[0, 1, 2], [0, 2, 3]], 20., 2.15, ell,
                                  calibration=self.calibration, mesh_size_mm=h)

    def repeated(self, F):
        return elastic_state(np.repeat(np.asarray(F, float)[None], 2, axis=0), self.config)

    def free_uniaxial(self, axial):
        material = from_config(self.config)
        lateral = brentq(lambda q: principal(np.array([[np.log(axial), q, q]]), material)[1][0, 1],
                         -np.log(axial)-1., 1., xtol=1e-13)
        return np.diag([axial, np.exp(lateral), np.exp(lateral)])

    def test_explicit_calibration_and_consistent_stress_measures(self):
        c, details = self.calibration, self.details
        uniaxial = np.diag(details['uniaxial_stretches'])
        state = self.repeated(uniaxial)
        np.testing.assert_allclose(state.second_pk_mpa[0], np.diag([c.uniaxial_second_pk_mpa, 0., 0.]), atol=1e-10)
        self.assertAlmostEqual(state.energy_mpa[0], c.uniaxial_energy_mpa, places=11)
        self.assertAlmostEqual(c.uniaxial_second_pk_mpa*uniaxial[0, 0], .4, places=10)
        hydro = self.repeated(details['hydrostatic_stretch']*np.eye(3))
        np.testing.assert_allclose(hydro.second_pk_mpa[0], 2.*np.eye(3), atol=1e-10)
        self.assertAlmostEqual(hydro.energy_mpa[0], c.hydrostatic_energy_mpa, places=11)
        self.assertGreater(c.uniaxial_energy_mpa, 0.)

    def test_catalogue_nominal_strength_is_not_silently_mapped_to_wrong_branch(self):
        with self.assertRaisesRegex(ValueError, 'first stretch=1.9378.*requested stretch=7.8625'):
            calibration_from_config(FEMConfig(), hydrostatic_second_pk_mpa=2.,
                                    provenance='synthetic hs, catalogue nominal strength compatibility audit',
                                    parameter_status='assumed')
        axial = np.linspace(1., 8., 141)
        path = elastic_state(np.array([self.free_uniaxial(a) for a in axial]), self.config)
        target = self.repeated(self.free_uniaxial(7.86250225272)).second_pk_mpa[0, 0, 0]
        crossed = np.flatnonzero(path.second_pk_mpa[:, 0, 0] > target)
        self.assertLess(axial[crossed[0]], 2.)
        self.assertLess(path.second_pk_mpa[-1, 0, 0], target)

    def test_arbitrarily_close_post_peak_targets_are_rejected(self):
        # Independent scalar maximization checks the analytic-tangent guard,
        # including targets closer to the peak than any coarse stress scan.
        def second(q):
            return self.repeated(self.free_uniaxial(np.exp(q))).second_pk_mpa[0, 0, 0]
        maximum = minimize_scalar(lambda q: -second(q), bounds=(.8, 1.8),
                                  method='bounded', options={'xatol': 1e-12}).x
        for offset in (1e-4, 1e-3, .003, .01):
            stretch = np.exp(maximum+offset)
            config = FEMConfig(tensile_strength_n_mm2=second(maximum+offset)*stretch)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'earlier crossing'):
                calibration_from_config(config, hydrostatic_second_pk_mpa=2.,
                                        provenance='synthetic post-peak nominal target', parameter_status='synthetic')

    def test_missing_strength_provenance_and_invalid_data_rejected(self):
        with self.assertRaises(TypeError):
            calibration_from_config(self.config, provenance='x', parameter_status='assumed')
        for change in ({'provenance': ''}, {'hydrostatic_second_pk_mpa': 0.},
                       {'parameter_status': 'calibrated'}, {'uniaxial_energy_mpa': float('nan')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                StrengthCalibration(**{**self.calibration.to_dict(), **change})
        with self.assertRaises(ValueError):
            self.make(h=float('nan'))

    def test_constitutive_adapter_matches_existing_plane_strain(self):
        F = np.array([[[1.3, .2], [.1, .9]], [[.8, .03], [.05, 1.1]]])
        q = response(F, self.config)
        full = np.tile(np.eye(3), (2, 1, 1)); full[:, :2, :2] = F
        _, second = stress_measures(full, q['cauchy'])
        state = elastic_state(F, self.config)
        np.testing.assert_allclose(state.energy_mpa, q['energy'], atol=1e-12)
        np.testing.assert_allclose(state.second_pk_mpa, second, atol=1e-11)

    def test_exact_tensile_thresholds_for_multiple_lengths_and_mesh_corrections(self):
        axial = np.diag(self.details['uniaxial_stretches'])
        hydro = self.details['hydrostatic_stretch']*np.eye(3)
        for ell in (.01, .04, .1):
            for h in (0., ell/5, ell):
                p = self.make(ell, h)
                for F, mode in ((axial, 'uniaxial'), (hydro, 'hydrostatic')):
                    with self.subTest(ell=ell, h=h, F=F.diagonal()):
                        residual = p.terms(np.zeros(4), self.repeated(F))[1]/p.lumped
                        np.testing.assert_allclose(residual, 0., atol=5e-9)
                        if mode == 'uniaxial':
                            below = self.free_uniaxial(.999*F[0, 0])
                            above = self.free_uniaxial(1.001*F[0, 0])
                        else:
                            below = np.eye(3)+.999*(F-np.eye(3))
                            above = np.eye(3)+1.001*(F-np.eye(3))
                        self.assertGreater(p.terms(np.zeros(4), self.repeated(below))[1].min(), 0.)
                        self.assertLess(p.terms(np.zeros(4), self.repeated(above))[1].max(), 0.)

    def test_hydrostatic_compression_does_not_damage(self):
        p = self.make()
        for stretch in (.99, .9, .7):
            state = self.repeated(stretch*np.eye(3))
            d, info = p.solve(state)
            np.testing.assert_array_equal(d, 0.)
            self.assertEqual(info['kkt_residual_mpa'], 0.)

    def test_overlarge_length_compression_failure_is_explicit(self):
        # A distinct synthetic calibration exposes beta1>0 at large ell.
        calibration = StrengthCalibration(.55, 2., 17., .06,
                                          'synthetic compression-applicability counterexample', 'synthetic')
        with self.assertRaisesRegex(ValueError, 'hydrostatic-compression'):
            StrengthPhaseField([[0., 0.], [1., 0.], [0., 1.]], [[0, 1, 2]],
                               1., 2.15, 10., calibration=calibration)

    def test_objectivity_under_superposed_rigid_rotation(self):
        F = np.array([[[1.3, .2, 0.], [.1, .9, 0.], [0., 0., 1.]],
                      [[.8, .03, 0.], [.05, 1.1, 0.], [0., 0., 1.]]])
        a = .7; Q = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
        first = elastic_state(F, self.config); rotated = elastic_state(Q@F, self.config)
        np.testing.assert_allclose(first.energy_mpa, rotated.energy_mpa, atol=1e-12)
        np.testing.assert_allclose(first.second_pk_mpa, rotated.second_pk_mpa, atol=1e-11)
        p = self.make(); d = np.array([.1, .3, .4, .2])
        np.testing.assert_allclose(p.terms(d, first)[1], p.terms(d, rotated)[1], atol=1e-10)

    def test_fixed_deformation_potential_gradient_and_tangent(self):
        p = self.make(h=.008)
        state = elastic_state(np.array([[[1.3, .2], [.1, .9]], [[.8, .03], [.05, 1.1]]]), self.config)
        d = np.array([.1, .3, .4, .2]); v = np.array([.3, -.2, .1, .4]); eps = 1e-5
        E, r, H = p.terms(d, state)
        plus = p.terms(d+eps*v, state); minus = p.terms(d-eps*v, state)
        np.testing.assert_allclose((plus[0]-minus[0])/(2*eps), r@v, rtol=1e-9, atol=2e-7)
        np.testing.assert_allclose((plus[1]-minus[1])/(2*eps), H@v, atol=3e-7, rtol=1e-8)
        np.testing.assert_allclose(H.toarray(), H.toarray().T, atol=1e-12)
        self.assertAlmostEqual(p.terms(np.zeros(4), state)[0], 0., places=12)
        with self.assertRaisesRegex(NotImplementedError, 'physical'):
            p.energy(d, state)

    def test_fresh_damage_dependent_source_matches_full_equation(self):
        p = self.make(); d = np.array([.1, .3, .4, .2])
        state = self.repeated(np.diag([1.3, .9, 1.]))
        z = 1-d[p.tri]@p.shape.T
        S = z[:, :, None, None]**2*state.second_pk_mpa[:, None]
        trace = np.trace(S, axis1=2, axis2=3)
        dev = S-trace[:, :, None, None]*np.eye(3)/3
        chat = (p.beta2*np.sqrt(.5*np.sum(dev*dev, axis=(2, 3)))+p.beta1*trace
                +z*(1-np.sign(trace))*state.energy_mpa[:, None])
        np.testing.assert_allclose(p.driving_force(d, state)['chat_mpa'], chat, atol=1e-12)

    def test_irreversibility_and_bounded_solver(self):
        p = self.make(); F = 1.015*self.details['hydrostatic_stretch']*np.eye(3)
        state = self.repeated(F)
        d, info = p.solve(state)
        self.assertGreater(d.min(), 0.)
        self.assertLessEqual(d.max(), 1.)
        self.assertLessEqual(info['kkt_residual_mpa'], 1e-6)
        old = d.copy()
        unloaded = self.repeated(np.eye(3))
        d, _ = p.solve(unloaded, old=old)
        np.testing.assert_array_equal(d, old)
        np.testing.assert_allclose(old, old[0], atol=1e-8)
        with self.assertRaisesRegex(ValueError, 'healing'):
            p.kkt_residual(old/2, unloaded, old)

    def test_zero_trace_extension_and_fully_broken_source_are_finite(self):
        p = self.make()
        state = ElasticPhaseState(np.array([1., 0.]), np.array([np.diag([1., -1., 0.]), np.zeros((3, 3))]))
        for d in (np.zeros(4), np.ones(4)):
            E, r, H = p.terms(d, state)
            self.assertTrue(np.isfinite(E) and np.isfinite(r).all() and np.isfinite(H.data).all())
        np.testing.assert_allclose(p.driving_force(np.ones(4), state)['chat_mpa'], 0., atol=1e-12)


if __name__ == '__main__':
    unittest.main()
