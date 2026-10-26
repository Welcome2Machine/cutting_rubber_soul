"""Measurement invariants; no inference/model dependencies."""
import unittest

import numpy as np

from src.analysis.contact_material_mapping import local_material_map, LocalFitGeometry, fit_local


class ContactMaterialMappingTests(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.arange(-60., 61., 12.), np.arange(-60., 61., 12.))
        self.reference = np.column_stack((x.ravel(), y.ravel()))
        self.supported = np.ones(len(self.reference), dtype=bool)

    def test_translation_rotation_and_affine_roundtrip(self):
        theta = 0.4
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for A in (np.eye(2), rotation, rotation @ np.array([[1.25, .12], [0., .8]])):
            with self.subTest(A=A):
                b = np.array([10., -8.])
                current = self.reference @ A.T+b
                query = np.array([3., 4.])
                forward = local_material_map(self.reference, current, self.supported, query, query_space="reference")
                self.assertTrue(forward["valid"], forward["reason"])
                np.testing.assert_allclose(forward["mapped_point"], A @ query+b, atol=1e-9)
                inverse = local_material_map(self.reference, current, self.supported, forward["mapped_point"])
                self.assertTrue(inverse["valid"], inverse["reason"])
                np.testing.assert_allclose(inverse["mapped_point"], query, atol=1e-9)

    def test_unsupported_predictions_do_not_influence_map(self):
        current = self.reference + [8., 4.]
        bad = np.arange(len(current)) % 3 == 0
        current[bad] += [18., 22.]
        result = local_material_map(self.reference, current, ~bad, [8., 4.])
        self.assertTrue(result["valid"], result["reason"])
        np.testing.assert_allclose(result["mapped_point"], [0., 0.], atol=1e-9)
        self.assertFalse(np.any(bad[result["support_indices"]]))
        self.assertFalse(local_material_map(self.reference, current, np.zeros(len(current), bool), [8., 4.])["valid"])

    def test_collapsed_and_reflected_maps_are_rejected(self):
        for A in (np.array([[1., 0.], [0., 0.]]), np.diag([1., -1.])):
            result = local_material_map(self.reference, self.reference @ A.T, self.supported, [0., 0.])
            self.assertFalse(result["valid"])
            self.assertEqual(result["reason"], "noninvertible_or_implausible_local_map")

    def test_controlled_boundary_extrapolation_and_limit(self):
        # Closest actual nodes lie at y=12; the surface at y=0 needs 12 px.
        ref = self.reference[self.reference[:, 1] >= 12]
        supported = np.ones(len(ref), bool)
        at_surface = local_material_map(ref, ref, supported, [0., 0.])
        self.assertTrue(at_surface["valid"], at_surface["reason"])
        self.assertAlmostEqual(at_surface["extrapolation_px"], 12.)
        beyond = local_material_map(ref, ref, supported, [0., -1.])
        self.assertFalse(beyond["valid"])
        self.assertEqual(beyond["reason"], "extrapolation_exceeds_allowance")

    def test_same_branch_selection_prevents_cross_crack_average(self):
        current = self.reference.copy()
        current[self.reference[:, 0] >= 0] += [0., 15.]
        side = {"point": [0., 0.], "normal": [1., 0.], "sign": 1, "space": "reference"}
        result = local_material_map(self.reference, current, self.supported, [0., 0.],
                                    query_space="reference", side=side)
        self.assertTrue(result["valid"], result["reason"])
        np.testing.assert_allclose(result["mapped_point"], [0., 15.], atol=1e-9)

    def test_isolated_bad_track_is_downweighted_and_recorded(self):
        current = self.reference + [3., 5.]
        bad = np.argmin(np.linalg.norm(self.reference-[12., 0.], axis=1))
        current[bad] += [0., 18.]
        result = local_material_map(self.reference, current, self.supported, [0., 0.], query_space="reference")
        self.assertTrue(result["valid"], result["reason"])
        np.testing.assert_allclose(result["mapped_point"], [3., 5.], atol=.05)
        self.assertGreaterEqual(result["outlier_count"], 1)

    def test_skinny_support_rejected_even_if_exact_affine(self):
        reference = self.reference * [1., .001]
        result = local_material_map(reference, reference, self.supported, [0., 0.])
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "nearly_collinear_support")


    def test_inconsistent_nonaffine_patch_fails_heldout_check(self):
        current = self.reference.copy()
        current[:, 1] += .02*self.reference[:, 0]**2
        result = local_material_map(self.reference, current, self.supported, [0., 0.],
                                    query_space="reference", max_fit_residual_px=2., max_heldout_residual_px=2.)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "local_map_residual_too_large")

def points():
    x,y=np.meshgrid(np.linspace(-12,12,5),np.linspace(1.5,19.5,4))
    return np.column_stack([x.ravel(),y.ravel()])

def test_local_map_recovers_rigid_rotation_and_translation():
    p=points();theta=.7;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    q=p@R.T+[30,50]
    fit=fit_local(p,q,np.ones(len(p)),np.zeros(2),np.eye(2),check_cv=True)
    np.testing.assert_allclose(fit['F'],R,atol=1e-10)
    np.testing.assert_allclose(fit['xy'],[30,50],atol=1e-10)


def test_surface_curvature_is_reconstructed_without_projecting_onto_tip():
    p=points();q=np.column_stack([p[:,0]*1.8,p[:,1]+.025*p[:,0]**2])+[7,9]
    fit=fit_local(p,q,np.ones(len(p)),np.zeros(2),np.eye(2),quadratic=True,check_cv=True)
    np.testing.assert_allclose(fit['xy'],[7,9],atol=1e-10)
    np.testing.assert_allclose(fit['F'],[[1.8,0],[0,1]],atol=1e-10)
    assert fit['cv']<1e-9

def test_fit_cache_preserves_results_with_changing_missing_nodes():
    rng=np.random.default_rng(9);ref=rng.normal(size=(24,2))*12
    for quadratic in (False,True):
        geometry=LocalFitGeometry(ref,np.zeros(2),np.eye(2),quadratic)
        for i in range(80):
            q=ref@np.array([[1.8,.3],[-.4,.9]])+[10,20]+rng.normal(0,.2,ref.shape)
            q[i%24]=np.nan;weights=rng.choice([0.,.15,1.],len(ref),p=[.1,.2,.7])
            args=(ref,q,weights,np.zeros(2),np.eye(2))
            a=fit_local(*args,quadratic=quadratic,check_cv=True)
            b=fit_local(*args,quadratic=quadratic,check_cv=True,prepared=geometry)
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            assert len(geometry.ranks)<=64
