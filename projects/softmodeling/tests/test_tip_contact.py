"""Synthetic geometry checks; no native IPC package or simulation is needed."""

import json
from types import SimpleNamespace
import unittest

import numpy as np

from prototype.tip_contact import TipContactTracker, _capsule_intervals


def fixture(angle=0.0, dhat=0.01, grid=(0., 1., 2., 3., 4.), facets=64):
    config = SimpleNamespace(inclination_deg=angle, height_mm=1., length_mm=4., drop_s_mm=2.)
    theta = np.deg2rad(angle)
    es, en = np.array([np.cos(theta), -np.sin(theta)]), np.array([np.sin(theta), np.cos(theta)])
    body = np.array([s * es + n * en for s in grid for n in (0., 1.)])
    triangles = np.array([t for i in range(len(grid) - 1)
                          for t in ((2*i, 2*i+2, 2*i+1), (2*i+2, 2*i+3, 2*i+1))])
    radius = .2
    start = -2 * dhat / max(np.cos(theta), .1)
    center = 2 * es + (1 + radius) * en - np.array([0., start])
    arc_theta = np.linspace(-np.pi + .2, -.2, facets + 1)
    arc = center + radius * np.column_stack((np.cos(arc_theta), np.sin(arc_theta)))
    blade = np.vstack((center + [-.5, 3.], center + [-.5, 1.], arc,
                       center + [.5, 1.], center + [.5, 3.]))
    reference = np.vstack((body, blade))
    tracker = TipContactTracker(reference, triangles, len(body), config, dhat, facets)
    return tracker, reference, triangles, len(body), config


def place_pole(tracker, reference, body_nodes, point):
    x = reference.copy()
    x[body_nodes:] += np.asarray(point) - x[tracker.tip_node]
    return x



def prescribed_address_observations():
    """Translate a circular nose across a stretched material surface, then rotate the entire scene."""
    rows=[]
    samples=[1.71,1.94,2.,2.07,2.34]
    for h in (.25,.125,.0625):
        tracker,ref,_,n,_=fixture(grid=tuple(np.arange(0.,4.+h/2,h)),dhat=.01)
        stretch=1.25;gap=.003;radius=.2
        true_half=np.sqrt((radius+.01)**2-(radius+gap)**2)/stretch
        for theta in (0.,.58,.99):
            Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            for s in samples:
                x=ref.copy();x[:n,0]=stretch*ref[:n,0]+.3
                x=place_pole(tracker,x,n,[stretch*s+.3,1+gap])
                x=x@Q.T + [1.4,-.7]
                contact=tracker.observe(x,.1)
                native=dict(edge_nodes=contact['edge_nodes'],edge_alpha=contact['edge_alpha'],
                    nose_arc_mm=tracker.anchor_arc,tip_contact_point_mm=contact['current_point_mm'],
                    blade_point_mm=x[tracker.tip_node].tolist(),gap_mm=gap)
                sa=tracker.sliding_amount(x,contact,[native])
                intervals=sa['SA_intervals_mm'];expected=s-2.
                assert len(intervals)==1 and sa['status']=='native_nose_contact'
                rows.append(dict(h_mm=h,rotation_rad=theta,reference_S_mm=s,expected_SA_mm=expected,
                    observed_SA_mm=sa['SA_signed_mm'],observed_interval_mm=intervals[0],
                    ideal_circle_interval_mm=[expected-true_half,expected+true_half],
                    address_error_um=1000*abs(sa['SA_signed_mm']-expected),
                    set_Hausdorff_um=1000*max(abs(np.array(intervals[0])-[expected-true_half,expected+true_half]))))
    return rows

class CapsuleIntervalTests(unittest.TestCase):
    def test_manufactured_address_and_circle_interval_cross_element_boundaries(self):
        rows=prescribed_address_observations()
        self.assertLess(max(r['address_error_um'] for r in rows),1.e-8)
        self.assertLess(max(r['set_Hausdorff_um'] for r in rows),2.)


    def test_reference_integrals_and_landing_marker_survive_redistribution(self):
        for grid in ((0., 1., 2., 3., 4.), (0., .2, 1.1, 1.8, 1.85, 2.7, 3.4, 4.)):
            tracker, ref, _, n, _ = fixture(grid=grid)
            x = ref.copy()
            x[:n, 0] += .3*tracker.s+.2
            values = tracker.fixed_material_motion(x, tracker.fixed_material_definition())
            self.assertAlmostEqual(values['reference_arclength_mean']['left_near']['u_s_mm'], .65)
            self.assertAlmostEqual(values['fixed_offset_points']['0.0']['u_s_mm'], .8)

    def test_support_evidence_preserves_disconnected_branches_and_same_material_velocity(self):
        tracker, ref, _, n, _ = fixture()
        x = place_pole(tracker, ref, n, [2.3, 1.005])
        contact = tracker.observe(x, .1)
        row = dict(edge_nodes=contact['edge_nodes'], edge_alpha=contact['edge_alpha'],
                   tip_contact_point_mm=contact['current_point_mm'], nose_arc_mm=tracker.anchor_arc, gap_mm=.005)
        sa = tracker.sliding_amount(x, contact, [row])
        sa['S_tip_intervals_mm'] = [[1., 1.2], [2.2, 2.4]]
        sa['branch_id'] = 1
        previous = x.copy()
        previous[:n, 0] -= .05
        evidence = tracker.nose_support_evidence(sa, [row], x, previous, .5, [[.9, 2.5]])
        self.assertEqual([b['parent_indices'] for b in evidence['branches']], [[0], [0]])
        self.assertEqual(evidence['branches'][0]['native_S_mm'], [])
        np.testing.assert_allclose(evidence['branches'][1]['native_S_mm'], [2.3])
        np.testing.assert_allclose(evidence['selected_material']['material_velocity_mm_s'], [.1, 0])

    def test_pair_tangent_follows_original_edge_through_vertical(self):
        tracker, ref, _, n, _ = fixture()
        prior = None
        for degrees in (89.,91.):
            phi = np.deg2rad(degrees)
            edge = np.array([np.cos(phi), np.sin(phi)])
            center = np.array([2.,1.])
            nodes = tracker.top_node_edge[5]
            x = ref.copy()
            x[nodes] = center + np.array([[-.5],[.5]])*edge
            x = place_pole(tracker, x, n, center + .005*np.array([-edge[1], edge[0]]))
            old = x.copy()
            old[:n] -= [.01,.03]
            row = dict(edge_nodes=nodes.tolist(), edge_alpha=.5, nose_arc_mm=tracker.anchor_arc,
                       tip_contact_point_mm=center.tolist(), blade_point_mm=x[tracker.tip_node].tolist())
            sa = dict(S_tip_intervals_mm=[[2.,3.]], selected=row, S_tip_mm=2.5, branch_id=0)
            evidence = tracker.nose_support_evidence(sa, [row], x, old, .5, [[2.,3.]], prior)
            sample = evidence['selected_material']
            np.testing.assert_allclose(sample['pair_tangent'], edge, atol=1e-12)
            np.testing.assert_allclose(sample['relative_body_minus_tool_mm_s'], [.02,.06], atol=1e-12)
            self.assertEqual(evidence['selected_branch_transition'], 'continued' if prior else 'acquired')
            prior = evidence
        switched = tracker.nose_support_evidence(sa, [row], x, old, .5, [[1.,1.5]], prior)
        self.assertEqual(switched['selected_branch_transition'], 'switched')

    def test_idealized_blades_use_the_full_outline_and_fixed_pole(self):
        from dataclasses import replace
        from prototype.model_config import FEMConfig, _rotation_vectors
        from prototype.geometry import GeometryOptions, rounded_outline
        for angle in (0.,33.,57.):
            cfg=replace(FEMConfig(),inclination_deg=angle,blade_tip_angle_deg=13.31684633759535,
                        blade_thickness_mm=.24577088241513723)
            es,en=_rotation_vectors(angle)
            for radius in (.002,.020):
                shapes={}
                for shape in ('double','single_up','single_down'):
                    contour,marks=rounded_outline(cfg,GeometryOptions(tip_radius_mm=radius),0.,64,shape)
                    shapes[shape]=contour
                    self.assertAlmostEqual(np.ptp(contour[:,0]),cfg.blade_thickness_mm,places=10)
                    self.assertAlmostEqual((contour@en).min(),cfg.height_mm,places=10)
                    self.assertAlmostEqual(contour[np.argmin(contour@en)]@es,cfg.drop_s_mm,places=9)
                    self.assertEqual(marks['pole_local'],int(np.argmin(contour[:,1])))
                    center=contour[marks['pole_local']]+[0,radius]
                    np.testing.assert_allclose(np.linalg.norm(contour[2:67]-center,axis=1),radius,atol=1e-11)
                    for i,j in (marks['left_flank'],marks['right_flank']):
                        nose=contour[j if i==1 else i]
                        self.assertAlmostEqual((nose-center)@(contour[j]-contour[i]),0.,places=11)
                axis=shapes['double'][np.argmin(shapes['double'][:,1]),0]
                mirror=shapes['single_up'][::-1].copy()
                mirror[:,0]=2*axis-mirror[:,0]
                np.testing.assert_allclose(mirror,shapes['single_down'],atol=1e-12)


    def test_straight_blade_subdivision_preserves_shape_and_contact_identity(self):
        from prototype.geometry import subdivide_blade
        tracker,reference,triangles,n,config=fixture()
        blade,edges=subdivide_blade(reference[n:],64,.01)
        np.testing.assert_array_equal(blade[:69],reference[n:])
        np.testing.assert_array_equal(np.bincount(edges.ravel()),np.full(len(blade),2))
        old_length=np.linalg.norm(np.roll(reference[n:],-1,axis=0)-reference[n:],axis=1).sum()
        new_length=np.linalg.norm(blade[edges[:,1]]-blade[edges[:,0]],axis=1).sum()
        self.assertAlmostEqual(old_length,new_length,places=12)
        refined=np.vstack((reference[:n],blade))
        other=TipContactTracker(refined,triangles,n,config,.01,64)
        a=place_pole(tracker,reference,n,[2.,1.005])
        b=place_pole(other,refined,n,[2.,1.005])
        self.assertEqual(tracker.observe(a,0.),other.observe(b,0.))
        self.assertEqual(tracker.observe_footprint(a,0.),other.observe_footprint(b,0.))
        invalid=refined.copy();invalid[-1]+=[.1,.2]
        with self.assertRaisesRegex(ValueError,'Extra blade nodes'):
            TipContactTracker(invalid,triangles,n,config,.01,64)

    def test_nose_address_uses_reference_length_and_fixed_landing(self):
        tr,ref,_,n,_ = fixture()
        x = place_pole(tr,ref,n,[2.3,1.005])
        pole = tr.observe(x,.1)
        row = dict(edge_nodes=pole['edge_nodes'],edge_alpha=pole['edge_alpha'],
                   nose_arc_mm=tr.anchor_arc,gap_mm=.005)
        sa = tr.sliding_amount(x,pole,[row])
        self.assertAlmostEqual(sa['S_tip_mm'],2.3)
        self.assertAlmostEqual(sa['SA_signed_mm'],.3)
        stretched=x.copy()
        stretched[:,0]*=1.25
        other=tr.sliding_amount(stretched,tr.observe(stretched,.1),[row])
        self.assertAlmostEqual(other['SA_signed_mm'],.3)
        self.assertIsNone(tr.sliding_amount(x,pole)['SA_signed_mm'])
        self.assertIsNotNone(tr.sliding_amount(x,pole)['proximity_estimate'])

    def test_disjoint_support_has_no_scalar_and_preserves_signed_ranges(self):
        tr,x,_,_,_=fixture()
        contact=dict(nose_reference_intervals_mm=[[1.8,1.9],[2.1,2.2]],
                     nose_offset_intervals_mm=[[-.2,-.1],[.1,.2]])
        sa=tr.sliding_amount(x,contact)
        self.assertIsNone(sa['S_tip_mm'])
        self.assertEqual(sa['status'],'ambiguous_material_branches')
        self.assertEqual(sa['SA_intervals_mm'],contact['nose_offset_intervals_mm'])

    def test_small_contact_in_middle_of_long_edge(self):
        a, b = np.array([-10., .05]), np.array([10., .05])
        c, d = np.array([-.1, 0.]), np.array([.1, 0.])
        result = _capsule_intervals(a, b, c, d, .1)
        extent = .1 + np.sqrt(.1**2 - .05**2)
        np.testing.assert_allclose(result, [[(10-extent)/20, (10+extent)/20]], atol=1e-14)

    def test_strict_threshold_and_isolated_tangency(self):
        for a, b in (([-1., .1], [1., .1]), ([-1., -.2], [-1., .2])):
            radius = .1 if a[1] == .1 else 1.
            self.assertEqual(_capsule_intervals(np.array(a), np.array(b), np.array([0., 0.]),
                                               np.array([1., 0.]), radius), [])

    def test_reversal_and_oblique_endpoint_regions(self):
        a, b, c, d = map(np.array, ([-1., -.3], [1., .7], [-.3, 0.], [.2, .1]))
        expected = _capsule_intervals(a, b, c, d, .2)
        reverse = _capsule_intervals(b, a, d, c, .2)
        np.testing.assert_allclose(expected, [[1-hi, 1-lo] for lo, hi in reversed(reverse)], atol=1e-14)
        u = np.linspace(0, 1, 10001)
        p, w = a + u[:, None]*(b-a), d-c
        t = np.clip((p-c) @ w / (w@w), 0, 1)
        distance = np.linalg.norm(p-c-t[:, None]*w, axis=1)
        inside = distance < .2
        inferred = np.zeros(len(u), dtype=bool)
        for lo, hi in expected:
            inferred |= (u > lo) & (u < hi)
        away_from_roundoff = abs(distance - .2) > 1e-12
        np.testing.assert_array_equal(inside[away_from_roundoff], inferred[away_from_roundoff])


class TipContactTests(unittest.TestCase):
    def test_no_contact_is_null_not_nearest_projection(self):
        tracker, reference, _, _, _ = fixture()
        result = tracker.observe(reference, -.02)
        self.assertEqual(result['status'], 'no_tip_contact')
        self.assertIsNone(result['reference_offset_mm'])
        self.assertEqual(result['nose_status'], 'no_nose_contact')
        self.assertTrue(result['nearest_projection_candidates'])
        json.dumps(result, allow_nan=False)
        json.dumps(tracker.metadata, allow_nan=False)

    def test_shared_vertex_is_one_material_point(self):
        tracker, reference, _, n, _ = fixture()
        result = tracker.observe(place_pole(tracker, reference, n, [2., 1.005]), .1)
        self.assertEqual(result['status'], 'unique')
        self.assertEqual(len(result['candidates']), 1)
        self.assertAlmostEqual(result['reference_offset_mm'], 0.)
        self.assertEqual(result['nose_status'], 'connected')
        lo, hi = result['nose_offset_intervals_mm'][0]
        self.assertAlmostEqual(lo, -hi)
        self.assertGreater(hi, 0.)
        self.assertLess(hi, .2)

    def test_interpolation_uses_reference_arclength_after_stretch_and_shear(self):
        tracker, reference, _, n, _ = fixture(grid=(0., 4.))
        x = place_pole(tracker, reference, n, [9.8, 1.005])
        x[:n, 0] = 2 * reference[:n, 0] + 5 * reference[:n, 1]
        result = tracker.observe(x, 1.)
        self.assertEqual(result['status'], 'unique')
        self.assertAlmostEqual(result['edge_alpha'], .6)
        self.assertAlmostEqual(result['reference_offset_mm'], .4)
        np.testing.assert_allclose(result['reference_point_mm'], [2.4, 1.])
        np.testing.assert_allclose(result['current_point_mm'], [9.8, 1.])

    def test_current_rigid_transform_preserves_material_identity(self):
        tracker, reference, _, n, _ = fixture()
        x = place_pole(tracker, reference, n, [2.3, 1.005])
        before = tracker.observe(x, 1.)
        angle = 1.7  # Old lowest point is no longer current global z-minimum.
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        y = x @ rotation.T + np.array([4., -7.])
        after = tracker.observe(y, 1.)
        self.assertEqual(after['status'], 'unique')
        self.assertAlmostEqual(before['reference_offset_mm'], after['reference_offset_mm'])
        self.assertAlmostEqual(before['gap_mm'], after['gap_mm'])
        np.testing.assert_allclose(before['nose_reference_intervals_mm'], after['nose_reference_intervals_mm'], atol=1e-10)
        np.testing.assert_allclose(after['knife_tip_mm'], y[tracker.tip_node])

    def test_distinct_material_branches_are_ambiguous(self):
        tracker, reference, _, n, _ = fixture(dhat=.02)
        x = place_pole(tracker, reference, n, [0., 0.])
        # Two vertical branches project to different material labels at the
        # same positive minimum distance; other edges stay farther away.
        x[np.arange(1, n, 2)] = [[-.01, 1.], [-.01, -1.], [.01, -1.], [.01, 1.], [1., 2.]]
        result = tracker.observe(x, 1.)
        self.assertEqual(result['status'], 'ambiguous')
        self.assertIsNone(result['reference_offset_mm'])
        self.assertEqual(len(result['candidates']), 2)
        self.assertEqual(result['nose_status'], 'disconnected')
        self.assertEqual(len(result['nose_reference_intervals_mm']), 2)

    def test_threshold_equality_does_not_create_contact(self):
        tracker, reference, _, n, _ = fixture(dhat=.125)
        result = tracker.observe(place_pole(tracker, reference, n, [2., 1.125]), 0.)
        self.assertEqual(result['status'], 'no_tip_contact')
        self.assertEqual(result['nose_status'], 'no_nose_contact')

    def test_gap_crossing_retains_diagnostic_address_without_extending_migration(self):
        tracker, reference, _, n, _ = fixture(dhat=.125)
        for gap, status in ((.124, 'unique'), (.125, 'no_tip_contact'), (.126, 'no_tip_contact')):
            with self.subTest(gap=gap):
                observed = tracker.observe(place_pole(tracker, reference, n, [2.3, 1.+gap]), 0.)
                diagnostic = tracker.pole_diagnostics(observed)
                self.assertEqual(observed['status'], status)
                self.assertEqual(diagnostic['nearest_is_migration'], status == 'unique')
                self.assertAlmostEqual(diagnostic['nearest_projections'][0]['reference_s_mm'], 2.3)
                self.assertAlmostEqual(diagnostic['gap_minus_dhat_mm'], gap-.125)
                if status != 'unique':
                    self.assertIsNone(observed['reference_offset_mm'])
                    self.assertEqual(diagnostic['reason'], 'gap_not_below_dhat')

    def test_reordered_triangle_edges_do_not_change_result(self):
        tracker, reference, triangles, n, config = fixture()
        alternative = TipContactTracker(reference, triangles[::-1, ::-1], n, config, .01, 64)
        x = place_pole(tracker, reference, n, [2.3, 1.005])
        self.assertEqual(tracker.observe(x, .1), alternative.observe(x, .1))

    def test_oblique_first_contact_is_on_arc_not_pole(self):
        tracker, reference, _, n, config = fixture(angle=60.)
        rest_depth = -2 * tracker.dhat / np.cos(np.deg2rad(60.))
        zero = reference.copy()
        zero[n:, 1] += rest_depth
        result = tracker.observe(zero, 0.)
        self.assertEqual(result['status'], 'no_tip_contact')
        self.assertEqual(result['nose_status'], 'connected')
        self.assertAlmostEqual(result['nearest_projection_candidates'][0]['reference_offset_mm'], .2*np.sin(np.deg2rad(60.)))
        lo, hi = tracker.metadata['faceted_first_contact_s_interval_mm']
        self.assertLess(abs((lo+hi)/2-config.drop_s_mm), .02)
        self.assertGreater(tracker.metadata['faceted_first_contact_depth_mm'], 0.)

    def test_invalid_inputs_and_flat_punch_rejected(self):
        tracker, reference, triangles, n, config = fixture()
        with self.assertRaisesRegex(ValueError, 'topology'):
            TipContactTracker(reference[:n+4], triangles, n, config, .01, 64)
        with self.assertRaises(ValueError):
            tracker.observe(reference, float('nan'))
        x = reference.copy()
        x[tracker.edges[0, 1]] = x[tracker.edges[0, 0]]
        with self.assertRaisesRegex(ValueError, 'Degenerate current top'):
            tracker.observe(x, .1)
        x = reference.copy()
        x[0, 0] = float('inf')
        with self.assertRaises(ValueError):
            tracker.observe(x, .1)


class BladeFootprintTests(unittest.TestCase):
    def test_no_support_has_empty_intervals_and_zero_lengths(self):
        tracker, reference, _, _, _ = fixture()
        result = tracker.observe_footprint(reference, -.02)
        self.assertEqual(result['status'], 'no_support')
        self.assertEqual(result['reference_s_intervals_mm'], [])
        self.assertEqual(result['supported_reference_length_mm'], 0.)
        self.assertEqual(result['supported_current_length_mm'], 0.)
        self.assertEqual(result['supporting_top_edge_count'], 0)
        json.dumps(result, allow_nan=False)

    def test_nose_only_matches_existing_observable(self):
        tracker, reference, _, n, _ = fixture(grid=(0., 4.))
        x = place_pole(tracker, reference, n, [2., 1.005])
        before = tracker.observe(x, .1)
        result = tracker.observe_footprint(x, .1)
        self.assertEqual(before, tracker.observe(x, .1))
        self.assertEqual(result['status'], 'connected')
        np.testing.assert_allclose(result['reference_s_intervals_mm'], before['nose_reference_intervals_mm'])
        np.testing.assert_allclose(result['parts']['nose']['reference_s_intervals_mm'], before['nose_reference_intervals_mm'])
        self.assertAlmostEqual(result['supported_current_length_mm'], result['supported_reference_length_mm'])
        # Neither endpoint of this long material edge is near the nose.
        self.assertGreater(result['supported_current_length_mm'], 0.)
        self.assertLess(result['supported_current_length_mm'], .4)

    def test_shank_support_is_detected_without_lowest_tip_support(self):
        tracker, reference, _, n, _ = fixture(grid=(0., 4.))
        x = reference.copy()
        shoulder, upper = x[n + 1], x[n]
        midpoint = .5 * (shoulder + upper) + [-.005, 0.]
        x[1], x[3] = midpoint + [0., -.1], midpoint + [0., .1]
        result = tracker.observe_footprint(x, .1)
        self.assertEqual(tracker.observe(x, .1)['status'], 'no_tip_contact')
        self.assertEqual(result['parts']['left_shank']['status'], 'connected')
        self.assertEqual(result['parts']['nose']['status'], 'no_support')
        np.testing.assert_allclose(result['reference_s_intervals_mm'], [[0., 4.]])
        self.assertAlmostEqual(result['supported_reference_length_mm'], 4.)
        self.assertAlmostEqual(result['supported_current_length_mm'], .2)

    def test_shared_shoulder_part_overlap_is_counted_once_in_total(self):
        tracker, reference, _, n, _ = fixture(grid=(0., 4.))
        x = reference.copy()
        shoulder = x[n + 1].copy()
        x[1], x[3] = shoulder + [-.005, -.006], shoulder + [-.005, .006]
        result = tracker.observe_footprint(x, .1)
        self.assertGreater(result['parts']['left_shank']['supported_current_length_mm'], 0.)
        self.assertGreater(result['parts']['left_flank']['supported_current_length_mm'], 0.)
        self.assertGreater(sum(v['supported_current_length_mm'] for v in result['parts'].values()),
                           result['supported_current_length_mm'])
        self.assertAlmostEqual(result['supported_current_length_mm'], .012)
        self.assertAlmostEqual(result['supported_reference_length_mm'], 4.)

    def test_current_rigid_transform_preserves_footprint_and_lengths(self):
        tracker, reference, _, n, _ = fixture()
        x = place_pole(tracker, reference, n, [2.3, 1.005])
        before = tracker.observe_footprint(x, .1)
        angle = 1.7
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        after = tracker.observe_footprint(x @ rotation.T + [4., -7.], .1)
        np.testing.assert_allclose(before['reference_s_intervals_mm'], after['reference_s_intervals_mm'], atol=1e-12)
        self.assertAlmostEqual(before['supported_current_length_mm'], after['supported_current_length_mm'])

    def test_threshold_and_degenerate_blade_guard(self):
        tracker, reference, _, n, _ = fixture(dhat=.125)
        x = place_pole(tracker, reference, n, [2., 1.125])
        self.assertEqual(tracker.observe_footprint(x, .1)['status'], 'no_support')
        x[n] = x[n + 1]
        with self.assertRaisesRegex(ValueError, 'Degenerate current blade'):
            tracker.observe_footprint(x, .1)


if __name__ == '__main__':
    unittest.main()

class MaterialIdentityMotionTests(unittest.TestCase):
    def test_translation_known_sliding_and_affine_stretch(self):
        tracker, reference, _, n, _ = fixture()
        old = place_pole(tracker, reference, n, [1.3, 1.005])
        translated = old+[.2, -.1]
        result = tracker.relative_motion(translated, old, tracker.observe(translated, .1), .5)
        self.assertAlmostEqual(result['raw_mm_s'], 0., places=12)
        self.assertAlmostEqual(result['transported_mm_s'], 0., places=12)
        sliding = old.copy();sliding[n:, 0] += .1
        result = tracker.relative_motion(sliding, old, tracker.observe(sliding, .1), .5)
        self.assertAlmostEqual(result['raw_mm_s'], .2, places=12)
        stretched = old.copy();stretched[:n, 0] *= 1.1
        contact = tracker.observe(stretched, .1)
        result = tracker.relative_motion(stretched, old, contact, .5)
        expected = -.1*contact['reference_s_mm']/.5
        self.assertAlmostEqual(result['raw_mm_s'], expected, places=12)
        self.assertAlmostEqual(result['local_edge_stretch'], 1.1)

    def test_finite_gap_corotation_raw_artifact_and_objective_transport(self):
        tracker, reference, _, n, _ = fixture()
        old = place_pole(tracker, reference, n, [1.3, 1.005])
        theta = .1
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        current = old@R.T+[.3, -.2]
        result = tracker.relative_motion(current, old, tracker.observe(current, .1), .2)
        self.assertTrue(result['valid'])
        self.assertGreater(abs(result['raw_mm_s']), 1e-4)
        self.assertAlmostEqual(result['transported_mm_s'], 0., places=11)

class FastProjectionCrossingTests(unittest.TestCase):
    def test_multiple_adjacent_material_edges_do_not_reduce_cumulative_motion(self):
        from prototype.contact_trace import accumulate_motion
        tracker,reference,_,n,_=fixture()
        old=place_pole(tracker,reference,n,[.3,1.005])
        current=place_pole(tracker,reference,n,[2.3,1.005])
        a,b=tracker.observe(old,0.),tracker.observe(current,.1)
        self.assertFalse(set(a['edge_nodes']).intersection(b['edge_nodes']))
        continuity=tracker.projection_continuity(old,current,a,b)
        self.assertTrue(continuity['continuous'])
        self.assertTrue(continuity['intermediate_geometry_only'])
        motion=tracker.relative_motion(current,old,b,1.)
        cumulative=dict(segment_id=0,signed=0.,absolute=0.)
        result=accumulate_motion(tracker,motion,b,a,dict(valid=True),cumulative,1.,continuity=continuity)
        self.assertIsNone(result['break_reason'])
        self.assertAlmostEqual(result['signed_segment_mm'],2.)
        self.assertAlmostEqual(result['absolute_segment_mm'],2.)

    def test_intermediate_gap_is_not_healed_by_valid_endpoints(self):
        from unittest.mock import patch
        tracker,reference,_,n,_=fixture()
        old=place_pole(tracker,reference,n,[.3,1.005]);current=place_pole(tracker,reference,n,[2.3,1.005])
        a,b=tracker.observe(old,0.),tracker.observe(current,.1)
        missing=dict(status='no_tip_contact',edge_nodes=None,reference_s_mm=None,gap_mm=.1)
        with patch.object(tracker,'observe',return_value=missing):
            continuity=tracker.projection_continuity(old,current,a,b)
        self.assertFalse(continuity['continuous'])
        self.assertEqual(continuity['status'],'chord_gap_or_ambiguous_projection')
