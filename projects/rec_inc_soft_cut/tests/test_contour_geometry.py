import cv2
import numpy as np

from src.analysis.contour_geometry import ContourGeometryTracker, _clear


def material_scene(edge=70, mask_edge=None):
    image = np.zeros((180, 240, 3), np.uint8)
    image[edge:] = 155
    labels = np.zeros(image.shape[:2], np.uint8)
    labels[edge if mask_edge is None else mask_edge:] = 1
    return image, labels


def test_image_edge_improves_coarse_segmentation_without_making_material_ids():
    image, labels = material_scene(edge=70, mask_edge=74)
    result = ContourGeometryTracker().estimate(image, labels, np.array([120., 69.]), np.array([1., 0.]))
    middle = result['path_xy'][(result['path_xy'][:, 0] > 30) & (result['path_xy'][:, 0] < 210)]
    assert abs(np.median(middle[:, 1])-69.5) < 1.
    assert result['tip_gap_px'] < result['raw_tip_gap_px']
    assert result['contact_state'] == 2
    assert not result['diagnostics']['contact_is_material_identity']
    assert result['diagnostics']['refined_normal_gradient_mean'] > result['diagnostics']['raw_normal_gradient_mean']


def test_complete_outline_survives_textureless_scene_and_marks_frame_clipping():
    image, labels = material_scene()
    image[:] = 70
    result = ContourGeometryTracker().estimate(image, labels, [120., 69.], [1., 0.])
    closed = result['closed_xy']
    assert closed[:, 0].min() == 0 and closed[:, 0].max() == 239
    assert closed[:, 1].max() == 179
    assert np.any(result['closed_state'] == 0)
    assert np.any(result['closed_state'] == 1)
    assert not np.any(result['closed_state'] == 2)
    assert np.all(result['path_state'] != 0)
    assert len(closed) > 50


def test_distant_tip_is_not_forced_into_contact():
    image, labels = material_scene()
    result = ContourGeometryTracker().estimate(image, labels, [120., 25.], [1., 0.])
    assert result['contact_state'] == 0
    assert result['tip_gap_px'] > 40.
    assert np.linalg.norm(result['contact_xy']-[120., 25.]) > 40.


def test_blade_notch_is_preserved_without_any_curve_crossing():
    image, labels = material_scene(edge=40)
    knife = np.array([[104, 0], [136, 0], [121, 120], [118, 120]], np.int32)
    cv2.fillPoly(labels, [knife], 2)
    image[labels == 2] = 245
    result = ContourGeometryTracker().estimate(image, labels, [119.5, 120.], [1., 0.])
    assert result['diagnostics']['blade_crossing_edges'] == 0
    assert np.any(result['closed_state'] == 3)
    for a, b in zip(result['closed_xy'], np.roll(result['closed_xy'], -1, axis=0)):
        assert _clear(labels == 2, a, b)
    assert result['tip_gap_px'] < 4.


def test_empty_material_and_invalid_tip_do_not_invent_a_contour():
    image, labels = material_scene()
    labels[:] = 0
    result = ContourGeometryTracker().estimate(image, labels, [120., 50.], [1., 0.])
    assert len(result['closed_xy']) == 0 and result['contact_state'] == 0


def test_temporal_prediction_is_optional_and_does_not_require_surviving_identity_everywhere():
    tracker = ContourGeometryTracker()
    image, labels = material_scene(edge=70)
    tracker.estimate(image, labels, [120., 69.], [1., 0.])
    xx, yy = np.meshgrid(np.arange(10., 240., 20.), np.arange(72., 170., 20.))
    previous = np.column_stack((xx.ravel(), yy.ravel()))
    current = previous + [0., 3.]
    image2, labels2 = material_scene(edge=73)
    support = np.ones(len(current), bool); support[::2] = False
    result = tracker.estimate(image2, labels2, [120., 72.], [1., 0.], previous, current, support)
    assert result['diagnostics']['temporal_predicted_stations'] > 0
    assert len(result['closed_xy']) > 50
    assert result['tip_gap_px'] < 2.


def test_flow_prediction_cannot_move_an_outline_without_image_evidence():
    tracker = ContourGeometryTracker()
    image, labels = material_scene(edge=70)
    image[:] = 70
    tracker.estimate(image, labels, [120., 69.], [1., 0.])
    xx, yy = np.meshgrid(np.arange(10., 240., 20.), np.arange(72., 170., 20.))
    previous = np.column_stack((xx.ravel(), yy.ravel()))
    result = tracker.estimate(image, labels, [120., 69.], [1., 0.], previous, previous+[0., 6.],
                              np.ones(len(previous), bool))
    assert result['diagnostics']['temporal_predicted_stations'] > 0
    assert result['diagnostics']['image_corrected_stations'] == 0
    assert result['diagnostics']['visible_correction_max_px'] == 0.
