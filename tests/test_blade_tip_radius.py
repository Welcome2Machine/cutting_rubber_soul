"""Independent circle truth, raster recovery and measurement provenance checks."""
import json
from pathlib import Path

import numpy as np
import pytest

from softcut_solution.blade_tip_radius import (
    edge_contour, extract, fit_circle, resample_arc, select_cap, subtract_background,
)


@pytest.mark.parametrize('radius', [2., 4., 15.])
def test_exact_partial_circle_recovers_radius_independent_of_translation(radius):
    angle = np.linspace(.65*np.pi, 1.35*np.pi, 70)
    for offset in (np.array([10., 20.]), np.array([10000., -9000.])):
        points = offset+radius*np.column_stack((np.cos(angle), np.sin(angle)))
        q = fit_circle(points)
        assert q['radius_px'] == pytest.approx(radius, abs=1e-7)
        assert q['centre_px'] == pytest.approx(offset, abs=1e-7)
        assert q['rms_px'] < 1e-7
        assert not q['bound_active']


def test_flat_contour_cannot_identify_radius():
    with pytest.raises(ValueError, match='Collinear'):
        fit_circle(np.column_stack((np.zeros(10), np.arange(10.))))


def test_raster_round_nose_recovers_independent_radius_with_gradient_background():
    y, x = np.mgrid[:90, :80]
    # Analytic signed-distance image, 4 pixel radius, half-disc joining a shaft.
    distance = np.where(x <= 30, np.hypot(x-30, y-45)-4, abs(y-45)-4)
    gray = 40+.1*x+.2*y+80*np.clip(.5-distance, 0, 1)
    contrast = subtract_background(gray, [[25, 30], [60, 65]])
    edge = edge_contour(contrast, [15, 33, 60, 57], 40, [26, 45])
    points = select_cap(resample_arc(edge), [33, 57], 3)
    q = fit_circle(points)
    assert q['radius_px'] == pytest.approx(4, abs=.2)
    assert q['rms_px'] < .15


def test_small_residual_on_nearly_flat_arc_does_not_imply_identified_radius():
    angle = np.linspace(.98*np.pi, 1.02*np.pi, 30)
    q = fit_circle(80*np.column_stack((np.cos(angle), np.sin(angle))))
    assert q['rms_px'] < 1e-7
    assert q['condition'] > 1000
    assert q['angular_coverage_deg'] < 8


def test_hash_mismatch_rejected_before_output_creation(tmp_path):
    source = tmp_path/'changed.png'
    source.write_bytes(b'changed source')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'observations': [{'local_source': 'changed.png', 'source_sha256': 'wrong'}]}))
    with pytest.raises(ValueError, match='Source hash mismatch'):
        extract(tmp_path, manifest, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_final_identity_and_no_force_based_radius_qualification():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'configs/blade_tip_closeups.json').read_text(encoding='utf-8'))
    assert [q['performance'] for q in manifest['observations']] == ['low_performance', 'high_performance']
    assert not manifest['experimental_force_used_for_identification']
    assert not manifest['numerical_geometry_update']
    for q in manifest['observations']:
        assert q['tip_radius_mm'] is None
        assert not q['physical_prediction_qualified']
