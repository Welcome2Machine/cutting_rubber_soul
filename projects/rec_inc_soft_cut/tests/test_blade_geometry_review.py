import numpy as np

from src.analysis.blade_geometry_review import detect_blade, fill_short_gaps, isolated_temporal_outliers


def test_watermark_larger_than_blade_is_not_tip():
    mask = np.zeros((400, 600), np.uint8)
    mask[40:260, 190:210] = 2
    mask[40:100, 300:590] = 2
    tip, source, _ = detect_blade(mask, np.zeros_like(mask), 200, 20)
    assert source == 1
    np.testing.assert_allclose(tip, [199.5, 259])


def test_watermark_only_is_missing_not_false_blade():
    mask = np.zeros((400, 600), np.uint8)
    mask[40:82, 238:590] = 2
    tip, source, _ = detect_blade(mask, np.zeros_like(mask), 200, 30)
    assert source == 0
    assert np.isnan(tip).all()


def test_real_candidate_outside_old_tip_is_accepted():
    mask = np.zeros((400, 600), np.uint8)
    mask[40:260, 235:255] = 2
    tip, source, _ = detect_blade(mask, np.zeros_like(mask), 200, 20)
    assert source == 1
    np.testing.assert_allclose(tip, [244.5, 259])


def test_raw_bright_shaft_recovers_missing_semantic_component():
    mask = np.zeros((400, 600), np.uint8)
    mask[40:82, 238:590] = 2
    gray = np.full_like(mask, 20)
    gray[0:250, 190:210] = 220
    tip, source, _ = detect_blade(mask, gray, 200, 20)
    assert source == 2
    np.testing.assert_allclose(tip, [199.5, 249])


def test_raw_top_connected_shaft_wins_over_longer_disconnected_streak():
    mask=np.zeros((768,1024),np.uint8);gray=np.full_like(mask,20)
    gray[:105,330:346]=220
    gray[180:470,330:346]=220
    tip,source,_=detect_blade(mask,gray,338,16)
    assert source==2
    assert tip[1]==104


def test_recorded_vertical_grid_cannot_extend_tip_to_image_bottom():
    mask = np.zeros((768, 1024), np.uint8)
    gray = np.full_like(mask, 40)
    gray[:420, 490:514] = 220
    gray[:, 511] = 180
    tip, source, _ = detect_blade(mask, gray, 505, 37)
    assert source == 2
    assert tip[1] == 419


def test_recorded_grid_through_narrow_apex_does_not_erase_tip():
    mask=np.zeros((768,1024),np.uint8)
    gray=np.full_like(mask,40)
    gray[:90,337:346]=220
    gray[90:102,340:343]=220
    gray[:,341]=85
    tip,source,_=detect_blade(mask,gray,341,10)
    assert source==2
    np.testing.assert_allclose(tip,[341,101])


def test_image_edge_truncated_blade_has_no_observable_tip():
    mask = np.zeros((400, 600), np.uint8)
    mask[40:, 190:210] = 2
    tip, source, _ = detect_blade(mask, np.zeros_like(mask), 200, 20)
    assert source == 0
    assert np.isnan(tip).all()


def test_semantic_extension_into_dark_slit_is_checked_against_image():
    mask = np.zeros((400, 600), np.uint8)
    mask[40:295, 190:210] = 2
    gray = np.full_like(mask, 20)
    gray[:250, 190:210] = 220
    tip, source, _ = detect_blade(mask, gray, 200, 20)
    assert source == 2
    assert tip[1] == 249


def test_semantic_truncation_or_raw_glare_is_explicitly_unsupported():
    mask = np.zeros((400, 600), np.uint8)
    mask[40:240, 190:210] = 2
    gray = np.full_like(mask, 20)
    gray[:280, 190:210] = 220
    tip, source, diagnostics = detect_blade(mask, gray, 200, 20)
    assert source == 0
    assert np.isnan(tip).all()
    assert diagnostics['reason'] == 'ambiguous_semantic_image_endpoint'


def test_temporal_rejection_does_not_enforce_monotonicity():
    tips = np.array([[2,0],[2,3],[2,8],[2,10],[2,8],[2,3],[2,0]], float)
    assert not isolated_temporal_outliers(tips, np.ones(len(tips),bool)).any()
    moved = np.array([[2,0],[2,0],[40,20],[40,21],[40,19]],float)
    assert not isolated_temporal_outliers(moved,np.ones(len(moved),bool)).any()


def test_short_gap_flag_does_not_turn_into_support():
    tips = np.array([[0,0],[np.nan,np.nan],[2,2],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan],[6,6]])
    support = np.isfinite(tips).all(axis=1)
    filled, interpolated = fill_short_gaps(tips,support)
    np.testing.assert_allclose(filled[1],[1,1])
    assert interpolated.tolist() == [False,True,False,False,False,False,False]
    assert not support[1]
    assert np.isnan(filled[3:6]).all()
def test_initial_shaft_short_wide_false_positive_cannot_outvote_real_blade():
    import numpy as np
    from src.analysis.blade_geometry_review import estimate_initial_shaft_adaptive
    masks=[]
    for i in range(20):
        m=np.zeros((768,1024),np.uint8)
        if i<8:m[40:87,405:474]=2
        else:m[40:140,332:348]=2
        masks.append(m)
    center,width,_=estimate_initial_shaft_adaptive(list(range(len(masks))),masks.__getitem__)
    assert center==339.5
    assert width==16


from src.analysis.blade_geometry_review import estimate_initial_shaft_adaptive
import pytest

def shaft_mask(center=None, width=16):
    result=np.zeros((240,640),np.uint8)
    if center is not None:
        result[25:150,center-width//2:center+width//2]=2
    return result


def test_early_calibration_retains_exact_original_votes():
    frames={i:shaft_mask(300+i//32) for i in range(0,160,8)}
    x,w,review=estimate_initial_shaft_adaptive(list(frames),frames.__getitem__)
    assert (x,w)==(300.5,16.)
    assert review['mode']=='original_early_window'


def test_late_entry_uses_first_stable_run_not_later_motion():
    frames={i:shaft_mask(None if i<144 else (300 if i<184 else 400)) for i in range(0,320,8)}
    x,w,review=estimate_initial_shaft_adaptive(list(frames),frames.__getitem__)
    assert (x,w)==(299.5,16.)
    assert [v['frame'] for v in review['witnesses']]==[144,152,160,168,176]
    assert review['contact_assigned_from_shaft_entry'] is False


def test_isolated_false_component_and_missing_frame_do_not_establish_corridor():
    frames={i:shaft_mask() for i in range(0,240,8)}
    for i in (128,136,144,152): frames[i]=shaft_mask(200)
    for i in (168,176,184,192,200): frames[i]=shaft_mask(330)
    x,_,review=estimate_initial_shaft_adaptive(list(frames),frames.__getitem__)
    assert x==329.5
    assert review['witnesses'][0]['frame']==168


def test_unstable_or_absent_candidates_fail_without_inventing_corridor():
    for frames in (
        {i:shaft_mask() for i in range(0,240,8)},
        {i:shaft_mask(None if i<120 else (220 if (i//8)%2 else 320)) for i in range(0,240,8)},
    ):
        with pytest.raises(ValueError,match='no stable observed entry'):
            estimate_initial_shaft_adaptive(list(frames),frames.__getitem__)


from src.analysis.blade_geometry_review import ContactGeometryConfig, audit_blade_contact


def contact_sequence(count=60, semantic_offset=0):
    frames, images, masks, tips = np.arange(count), {}, {}, []
    for i in frames:
        tip_y = 190+i
        gray = np.full((360, 400), 20, np.uint8)
        gray[:tip_y+1, 190:210] = 220
        mask = np.zeros_like(gray)
        mask[235:] = 1
        mask[40:tip_y+semantic_offset+1, 190:210] = 2
        images[i], masks[i] = gray, mask
        tips.append([199.5, tip_y])
    return frames, images, masks, np.asarray(tips)


def run_contact_audit(sequence, **kwargs):
    frames, images, masks, tips = sequence
    if kwargs.get('material_displacement') is not None and kwargs.get('material_supported') is not None:
        kwargs.setdefault('times_s', frames/12.)
    return audit_blade_contact(frames, tips, images.__getitem__, masks.__getitem__,
                               shaft_x=200, shaft_width=20,
                               config=ContactGeometryConfig(grid_enabled=False), **kwargs)


def test_geometry_proximity_alone_cannot_assign_contact_or_modify_tip():
    sequence = contact_sequence()
    original = sequence[-1].copy()
    result, arrays = run_contact_audit(sequence)
    assert result['geometry']['available']
    assert result['geometry']['candidate_frame'] == 43
    assert result['state'] == 'unresolved'
    assert not result['contact_observed']
    assert not result['depth_modified'] and not result['tip_modified']
    np.testing.assert_array_equal(sequence[-1], original)
    np.testing.assert_allclose(arrays['tip_agreement_px'], 0)


def test_independent_image_apex_exposes_semantic_overrun():
    result, arrays = run_contact_audit(contact_sequence(semantic_offset=15))
    np.testing.assert_allclose(arrays['tip_agreement_px'], 15)
    assert not arrays['geometric_qualified'].any()
    assert not result['geometry']['available']
    assert result['state'] == 'unresolved'


def test_registered_tip_bias_is_reported_not_corrected():
    frames, images, masks, tips = contact_sequence()
    tips[:, 1] += 25
    result, arrays = run_contact_audit((frames, images, masks, tips))
    np.testing.assert_allclose(arrays['registered_image_delta_px'][:, 1], 25)
    assert not result['tip_modified']
    assert result['geometry']['available']


def test_tip_disagreement_does_not_suppress_separate_motion_evidence():
    sequence = contact_sequence(semantic_offset=15)
    delta = np.zeros((60, 20, 2))
    delta[:, :, 1] = np.maximum(np.arange(60)-40, 0)[:, None]*.3
    result, _ = run_contact_audit(sequence, material_displacement=delta,
                                 material_supported=np.ones((60, 20), bool))
    assert result['material_response']['accepted']
    assert not result['geometry']['available']
    assert result['state'] == 'unresolved'


def test_already_contacting_window_is_left_censored():
    frames, images, masks, tips = contact_sequence()
    result, _ = run_contact_audit((frames[42:], images, masks, tips[42:]))
    assert result['geometry']['left_censored']
    assert result['state'] == 'unresolved'


def test_support_loss_does_not_change_material_response_cohort():
    sequence = contact_sequence()
    delta = np.zeros((60, 20, 2))
    delta[:, :5, 1] = 20
    supported = np.ones((60, 20), bool)
    supported[30:, :5] = False
    result, arrays = run_contact_audit(sequence, material_displacement=delta,
                                      material_supported=supported)
    np.testing.assert_allclose(arrays['material_displacement_px'], 0)
    assert len(result['material_response']['fixed_common_cohort_indices']) == 15
    assert not result['material_response']['accepted']


def test_material_motion_can_cross_check_geometry_without_forcing_zero():
    sequence = contact_sequence()
    delta = np.zeros((60, 20, 2))
    delta[:, :, 1] = np.maximum(np.arange(60)-40, 0)[:, None]*.3
    result, _ = run_contact_audit(sequence, material_displacement=delta,
                                 material_supported=np.ones((60, 20), bool))
    assert result['material_response']['accepted']
    assert result['material_response']['frame'] == 40
    assert result['state'] == 'cross_evidence_candidate'
    assert not result['contact_observed']
    assert not result['depth_modified']


def test_contact_audit_rejects_nondense_frames_and_unpaired_motion():
    frames, images, masks, tips = contact_sequence()
    with pytest.raises(ValueError, match='contiguous'):
        run_contact_audit((frames[::2], images, masks, tips[::2]))
    with pytest.raises(ValueError, match='together'):
        run_contact_audit((frames, images, masks, tips), material_displacement=np.zeros((60, 20, 2)))
    with pytest.raises(ValueError, match='exceed'):
        ContactGeometryConfig(clear_distance_px=1)
