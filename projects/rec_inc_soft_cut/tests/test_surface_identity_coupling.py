from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest

from src.analysis.surface_identity_coupling import (
    couple_origin_profile, conservative_address, couple_joint_transport_profiles,
)


def fixture(stretch=1.6,contact=15.):
    s,d=np.meshgrid(np.arange(-30.,61.,6.),np.arange(2.,42.,6.))
    s,d=s.ravel(),d.ravel()
    mapping=SimpleNamespace(s=s,d=d,addresses=np.array([-100.,100.]))
    beta=np.array([-stretch*contact,60.*stretch,12.,0.])
    values=stretch*(s-contact)+.2*d
    profile=dict(reference_s_px=contact,estimated_surface_stretch=stretch,
        low_s_px=contact-5.,high_s_px=contact+5.,rms_px=.5,heldout_px=1.,
        support_indices=np.arange(len(s)),model='linear_depth',origin_s_px=0.,beta=beta)
    return profile,mapping,values


def test_current_arc_is_converted_by_stretch_not_called_reference_slip():
    p,m,r=fixture()
    result=couple_origin_profile(p,m,r,0.,-24.,1.,1,0.)
    assert result['origin_used']
    assert abs(result['reference_s_px']-15.)<1.e-10
    assert result['reference_s_px']!=24.


def test_stationary_origin_and_zero_reference_contact_gives_zero_slip():
    p,m,r=fixture(contact=0.)
    result=couple_origin_profile(p,m,r,0.,0.,1.,1,0.)
    assert abs(result['reference_s_px'])<1.e-10


def test_long_gap_prediction_cannot_become_origin_observation():
    p,m,r=fixture()
    result=couple_origin_profile(p,m,r,0.,-60.,1.,4,0.)
    assert not result['origin_used']
    assert result['reference_s_px']==p['reference_s_px']


def test_contradictory_origin_does_not_force_material_map():
    p,m,r=fixture()
    result=couple_origin_profile(p,m,r,0.,-100.,1.,1,0.)
    assert not result['origin_used']
    assert result['reference_s_px']==15.


def test_repeated_correlated_models_do_not_shrink_uncertainty():
    mean,sigma=conservative_address([(10.,3.)]*20)
    assert mean==10. and abs(sigma-3.)<1.e-12


def test_duplicate_one_of_two_models_cannot_change_center_or_uncertainty():
    a=conservative_address([(-10.,3.),(10.,3.)])
    b=conservative_address([(-10.,3.)]+[(10.,3.)]*20)
    assert a==b and a[0] in (-10.,10.) and a[1]>=23.


def test_nonfinite_correspondence_cannot_pass_residual_gate():
    p,m,r=fixture();r[0]=np.nan
    result=couple_origin_profile(p,m,r,0.,-24.,1.,1,0.)
    assert not result['origin_used']


def test_other_boundary_side_cannot_supply_origin_anchor():
    p,m,r=fixture();p['side']=1
    result=couple_origin_profile(p,m,r,0.,-24.,1.,1,0.)
    assert not result['origin_used']


def motion_profile(index, *, contact=15., depth=36., score=1., position=True,
                   velocity=True, normal=True):
    p, mapping, arc = fixture(contact=contact)
    p.update(side=-1, depth_limit_px=depth, score=score, contact_compatible=position,
             support_ids=np.arange(len(mapping.s))+index*1000)
    flux = bool(velocity and normal)  # A usable flux need not meet stricter absolute-position gates.
    absolute = bool(position and flux)
    model = dict(position_profile=p, position_profile_index=index,
        position_profile_id=f'boundary:{index}', support_ids=p['support_ids'],
        position_absolute_compatible=position, velocity_fit_valid=velocity,
        normal_compatible=normal, flux_valid=flux, absolute_valid=absolute,
        joint_absolute_valid=absolute, reasons=[])
    return model, mapping, arc


def test_same_side_good_position_and_good_velocity_cannot_authorize_each_other():
    good_position, mapping, arc = motion_profile(0, score=.1, position=True, velocity=False)
    good_velocity, _, _ = motion_profile(1, score=1., position=False, velocity=True)
    # The aggregate side deliberately appears usable. Only the individual
    # pair may authorize an absolute update, and neither pair qualifies.
    record = dict(models=[[good_position, good_velocity], []],
                  sides=[dict(flux_valid=True, absolute_valid=True), {}], absolute_valid=True)
    result = couple_joint_transport_profiles(record, mapping, arc, 0., -24., 1., 1, 0.)
    assert not result['absolute_valid']
    assert result['selected'] == []
    assert not result['origin_used']
    assert all(not p['qualified_for_selection'] for p in result['profiles'])


def test_same_model_different_depths_keep_their_own_normal_qualification():
    bad, mapping, arc = motion_profile(0, contact=9., depth=18., score=.1, normal=False)
    good, _, _ = motion_profile(1, contact=15., depth=60., score=2.)
    assert bad['position_profile']['model'] == good['position_profile']['model']
    record = dict(models=[[bad, good], []], sides=[dict(flux_valid=True), {}])
    result = couple_joint_transport_profiles(record, mapping, arc, 0., -24., 1., 4, 0.)
    assert result['absolute_valid']
    assert result['chosen_position_profile_id'] == 'boundary:1'
    assert result['selected_position_profile_ids'] == ['boundary:1']
    assert result['absolute_s_px'] == 15.
    selected = result['selected'][0]
    assert selected['depth_limit_px'] == 60.
    assert selected['absolute_valid'] and selected['normal_compatible'] and selected['velocity_fit_valid']
    assert np.array_equal(selected['support_ids'], good['position_profile']['support_ids'])


def test_unqualified_geometry_cannot_fall_back_to_frame_absolute_summary():
    model, mapping, arc = motion_profile(0)
    result = couple_joint_transport_profiles(
        dict(models=[[model], []], absolute_valid=True, absolute_s_px=15., absolute_sigma_px=5.),
        mapping, arc, 0., -24., 1., 1, 0., geometry_qualified=False)
    assert not result['absolute_valid'] and result['selected'] == []
    assert not np.isfinite(result['absolute_s_px'])


def test_coupler_requires_exact_position_profiles_with_unique_frame_ids():
    model, mapping, arc = motion_profile(0)
    duplicate = deepcopy(model)
    duplicate['position_profile']['depth_limit_px'] = 60.
    with pytest.raises(ValueError, match='Duplicate'):
        couple_joint_transport_profiles(dict(models=[[model, duplicate], []]), mapping, arc, 0., -24., 1., 1, 0.)
    with pytest.raises(ValueError, match='return_models=True'):
        couple_joint_transport_profiles(dict(sides=[]), mapping, arc, 0., -24., 1., 1, 0.)
