import numpy as np
from src.analysis.blade_follow_camera import blade_follow_camera,point_in_display,crop_for_display


def test_camera_ignores_horizontal_tip_noise_and_follows_both_motion_phases():
    y=np.r_[np.linspace(200,650,201),np.linspace(650,210,200)]
    tips=np.column_stack((400+10*np.sin(np.arange(len(y))),y));saved=tips.copy()
    camera=blade_follow_camera(tips,shaft_x=400)
    q=camera['origin_xy']
    assert np.ptp(q[:,0])==0
    assert np.all(np.diff(q[:198,1])>=-1e-8)
    assert np.all(np.diff(q[204:,1])<=1e-8)
    np.testing.assert_array_equal(tips,saved)


def test_isolated_error_and_missing_tips_do_not_cause_camera_jump():
    tips=np.column_stack((np.full(100,450),300+np.arange(100)*.5))
    tips[40,1]=650;tips[60:65]=np.nan
    q=blade_follow_camera(tips,shaft_x=450)['origin_xy']
    assert np.isfinite(q).all()
    assert np.max(abs(np.diff(q[:,1])))<.7
    assert np.any(abs(q[:,1]-np.rint(q[:,1]))>.05)


def test_camera_stays_in_valid_image_region_and_preserves_fractional_mapping():
    tips=np.array([[500,20],[500,750]],float)
    q=blade_follow_camera(tips,shaft_x=500)['origin_xy']
    assert np.all((q[:,1]>=40)&(q[:,1]<=576))
    origin=np.array([12.25,40.375]);point=origin+[127.5,95.5]
    np.testing.assert_allclose(point_in_display(point,origin),[575.5,431.5])
    assert crop_for_display(np.zeros((768,1024,3),np.uint8),origin).shape==(864,1152,3)
