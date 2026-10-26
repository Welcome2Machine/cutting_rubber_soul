import numpy as np
from src.analysis.original_surface_cohort import original_surface_stations


def test_original_free_surface_and_inward_pair_share_reference_address():
    mask=np.zeros((240,360),np.uint8);mask[80:]=1
    cohort=original_surface_stations(mask,[180,80],[1,0],180,half_span=90)
    q=cohort['node_xy']
    assert np.all((q[:,:2,1]>79)&(q[:,:2,1]<86))
    assert np.all((q[:,2:,1]>91)&(q[:,2:,1]<99))
    np.testing.assert_allclose(q[:,2:]-q[:,:2],np.broadcast_to([0,12],q[:,:2].shape))
    assert np.ptp(q[:,:,0],axis=1).max()==0
