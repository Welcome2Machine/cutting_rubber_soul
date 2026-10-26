import numpy as np
from src.analysis.local_blade_mask_repair import LocalBladeMaskRepair


def data():
    tips = np.array([[50., 60.+i] for i in range(7)])
    masks = []
    for i in range(7):
        m = np.ones((100, 100), np.uint8)
        m[:61+i, 45:56] = 2
        masks.append(m)
    return masks, tips


def test_repairs_local_rigid_intersection_without_mutating_or_expanding_material():
    masks, tips = data()
    masks[3][:] = 1
    masks[3][:10,45:56] = 2
    saved = masks[3].copy()
    repair = LocalBladeMaskRepair(lambda i:masks[i], tips, tip_radius_px=25)
    result, estimated = repair.read(3)
    assert estimated
    assert result[63, 50] == 2
    np.testing.assert_array_equal(masks[3], saved)
    assert np.all((result == 1) <= (saved == 1))
    yy, xx = np.indices(result.shape)
    assert np.all(result[((xx-50)**2+(yy-63)**2)>25**2] == saved[((xx-50)**2+(yy-63)**2)>25**2])
    assert repair.diagnostics[3]['previous_source_frame'] == 2
    assert repair.diagnostics[3]['next_source_frame'] == 4


def test_preserves_good_mask_and_does_not_repair_without_two_neighbors():
    masks, tips = data()
    repair = LocalBladeMaskRepair(lambda i:masks[i], tips)
    result, estimated = repair.read(3)
    assert not estimated
    np.testing.assert_array_equal(result, masks[3])
    masks[0][:] = 1
    repair = LocalBladeMaskRepair(lambda i:masks[i], tips)
    result, estimated = repair.read(0)
    assert not estimated
    assert repair.diagnostics[0]['reason'] == 'two_reliable_original_neighbors_unavailable'


def test_unsupported_tip_and_distant_neighbors_are_not_repaired():
    masks, tips = data()
    masks[3][:] = 1
    valid = np.ones(7, bool)
    valid[3] = False
    repair = LocalBladeMaskRepair(lambda i:masks[i], tips, valid)
    assert not repair.read(3)[1]
    masks[2][:] = masks[4][:] = 1
    repair = LocalBladeMaskRepair(lambda i:masks[i], tips, max_neighbor_frames=1)
    assert not repair.read(3)[1]


def test_disagreeing_neighbor_masks_do_not_generate_filled_union():
    masks, tips = data()
    masks[3][:] = 1
    masks[2][:] = 1
    masks[4][:] = 1
    masks[2][:63,44:49] = 2
    masks[4][:65,52:57] = 2
    repair = LocalBladeMaskRepair(lambda i:masks[i], tips, max_neighbor_frames=1)
    result, estimated = repair.read(3)
    assert not estimated
    assert not (result == 2).any()
    assert repair.diagnostics[3]['reason'] == 'aligned_neighbor_intersection_insufficient'


def test_repaired_masks_are_never_used_as_neighbor_evidence():
    masks, tips = data()
    for i in (2,3,4):
        masks[i][:] = 1
    repair = LocalBladeMaskRepair(lambda i:masks[i], tips, max_neighbor_frames=3)
    for i in (2,3,4):
        assert repair.read(i)[1]
        assert repair.diagnostics[i]['previous_source_frame'] == 1
        assert repair.diagnostics[i]['next_source_frame'] == 5
