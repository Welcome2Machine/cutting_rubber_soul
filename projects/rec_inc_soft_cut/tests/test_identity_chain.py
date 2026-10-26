"""Known-image truth and graph-policy tests, not evidence of real rubber truth."""
import cv2
import numpy as np
import pytest

from src.tracking.identity_chain import InitialRootedIdentityChain, IdentityChainConfig
from src.tracking.material_patch_tracker import verify_material_candidates


def scene():
    rng = np.random.default_rng(51)
    texture = cv2.GaussianBlur(rng.random((180, 220)).astype(np.float32), (3, 3), .7)
    texture = np.clip((texture-.5)*1.8+.5, 0, 1)
    points = np.array([[65., 65.], [105., 100.], [130., 70.]], np.float32)
    return texture, points, np.ones(texture.shape, bool)


def warp(image, affine, offset):
    return cv2.warpAffine(image, np.column_stack((affine, offset)),
                          (image.shape[1], image.shape[0]))


def test_gradual_affine_keeps_original_ids_and_native_positions():
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points, point_ids=[9, 2, 17],
        config=IdentityChainConfig(keyframe_interval=3, root_check_interval=5))
    assert chain.initial_supported.all()
    conditional = False
    for frame in range(1, 16):
        affine = np.array([[1+.012*frame, .002*frame], [-.001*frame, 1-.004*frame]])
        offset = np.array([-.5*frame, .15*frame])
        truth = points @ affine.T + offset
        result = chain.step(warp(image, affine, offset), mask, frame_index=frame,
                            proposals=truth+.15)
        assert result['rooted_supported'].all(), result['state']
        assert np.max(np.linalg.norm(result['positions']-truth, axis=1)) < .5
        np.testing.assert_array_equal(result['point_ids'], [9, 2, 17])
        assert np.all(result['reference_frames'] < frame)
        conditional |= result['rooted_conditional'].any()
    assert conditional
    assert not result['external_affine_prior']


def test_long_gap_preserves_both_pre_gap_references_and_recovers():
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points,
        config=IdentityChainConfig(keyframe_interval=2, root_check_interval=5))
    for frame in range(1, 7):
        result = chain.step(warp(image, np.eye(2), [.3*frame, 0]), mask,
                            frame_index=frame, proposals=points+[.3*frame, 0])
        assert result['rooted_supported'].all()
    before_recent, before_key = chain.recent.age.copy(), chain.keyframe.age.copy()
    for frame in range(7, 30):
        result = chain.step(np.zeros_like(image), np.zeros_like(mask), frame_index=frame,
                            proposals=points+[.3*frame, 0])
        assert not result['rooted_supported'].any()
        assert np.isnan(result['positions']).all()
    np.testing.assert_array_equal(chain.recent.age, before_recent)
    np.testing.assert_array_equal(chain.keyframe.age, before_key)
    result = chain.step(warp(image, np.eye(2), [9., 0]), mask, frame_index=30,
                        proposals=points+[9., 0])
    assert result['rooted_supported'].all()
    assert result['recovered'].all()
    assert np.all(result['gap_before'] == 23)
    assert np.max(np.linalg.norm(result['positions']-points-[9, 0], axis=1)) < .3


def test_one_failed_id_is_not_replaced_as_other_ids_advance():
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points,
        config=IdentityChainConfig(keyframe_interval=2, root_check_interval=5))
    for frame in range(1, 6):
        result = chain.step(image, mask, frame_index=frame, proposals=points)
        assert result['rooted_supported'].all()
    old = (chain.recent.age[0], chain.keyframe.age[0])
    partial = mask.copy()
    partial[35:96, 35:96] = False
    for frame in range(6, 22):
        result = chain.step(image, partial, frame_index=frame, proposals=points)
        assert not result['rooted_supported'][0]
        assert result['rooted_supported'][1:].all()
    assert (chain.recent.age[0], chain.keyframe.age[0]) == old
    result = chain.step(image, mask, frame_index=22, proposals=points)
    assert result['recovered'][0]


def test_textureless_initial_seed_never_acquires_another_identity():
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(np.full_like(image, .5), mask, points)
    assert not chain.initial_supported.any()
    result = chain.step(image, mask, frame_index=1, proposals=points)
    assert not result['rooted_supported'].any()
    assert np.isnan(result['positions']).all()
    assert not chain.recent.trusted.any()


def test_periodic_competing_proposals_remain_unavailable():
    yy, xx = np.indices((160, 180))
    image = (.5+.2*np.sin(xx*2*np.pi/12)+.2*np.cos(yy*2*np.pi/15)).astype(np.float32)
    points = np.array([[70., 75.]], np.float32)
    mask = np.ones(image.shape, bool)
    chain = InitialRootedIdentityChain(image, mask, points)
    result = chain.step(image, mask, frame_index=1,
                        proposals=np.stack((points, points+[12, 0])))
    assert not result['rooted_supported'].any()
    assert chain.recent.age[0] == 0


def test_swapped_proposals_cannot_reassign_ids():
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points, point_ids=[7, 11, 3])
    result = chain.step(image, mask, frame_index=1, proposals=points[::-1])
    np.testing.assert_array_equal(result['point_ids'], [7, 11, 3])
    assert not result['rooted_supported'][[0, 2]].any()
    assert result['rooted_supported'][1]


def test_strict_time_cutoff_and_initial_query_provenance():
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points, point_ids=[2, 8, 1],
                                       frame_index=4, cutoff_frame=6)
    queries = chain.initial_queries()
    np.testing.assert_array_equal(queries['queries_txy'][:, 1:], points)
    np.testing.assert_array_equal(queries['point_ids'], [2, 8, 1])
    assert queries['source_frame_index'] == 4
    for frame in (4, 3, 7, 5.5):
        with pytest.raises(ValueError, match='Strictly increasing'):
            chain.step(image, mask, frame_index=frame)
    chain.step(image, mask, frame_index=5)
    with pytest.raises(ValueError, match='Strictly increasing'):
        chain.step(image, mask, frame_index=5)


def test_cycle_budget_is_an_explicit_diagnostic_gate_not_an_accuracy_bound():
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points,
        config=IdentityChainConfig(maximum_cycle_budget_px=1.e-8, cycle_budget_policy='reject'))
    result = chain.step(warp(image, np.eye(2), [.37, .23]), mask, frame_index=1,
                        proposals=points+[.37, .23])
    assert not result['rooted_supported'].any()
    assert (result['state'] == 'cycle_budget_exceeded').all()
    np.testing.assert_array_equal(chain.recent.age, 0)


def test_bank_storage_is_bounded_with_sequence_length():
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points,
        config=IdentityChainConfig(keyframe_interval=2, root_check_interval=4))
    sizes = [bank.gray_tiles.shape for bank in (chain.root, chain.recent, chain.keyframe)]
    for frame in range(1, 15):
        chain.step(image, mask, frame_index=frame)
    assert sizes == [bank.gray_tiles.shape for bank in (chain.root, chain.recent, chain.keyframe)]


def test_strong_total_compression_can_use_valid_incremental_edges():
    """Direct initial affine is outside its stretch gate; local edges are not."""
    rng = np.random.default_rng(51)
    image = cv2.GaussianBlur(rng.random((180, 220)).astype(np.float32), (0, 0), 1.5)
    image = np.clip((image-.5)*4+.5, 0, 1)
    points = np.array([[65., 65.], [105., 100.], [130., 70.]], np.float32)
    mask = np.ones(image.shape, bool)
    chain = InitialRootedIdentityChain(image, mask, points,
        config=IdentityChainConfig(keyframe_interval=3, root_check_interval=5))
    errors = []
    for frame in range(1, 43):
        affine = np.diag([1-.02*frame, 1.])
        offset = np.array([90*(1-affine[0, 0]), 0])
        truth = points @ affine.T + offset
        target = warp(image, affine, offset)
        result = chain.step(target, mask, frame_index=frame, proposals=truth)
        assert result['rooted_supported'].all()
        errors.extend(np.linalg.norm(result['positions']-truth, axis=1))
        assert result['reference_frames'].shape == (3, 3)
        assert result['reference_failure_flags'].shape == (3, 3)
    direct = verify_material_candidates(image, target, points, truth, mask, mask,
        initial_affine=np.repeat(affine[None], len(points), axis=0), boundary_mode='one-sided')
    assert not direct['accepted'].any()
    assert result['rooted_conditional'].all()
    assert max(errors) < .3


def test_disagreeing_trusted_paths_do_not_update_either_bank(monkeypatch):
    """Fault injection into one previously admitted edge checks the graph gate."""
    from src.tracking import identity_chain as module
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points,
        config=IdentityChainConfig(keyframe_interval=2, root_check_interval=10))
    for frame in range(1, 5):
        assert chain.step(image, mask, frame_index=frame)['rooted_supported'].all()
    before = chain.recent.age.copy(), chain.keyframe.age.copy()
    calls = []
    def faulty_reference(*args, **kwargs):
        result = verify_material_candidates(*args, **kwargs)
        calls.append(True)
        if len(calls) == 2:
            result['positions'] += [4., 0.]
        return result
    monkeypatch.setattr(module, 'verify_material_candidates', faulty_reference)
    result = chain.step(image, mask, frame_index=5)
    assert not result['rooted_supported'].any()
    assert (result['state'] == 'reference_conflict').all()
    np.testing.assert_array_equal(chain.recent.age, before[0])
    np.testing.assert_array_equal(chain.keyframe.age, before[1])


def test_rejected_high_score_reference_cannot_supply_a_supported_position(monkeypatch):
    from src.tracking import identity_chain as module
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points,
        config=IdentityChainConfig(keyframe_interval=2, root_check_interval=6))
    for frame in range(1, 6):
        assert chain.step(image, mask, frame_index=frame)['rooted_supported'].all()
    calls = []
    def ambiguous_root(*args, **kwargs):
        result = verify_material_candidates(*args, **kwargs)
        calls.append(True)
        if len(calls) == 3:
            result['accepted'][:] = False
            result['ambiguous'][:] = True
            result['score'][:] = 100.
            result['positions'] += [15., 0.]
        return result
    monkeypatch.setattr(module, 'verify_material_candidates', ambiguous_root)
    result = chain.step(image, mask, frame_index=6)
    assert result['rooted_conditional'].all()
    assert result['reference_ambiguous'][2].all()
    assert np.max(np.linalg.norm(result['positions']-points, axis=1)) < .01
    assert not (result['selected_reference_frame'] == 0).any()


@pytest.mark.parametrize('kwargs', [
    {'keyframe_interval': .5}, {'root_check_interval': np.inf},
    {'root_check_interval': 0}, {'agreement_px': np.nan},
    {'maximum_cycle_budget_px': -1}, {'maximum_correction_px': 0},
    {'boundary_mode': 'anything'}, {'center_tolerance_px': 2.1},
    {'cycle_budget_policy': 'anything'}])
def test_invalid_config_is_rejected_before_any_images_are_read(kwargs):
    with pytest.raises(ValueError):
        IdentityChainConfig(**kwargs)


def test_singular_reference_only_disables_that_material_id():
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points)
    for bank in (chain.root, chain.keyframe, chain.recent):
        bank.cumulative[0] = 0
    result = chain.step(image, mask, frame_index=1, proposals=points)
    assert not result['rooted_supported'][0]
    assert result['state'][0] == 'map_numerical_unavailable'
    assert result['rooted_supported'][1:].all()
    assert not result['reference_numerical_valid'][:, 0].any()
    assert chain.recent.age[0] == 0


def test_reference_cache_matches_uncached_sequence_with_gap_and_updates(monkeypatch):
    from src.tracking import identity_chain as module
    image, points, mask = scene()
    prepare = module.prepare_verification_image
    sequences = []
    for cached in (False, True):
        monkeypatch.setattr(module, 'prepare_verification_image',
                            prepare if cached else lambda *a, **k: None)
        chain = InitialRootedIdentityChain(image, mask, points,
            config=IdentityChainConfig(keyframe_interval=2, root_check_interval=3))
        results = []
        for frame in range(1, 9):
            offset = np.array([.27*frame, -.13*frame])
            current_mask = mask.copy()
            if frame in (4, 5):
                current_mask[35:96, 35:96] = False
            results.append(chain.step(warp(image, np.eye(2), offset), current_mask,
                frame_index=frame, proposals=points+offset))
        sequences.append(results)
    for plain, cached in zip(*sequences):
        for key, value in plain.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(cached[key], value, err_msg=key)


def test_cycle_budget_is_diagnostic_without_erasing_passing_image_observations():
    image, points, mask = scene()
    chain = InitialRootedIdentityChain(image, mask, points,
        config=IdentityChainConfig(maximum_cycle_budget_px=1.e-8,
                                  keyframe_interval=2, root_check_interval=50))
    warnings = []
    for frame in range(1, 16):
        offset = np.array([.23*frame, .17*frame])
        result = chain.step(warp(image, np.eye(2), offset), mask,
                            frame_index=frame, proposals=points+offset)
        assert result['rooted_supported'].all()
        assert np.max(np.linalg.norm(result['positions']-points-offset, axis=1)) < .3
        assert (chain.recent.age == frame).all()
        assert not (result['state'] == 'cycle_budget_exceeded').any()
        warnings.extend(result['cycle_budget_exceeded'])
    assert all(warnings)
    assert result['cycle_budget_policy'] == 'diagnostic'
    assert result['rooted_conditional'].all()
