"""Prepare native-pixel loading prefixes or complete imported videos.

Only blade calibration is sampled coarsely over the complete video. Every
frame in the tracking prefix has its own temporal segmentation and normalized
image. Sparse geometry interpolations are explicitly distinguished from image
observations; they are not material-motion measurements.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from src.io import ROOT, atomic_write_json, sha as _sha
from src.video_sources import video_directory, validate_case_names, input_profile
from src.analysis.blade_geometry_review import detect_blade, estimate_initial_shaft_adaptive, isolated_temporal_outliers
from src.analysis.original_surface_cohort import original_surface_stations, _labels_at

SEG = ROOT / "models/segmentation/clean_roi_20260909/checkpoint_best.pt"


def _save(path, value):
    atomic_write_json(Path(path), value, allow_nan=False)




def load_segmentation_session(device="cuda"):
    """Load the existing segmentation model once; no training or downloading."""
    import torch
    from src.segmentation import build_model
    torch.set_num_threads(4)
    payload = torch.load(SEG, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    config["preprocessing"]["grid_coordinate_space"] = "source_0915"
    model = build_model(config["model"]).to(device).eval()
    model.load_state_dict(payload["state_dict"])
    return dict(model=model, device=torch.device(device), config=config)


def session_for_input(session, profile):
    """Per-input preprocessing; never mutate a model shared with another case."""
    result = dict(session, config=copy.deepcopy(session['config']))
    preprocessing = result['config']['preprocessing']
    preprocessing.setdefault('grid_suppression', {})['enabled'] = profile['grid_enabled']
    preprocessing.setdefault('overlay_redaction', {})['enabled'] = profile['overlay_enabled']
    return result


def _segment_selected(video, indices, mask_dir, session, *, reuse=False):
    """Same temporal inputs/postprocessing as production, without review encode."""
    import torch
    from src.segmentation import SourceFrameCache, SegmentationFeatureCache
    from src.segmentation import conservative_postprocess, temporal_blade_cleanup
    from src.segmentation import refine_material_boundary
    mask_dir.mkdir(parents=True, exist_ok=True)
    indices = sorted(set(map(int, indices)))
    pending = [i for i in indices if not reuse or not (mask_dir / f"f{i:06d}.png").exists()]
    if not pending:
        return
    model, device, config = (session[k] for k in ("model", "device", "config"))
    cfg = config["inference"]
    model_cfg = config["model"]
    radius = int(model_cfg["sequence_length"]) // 2
    offsets = np.arange(-radius, radius + 1) * int(model_cfg["frame_stride"])
    reader = SourceFrameCache(video, cache_size=64)
    features = SegmentationFeatureCache(reader, tuple(model_cfg["input_size"]), config["preprocessing"])
    crop_top = int(config["preprocessing"].get("source_crop_top_px", 0))
    previous = None
    previous_index = -100
    processed = 0
    try:
        for number, centre in enumerate(indices):
            if reuse and (mask_dir / f"f{centre:06d}.png").exists():
                previous = _mask(mask_dir, centre)
                previous_index = centre
                continue
            reader.prefetch_through(centre + int(offsets[-1]))
            current = reader.read(centre)
            tensor = torch.from_numpy(np.stack([features.read(centre + int(d)) for d in offsets])[None]).to(device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                small = torch.softmax(model(tensor), dim=1)[0].float().cpu().numpy()
            probability = np.stack([cv2.resize(channel, (reader.width, reader.height - crop_top)) for channel in small])
            probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-8)
            if crop_top:
                full = np.zeros((3, reader.height, reader.width), probability.dtype)
                full[0, :crop_top] = 1
                full[:, crop_top:] = probability
                probability = full
            mask = conservative_postprocess(probability.argmax(axis=0).astype(np.uint8),
                int(cfg.get("max_material_hole_area", 700)), int(cfg.get("min_blade_component_area", 36)),
                int(cfg.get("min_isolated_material_area", 2500)), float(cfg.get("min_material_hole_aspect_ratio", 3)))
            # The production temporal cleanup assumes adjacent centre frames.
            # Do not pretend an eight-frame gap is one frame.
            mask = temporal_blade_cleanup(mask, previous if centre == previous_index + 1 else None,
                int(cfg.get("min_blade_component_area", 36)), int(cfg.get("blade_temporal_max_gap_px", 18)))
            previous, previous_index = mask.copy(), centre
            if crop_top:
                mask[:crop_top] = 255
            refinement = cfg.get("material_boundary_refinement", {})
            if refinement.get("enabled", False):
                mask, _ = refine_material_boundary(current, mask,
                    band_radius=int(refinement.get("band_radius_px", 6)),
                    protected_radius=int(refinement.get("protected_radius_px", 8)),
                    iterations=int(refinement.get("iterations", 3)))
            if not cv2.imwrite(str(mask_dir / f"f{centre:06d}.png"), mask):
                raise RuntimeError("Cannot save semantic mask")
            processed += 1
            if processed % 100 == 1 or processed == len(pending):
                print(f"prepare segmentation {Path(video).stem}: {processed}/{len(pending)} centres", flush=True)
    finally:
        reader.close()


def _mask(mask_dir, index):
    value = cv2.imread(str(mask_dir / f"f{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if value is None or value.shape != (768, 1024):
        raise ValueError(f"Missing native semantic mask {index}")
    return value


def _geometry_samples(video, indices, mask_dir, shaft, width, reference=None, guides=None,
                      *, require_visible_template=False, grid_enabled=True):
    from src.analysis.rigid_blade_reference import RigidBladeReference
    indices = sorted(set(map(int, indices)))
    tips = np.full((len(indices), 2), np.nan)
    sources = np.zeros(len(indices), np.uint8)
    gaps = np.full(len(indices), np.nan)
    scores = np.full(len(indices), np.nan)
    registration = RigidBladeReference(*reference, grid_enabled=grid_enabled) if reference is not None else None
    cap = cv2.VideoCapture(str(video))
    selected = set(indices)
    j = 0
    try:
        for f in range(indices[-1] + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Decode failed at {f}")
            if f not in selected:
                continue
            mask = _mask(mask_dir, f)
            tip, source, _ = detect_blade(mask, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), shaft, width, grid_enabled=grid_enabled)
            if registration is not None:
                guide = tip if source else guides[f]
                registered, score, agreement = registration.locate(frame, guide)
                scores[j] = score
                accepted = score >= .55 and agreement <= 5 and np.isfinite(registered).all()
                if require_visible_template and accepted:
                    # During withdrawal an off-screen template can correlate
                    # with the remaining shaft and falsely report a still apex.
                    # Require its complete edge bands and a nearby independent
                    # image endpoint; an extrapolated search guide is not proof.
                    dy = registered[1] - registration.tip[1]
                    accepted = (source > 0 and np.linalg.norm(registered - tip) <= 12
                                and registration.top + dy >= 38
                                and registration.bottom + dy < frame.shape[0])
                if accepted:
                    tip, source = registered, 3
                else:
                    tip, source = np.full(2, np.nan), 0
            tips[j], sources[j] = tip, source
            if source:
                x, y = np.rint(tip).astype(int)
                left, top = max(0, x - 100), max(40, y - 100)
                local = mask[top:min(768, y + 101), left:min(1024, x + 101)]
                yy, xx = np.nonzero(local == 1)
                if len(xx):
                    gaps[j] = np.hypot(xx + left - tip[0], yy + top - tip[1]).min()
            j += 1
    finally:
        cap.release()
    support = sources > 0
    support[isolated_temporal_outliers(tips, support)] = False
    tips[~support] = np.nan
    return np.asarray(indices), tips, support, gaps, sources, scores


def local_reversal_support(registered, semantic, raw, scores, band_agreement):
    """A new blade template is usable only with independent current endpoints.

    High self-correlation of a new template alone is not evidence of the apex.
    Both endpoint detectors must be near it, and both template bands must agree.
    The wider raw limit accounts for the visible bright tip ending before the
    dark steel apex; the discrepancy remains reported, not silently corrected.
    """
    registered, semantic, raw = (np.asarray(v, float) for v in (registered, semantic, raw))
    return (np.isfinite(registered).all(axis=-1) & np.isfinite(semantic).all(axis=-1)
            & np.isfinite(raw).all(axis=-1) & (np.asarray(scores) >= .85)
            & (np.asarray(band_agreement) <= 5.)
            & (np.linalg.norm(semantic - raw, axis=-1) <= 8.)
            & (np.linalg.norm(registered - semantic, axis=-1) <= 8.)
            & (np.linalg.norm(registered - raw, axis=-1) <= 10.))


def _local_reversal_reference(video, indices, mask_dir, shaft, width, guides, peak_frame, *, grid_enabled=True):
    """Recover a failed distant template from observed local blade appearance.

    This never invents a smooth full-video trajectory. It is restricted to the
    already segmented reversal interval; loading-prefix geometry keeps its
    original contact reference. The absolute apex is independently observed.
    """
    from src.analysis.rigid_blade_reference import RigidBladeReference
    indices = np.asarray(indices, int)
    images = []
    semantic = np.full((len(indices), 2), np.nan)
    raw = np.full_like(semantic, np.nan)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
    try:
        for i, frame_index in enumerate(indices):
            if i and frame_index != indices[i - 1] + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode reversal candidate {frame_index}")
            images.append(image)
            mask = _mask(mask_dir, int(frame_index))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            q, source, _ = detect_blade(mask, gray, shaft, width, grid_enabled=grid_enabled)
            r, raw_source, _ = detect_blade(np.zeros_like(mask), gray, shaft, width, grid_enabled=grid_enabled)
            # Source 2 on the first call would reuse the raw result, not give
            # two independent endpoint observations.
            if source == 1 and raw_source == 2:
                semantic[i], raw[i] = q, r
    finally:
        cap.release()
    candidate_indices = np.flatnonzero(np.isfinite(semantic).all(axis=1)
        & np.isfinite(raw).all(axis=1) & (np.linalg.norm(semantic - raw, axis=1) <= 8.))
    candidate_indices = sorted(candidate_indices, key=lambda i: abs(int(indices[i]) - int(peak_frame)))
    for candidate in candidate_indices[:3]:
        reference = RigidBladeReference(images[candidate], semantic[candidate], width, grid_enabled=grid_enabled)
        positions = np.full_like(semantic, np.nan)
        scores, agreement = np.zeros(len(indices)), np.full(len(indices), np.inf)
        for i, image in enumerate(images):
            guide = semantic[i] if np.isfinite(semantic[i]).all() else guides[indices[i]]
            positions[i], scores[i], agreement[i] = reference.locate(image, guide)
        accepted = local_reversal_support(positions, semantic, raw, scores, agreement)
        accepted[isolated_temporal_outliers(positions, accepted)] = False
        if accepted.sum() < 3:
            continue
        coordinates = positions.copy()
        coordinates[~accepted] = np.nan
        metadata = dict(mode="local_reversal_image_reference_after_contact_template_failure",
            reference_frame=int(indices[candidate]), reference_tip_px=semantic[candidate].tolist(),
            independent_raw_tip_px=raw[candidate].tolist(),
            endpoint_detector_disagreement_px=float(np.linalg.norm(semantic[candidate] - raw[candidate])),
            contact_reference_unchanged_for_loading_prefix=True, trajectory_interpolation_used=False,
            supported_frames=int(accepted.sum()), total_frames=len(indices),
            accepted_score_range=[float(scores[accepted].min()), float(scores[accepted].max())],
            max_accepted_band_disagreement_px=float(agreement[accepted].max()),
            maximum_accepted_semantic_gap_px=float(np.linalg.norm(positions[accepted] - semantic[accepted], axis=1).max()),
            maximum_accepted_raw_gap_px=float(np.linalg.norm(positions[accepted] - raw[accepted], axis=1).max()))
        result = (indices, coordinates, accepted, np.full(len(indices), np.nan),
                  np.where(accepted, 4, 0).astype(np.uint8), scores)
        return result, metadata
    raise ValueError("Cannot independently verify a local blade reversal reference")


def contact_candidates(indices, supported, gaps, consecutive=3):
    """Only consecutive selected observations establish a proximity run."""
    result = {}
    indices = np.asarray(indices)
    for threshold in (1., 2., 4., 6.):
        close = np.asarray(supported) & np.isfinite(gaps) & (np.asarray(gaps) <= threshold)
        runs = np.flatnonzero(np.convolve(close.astype(int), np.ones(consecutive, int), "valid") == consecutive)
        result[str(threshold)] = int(indices[runs[0]]) if len(runs) else None
    return result


def _choose_contact(candidates):
    for threshold in ("2.0", "4.0", "6.0"):
        if candidates[threshold] is not None:
            return candidates[threshold], threshold
    raise ValueError("No sustained image-supported contact proximity; inspect masks")


def prefix_endpoint(tips, supported, contact, reversal, target_mm=1., full_stroke_mm=3.):
    """Calibrate only from observed contact/reversal and choose observed endpoint."""
    tips = np.asarray(tips, float)
    supported = np.asarray(supported, bool)
    if not 0 < target_mm <= full_stroke_mm or not 0 <= contact < reversal < len(tips):
        raise ValueError("Invalid depth calibration interval")
    if not supported[contact] or not supported[reversal]:
        raise ValueError("Calibration endpoints must be image supported")
    stroke = tips[reversal, 1] - tips[contact, 1]
    if not 100 < stroke < 650:
        raise ValueError(f"Unexpected contact-to-reversal stroke {stroke} px")
    depth = (tips[:, 1] - tips[contact, 1]) * float(full_stroke_mm) / stroke
    # The calibration itself defines both endpoints exactly. Floating-point
    # division must not move the requested full-stroke endpoint past reversal.
    depth[contact], depth[reversal] = 0., float(full_stroke_mm)
    if np.isclose(target_mm, full_stroke_mm, rtol=0., atol=1e-10):
        return depth, int(reversal), float(stroke)
    eligible = supported & (np.arange(len(tips)) <= reversal) & (depth >= target_mm)
    hits = np.flatnonzero(eligible)
    if not len(hits):
        raise ValueError("No observed target-depth endpoint")
    return depth, int(hits[0]), float(stroke)


def sequence_endpoint(tips, supported, contact, reversal, target_mm=3., full_stroke_mm=3., *, full_video=False):
    """Separate calibrated loading depth from the last sample of a full video."""
    depth, endpoint, stroke = prefix_endpoint(tips, supported, contact, reversal, target_mm, full_stroke_mm)
    depth[~np.asarray(supported, bool)] = np.nan
    return depth, len(tips) - 1 if full_video else endpoint, stroke


def validated_reuse_inputs(reuse_dir, name, source_sha256, full_stroke_mm):
    """Read a compatible completed case; no old files are changed."""
    reuse_dir = Path(reuse_dir)
    plan = json.loads((reuse_dir / "plan.json").read_text(encoding="utf-8"))
    geometry = json.loads((reuse_dir / "geometry.json").read_text(encoding="utf-8"))
    if plan.get("name") != name or plan.get("source_sha256") != source_sha256:
        raise ValueError("Reuse video identity/checksum differs from requested source")
    if (plan.get("contact_to_reversal_mm") != full_stroke_mm
            or geometry.get("contact_to_reversal_mm") != full_stroke_mm):
        raise ValueError("Reuse depth calibration differs from requested full stroke")
    with np.load(reuse_dir / "geometry.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}
    contact, reversal = (int(geometry[k]) for k in ("contact_frame", "reversal_frame"))
    prefix_endpoint(arrays["tips_px"], arrays["supported"], contact, reversal, full_stroke_mm, full_stroke_mm)
    with np.load(reuse_dir / "queries.npz") as z:
        queries = {k: z[k].copy() for k in z.files}
    core = np.asarray(queries.get("is_core_point", np.ones(len(queries["point_ids"]), bool)), bool)
    if (len(queries["seed_points"]) != len(queries["point_ids"])
            or not np.isfinite(queries["seed_points"]).all()
            or len(np.unique(queries["point_ids"])) != len(queries["point_ids"])
            or core.shape != (len(queries["point_ids"]),)
            or len(np.unique(queries["depth_layer"][core])) != 25):
        raise ValueError("Reuse material identities/layers are not a valid fixed query set")
    return plan, geometry, arrays, queries


def _prepare_from_existing(name, out, video, video_hash, reuse_dir, target_mm,
                           full_stroke_mm, geometry_stride, session, *, full_video=False,
                           query_scope="near_blade", coarse_pitch_px=24):
    """Extend trusted 1-mm inputs without recalibrating contact or material IDs."""
    started = time.perf_counter()
    profile = input_profile(name)
    reuse_dir = Path(reuse_dir).resolve()
    if reuse_dir == out:
        raise ValueError("Reuse source and new output must be different directories")
    previous_plan, geometry, prior, queries = validated_reuse_inputs(reuse_dir, name, video_hash, full_stroke_mm)
    if previous_plan.get('input_profile', {}).get('grid_enabled', True) != profile['grid_enabled']:
        raise ValueError('Reuse acquisition grid profile differs from the requested input')
    count = int(previous_plan["full_video_frames"])
    contact, reversal = int(geometry["contact_frame"]), int(geometry["reversal_frame"])
    if not np.isclose(target_mm, full_stroke_mm, rtol=0., atol=1e-10):
        raise ValueError("Reuse extension currently requires the complete calibrated stroke")
    # No post-reversal frames are needed for this standalone full-stroke run.
    # The last tracking window may be shorter, as in the existing tracker.
    inference_frames, endpoint = (count, count - 1) if full_video else (reversal + 1, reversal)
    old_masks = reuse_dir / "segmentation" / name / "semantic_masks"
    mask_dir = out / "segmentation" / name / "semantic_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    reused_masks = 0
    for index in range(inference_frames):
        source = old_masks / f"f{index:06d}.png"
        target = mask_dir / source.name
        if source.exists():
            if not target.exists():
                # Independent copies prevent later writes from changing old
                # results through a hard link or symbolic link.
                shutil.copyfile(source, target)
            reused_masks += 1
    owned_session = session is None
    session = load_segmentation_session() if session is None else session
    session = session_for_input(session, profile)
    _segment_selected(video, range(inference_frames), mask_dir, session, reuse=True)
    if owned_session:
        del session
        import torch
        torch.cuda.empty_cache()
    shaft, shaft_width = float(geometry["shaft_center_x_px"]), float(geometry["shaft_width_px"])
    # Anchor the reference at the previously observed contact apex in that
    # same source image. Reusing this coordinate does not impose future motion.
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, contact)
    ok, reference_frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError("Cannot read reused contact reference image")
    reference = (reference_frame, prior["tips_px"][contact], shaft_width)
    guide = prior["full_tip_guides_px"].copy()
    if len(guide) < inference_frames:
        coarse_good = prior["coarse_supported"] & np.isfinite(prior["coarse_tips_px"]).all(axis=1)
        good_indices = prior["coarse_frame_indices"][coarse_good]
        if len(good_indices) < 2:
            raise ValueError("Insufficient full-video coarse geometry to guide withdrawal registration")
        guide = np.column_stack([np.interp(np.arange(inference_frames), good_indices,
            prior["coarse_tips_px"][coarse_good, d]) for d in range(2)])
        guide[:len(prior["full_tip_guides_px"])] = prior["full_tip_guides_px"]
    old_prefix = min(int(previous_plan["frames"]), inference_frames)
    tips = np.full((inference_frames, 2), np.nan)
    supported = np.zeros(inference_frames, bool)
    source_codes = np.zeros(inference_frames, np.uint8)
    registration_scores = np.full(inference_frames, np.nan)
    # For full-video extension the validated prefix needs no re-registration.
    start_registration = old_prefix if full_video else 0
    if start_registration < inference_frames:
        observed = _geometry_samples(video, range(start_registration, inference_frames), mask_dir,
            shaft, shaft_width, reference, guide, require_visible_template=full_video, grid_enabled=profile['grid_enabled'])
        tips[observed[0]], supported[observed[0]] = observed[1], observed[2]
        source_codes[observed[0]], registration_scores[observed[0]] = observed[4], observed[5]
    # Existing supported prefix and reversal observations are the same images
    # and calibration, not forecasts. Keep their original values unchanged.
    prior_length = min(len(prior["tips_px"]), inference_frames)
    reused_observations = np.zeros(inference_frames, bool)
    reused_observations[:prior_length] = prior["supported"][:prior_length]
    prior_tips = np.full_like(tips, np.nan)
    prior_tips[:prior_length] = prior["tips_px"][:prior_length]
    overlap = reused_observations & supported & np.isfinite(tips).all(axis=1)
    disagreement = np.linalg.norm(tips[overlap] - prior_tips[overlap], axis=1)
    endpoint_before_reuse = tips[reversal].copy()
    # Preserve the earlier prefix's explicit unknown states as well as its
    # observations. A different template must not retroactively turn them
    # into apparently measured early-tip positions.
    tips[:old_prefix] = prior["tips_px"][:old_prefix]
    supported[:old_prefix] = prior["supported"][:old_prefix]
    tips[reused_observations] = prior_tips[reused_observations]
    supported[reused_observations] = True
    if full_video:
        source_codes[:old_prefix] = prior.get("registration_source_code", np.zeros(old_prefix, np.uint8))[:old_prefix]
        registration_scores[:old_prefix] = prior.get("registration_score", np.full(old_prefix, np.nan))[:old_prefix]
    # Unsupported image registrations remain missing. Geometry guides are
    # retained separately and are never relabelled as measured coordinates.
    tips[~supported] = np.nan
    depth, endpoint, stroke = sequence_endpoint(tips, supported, contact, reversal,
        target_mm, full_stroke_mm, full_video=full_video)
    np.savez_compressed(out / "geometry.npz", tips_px=tips, supported=supported, depth_mm=depth,
        entry_px=tips[:, 1] - tips[contact, 1], full_tip_guides_px=guide[:inference_frames],
        geometry_interpolated=np.zeros(inference_frames, bool),
        reused_observation=reused_observations, registration_source_code=source_codes,
        registration_score=registration_scores, coarse_frame_indices=prior["coarse_frame_indices"],
        coarse_tips_px=prior["coarse_tips_px"], coarse_supported=prior["coarse_supported"])
    geometry = dict(geometry, frames=inference_frames, full_video_frames=count, reused_from=str(reuse_dir),
        contact_and_reversal_recalibrated=False, contact_reference_frame=contact,
        contact_reference_tip_px=prior["tips_px"][contact].tolist(),
        per_frame_image_registration=True, unsupported_coordinates="NaN, guides stored separately",
        sequence_scope="full_video" if full_video else "loading_prefix", frame_end=endpoint,
        depth_definition="signed blade y relative to first contact; may decrease or become negative after reversal",
        depth_is_monotonic=False, withdrawal_geometry_requires_complete_template=full_video,
        reused_observed_tip_frames=int(reused_observations.sum()),
        shared_observation_registration_difference_median_px=(float(np.median(disagreement)) if len(disagreement) else None),
        shared_observation_registration_difference_p95_px=(float(np.percentile(disagreement, 95)) if len(disagreement) else None),
        new_reversal_registration_difference_px=(float(np.linalg.norm(endpoint_before_reuse - tips[reversal]))
            if np.isfinite(endpoint_before_reuse).all() else None))
    _save(out / "geometry.json", geometry)
    if query_scope == "full_material":
        queries = add_full_material_queries(queries, _mask(mask_dir, 0), pitch=coarse_pitch_px, grid_enabled=profile['grid_enabled'])
    queries["source_frame_indices"] = np.arange(inference_frames)
    np.savez_compressed(out / "queries.npz", **queries)
    shutil.copyfile(reuse_dir / "original_surface_reference.npz", out / "original_surface_reference.npz")
    with np.load(video_directory(name) / f"{name}_alignment.npz") as aligned:
        source_indices = aligned["source_indices"][:inference_frames].copy()
    if len(source_indices) != inference_frames or np.any(np.diff(source_indices) <= 0):
        raise ValueError("Invalid original-frame mapping for extended prefix")
    np.savez_compressed(out / "frame_mapping.npz", frame_indices=np.arange(inference_frames), source_indices=source_indices,
        physical_time_s=source_indices / profile['acquisition_fps'], physical_acquisition_fps=profile['acquisition_fps'],
        acquisition_clock_status=profile['acquisition_clock_status'])
    normalized_video = out / "normalized_lossless.mkv"
    _save(out / "normalization.json", _normalize(video, mask_dir, inference_frames, normalized_video,
        reuse_video=previous_plan["normalized_video"] if full_video else None,
        reuse_frames=old_prefix if full_video else 0, grid_enabled=profile['grid_enabled']))
    roi = (dict(x=0, y=0, width=1024, height=768, native_pixels=True, resized=False,
                selected_points_inside_fraction=1., scope="full initial material coverage; full-frame tracking")
           if query_scope == "full_material" else
           dict(previous_plan.get("native_roi", {}), scope="initial coverage only; full-frame tracking required for 3-mm motion"))
    _save(out / "native_roi.json", roi)
    plan = dict(previous_plan, name=name, input_profile=profile, frames=inference_frames, inference_frames=inference_frames,
        full_video_frames=count, frame_end=endpoint, display_frames=endpoint + 1, video=str(video),
        normalized_video=str(normalized_video), source_sha256=video_hash, target_depth_mm=float(target_mm),
        actual_endpoint_depth_mm=float(depth[endpoint]) if np.isfinite(depth[endpoint]) else None,
        native_roi=roi, reuse_dir=str(reuse_dir), full_video=bool(full_video),
        sequence_scope="full_video" if full_video else "loading_prefix",
        processing_scope="full_video" if full_video else "loading_prefix", query_scope=query_scope,
        query_scope_description="full initial material, preserved 25 near-blade layers plus sparse outer coverage" if query_scope == "full_material" else
            "initial material surface +/-90 px, 25 inward layers",
        full_material_coarse_pitch_px=coarse_pitch_px if query_scope == "full_material" else None,
        calibrated_max_loading_depth_mm=float(full_stroke_mm),
        endpoint_definition="last original imported video frame" if full_video else "calibrated reversal",
        points=len(queries["seed_points"]), core_points=int(np.count_nonzero(queries.get("is_core_point", np.ones(len(queries["seed_points"]), bool)))),
        core_point_count=int(np.count_nonzero(queries.get("is_core_point", np.ones(len(queries["seed_points"]), bool)))),
        coarse_count=int(queries["coarse_count"]),
        coarse_points=int(queries["coarse_count"]),
        coarse_geometry_stride=geometry_stride, reused_mask_frames=reused_masks,
        new_mask_frames=inference_frames - reused_masks, query_ids_and_coordinates_reused_exactly=query_scope == "near_blade",
        original_query_ids_and_coordinates_preserved=True,
        calibrated_contact_and_reversal_reused=True, all_frames_preserved=True,
        registered_tip_support_fraction=float(supported.mean()), seconds=time.perf_counter() - started)
    _save(out / "plan.json", plan)
    return plan


def build_queries(mask0, shaft_x, *, half_span=90., pitch=6., layers=25):
    boundary = []
    for x in range(100, 930):
        ys = np.flatnonzero(mask0[:, x] == 1)
        if len(ys) and 48 < ys[0] < 650:
            boundary.append((x, float(ys[0])))
    if len(boundary) < 20:
        raise ValueError("Insufficient initial material boundary")
    line = cv2.fitLine(np.asarray(boundary, np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    tangent, center = line[:2].astype(float), line[2:].astype(float)
    if tangent[0] < 0:
        tangent = -tangent
    cohort = original_surface_stations(mask0, center, tangent, shaft_x, half_span=half_span, pitch=pitch)
    offsets = 1.5 + pitch * np.arange(layers)
    all_points = cohort["boundary_xy"][:, None, :] + offsets[None, :, None] * cohort["inward_normal"]
    valid = _labels_at(mask0, all_points) == 1
    if np.any(valid.sum(axis=0) == 0):
        raise ValueError("Every inward depth layer must contain actual material points")
    seeds = all_points[valid].astype(np.float32)
    station, layer = np.nonzero(valid)
    return dict(seed_points=seeds, point_ids=np.arange(len(seeds)), dense_ids=np.arange(len(seeds)),
                depth_layer=layer, surface_station=station, base_pitch_px=12, dense_pitch_px=pitch,
                density_multiplier=4, coarse_count=0), cohort, center, tangent


def add_full_material_queries(queries, mask0, *, pitch=24, grid_enabled=True):
    """Append sparse coverage of every safe material tile, preserving core IDs.

    Selection is geometric, without texture scores that would bias coverage
    toward bright/easy material. A tile already covered by a core point gets
    no extra point. The fixed acquisition grid and blade margin are excluded
    only for new queries; existing core identities are never relocated.
    """
    from src.tracking.fixed_grid import FixedGridProcessor, grid_0915_config
    if mask0.shape != (768, 1024) or int(pitch) != pitch or pitch < 12:
        raise ValueError("Full material queries require native mask and integer pitch >= 12")
    pitch = int(pitch)
    result = {key: np.asarray(value).copy() for key, value in queries.items()}
    old_points = np.asarray(result["seed_points"], np.float32)
    old_ids = np.asarray(result["point_ids"], np.int64)
    if (old_points.ndim != 2 or old_points.shape[1] != 2 or len(old_points) != len(old_ids)
            or not np.isfinite(old_points).all() or len(np.unique(old_ids)) != len(old_ids)):
        raise ValueError("Invalid original material identities")
    grid = FixedGridProcessor(grid_0915_config()).mask().astype(np.uint8)
    if not grid_enabled:
        grid[:] = 0
    grid_excluded = cv2.dilate(grid, np.ones((7, 7), np.uint8)) > 0
    blade_excluded = cv2.dilate((mask0 == 2).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    safe = (mask0 == 1) & ~grid_excluded & ~blade_excluded
    # Keep patch centres off image borders, where a symmetric image patch
    # would need invented outside-image values.
    safe[:4] = safe[-4:] = False
    safe[:, :4] = safe[:, -4:] = False
    occupied = {(int(y) // pitch, int(x) // pitch) for x, y in old_points}
    appended = []
    candidate_tiles = 0
    for top in range(0, mask0.shape[0], pitch):
        for left in range(0, mask0.shape[1], pitch):
            yy, xx = np.nonzero(safe[top:top + pitch, left:left + pitch])
            if not len(xx):
                continue
            candidate_tiles += 1
            if (top // pitch, left // pitch) in occupied:
                continue
            xy = np.column_stack([xx + left, yy + top])
            centre = np.array([left + (pitch - 1) / 2, top + (pitch - 1) / 2])
            selected = xy[np.argmin(np.sum((xy - centre) ** 2, axis=1))]
            appended.append(selected)
    extra = np.asarray(appended, np.float32).reshape(-1, 2)
    new_ids = np.arange(int(old_ids.max()) + 1, int(old_ids.max()) + 1 + len(extra), dtype=np.int64)
    result["seed_points"] = np.vstack([old_points, extra])
    result["point_ids"] = np.r_[old_ids, new_ids]
    for field in ("depth_layer", "surface_station"):
        result[field] = np.r_[result[field], np.full(len(extra), -1, dtype=np.int64)]
    result["is_core_point"] = np.r_[result.get("is_core_point", np.ones(len(old_points), bool)), np.zeros(len(extra), bool)]
    result["coarse_count"] = int(result.get("coarse_count", 0)) + len(extra)
    result["coarse_pitch_px"] = pitch
    result["initial_safe_material_tiles"] = candidate_tiles
    rounded = np.rint(old_points).astype(int)
    inside = (rounded[:, 0] >= 0) & (rounded[:, 0] < 1024) & (rounded[:, 1] >= 0) & (rounded[:, 1] < 768)
    old_grid = grid_excluded[rounded[inside, 1], rounded[inside, 0]]
    result["preserved_core_points_near_grid"] = int(old_grid.sum())
    return result


def _normalize(video, mask_dir, count, output, *, reuse_video=None, reuse_frames=0, grid_enabled=True):
    from src.tracking.shadow_normalization import MaterialShadowNormalizer
    normalizer = MaterialShadowNormalizer(grid_enabled=grid_enabled)
    partial = output.with_name("normalized_lossless.encoding.mkv")
    cap = cv2.VideoCapture(str(video))
    reused_cap = cv2.VideoCapture(str(reuse_video)) if reuse_video is not None else None
    if reuse_frames < 0 or reuse_frames > count or (reuse_frames and reused_cap is None):
        raise ValueError("Invalid normalized prefix reuse")
    command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1024x768",
        "-r", "12", "-i", "-", "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-threads", "3", str(partial)]
    log = output.with_suffix(".encoder.log").open("wb")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    samples = {}
    try:
        for f in range(count):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Normalization decode failed at {f}")
            corrected, _ = normalizer.process(frame, _mask(mask_dir, f))
            if f < reuse_frames:
                reused_ok, cached = reused_cap.read()
                if not reused_ok:
                    raise RuntimeError(f"Cached normalization decode failed at {f}")
                # Replay gain estimation to restore temporal state exactly.
                # Pixel-exact checks detect changed masks or normalization
                # configuration before adjoining the newly normalized tail.
                if f in (0, reuse_frames // 2, reuse_frames - 1):
                    np.testing.assert_array_equal(corrected, cached)
                corrected = cached
            process.stdin.write(corrected.tobytes())
            if f in (0, count // 2, count - 1):
                samples[f] = corrected.copy()
        process.stdin.close()
        if process.wait(timeout=180):
            raise RuntimeError("Lossless encoder failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        cap.release()
        if reused_cap is not None:
            reused_cap.release()
        log.close()
    cap = cv2.VideoCapture(str(partial))
    actual = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if actual in samples:
                np.testing.assert_array_equal(image, samples[actual])
            actual += 1
    finally:
        cap.release()
    if actual != count:
        raise ValueError("Lossless prefix frame count mismatch")
    partial.replace(output)
    return dict(config=normalizer.metadata(), pixel_exact_sample_frames=list(samples), frames=count,
        reused_normalized_video=str(reuse_video) if reuse_video is not None else None,
        reused_pixel_frames=reuse_frames,
        prefix_state_continuation="replay from frame zero, no gain reset at extension boundary" if reuse_frames else None)


def prepare_case(name, output_dir, target_mm=1., *, session=None, full_stroke_mm=3., geometry_stride=8,
                 reuse_dir=None, full_video=False, query_scope="near_blade", coarse_pitch_px=24):
    """Prepare a case; reuse a loaded segmentation session across a batch."""
    # Existing batch CLI calls the original dense near-blade set "dense".
    query_scope = "near_blade" if query_scope == "dense" else query_scope
    if not 0 < target_mm <= full_stroke_mm or int(geometry_stride) != geometry_stride or geometry_stride < 1:
        raise ValueError("Invalid requested depth or geometry sampling stride")
    if query_scope not in ("near_blade", "full_material"):
        raise ValueError("Unknown material query scope")
    if int(coarse_pitch_px) != coarse_pitch_px or coarse_pitch_px < 12:
        raise ValueError("Full-material coarse pitch must be an integer >= 12")
    if full_video and reuse_dir is None:
        raise ValueError("Full-video preparation requires previously verified contact/reversal reuse inputs")
    validate_case_names([name])
    out = Path(output_dir).resolve()
    out.relative_to((ROOT / "results").resolve())
    out.mkdir(parents=True, exist_ok=True)
    data_directory = video_directory(name)
    video = data_directory / f"{name}.mp4"
    imported = json.loads((data_directory / f"{name}.json").read_text(encoding="utf-8"))
    profile = input_profile(name)
    video_hash = _sha(video)
    if video_hash != imported["output_sha256"]:
        raise ValueError("Imported local video checksum changed")
    if (out / "plan.json").exists():
        plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        if (plan.get("source_sha256") == video_hash and plan.get("target_depth_mm") == target_mm
                and plan.get("contact_to_reversal_mm") == full_stroke_mm
                and plan.get('input_profile', {}).get('grid_enabled', True) == profile['grid_enabled']
                and bool(plan.get("full_video", False)) == bool(full_video)
                and (plan.get("query_scope") == query_scope or
                     (query_scope == "near_blade" and "full_material" not in plan.get("query_scope", "")))
                and (query_scope != "full_material" or plan.get("full_material_coarse_pitch_px") == coarse_pitch_px)
                and plan.get("coarse_geometry_stride") == geometry_stride and Path(plan["normalized_video"]).exists()):
            return plan
        raise ValueError("Existing preparation does not match requested inputs")
    if reuse_dir is not None:
        return _prepare_from_existing(name, out, video, video_hash, reuse_dir, target_mm,
                                      full_stroke_mm, geometry_stride, session, full_video=full_video,
                                      query_scope=query_scope, coarse_pitch_px=coarse_pitch_px)
    started = time.perf_counter()
    owned_session = session is None
    session = load_segmentation_session() if session is None else session
    session = session_for_input(session, profile)
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    cap.release()
    if (width, height) != (1024, 768):
        raise ValueError("Validation requires existing native 1024x768 imports")
    mask_dir = out / "segmentation" / name / "semantic_masks"
    coarse = np.unique(np.r_[np.arange(0, count, geometry_stride), count - 1]).astype(int)
    _segment_selected(video, coarse, mask_dir, session, reuse=True)
    shaft, shaft_width, shaft_review = estimate_initial_shaft_adaptive(coarse, lambda i: _mask(mask_dir, i))
    _save(out / "shaft_initialization.json", shaft_review)
    samples = _geometry_samples(video, coarse, mask_dir, shaft, shaft_width, grid_enabled=profile['grid_enabled'])
    indices, values, support, gaps, sources, scores = samples
    if support.sum() < len(coarse) // 2:
        raise ValueError("Insufficient supported coarse blade geometry")
    coarse_contact, _ = _choose_contact(contact_candidates(indices, support, gaps))
    good = indices[support]
    guide = np.column_stack([np.interp(np.arange(count), good, values[support, d]) for d in range(2)])
    coarse_reversal = int(good[np.argmax(values[support, 1])])
    # Refine calibration near reversal; coarse sampling is never used as flow.
    fine_reversal = np.arange(max(coarse_contact + 1, coarse_reversal - 2 * geometry_stride), min(count, coarse_reversal + 2 * geometry_stride + 1))
    _segment_selected(video, fine_reversal, mask_dir, session, reuse=True)
    reference_index = int(good[good <= coarse_contact][-1])
    reference_tip = values[np.flatnonzero(indices == reference_index)[0]]
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, reference_index)
    ok, reference_frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError("Cannot read blade reference")
    reference = (reference_frame, reference_tip, shaft_width)
    rev = _geometry_samples(video, fine_reversal, mask_dir, shaft, shaft_width, reference, guide, grid_enabled=profile['grid_enabled'])
    reversal_reference = dict(mode="same_contact_reference", reference_frame=reference_index)
    if rev[2].sum() < 3:
        rev, reversal_reference = _local_reversal_reference(video, fine_reversal, mask_dir,
            shaft, shaft_width, guide, coarse_reversal, grid_enabled=profile['grid_enabled'])
        _save(out / "reversal_reference_review.json", reversal_reference)
    reversal = int(rev[0][rev[2]][np.argmax(rev[1][rev[2], 1])])
    reversal_tip = rev[1][np.flatnonzero(rev[0] == reversal)[0]]
    estimated_depth = (guide[:, 1] - guide[coarse_contact, 1]) * full_stroke_mm / (reversal_tip[1] - guide[coarse_contact, 1])
    full_stroke_requested = bool(np.isclose(target_mm, full_stroke_mm, rtol=0., atol=1e-10))
    endpoints = (np.array([reversal]) if full_stroke_requested else
                 np.flatnonzero((estimated_depth >= target_mm) & (np.arange(count) <= reversal)))
    if not len(endpoints):
        raise ValueError("Target depth cannot be located")
    prefix_count = min(count, int(endpoints[0]) + 2 * geometry_stride + 25)
    # A prefix keeps all samples. Extra context allows exact endpoint refinement.
    for attempt in range(3):
        _segment_selected(video, range(prefix_count), mask_dir, session, reuse=False)
        dense = _geometry_samples(video, range(prefix_count), mask_dir, shaft, shaft_width, reference, guide, grid_enabled=profile['grid_enabled'])
        candidates = contact_candidates(dense[0], dense[2], dense[3])
        contact, contact_threshold = _choose_contact(candidates)
        full_tips = guide.copy()
        full_supported = np.zeros(count, bool)
        full_tips[:prefix_count] = dense[1]
        full_supported[:prefix_count] = dense[2]
        full_tips[rev[0]] = rev[1]
        full_supported[rev[0]] = rev[2]
        try:
            depth, endpoint, stroke = prefix_endpoint(full_tips, full_supported, contact, reversal, target_mm, full_stroke_mm)
            if endpoint < prefix_count and (full_stroke_requested or endpoint < prefix_count - 8 or prefix_count == count):
                break
        except ValueError as error:
            if "No observed target-depth" not in str(error):
                raise
        prefix_count = min(count, prefix_count + 64)
    else:
        raise ValueError("Could not establish a supported prefix endpoint")
    inference_frames = min(prefix_count, max(endpoint + 1, 48 + 24 * int(np.ceil((endpoint + 1 - 48) / 24))))
    queries, cohort, center, tangent = build_queries(_mask(mask_dir, 0), shaft)
    if query_scope == "full_material":
        queries = add_full_material_queries(queries, _mask(mask_dir, 0), pitch=coarse_pitch_px, grid_enabled=profile['grid_enabled'])
    queries["source_frame_indices"] = np.arange(inference_frames)
    np.savez_compressed(out / "queries.npz", **queries)
    np.savez_compressed(out / "original_surface_reference.npz", **cohort)
    np.savez_compressed(out / "geometry.npz", tips_px=full_tips, supported=full_supported,
        depth_mm=depth, entry_px=full_tips[:, 1] - full_tips[contact, 1],
        coarse_frame_indices=indices, coarse_tips_px=values, coarse_supported=support,
        geometry_interpolated=np.isfinite(full_tips).all(axis=1) & ~full_supported, full_tip_guides_px=guide)
    geometry = dict(shaft_center_x_px=shaft, shaft_width_px=shaft_width, contact_frame=contact,
        reversal_frame=reversal, contact_threshold_px=float(contact_threshold), contact_threshold_candidates=candidates,
        surface_center=center.tolist(), surface_tangent=tangent.tolist(),
        surface_angle_deg=float(np.degrees(np.arctan2(tangent[1], tangent[0]))),
        contact_to_reversal_px=stroke, contact_to_reversal_mm=float(full_stroke_mm),
        calibration=(f"approximately {full_stroke_mm:g} mm user-supplied contact-to-reversal; image-registered blade"
            if profile.get('physical_scale_status') != 'pending_user_contact_to_reversal_stroke' else
            f"UNCONFIRMED {full_stroke_mm:g} mm normalization for complete loading selection only; pixel tracks usable; physical measurements pending"),
        physical_scale_status=profile.get('physical_scale_status', 'historical_calibration'),
        reversal_registration_reference=reversal_reference,
        geometry_sampling_stride=geometry_stride, prefix_geometry_sampling_stride=1,
        contact_status="proximity estimate, not a fracture-onset measurement", frames=count)
    _save(out / "geometry.json", geometry)
    with np.load(data_directory / f"{name}_alignment.npz") as aligned:
        source_indices = aligned["source_indices"][:inference_frames].copy()
    if len(source_indices) != inference_frames or np.any(np.diff(source_indices) <= 0):
        raise ValueError("Invalid saved original-frame mapping")
    np.savez_compressed(out / "frame_mapping.npz", frame_indices=np.arange(inference_frames), source_indices=source_indices,
                        physical_time_s=source_indices / profile['acquisition_fps'],
                        physical_acquisition_fps=profile['acquisition_fps'],
                        acquisition_clock_status=profile['acquisition_clock_status'])
    all_interest = np.vstack([queries["seed_points"], dense[1][dense[2]]])
    low, high = np.min(all_interest, axis=0), np.max(all_interest, axis=0)
    roi_xy = np.clip(np.rint((low + high) / 2 - [256, 192]), [0, 0], [512, 384]).astype(int)
    roi = dict(x=int(roi_xy[0]), y=int(roi_xy[1]), width=512, height=384, native_pixels=True,
        scope="initial seed and observed tip coverage only; use full frames or moving crops for tracking",
        resized=False, selected_points_inside_fraction=float(np.mean(np.all((queries["seed_points"] >= roi_xy) & (queries["seed_points"] < roi_xy + [512, 384]), axis=1))))
    _save(out / "native_roi.json", roi)
    normalized_video = out / "normalized_lossless.mkv"
    normalization = _normalize(video, mask_dir, inference_frames, normalized_video, grid_enabled=profile['grid_enabled'])
    _save(out / "normalization.json", normalization)
    plan = dict(name=name, input_profile=profile, frames=inference_frames, full_video_frames=count, frame_end=endpoint,
        display_frames=endpoint + 1, inference_frames=inference_frames, video=str(video), normalized_video=str(normalized_video),
        source_sha256=video_hash, target_depth_mm=float(target_mm), actual_endpoint_depth_mm=float(depth[endpoint]),
        contact_frame=contact, reversal_frame=reversal, contact_to_reversal_mm=float(full_stroke_mm),
        points=len(queries["seed_points"]), query_start_frame=0, depth_layers=25, density_multiplier=4,
        query_scope=query_scope, full_video=False, processing_scope="loading_prefix",
        query_scope_description="full initial material, original 25 near-blade layers plus sparse outer coverage" if query_scope == "full_material" else
            "initial material surface +/-90 px, 25 inward layers; fixed original material IDs throughout the loading prefix",
        core_point_count=int(np.count_nonzero(queries.get("is_core_point", np.ones(len(queries["seed_points"]), bool)))),
        coarse_count=int(queries["coarse_count"]),
        full_material_coarse_pitch_px=coarse_pitch_px if query_scope == "full_material" else None,
        native_roi=roi, segmentation_model=str(SEG), coarse_geometry_stride=geometry_stride,
        flow_frame_stride=1, retraining=False, all_inputs_local=True, seconds=time.perf_counter() - started)
    _save(out / "plan.json", plan)
    if owned_session:
        del session
        import torch
        torch.cuda.empty_cache()
    return plan
