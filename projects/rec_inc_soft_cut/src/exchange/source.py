"""Read provenance for the local optical-flow exchange without rerunning tracking.

The selected source is the unannotated local MP4, not its upstream AVI and not
a review video. Arrays returned by ``inspect_case`` are indexed by the saved
``frame_mapping.frame_indices``. ``display_mask`` excludes inference tail frames.
No experimental laboratory transform or acquisition clock is manufactured here.
"""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

from src.io import sha as _sha256, clean as _json_finite, read_json


def _read_json(path: Path, *, optional: bool = False) -> dict:
    if optional and not path.is_file():
        return {}
    return read_json(path)




def _probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open selected source video: {path}")
        result = dict(width_px=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      height_px=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                      fps_nominal=float(cap.get(cv2.CAP_PROP_FPS)))
    finally:
        cap.release()
    if (min(result["width_px"], result["height_px"], result["frame_count"]) <= 0
            or not np.isfinite(result["fps_nominal"]) or result["fps_nominal"] <= 0):
        raise ValueError(f"Invalid video container properties: {path}")
    result["probe_method"] = "OpenCV container properties; this inspection does not decode every frame"
    return result


def _path(repo: Path, value: str | Path) -> Path:
    value = Path(value)
    return value.resolve() if value.is_absolute() else (repo / value).resolve()




def inspect_case(repo: Path, case_config: dict) -> dict:
    """Inspect one manifest case and return JSON metadata plus small NumPy arrays.

    Required config: ``name`` (or ``case_id``), ``tracks_root``. Optional config:
    ``measurements_root``, ``stroke_status``, ``data_role`` and the original
    manifest fields. No tracks.npz data or normalized-video data are read.

    Important output fields:
      frames, avi_frames, time_s, tips, geometry_supported,
      geometry_interpolated, depth_estimate_mm, entry_px: aligned frame arrays;
      display_mask/display_frames: published frame range excluding inference tail;
      contact_frame/reversal_frame: estimated local-MP4 frame IDs;
      scale_mm_px: conditional stroke-derived image scale, never a lab transform;
      metadata/json_evidence: finite JSON dictionaries; paths: absolute strings.
    """
    repo = Path(repo).resolve()
    name = str(case_config.get("name", case_config.get("case_id", "")))
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Expected a case filename stem")
    track_dir = _path(repo, case_config["tracks_root"]) / name
    measurement_dir = (_path(repo, case_config["measurements_root"]) / name
                       if case_config.get("measurements_root") else track_dir.parent.parent / "measurements" / name)
    plan = _read_json(track_dir / "plan.json")
    # Saved plans identify later acquisitions too; never silently select a 0930
    # file for a 1005/1014 case. The legacy location remains a compatibility fallback.
    source_video = _path(repo, case_config.get("video", plan.get("video",
                          repo / "data" / "videos" / "0930" / f"{name}.mp4")))
    paths = dict(source_video=source_video, source_metadata=source_video.with_suffix(".json"),
                 alignment=source_video.with_name(f"{name}_alignment.npz"),
                 tracks_dir=track_dir, tracks=track_dir / "tracking" / "tracks.npz",
                 measurements_dir=measurement_dir, geometry=track_dir / "geometry.npz",
                 geometry_metadata=track_dir / "geometry.json", mapping=track_dir / "frame_mapping.npz",
                 queries=track_dir / "queries.npz", normalization=track_dir / "normalization.json",
                 tracking_complete=track_dir / "tracking" / "complete.json", plan=track_dir / "plan.json",
                 masks=track_dir / "segmentation" / name / "semantic_masks")
    imported = _read_json(paths["source_metadata"])
    geometry = _read_json(paths["geometry_metadata"])
    normalization = _read_json(paths["normalization"], optional=True)
    tracking = _read_json(paths["tracking_complete"], optional=True)
    probe = _probe_video(source_video)
    actual_sha = _sha256(source_video)
    for recorded in (imported.get("output_sha256"), plan.get("source_sha256")):
        if recorded is not None and actual_sha != recorded:
            raise ValueError(f"Source video SHA-256 differs from recorded provenance: {name}")
    for field, container_field in (("width", "width_px"), ("height", "height_px"),
                                   ("frames", "frame_count"), ("fps", "fps_nominal")):
        if field in imported and not np.isclose(float(imported[field]), probe[container_field], rtol=0, atol=1e-5):
            raise ValueError(f"Source video {field} differs from import metadata: {name}")
    if plan.get("name", name) != name:
        raise ValueError("Case identity differs from tracking plan")
    with np.load(paths["mapping"], allow_pickle=False) as saved:
        frames = saved["frame_indices"].astype(np.int64)
        avi_frames = saved["source_indices"].astype(np.int64)
        legacy_times = saved["physical_time_s"].astype(float) if "physical_time_s" in saved else np.full(len(frames), np.nan)
        legacy_fps = float(saved["physical_acquisition_fps"]) if "physical_acquisition_fps" in saved else None
    if (frames.ndim != 1 or len(frames) == 0 or avi_frames.shape != frames.shape
            or legacy_times.shape != frames.shape or np.any(np.diff(frames) <= 0)
            or np.any(np.diff(avi_frames) <= 0) or frames[0] < 0
            or frames[-1] >= probe["frame_count"]):
        raise ValueError("Invalid frame mapping")
    with np.load(paths["alignment"], allow_pickle=False) as saved:
        imported_mapping = saved["source_indices"]
        imported_frame_ids = saved["frame_indices"] if "frame_indices" in saved else np.arange(len(imported_mapping))
        if not np.array_equal(imported_frame_ids, np.arange(probe["frame_count"])):
            raise ValueError("Import alignment does not enumerate the selected MP4")
        if not np.array_equal(imported_mapping[frames], avi_frames):
            raise ValueError("Tracking and import AVI frame mappings disagree")
    with np.load(paths["geometry"], allow_pickle=False) as saved:
        if saved["tips_px"].shape != (probe["frame_count"], 2):
            raise ValueError("Geometry and selected video dimensions disagree")
        tips = saved["tips_px"][frames].astype(float)
        supported = saved["supported"][frames].astype(bool)
        interpolated = (saved["geometry_interpolated"][frames].astype(bool)
                        if "geometry_interpolated" in saved else np.isfinite(tips).all(1) & ~supported)
        depths = saved["depth_mm"][frames].astype(float)
        entry = saved["entry_px"][frames].astype(float)
        all_tips = saved["tips_px"]
        contact = int(geometry["contact_frame"])
        reversal = int(geometry["reversal_frame"])
        if not 0 <= contact < reversal < len(all_tips):
            raise ValueError("Contact/reversal landmarks are outside source geometry")
        tip_at_contact = all_tips[contact].astype(float)
    with np.load(paths["queries"], allow_pickle=False) as saved:
        queries = {key: saved[key].copy() for key in ("seed_points", "point_ids", "is_core_point",
                   "depth_layer", "surface_station", "base_pitch_px", "dense_pitch_px", "coarse_pitch_px") if key in saved}
        if "source_frame_indices" in saved and not np.array_equal(saved["source_frame_indices"], frames):
            raise ValueError("Query and tracking frame mappings disagree")
    points, ids = queries["seed_points"], queries["point_ids"]
    if points.shape != (len(ids), 2) or not np.isfinite(points).all() or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid fixed reference material identities")
    scale = float(geometry["contact_to_reversal_mm"]) / float(geometry["contact_to_reversal_px"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid conditional stroke scale")
    finite_depth = np.isfinite(depths) & np.isfinite(entry)
    if not np.allclose(depths[finite_depth], entry[finite_depth] * scale, rtol=1e-7, atol=1e-7):
        raise ValueError("Saved conditional depth disagrees with stroke scale")
    display_frames = int(plan.get("display_frames", plan.get("frame_end", reversal) + 1))
    display_mask = frames < display_frames
    if display_frames > probe["frame_count"] or display_frames <= 0 or not display_mask.any():
        raise ValueError("Invalid published/display frame range")
    reference_frame = int(plan.get("query_start_frame", 0))
    if reference_frame not in frames:
        raise ValueError("Reference frame is absent from returned mapping")
    header_fps = imported.get("source_header_fps")
    header_fps = float(header_fps) if header_fps is not None and float(header_fps) > 0 else None
    formula_mapping_matches = bool(np.array_equal(avi_frames, 5*(frames//2) + 3*(frames % 2)))
    fingerprints = {}
    for key in ("source_video", "source_metadata", "alignment", "geometry", "geometry_metadata",
                "mapping", "queries", "normalization", "tracking_complete", "plan"):
        path = paths[key]
        if path.is_file():
            fingerprints[key] = dict(path=str(path), bytes=path.stat().st_size,
                                     sha256=actual_sha if key == "source_video" else _sha256(path))
    default_python = repo / ".venv" / "Scripts" / "python.exe"
    interpreter = dict(executable=str(Path(sys.executable).resolve()),
                       expected_project_interpreter=str(default_python),
                       expected_project_interpreter_exists=default_python.is_file(),
                       uses_expected_project_interpreter=Path(sys.executable).resolve() == default_python.resolve(),
                       is_virtual_environment=sys.prefix != sys.base_prefix,
                       reason="Read/export of existing arrays only; the active interpreter is recorded explicitly because repository environment provenance must not be inferred")
    unknowns = [
        "Physical acquisition clock and capture-to-playback mapping are unverified; historical 200 fps is an assumption.",
        "No independent fixed-base/camera registration, laboratory-axis rotation, mirror audit, or lens-distortion calibration is available.",
        "Core depth_mm and laboratory coordinates remain unavailable; depth_video_estimate_mm is a conditional image-vertical estimate only.",
        "First contact is a segmentation proximity estimate; no reviewed last-separated/first-contact evidence interval is available.",
        "Stroke length is approximately 3 mm; its measurement uncertainty and isotropic image scale validity are not established.",
        "Blade tip is transferred from a template by translation; blade rotation and exact instantaneous apex visibility are not measured.",
        "Tracking confidence is an image consistency score, not a calibrated probability of correct material identity.",
        "Out-of-plane motion, refraction/lubricant interference, and visible free-face versus plane-strain equivalence are unresolved.",
        "Specimen dimensions, clamp motion, temperature, laboratory inclination, blade identity/pose, and lubrication condition are not established by filename.",
        "No physical fracture onset is certified by this export; external analysis cutoffs do not prove an intact interval.",
    ]
    data_role = case_config.get("data_role", "undesignated")
    if data_role not in ("undesignated", "validation_only", "calibration_allowed"):
        raise ValueError("Unknown exchange data_role")
    metadata = dict(
        schema_version="1.0", case_id=name, data_role=data_role,
        source=dict(video_name=source_video.name, path=str(source_video), sha256=actual_sha, **probe,
                    time_source="Selected local MP4 CFR container time, frame_id / fps_nominal; not physical acquisition time",
                    capture_to_playback_time_mapping=None,
                    upstream_avi=dict(path=imported.get("source", {}).get("path"), sha256=None,
                                      reported_bytes=imported.get("source", {}).get("bytes"),
                                      header_fps=header_fps, header_frame_count=imported.get("source_header_frames"),
                                      sequentially_decoded_frames=imported.get("source_sequentially_decoded_frames"),
                                      import_mapping=imported.get("frame_mapping"),
                                      mapping_formula_verified=formula_mapping_matches,
                                      discarded_terminal_header_frames=imported.get("discarded_unreadable_terminal_header_frame")),
                    historical_capture_fps_assumption=legacy_fps, physical_capture_time_verified=False),
        experiment=dict(specimen_dimensions_mm=None, inclination_deg=None, angle_definition=None,
                        blade_id=None, lubrication=None, temperature_C=None, loading_speed_mm_s=None, fixation=None,
                        sources_and_unknowns=["Filename is an identifier, not an independently calibrated condition.",
                                              f"Contact-to-reversal stroke status in batch manifest: {case_config.get('stroke_status', 'unknown')}"]),
        coordinates=dict(pixel_convention="source_pixel_centers_x_right_y_down_origin_0_0",
                         lab_convention="x_right_y_up_mm_fixed_to_base", observed_face=None, mirrored=None,
                         source_to_tracking_pixel_transform=np.eye(3).tolist(), pixel_to_lab_mapping=None,
                         calibration_method=None, calibration_source=None, camera_motion_correction=None,
                         conditional_image_scale_mm_px=scale,
                         conditional_image_scale_method="User-supplied approximate contact-to-reversal stroke divided by registered image-vertical blade travel",
                         conditional_image_scale_stroke_status=case_config.get("stroke_status", "unknown"),
                         uncertainty_definition="Not evaluated; null uncertainty does not mean zero",
                         processing_transform_chain=[
                             dict(order=1, operation="Temporal AVI selection and H.264 recompression", coordinate_change=False,
                                  mapping="See exact AVI index mapping; no crop/scale/rotation/mirror reported at import",
                                  original_dimensions_preserved=imported.get("original_dimensions_preserved"), crf=imported.get("crf")),
                             dict(order=2, operation="Fixed acquisition-overlay grid replacement", coordinate_change=False,
                                  parameters=normalization.get("config", {}).get("grid_preprocessing", {})),
                             dict(order=3, operation="Masked bounded illumination gain", coordinate_change=False,
                                  parameters=normalization.get("config", {})),
                             dict(order=4, operation="Native full-frame material tracking", coordinate_change=False,
                                  roi_note="native_roi describes seed/display coverage, not an offset to add to tracks; review camera is display-only")]),
        reference=dict(frame_id=reference_frame, state="precontact_estimated_not_independently_certified_undeformed",
                       last_separated_frame=None, first_contact_evidence_frame=None,
                       contact_zero_method="First sustained 2-pixel (or configured fallback) proximity between registered blade apex and segmented material; not mechanical contact proof",
                       estimated_contact_frame=contact, estimated_reversal_frame=reversal,
                       estimated_contact_blade_source_xy_px=tip_at_contact.tolist(),
                       initial_contact_source_xy_px=None, initial_surface_direction_lab=None,
                       initial_surface_direction_image=geometry.get("surface_tangent"),
                       initial_surface_center_image_px=geometry.get("surface_center"),
                       initial_surface_angle_image_deg=geometry.get("surface_angle_deg"),
                       initial_surface_angle_definition="atan2(tangent_y,tangent_x), positive image-down from image +x; robust initial material-mask line fit",
                       contact_threshold_px=geometry.get("contact_threshold_px"),
                       contact_threshold_sensitivity_frames=geometry.get("contact_threshold_candidates"),
                       sensitivity_is_contact_evidence_interval=False),
        depth=dict(core_depth_mm_available=False, exported_estimate_column="depth_video_estimate_mm",
                   estimate_formula="(blade_tip_y_px - estimated_contact_tip_y_px) * conditional_image_scale_mm_px",
                   source="derived", fixed_base_verified=False, image_vertical_is_lab_vertical_verified=False,
                   contact_to_reversal_mm=geometry["contact_to_reversal_mm"], uncertainty_mm=None,
                   unsupported_geometry_policy="Keep conditional estimate only if finite; geometry status remains separate, never treat interpolation as observed"),
        tracking=dict(method=tracking.get("method", "independent multi-reference affine material patch registration"),
                      version_or_commit=tracking.get("code_sha256"), parameters=tracking.get("config", {}),
                      trajectory_composition="Immutable initial material IDs; incremental and keyframe affine registration, cumulative deformation composition, bilinear image sampling, reference checks and reverse-cycle checks",
                      confidence_definition="Recorded photometric/texture/reverse-consistency diagnostics; not a calibrated material-identity probability",
                      validity_rules="Saved tracking supported and postprocessed measurement_supported must be kept distinct; forecasts and occlusion are not valid observations",
                      uncertainty_definition="No calibrated per-point position uncertainty is available",
                      postprocessing=["Measurement support excludes blade/opaque-core contamination; original positions and IDs are preserved"],
                      reference_frame_id=reference_frame, point_count=len(ids), display_frames=display_frames,
                      saved_inference_frames=len(frames), neural_proposals=tracking.get("neural_proposals"),
                      physical_truth_validated=False,
                      seed_sampling=dict(dense_pitch_px=float(queries["dense_pitch_px"]) if "dense_pitch_px" in queries else None,
                                         coarse_pitch_px=float(queries["coarse_pitch_px"]) if "coarse_pitch_px" in queries else None,
                                         scope=plan.get("query_scope_description"))),
        blade=dict(method="Fixed blade-edge template registration; inferred apex follows template translation",
                   supported_status="rigid_inferred", interpolated_status="predicted", rotation_measured=False,
                   absolute_apex_initialization="Segmentation/bright-ridge geometry; no per-frame direct-apex evidence preserved",
                   reversal_registration_reference=geometry.get("reversal_registration_reference")),
        force_sync=dict(available=None, mapping=None, method=None, time_uncertainty_s=None,
                        note="Source inspection does not establish force identity or synchronization; force exporter supplies its own-clock data separately"),
        files=dict(frames="frames.csv", tracks="tracks.csv", quality_report="quality_report.md"),
        unavailable_items=unknowns)
    json_evidence = dict(case_config=case_config, import_metadata=imported, geometry_metadata=geometry,
                         plan=plan, normalization=normalization,
                         tracking_metadata={k: v for k, v in tracking.items() if k != "specification"},
                         fingerprints=fingerprints, interpreter=interpreter,
                         checks=dict(source_sha256_matches_import=True if imported.get("output_sha256") else None,
                                     source_sha256_matches_plan=True if plan.get("source_sha256") else None,
                                     video_container_matches_import=True, source_mapping_matches_import=True,
                                     formula_selection_matches=formula_mapping_matches,
                                     conditional_depth_matches_saved_scale=True,
                                     tracks_data_read=False, optical_flow_recomputed=False))
    return dict(case_id=name, frames=frames, avi_frames=avi_frames, time_s=frames / probe["fps_nominal"],
                avi_header_time_s=avi_frames / header_fps if header_fps else np.full(len(frames), np.nan),
                legacy_assumed_capture_time_s=legacy_times, tips=tips, geometry_supported=supported,
                geometry_interpolated=interpolated, depth_estimate_mm=depths, entry_px=entry,
                contact_frame=contact, reversal_frame=reversal, scale_mm_px=scale,
                angle_image_deg=float(geometry["surface_angle_deg"]), point_reference_frame_id=reference_frame,
                display_frames=display_frames, display_mask=display_mask, **queries,
                metadata=_json_finite(metadata), paths={k: str(v) for k, v in paths.items()},
                json_evidence=_json_finite(json_evidence))
