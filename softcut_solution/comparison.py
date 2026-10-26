"""Read-only observation operators between saved FE states and fixed-ID video.

This module does not solve FE, recalibrate a camera, or certify material identity.
It keeps the original acquisition order and samples the same reference triangle
and barycentric weights throughout a comparison.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA = "softcut-comparison/1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _bundle_path(path):
    path = Path(path).resolve()
    return path / "result.bundle.json" if path.is_dir() else path


def _read_fe(path):
    path = _bundle_path(path)
    bundle = json.loads(path.read_text(encoding="utf-8-sig"))
    if bundle.get("schema_version") != "softcut-result/1":
        raise ValueError("FE input must be an exported softcut-result/1 bundle")
    series = bundle.get("series", [])
    if not series:
        raise ValueError("FE bundle has no accepted-state series")
    depths = np.asarray([s["depth_mm"] for s in series], float)
    if not np.isfinite(depths).all() or np.any(np.diff(depths) <= 0):
        raise ValueError("FE accepted depths must be finite and strictly increasing")
    prefix = 0
    for row in series:
        if row.get("qualified") is not True or row.get("qualification") != "audited_prefix":
            break
        prefix += 1
    declared = bundle.get("metadata", {}).get("qualification", {}).get("qualified_states")
    if isinstance(declared, bool) or not isinstance(declared, int) or not 0 <= declared <= len(series):
        raise ValueError("FE bundle must declare its audited qualified-state count")
    prefix = min(prefix, declared)
    mesh = bundle.get("mesh")
    frames = bundle.get("frames", [])
    qualified_frames = []
    previous_index = -1
    for frame in frames:
        index = frame.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(series):
            raise ValueError("Invalid FE frame index")
        if index <= previous_index:
            raise ValueError("FE saved frames must retain accepted-state order without duplicates")
        previous_index = index
        if not np.isclose(float(frame["depth_mm"]), depths[index], atol=1e-10, rtol=0):
            raise ValueError("FE saved frame depth differs from its accepted state")
        if index < prefix:
            qualified_frames.append(frame)
    if mesh is not None:
        reference = np.asarray(mesh["reference_positions"], float)
        raw_triangles = np.asarray(mesh["triangles"])
        triangles = np.asarray(raw_triangles, int)
        if reference.ndim != 2 or reference.shape[1] != 2 or not np.isfinite(reference).all():
            raise ValueError("Invalid FE reference coordinates")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or not np.array_equal(raw_triangles, triangles):
            raise ValueError("FE triangles must contain integer node indices")
        if triangles.size == 0 or triangles.min() < 0 or triangles.max() >= len(reference):
            raise ValueError("FE triangle index outside the reference mesh")
        for frame in qualified_frames:
            positions = np.asarray(frame["positions"], float)
            if positions.shape != reference.shape or not np.isfinite(positions).all():
                raise ValueError("FE frame positions do not match the reference mesh")
    return path, bundle, depths, prefix, qualified_frames


def inspect_fe(path):
    """Summarize an existing bundle without loading a solver or changing files."""
    path, bundle, depths, prefix, frames = _read_fe(path)
    meta = bundle.get("metadata", {})
    return dict(schema_version=bundle["schema_version"], source_path=str(path),
                name=meta.get("name", path.parent.name), angle_deg=meta.get("angle_deg"),
                accepted_states=len(depths), qualified_states=prefix,
                qualified_end_mm=float(depths[prefix - 1]) if prefix else None,
                saved_frame_count=len(frames), saved_frame_indices=[f["index"] for f in frames],
                saved_frame_depths_mm=[f["depth_mm"] for f in frames],
                source_sha256=_sha(path), qualification=meta.get("qualification", {}),
                limitations=meta.get("limitations", []))


def _load_video(path):
    from .video import load_fixed_id
    return load_fixed_id(path)


def _map_points(reference, triangles, points):
    """Locate once in the actual accepted mesh; outside points stay unmapped."""
    from matplotlib.tri import Triangulation
    reference, triangles, points = np.asarray(reference, float), np.asarray(triangles, int), np.asarray(points, float)
    tri_index = np.full(len(points), -1, int)
    weights = np.full((len(points), 3), np.nan)
    finite = np.isfinite(points).all(axis=1)
    triangulation = Triangulation(reference[:, 0], reference[:, 1], triangles)
    tri_index[finite] = triangulation.get_trifinder()(points[finite, 0], points[finite, 1])
    valid = tri_index >= 0
    vertices = reference[triangles[tri_index[valid]]]
    matrices = np.stack((vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=2)
    uv = np.linalg.solve(matrices, (points[valid] - vertices[:, 0])[..., None])[..., 0]
    weights[valid] = np.column_stack((1 - uv.sum(axis=1), uv))
    if not np.allclose(weights[valid].sum(axis=1), 1, atol=1e-12):
        raise ValueError("Barycentric weights do not sum to one")
    if np.any(weights[valid] < -1e-9):
        raise ValueError("Invalid negative barycentric weight")
    reconstructed = np.einsum("ni,nij->nj", weights[valid], vertices)
    if not np.allclose(reconstructed, points[valid], atol=1e-9, rtol=1e-10):
        raise ValueError("Reference point reconstruction failed")
    return tri_index, weights


def _sample(reference_positions, frame_positions, triangles, tri_index, weights):
    answer = np.full((len(tri_index), 2), np.nan)
    valid = tri_index >= 0
    nodes = triangles[tri_index[valid]]
    answer[valid] = np.einsum("ni,nij->nj", weights[valid], frame_positions[nodes] - reference_positions[nodes])
    return answer


def _bracket(depth, frame_depths):
    """No extrapolation. Only interpolation between actual saved states."""
    if not np.isfinite(depth) or depth < frame_depths[0] - 1e-10 or depth > frame_depths[-1] + 1e-10:
        return None
    exact = np.flatnonzero(np.isclose(frame_depths, depth, atol=1e-10, rtol=0))
    if len(exact):
        return int(exact[0]), int(exact[0]), 0.0
    right = int(np.searchsorted(frame_depths, depth))
    left = right - 1
    return left, right, float((depth - frame_depths[left]) / (frame_depths[right] - frame_depths[left]))


def _cell(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return value


def _csv(path, rows, columns=None):
    rows = iter(rows)
    first = next(rows, None)
    columns = columns or list(first or {})
    if not columns:
        raise ValueError("CSV requires a defined schema")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        if first is not None:
            writer.writerow({k: _cell(v) for k, v in first.items()})
        for row in rows:
            writer.writerow({k: _cell(v) for k, v in row.items()})


def _svg(rows):
    specs = [("observed_mean_y_mm", "#007c91", "Video y"), ("predicted_mean_y_mm", "#d76322", "FE y"),
             ("observed_mean_z_mm", "#7054a4", "Video z"), ("predicted_mean_z_mm", "#527c26", "FE z")]
    values = [r[k] for r in rows for k, _, _ in specs if r[k] is not None]
    if not values:
        return "<p>No supported common observations are available.</p>"
    xx = [r["depth_mm"] for r in rows]
    xmin, xmax, ymin, ymax = min(xx), max(xx), min(values), max(values)
    if xmax == xmin:
        xmax += .01
    if ymax == ymin:
        ymax += .01
    def xy(x, y):
        return f"{65 + 770*(x-xmin)/(xmax-xmin):.2f},{235 - 185*(y-ymin)/(ymax-ymin):.2f}"
    elements = ['<svg viewBox="0 0 880 300" role="img" aria-label="Mean displacement in acquisition order">',
                '<path d="M65 40V235H835" fill="none" stroke="#7c8992"/>']
    for n, (key, color, label) in enumerate(specs):
        segment = []
        for row in rows + [{key: None}]:
            if row[key] is not None:
                segment.append(xy(row["depth_mm"], row[key]))
            elif segment:
                elements.append(f'<polyline points="{" ".join(segment)}" stroke="{color}" stroke-width="2" fill="none"/>')
                segment = []
        elements.append(f'<text x="{70+n*185}" y="22" fill="{color}" font-size="14">{label}</text>')
    elements += [f'<text x="65" y="260" font-size="12">{xmin:.3g}</text>',
                 f'<text x="805" y="260" font-size="12">{xmax:.3g}</text>',
                 f'<text x="5" y="50" font-size="12">{ymax:.3g}</text>',
                 f'<text x="5" y="237" font-size="12">{ymin:.3g}</text>',
                 '<text x="350" y="285" font-size="13">Conditional video blade travel (mm)</text></svg>']
    return "".join(elements)


def compare(fe_bundle, video_package, output_dir, options=None):
    """Export a traceable, conditional fixed-material-point displacement comparison."""
    options = dict(options or {})
    permitted = {"registration", "rotation_deg", "translation_y_mm", "translation_z_mm", "depth_offset_mm",
                 "max_depth_mm", "fe_reference_index", "support_mode", "intact_status", "allow_angle_mismatch"}
    if set(options) - permitted:
        raise ValueError(f"Unknown comparison options: {sorted(set(options)-permitted)}")
    registration = options.get("registration", "explicit")
    if registration not in ("tip_translation", "explicit"):
        raise ValueError("registration must be tip_translation or explicit")
    mode = options.get("support_mode", "continuous")
    if mode not in ("continuous", "endpoint"):
        raise ValueError("support_mode must be continuous or endpoint")
    if options.get("intact_status", "unknown") != "unknown":
        raise ValueError("This comparison cannot certify material intactness; use intact_status=unknown")
    maximum = _number(options.get("max_depth_mm", 1.0), "max_depth_mm")
    if not 0 < maximum <= 2:
        raise ValueError("max_depth_mm must be in (0, 2]")
    offset = _number(options.get("depth_offset_mm", 0.0), "depth_offset_mm")
    rotation = _number(options.get("rotation_deg", 0.0), "rotation_deg")
    ref_index = options.get("fe_reference_index", 0)
    if isinstance(ref_index, bool) or not isinstance(ref_index, int) or ref_index < 0:
        raise ValueError("fe_reference_index must be a nonnegative accepted-state index")
    fe_initial_sha = _sha(_bundle_path(fe_bundle))
    path, bundle, series_depths, prefix, fe_frames = _read_fe(fe_bundle)
    if not prefix or not fe_frames or bundle.get("mesh") is None:
        raise ValueError("No audited FE saved frames are available")
    references = [frame for frame in fe_frames if frame["index"] == ref_index]
    if not references:
        raise ValueError("Selected FE reference index must be an exported qualified saved frame")
    reference = references[0]
    output_dir = Path(output_dir).resolve()
    video_path = Path(video_package).resolve()
    video_dir = video_path if video_path.is_dir() else video_path.parent
    video_file = video_dir / "material_displacements.npz" if video_path.is_dir() else video_path
    input_files = [path, video_dir / "metadata.json", video_file]
    if any(output_dir == f.parent or output_dir in f.parents for f in input_files):
        raise ValueError("Comparison output must be separate from all source directories")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Comparison output directory must be empty; use a new run directory")
    before = {str(f): _sha(f) for f in input_files if f.is_file()}
    if before[str(path)] != fe_initial_sha:
        raise ValueError("FE bundle changed while it was being read")
    video = _load_video(video_package)
    if "source_path" in video and Path(video["source_path"]).resolve() != video_file.resolve():
        raise ValueError("Loaded video source differs from the fingerprinted input file")
    metadata = video["metadata"]
    video_angle = metadata.get("requested_nominal_angle_deg")
    fe_angle = bundle.get("metadata", {}).get("angle_deg")
    angle_status = "unknown"
    if video_angle is not None and fe_angle is not None:
        angle_status = "matched_nominal_labels" if np.isclose(float(video_angle), float(fe_angle), atol=1e-6) else "explicit_mismatch"
        if angle_status == "explicit_mismatch" and options.get("allow_angle_mismatch") is not True:
            raise ValueError(f"Video nominal angle {video_angle} differs from FE angle {fe_angle}")
    points = np.asarray(video["contact_image_xright_yup_mm"], float)
    displacement = np.asarray(video["displacement_from_contact_xright_yup_mm"], float)
    point_ids = np.asarray(video["point_ids"])
    depths = np.asarray(video["indentation_estimate_mm"], float)
    source_frames = np.asarray(video["source_frames"])
    endpoint = np.asarray(video["endpoint_valid"], bool)
    continuous = np.asarray(video["continuous_valid"], bool)
    count, point_count = len(depths), len(points)
    if count < 1 or point_count < 1 or points.shape != (point_count, 2) or displacement.shape != (count, point_count, 2):
        raise ValueError("Inconsistent fixed-ID position/displacement shapes")
    if endpoint.shape != (count, point_count) or continuous.shape != endpoint.shape or len(point_ids) != point_count:
        raise ValueError("Inconsistent fixed-ID support shapes")
    if len(np.unique(point_ids)) != point_count or source_frames.shape != (count,) or np.any(np.diff(source_frames) <= 0):
        raise ValueError("Point IDs must be unique and source frames must retain acquisition order")
    if not np.isfinite(depths).all():
        raise ValueError("Video depth contains nonfinite values")
    if not np.isclose(depths[0] + offset, reference["depth_mm"], atol=1e-8, rtol=0):
        raise ValueError("Video reference depth + depth_offset_mm must equal the selected FE reference depth")
    c, s = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
    Q = np.array([[c, -s], [s, c]])
    blade = np.asarray(reference["blade"], float)
    if blade.ndim != 2 or blade.shape[1] != 2 or not np.isfinite(blade).all():
        raise ValueError("FE reference blade outline is invalid")
    fe_tip = blade[np.isclose(blade[:, 1], blade[:, 1].min(), atol=1e-10, rtol=0)].mean(axis=0)
    if registration == "tip_translation":
        coordinates = metadata.get("coordinates", {})
        origin = np.asarray(coordinates.get("image_origin_source_px", video.get("image_origin_source_px", [])), float)
        scale = metadata.get("scale_um_per_px")
        if scale is None:
            raise ValueError("Tip registration requires declared metadata.scale_um_per_px")
        scale = _number(scale, "scale_um_per_px") / 1000
        if scale <= 0 or origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("Tip registration needs a finite image origin and positive declared scale")
        if "image_origin_source_px" in video and not np.allclose(video["image_origin_source_px"], origin, atol=1e-8, rtol=0):
            raise ValueError("Video NPZ image origin differs from its metadata")
        if "scale_um_per_px" in video and not np.isclose(float(video["scale_um_per_px"]), scale * 1000):
            raise ValueError("Video NPZ scale differs from its metadata")
        if "contact_positions_source_px" in video:
            source_reference = (np.asarray(video["contact_positions_source_px"], float) - origin) * [scale, -scale]
            finite_reference = np.isfinite(source_reference).all(axis=1) & np.isfinite(points).all(axis=1)
            if not np.allclose(source_reference[finite_reference], points[finite_reference], atol=1e-7, rtol=1e-8):
                raise ValueError("Video reference millimetres disagree with the declared image origin and scale")
        tip = np.asarray(video["blade_tip_source_px"], float)[0]
        if tip.shape != (2,) or not np.isfinite(tip).all():
            raise ValueError("Reference video blade tip is unavailable; use independently specified explicit registration")
        tip_mm = (tip - origin) * [scale, -scale]
        translation = fe_tip - Q @ tip_mm
        if options.get("translation_y_mm") is not None or options.get("translation_z_mm") is not None:
            raise ValueError("Explicit translations cannot be combined with tip_translation")
    else:
        translation = np.array([_number(options.get("translation_y_mm"), "translation_y_mm"),
                                _number(options.get("translation_z_mm"), "translation_z_mm")])
    registered = points @ Q.T + translation
    observed = displacement @ Q.T
    xref = np.asarray(reference["positions"], float)
    triangles = np.asarray(bundle["mesh"]["triangles"], int)
    tri_index, weights = _map_points(xref, triangles, registered)
    mapped = tri_index >= 0
    finite = np.isfinite(observed).all(axis=2) & np.isfinite(registered).all(axis=1)[None, :]
    endpoint &= finite
    continuous &= finite
    if np.any(np.linalg.norm(observed[0, endpoint[0]], axis=1) > 1e-8):
        raise ValueError("Supported video reference displacement is not zero")
    frame_depths = np.asarray([f["depth_mm"] for f in fe_frames], float)
    mapped_depths = depths + offset
    brackets, statuses = [], []
    cutoff_reached = False
    for d, fd in zip(depths, mapped_depths):
        bracket = _bracket(fd, frame_depths)
        if cutoff_reached:
            bracket, status = None, "after_cutoff"
        elif d < depths[0] - 1e-10 or d > maximum + 1e-10 or fd > 2 + 1e-10:
            bracket, status = None, "outside_requested_window"
            cutoff_reached = True
        elif bracket is None:
            status = "outside_qualified_saved_frame_range"
            cutoff_reached = True
        else:
            status = "sampled_saved_states"
        brackets.append(bracket)
        statuses.append(status)
    eligible = np.array([b is not None for b in brackets])
    if not eligible.any():
        raise ValueError("No shared video/qualified-FE depth window")
    support = continuous if mode == "continuous" else endpoint
    cohort = mapped & np.all(support[eligible], axis=0) if mode == "continuous" else mapped
    predicted_saved = [_sample(xref, np.asarray(f["positions"], float), triangles, tri_index, weights) for f in fe_frames]
    observations, frame_rows, predictions, masks = [], [], [], []
    capture_times = np.asarray(video.get("capture_time_from_contact_s", np.full(count, np.nan)), float)
    if capture_times.shape != (count,):
        raise ValueError("Capture-time array does not match video frames")
    avi_frames = np.asarray(video.get("avi_source_frames", np.full(count, -1)))
    for i, bracket in enumerate(brackets):
        prediction = np.full((point_count, 2), np.nan)
        used = np.zeros(point_count, bool)
        left_index = right_index = fraction = None
        if bracket is not None:
            left, right, fraction = bracket
            left_index, right_index = fe_frames[left]["index"], fe_frames[right]["index"]
            prediction = (1 - fraction) * predicted_saved[left] + fraction * predicted_saved[right]
            used = cohort & support[i] & np.isfinite(prediction).all(axis=1)
        predictions.append(prediction)
        masks.append(used)
        n = int(used.sum())
        fields = dict(observed_mean_y_mm=None, observed_mean_z_mm=None, predicted_mean_y_mm=None,
                      predicted_mean_z_mm=None, mean_error_y_mm=None, mean_error_z_mm=None, vector_rmse_mm=None)
        if n:
            obs_mean, pred_mean = observed[i, used].mean(axis=0), prediction[used].mean(axis=0)
            errors = prediction[used] - observed[i, used]
            fields.update(observed_mean_y_mm=float(obs_mean[0]), observed_mean_z_mm=float(obs_mean[1]),
                          predicted_mean_y_mm=float(pred_mean[0]), predicted_mean_z_mm=float(pred_mean[1]),
                          mean_error_y_mm=float(errors[:, 0].mean()), mean_error_z_mm=float(errors[:, 1].mean()),
                          vector_rmse_mm=float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))))
        observations.append(dict(frame_index=i, source_frame_id=int(source_frames[i]), depth_mm=float(depths[i]),
                                 fe_depth_mm=float(mapped_depths[i]), **fields, valid_point_count=n,
                                 coverage_fraction=n / point_count))
        frame_rows.append(dict(frame_index=i, source_frame_id=int(source_frames[i]), avi_source_frame_id=int(avi_frames[i]),
                               capture_time_from_video_reference_s=capture_times[i], depth_mm=float(depths[i]),
                               fe_depth_mm=float(mapped_depths[i]), fe_left_state_index=left_index,
                               fe_right_state_index=right_index, fe_right_weight=fraction,
                               status=statuses[i], support_mode=mode, used_point_count=n))
    after = {str(f): _sha(f) for f in input_files if f.is_file()}
    if before != after:
        raise ValueError("Source files changed during comparison; no report exported")
    output_dir.mkdir(parents=True, exist_ok=True)
    _csv(output_dir / "frames.csv", frame_rows)
    _csv(output_dir / "observables.csv", observations)
    _csv(output_dir / "points.csv", (dict(point_id=int(point_ids[i]), video_reference_y_mm=points[i, 0],
        video_reference_z_mm=points[i, 1], registered_y_mm=registered[i, 0], registered_z_mm=registered[i, 1]) for i in range(point_count)))
    _csv(output_dir / "mapping.csv", (dict(point_id=int(point_ids[i]), mapped=bool(mapped[i]),
        triangle_index=int(tri_index[i]) if mapped[i] else None, b0=weights[i, 0], b1=weights[i, 1], b2=weights[i, 2],
        reason="inside_reference_accepted_mesh" if mapped[i] else "outside_reference_domain_or_nonfinite") for i in range(point_count)))
    _csv(output_dir / "cohorts.csv", (dict(point_id=int(point_ids[i]), mapped=bool(mapped[i]), selected=bool(cohort[i]),
        membership="fixed_across_comparison_window" if mode == "continuous" else "endpoint_selection_per_frame",
        supported_frames_in_window=int(support[eligible, i].sum()), window_frames=int(eligible.sum()),
        weight=1.0 / int(cohort.sum()) if mode == "continuous" and cohort[i] else None) for i in range(point_count)))
    def trajectory_rows():
        for i in range(count):
            for j in range(point_count):
                used = masks[i][j]
                yield dict(frame_index=i, source_frame_id=int(source_frames[i]), point_id=int(point_ids[j]),
                           endpoint_valid=endpoint[i, j], continuous_valid=continuous[i, j], mapped=mapped[j], used=used,
                           observed_y_mm=observed[i, j, 0] if used else None, observed_z_mm=observed[i, j, 1] if used else None,
                           predicted_y_mm=predictions[i][j, 0] if used else None, predicted_z_mm=predictions[i][j, 1] if used else None)
    _csv(output_dir / "trajectories.csv.gz", trajectory_rows())
    normalized = dict(registration=registration, rotation_deg=rotation, translation_y_mm=float(translation[0]),
                      translation_z_mm=float(translation[1]), depth_offset_mm=offset, max_depth_mm=maximum,
                      fe_reference_index=ref_index, support_mode=mode, intact_status="unknown",
                      allow_angle_mismatch=options.get("allow_angle_mismatch") is True)
    warnings = ["Conditional image-axis registration and scale; not independent laboratory calibration.",
                "Video motion-onset reference is not certified mechanical first contact.",
                "No time synchronization or force-video synchronization is inferred.",
                "FE is intact plane strain; video material intactness and side-surface equivalence remain unknown.",
                "Tracking support does not certify original material identity.",
                "Saved-state depth interpolation is an observation operator, not a new equilibrium solution."]
    if mode == "endpoint":
        warnings.append("Endpoint cohorts can change by frame; their mean curves do not represent one fixed material cohort.")
    validation = dict(schema_version=SCHEMA, implementation_checks_passed=True,
                      laboratory_calibration_established=False, physical_accuracy_established=False,
                      time_synchronization_established=False, intact_status="unknown",
                      source_files_unchanged=True, source_frame_order_preserved=True,
                      video_depth_nonmonotone_steps=int(np.sum(np.diff(depths) < 0)),
                      total_video_points=point_count, mapped_points=int(mapped.sum()),
                      selected_cohort_points=int(cohort.sum()), comparison_frames=int(eligible.sum()),
                      frames_with_observations=sum(r["valid_point_count"] > 0 for r in observations),
                      has_supported_observations=any(r["valid_point_count"] > 0 for r in observations),
                      all_requested_frames_have_fe_support=bool(np.all(eligible[(depths >= depths[0]) & (depths <= maximum)])),
                      limitations=warnings)
    manifest = dict(schema_version=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
                    fe_source=str(path), video_source=str(video_path), sources_sha256=before, parameters=normalized,
                    implementation_sha256={p.name: _sha(p) for p in (Path(__file__), Path(__file__).with_name("video.py")) if p.is_file()},
                    angle=dict(fe_angle_deg=fe_angle, video_nominal_angle_deg=video_angle, status=angle_status),
                    qualification=dict(qualified_prefix_states=prefix, available_saved_indices=[f["index"] for f in fe_frames],
                                       original=bundle.get("metadata", {}).get("qualification", {})),
                    reference=dict(kind="accepted_fe_state", fe_state_index=ref_index, fe_depth_mm=reference["depth_mm"],
                                   video_source_frame=int(source_frames[0]), fe_tip_yz_mm=fe_tip.tolist(),
                                   position_rule="Q @ video_contact_image_position + translation",
                                   displacement_rule="same fixed barycentric weights applied to x_FE(t)-x_FE(reference)"),
                    units=dict(position="mm", displacement="mm", depth="mm; downward-positive blade travel", force="N"),
                    video_calibration=dict(scale_um_per_px=metadata.get("scale_um_per_px"), scale_status=metadata.get("scale_status"),
                                           coordinates=metadata.get("coordinates"), time=metadata.get("time"),
                                           contact_zero_method=metadata.get("contact_zero_method")),
                    validation=validation, files=["frames.csv", "points.csv", "mapping.csv", "cohorts.csv", "observables.csv", "trajectories.csv.gz", "validation.json", "index.html"])
    for name, value in (("manifest.json", manifest), ("validation.json", validation)):
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    warning_html = "".join(f"<li>{html.escape(text)}</li>" for text in warnings)
    links = " · ".join(f'<a href="{name}">{name}</a>' for name in manifest["files"] if name != "index.html")
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>视频—有限元材料位移比较</title><style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1060px;margin:40px auto;padding:0 24px;color:#223340;background:#f4f6f7}}section{{background:white;padding:24px;margin:18px 0;border-radius:10px}}h1{{font-size:28px}}svg{{width:100%;background:#fff}}a{{color:#006b83}}code{{overflow-wrap:anywhere}}.note{{border-left:4px solid #c38128}}</style>
<h1>视频—有限元材料位移比较</h1><p>条件诊断 · 固定材料点 · 原采集顺序 · {html.escape(mode)} 支持规则</p>
<section><strong>{html.escape(str(metadata.get('case_id', 'Video')))} ↔ {html.escape(str(bundle.get('metadata',{}).get('name','FE')))}</strong>
<p>参考域映射 {int(mapped.sum())}/{point_count} 点；{int(eligible.sum())}/{count} 帧位于共同范围。连续固定集合 {int(cohort.sum()) if mode == 'continuous' else '按端点逐帧变化'} 点。角度状态：{angle_status}。</p>
{_svg(observations)}<p>y：向右；z：向上。位移均相对各自选定参考帧。误差定义为 FE − 视频；空值不连线。</p></section>
<section class="note"><strong>本报告没有认证标定、首触、同步或完整性</strong><ul>{warning_html}</ul></section>
<section><h2>数据与来源</h2><p>{links} · <a href="manifest.json">manifest.json</a></p><p>所有原始输入只读，源 SHA256、坐标变换、资格范围与实际插值状态记录于 manifest 和 frames。</p></section></html>'''
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return dict(schema_version=SCHEMA, report_path=str(output_dir / "index.html"), output_dir=str(output_dir),
                manifest_path=str(output_dir / "manifest.json"), validation=validation,
                observables_path=str(output_dir / "observables.csv"))
