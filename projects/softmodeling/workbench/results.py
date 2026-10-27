"""Portable intact-Maxwell results from accepted, audited saved states.

This is a read-only postprocessor, not a material update or loading solver.
The alpha=2 isochoric Maxwell stress uses the COMMITTED inverse viscous metric.
Unsupported constitutive laws fail closed. An old source hash is provenance,
not grounds to discard a saved field whose explicit equation is supported here.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

SCHEMA_VERSION = "softcut-result/1"
FIELD_INFO = {
    "tensile_MPa": dict(label="Maximum principal tensile Cauchy stress", unit="MPa", association="element"),
    "green_max": dict(label="Maximum principal Green-Lagrange strain (3D plane strain)", unit="1", association="element"),
    "green_min": dict(label="Minimum principal Green-Lagrange strain (3D plane strain)", unit="1", association="element"),
    "J": dict(label="Deformation Jacobian", unit="1", association="element"),
    "displacement_mm": dict(label="Displacement magnitude from undeformed reference", unit="mm", association="node"),
}


def _digest(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _material_parameters(material):
    if (material.get("viscous_flow") != "isochoric_maxwell"
            or material.get("volumetric") not in ("quadratic_j", "log_j")
            or material.get("alpha_eq") not in ([2.], (2.,))
            or material.get("alpha_ov") not in ([2.], (2.,))
            or len(material.get("mu_eq", [])) != 1 or len(material.get("mu_ov", [])) != 1):
        raise ValueError("Field export supports one alpha=2 equilibrium/isochoric-Maxwell branch only")
    ge, gv, nu = float(material["mu_eq"][0]), float(material["mu_ov"][0]), float(material["poisson"])
    if not np.isfinite([ge, gv, nu]).all() or ge <= 0 or gv <= 0 or not 0 < nu < .5:
        raise ValueError("Invalid saved material parameters")
    return ge, gv, 2 * ge * (1 + nu) / (3 * (1 - 2 * nu))


def saved_state_fields(reference, positions, triangles, metric, material, *, include_mechanics=False):
    """Return unsmoothed P1 fields and stored energy density, without integration.

    Cauchy stress = J^(-5/3) [Ge dev(FF^T) + Gv dev(F A F^T)]
                   + volumetric pressure I, with F33=1 and det(A)=1.
    Principal strains include the zero out-of-plane Green strain. Energy is per
    reference volume (N/mm²), integrated separately with reference area * width.
    """
    ge, gv, bulk = _material_parameters(material)
    X, x, tri, A = (np.asarray(reference, dtype=float), np.asarray(positions, dtype=float),
                     np.asarray(triangles, dtype=int), np.asarray(metric, dtype=float))
    if X.ndim != 2 or X.shape[1] != 2 or x.shape != X.shape or not np.isfinite([X, x]).all():
        raise ValueError("Expected finite body reference and current xy arrays")
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or tri.min() < 0 or tri.max() >= len(X):
        raise ValueError("Invalid body triangle indices")
    if A.shape != (len(tri), 3, 3) or not np.isfinite(A).all():
        raise ValueError("Missing/invalid committed material metric")
    if (not np.allclose(A, A.transpose(0, 2, 1), atol=1e-12, rtol=1e-10)
            or np.min(np.linalg.eigvalsh(A)) <= 0 or np.max(np.abs(np.linalg.det(A) - 1)) > 1e-8):
        raise ValueError("Committed metric must be symmetric positive definite and isochoric")
    def edges(points):
        return np.stack((points[tri[:, 1]] - points[tri[:, 0]],
                         points[tri[:, 2]] - points[tri[:, 0]]), axis=-1)
    Dm = edges(X)
    if np.any(np.linalg.det(Dm) <= 0):
        raise ValueError("Non-positive reference element area")
    F = np.broadcast_to(np.eye(3), (len(tri), 3, 3)).copy()
    F[:, :2, :2] = edges(x) @ np.linalg.inv(Dm)
    J = np.linalg.det(F)
    if np.any(J <= 0):
        raise ValueError("Non-positive saved deformation Jacobian")
    B, Be = F @ F.transpose(0, 2, 1), F @ A @ F.transpose(0, 2, 1)
    def dev(tensor):
        return tensor - np.trace(tensor, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    stress = J[:, None, None]**(-5/3) * (ge * dev(B) + gv * dev(Be))
    energy = .5 * J**(-2/3) * (ge * np.trace(B, axis1=1, axis2=2)
                              + gv * np.trace(Be, axis1=1, axis2=2)) - 1.5 * (ge + gv)
    if material["volumetric"] == "quadratic_j":
        pressure = bulk * (J - 1)
        energy += .5 * bulk * (J - 1)**2
    else:
        pressure = .5 * bulk * (J - 1/J)
        energy += .25 * bulk * (J*J - 1 - 2*np.log(J))
    stress += pressure[:, None, None] * np.eye(3)
    strain = .5 * (F.transpose(0, 2, 1) @ F - np.eye(3))
    strain_eigenvalues = np.linalg.eigvalsh(strain)
    fields = dict(tensile_MPa=np.maximum(np.linalg.eigvalsh(stress)[:, -1], 0),
                  green_max=strain_eigenvalues[:, -1], green_min=strain_eigenvalues[:, 0],
                  J=J, displacement_mm=np.linalg.norm(x-X, axis=1))
    if include_mechanics:
        principal_stress, directions = np.linalg.eigh(stress)
        equilibrium = .5 * ge * (J**(-2/3) * np.trace(B, axis1=1, axis2=2) - 3)
        nonequilibrium = .5 * gv * (J**(-2/3) * np.trace(Be, axis1=1, axis2=2) - 3)
        # The non-relaxing volumetric energy belongs to the equilibrium branch.
        equilibrium += energy - equilibrium - nonequilibrium
        inplane, inplane_directions = np.linalg.eigh(stress[:, :2, :2])
        fields.update(F=F, cauchy_MPa=stress, principal_stress_MPa=principal_stress,
                      principal_directions=directions,
                      principal_stretch=np.sqrt(np.maximum(1+2*strain_eigenvalues, 0)),
                      inplane_tensile_MPa=np.maximum(inplane[:, -1], 0),
                      inplane_normal=inplane_directions[:, :, -1],
                      equilibrium_density_MPa=equilibrium,
                      nonequilibrium_density_MPa=nonequilibrium)
    if not all(np.isfinite(value).all() for value in fields.values()):
        raise ValueError("Non-finite reconstructed fields")
    return fields, energy, .5 * np.linalg.det(Dm)


def _qualified_prefix(report, audit):
    """Use the existing audit's continuous passed prefix; do not fill holes."""
    rows = audit.get("rows", []) if audit else []
    if audit and audit.get("history_chain_checks_passed") is False:
        return 0, "audit reports an invalid material/friction history chain"
    prefix, reason = 0, "audit absent or missing row"
    declared_prefix = audit.get("qualified_prefix_states") if audit else None
    for i, step in enumerate(report["steps"]):
        if declared_prefix is not None and i >= declared_prefix:
            reason = "end of explicitly qualified audit prefix"
            break
        if i >= len(rows):
            break
        row = rows[i]
        if not row.get("passed"):
            reason = "first failed audit row; later states are not a qualified history"
            break
        for key in ("depth_mm", "reaction_fz_n", "min_j", "residual_n"):
            if key not in row or not np.isclose(float(row[key]), float(step[key]), rtol=1e-8, atol=1e-8):
                raise ValueError(f"Audit/result mismatch at state {i}: {key}")
        if not isinstance(row.get("checkpoint_sha256"), str):
            raise ValueError(f"Audit row {i} has no checkpoint fingerprint")
        prefix += 1
    if prefix == len(report["steps"]):
        if audit.get("passed") is False:
            return 0, "audit failed outside per-state rows; no complete qualification"
        reason = None
    return prefix, reason


def _case_label(report):
    """Physical labels distinguish exported case folders across parameter sets."""
    signature = report.get("signature", {})
    config, material = signature.get("config", {}), signature.get("material", {})
    def shown(value):
        return "?" if value is None else f"{float(value):.4g}"
    angle = shown(config.get("inclination_deg"))
    ge, gv = shown(material["mu_eq"][0]), shown(material["mu_ov"][0])
    mu = shown(config.get("friction_coefficient"))
    radius = shown(report.get("settings", {}).get("tip_radius"))
    return f"{angle}° · G∞/Gv={ge}/{gv} MPa · μ={mu} · r={radius} mm"


def export_case(source, output, *, max_frames=11, movie=False, audit_path=None, fps=8):
    """Write curves.csv and a portable result.bundle.json, returning that bundle.

    Select actual saved frames, always including both qualified endpoints. All
    force rows are retained; rows beyond the audit's contiguous passed prefix
    have null public quantities. Raw source files remain unchanged. The existing
    audit is reused, not rerun, and selected checkpoint fingerprints are checked.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_file():
        source = source.parent
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or not 2 <= max_frames <= 201:
        raise ValueError("max_frames must be an integer in [2, 201]")
    result_path = source / "result.json"
    result_bytes = result_path.read_bytes()
    report = json.loads(result_bytes)
    steps = report.get("steps", [])
    if report.get("schema") == "softcut-cohesive-result/1" or any("fracture" in step for step in steps):
        raise ValueError("Intact result export cannot consume a coupled fracture history; use its fracture report")
    if not steps:
        raise ValueError("No accepted steps in result.json")
    depths = np.asarray([s["depth_mm"] for s in steps], float)
    if not np.isfinite(depths).all() or np.any(np.diff(depths) <= 0):
        raise ValueError("Accepted depths must be finite and strictly increasing")
    if report.get("discretization", {}).get("element", "p1") != "p1":
        raise ValueError("Only saved P1 fields are supported")
    signature = report.get("signature", {})
    material = signature.get("material", {})
    _material_parameters(material)
    audit_path = Path(audit_path).resolve() if audit_path else source / "audit.json"
    audit_bytes = audit_path.read_bytes() if audit_path.is_file() else None
    audit = json.loads(audit_bytes) if audit_bytes is not None else None
    prefix, reason = _qualified_prefix(report, audit)
    selected = np.unique(np.linspace(0, prefix - 1, min(prefix, max_frames)).round().astype(int)) if prefix else []
    series = []
    for i, step in enumerate(steps):
        qualified = i < prefix
        def number(value):
            if not qualified or value is None:
                return None
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"Non-finite accepted quantity at state {i}")
            return value
        raw_fz, raw_fy = step.get("reaction_fz_n"), step.get("reaction_x_n")
        if raw_fz is not None and not np.isfinite(raw_fz) or raw_fy is not None and not np.isfinite(raw_fy):
            raise ValueError(f"Non-finite force in source state {i}")
        fz, fy = number(raw_fz), number(raw_fy)
        contact, mat = step.get("tip_contact") or {}, step.get("material") or {}
        series.append(dict(depth_mm=float(step["depth_mm"]), qualified=qualified,
            qualification="audited_prefix" if qualified else "unqualified_masked",
            fz_N=fz, fy_N=fy, fyz_N=None if fz is None or fy is None else float(np.hypot(fz, fy)),
            raw_fz_N=raw_fz, raw_fy_N=raw_fy,
            raw_fyz_N=None if raw_fz is None or raw_fy is None else float(np.hypot(raw_fz, raw_fy)),
            time_s=number(mat.get("time_s")), residual_N=number(step.get("residual_n")),
            dt_s=number(mat.get("dt_s")),
            min_J=number(step.get("min_j")), stored_energy_Nmm=number(mat.get("stored_energy_n_mm")),
            equilibrium_energy_Nmm=number(mat.get("equilibrium_energy_n_mm")),
            overstress_energy_Nmm=number(mat.get("overstress_energy_n_mm")),
            viscous_dissipation_Nmm=number(mat.get("cumulative_viscous_dissipation_n_mm")),
            viscous_dissipation_increment_Nmm=number(mat.get("viscous_dissipation_increment_n_mm")),
            friction_work_increment_Nmm=number(step.get("friction_work_n_mm")),
            normal_fz_N=number(step.get("reaction_normal_fz_n")),
            friction_fz_N=number(step.get("reaction_friction_fz_n")),
            tip_contact_status=contact.get("status", "not_recorded") if qualified else "unqualified",
            contact_reference_offset_mm=number(contact.get("reference_offset_mm")) if contact.get("status") == "unique" else None))
    frames, hashes, mesh = [], {}, None
    checks = []
    for index in selected:
        index = int(index)
        path = source / "states" / f"{index:04d}.npz"
        sha = _digest(path)
        if sha != audit["rows"][index]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint fingerprint mismatch: {path.name}")
        hashes[path.relative_to(source).as_posix()] = sha
        with np.load(path, allow_pickle=False) as state:
            count = int(state["body_nodes"])
            X, x, tri = state["reference_xy"][:count], state["positions"][:count], state["triangles"]
            if not np.isclose(float(state["depth_mm"]), depths[index], rtol=0, atol=1e-12):
                raise ValueError("Saved state depth differs from accepted record")
            fields, energy, areas = saved_state_fields(X, x, tri, state["material_metric"], material)
            current_mesh = dict(reference_positions=X.tolist(), triangles=tri.tolist())
            if mesh is None:
                mesh = current_mesh
            elif current_mesh != mesh:
                raise ValueError("Selected states do not share an identical reference mesh")
            outline = report.get("blade_discretization", {}).get("outline_vertex_count")
            blade_count = len(state["positions"]) - count
            if outline is None:
                # Older unsubdivided outputs contain only a canonical polygon.
                if report.get("blade_discretization", {}).get("spacing_mm") is not None:
                    raise ValueError("Subdivided blade requires an explicit ordered outline count")
                outline = blade_count
            outline = int(outline)
            if outline < 3 or outline > blade_count:
                raise ValueError("Invalid blade outline count")
            blade = state["positions"][count:count+outline].tolist()
        if not np.isclose(fields["J"].min(), steps[index]["min_j"], atol=1e-10, rtol=1e-9):
            raise ValueError("Reconstructed minimum J differs from accepted state")
        width = float(signature["config"]["width_mm"])
        stored = float(width * (areas @ energy))
        expected = float(steps[index]["material"]["stored_energy_n_mm"])
        if not np.isclose(stored, expected, rtol=1e-8, atol=1e-9):
            raise ValueError("Committed-state stored energy does not reproduce accepted record")
        checks.append(dict(index=index, energy_difference_Nmm=stored-expected))
        frames.append(dict(index=index, depth_mm=float(depths[index]), positions=x.tolist(), blade=blade,
                           fields={key: value.tolist() for key, value in fields.items()},
                           tip_contact=steps[index].get("tip_contact")))
    info = {key: dict(value, range=[min(min(f["fields"][key]) for f in frames),
                                  max(max(f["fields"][key]) for f in frames)])
            for key, value in FIELD_INFO.items()} if frames else {}
    bundle = dict(schema_version=SCHEMA_VERSION, metadata=dict(
        name=_case_label(report), source=str(source), exported_utc=datetime.now(timezone.utc).isoformat(),
        result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        audit_sha256=hashlib.sha256(audit_bytes).hexdigest() if audit_bytes is not None else None,
        source_sha256=report.get("source_sha256", {}), signature=signature,
        angle_deg=signature.get("config", {}).get("inclination_deg"),
        target_mm=report.get("target_mm"), completed=bool(report.get("completed")),
        qualification=dict(qualified_states=prefix, accepted_states=len(steps),
                           qualified_end_mm=float(depths[prefix-1]) if prefix else None,
                           complete_path_qualified=prefix == len(steps) and bool(report.get("completed")),
                           stop_reason=reason, audit_path=str(audit_path) if audit else None,
                           scope="Existing physical audit; selected saved-state SHA/J/energy rechecked. No new equilibrium/history solve."),
        coordinates=dict(positions="[y,z] mm; public y equals solver x, z upward", depth="vertical downward blade travel, mm",
                         fy_N="reaction_x_n: signed force on blade along laboratory horizontal (solver x)",
                         fz_N="reaction_fz_n: signed upward force on blade; positive indentation resistance",
                         fyz_N="sqrt(fy_N^2+fz_N^2); in-plane resultant, no third component",
                         raw_forces="raw_fz_N/raw_fy_N/raw_fyz_N preserve accepted-record values even outside qualified prefix; not certified",
                         force_zero="Raw simulated forces retained, including finite barrier preload at depth zero"),
        field_method="Direct alpha=2 stress from saved F and committed metric, F33=1; no return map, smoothing or history update",
        field_checks=checks, selected_checkpoints_sha256=hashes,
        animation="Selected accepted states only; uniform playback is not physical time; displacement scale 1",
        limitations=["Intact model: no crack prediction", "Computational qualification is not experimental calibration",
                     "Contact reference-material offset is not accumulated slip", "Extrema depend on mesh and tip discretization"],
        movie=dict(status="not_requested")), series=series, mesh=mesh, frames=frames, field_info=info)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "curves.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(series[0]))
        writer.writeheader()
        writer.writerows(series)
    if movie:
        partial_movie = output / "indentation.partial.mp4"
        try:
            render_movie(bundle, partial_movie, fps=fps)
            partial_movie.replace(output / "indentation.mp4")
            bundle["metadata"]["movie"] = dict(status="completed", file="indentation.mp4", fps=fps)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            partial_movie.unlink(missing_ok=True)
            bundle["metadata"]["movie"] = dict(status="failed", reason=str(error))
    _write_json(output / "result.bundle.json", bundle)
    return bundle


def render_movie(bundle, destination, *, field="tensile_MPa", fps=8):
    """MP4 of the actual selected frames, with one fixed, explicit field scale."""
    if not bundle.get("frames"):
        raise ValueError("No qualified frames available for a movie")
    if not isinstance(fps, int) or isinstance(fps, bool) or not 1 <= fps <= 60:
        raise ValueError("fps must be an integer in [1, 60]")
    info = bundle["field_info"].get(field)
    if info is None or info["association"] != "element":
        raise ValueError("Movie field must be an exported element scalar")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg is not on PATH; JSON/CSV export remains available")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import animation, colors, pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Polygon
    frames, triangles = bundle["frames"], np.asarray(bundle["mesh"]["triangles"], dtype=int)
    points = np.concatenate([np.asarray(frame["positions"]) for frame in frames])
    limits = [points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()]
    lower, upper = info["range"]
    if upper == lower:
        upper = lower + max(1e-12, abs(lower)*1e-6)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    first_blade, last_blade = np.asarray(frames[0]["blade"]), np.asarray(frames[-1]["blade"])
    tip = first_blade[np.argmin(first_blade[:, 1])]
    contact_limits = [tip[0]-2., tip[0]+2., last_blade[:, 1].min()-1.5, tip[1]+.8]
    with plt.rc_context({"animation.ffmpeg_path": ffmpeg, "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained",
                                 gridspec_kw={"width_ratios": [.85, 1.3]})
        artists = []
        for ax, bounds, label in zip(axes, [limits, contact_limits], ["Whole specimen", "Contact region (fixed view)"]):
            body = PolyCollection([], cmap="viridis", norm=colors.Normalize(lower, upper),
                                  edgecolors="none", linewidths=0, antialiaseds=False)
            ax.add_collection(body)
            blade = Polygon(first_blade, color="#555c68")
            ax.add_patch(blade)
            ax.set(xlim=(bounds[0]-.1, bounds[1]+.1), ylim=(bounds[2]-.1, bounds[3]+.1),
                   aspect="equal", xlabel="Laboratory y (mm; solver x)", ylabel="Laboratory z (mm)", title=label)
            artists.append((body, blade))
        title = fig.suptitle("")
        fig.colorbar(body, ax=axes[1], label=f"{info['label']} ({info['unit']})")
        fig.supxlabel("Accepted sampled states; no displacement amplification or stress smoothing. Playback is not physical time.", fontsize=8)
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2000,
                                        extra_args=["-pix_fmt", "yuv420p", "-threads", "2"])
        try:
            with writer.saving(fig, str(destination), dpi=120):
                for frame in frames:
                    for body, blade in artists:
                        body.set_verts(np.asarray(frame["positions"])[triangles])
                        body.set_array(np.asarray(frame["fields"][field]))
                        blade.set_xy(np.asarray(frame["blade"]))
                    title.set_text(f"Intact indentation | depth {frame['depth_mm']:.3f} mm")
                    writer.grab_frame(facecolor="white")
        finally:
            plt.close(fig)
    return destination
