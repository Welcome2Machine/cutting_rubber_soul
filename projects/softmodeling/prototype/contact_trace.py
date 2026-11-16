"""Export and query material coordinates of accepted intact IPC tip contact.

No native contact library or new equilibrium solve is needed for offline export.
The query returns accepted depths (or brackets), never inventing a contact point.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_support import write_json
from .tip_contact import TipContactTracker


def write_trace_csv(path, steps):
    """Atomically rewrite the small accepted-state table; blank means undefined."""
    path = Path(path)
    fields = ["depth_mm", "status", "reference_offset_mm", "absolute_reference_offset_mm",
              "reference_s_mm", "gap_mm", "edge_alpha", "edge_nodes", "reference_point_mm",
              "current_point_mm", "knife_tip_mm", "nose_status", "nose_offset_intervals_mm"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            item = dict(step.get("tip_contact") or {}, depth_mm=step["depth_mm"])
            row = {key: item.get(key) for key in fields}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, separators=(",", ":"), allow_nan=False)
            writer.writerow(row)
    temporary.replace(path)


def query_depth(steps, depth):
    depths = np.asarray([step["depth_mm"] for step in steps], dtype=float)
    if (not np.isfinite(depth) or len(depths) == 0 or not np.isfinite(depths).all()
            or np.any(np.diff(depths) <= 0)):
        raise ValueError("Depths must be finite and strictly increasing")
    exact = np.flatnonzero(np.isclose(depths, depth, atol=1e-10, rtol=0))
    if len(exact):
        indices, kind = [int(exact[0])], "exact_accepted_depth"
    elif depth < depths[0] or depth > depths[-1]:
        raise ValueError(f"Depth outside accepted range [{depths[0]}, {depths[-1]}] mm")
    else:
        upper = int(np.searchsorted(depths, depth))
        indices, kind = [upper - 1, upper], "bracketing_accepted_depths"
    return dict(requested_depth_mm=float(depth), kind=kind, interpolated=False,
                samples=[dict(depth_mm=steps[i]["depth_mm"],
                              tip_contact=steps[i].get("tip_contact")) for i in indices])


def validate_loading_schedule(report, *, require_complete=False):
    """Validate recorded accepted depths, including adaptive substeps.

    A partial adaptive history may end between output anchors. Every anchor
    already crossed must be a saved accepted state; nothing is interpolated.
    The report's original target is used even when a caller plans an extension.
    """
    tolerance = 1e-10
    depths = np.asarray([step["depth_mm"] for step in report["steps"]], dtype=float)
    increment = float(report.get("loading_increment_mm", report.get("settings", {}).get("increment", np.nan)))
    target = float(report["target_mm"])
    if (not np.isfinite(increment) or increment <= 0 or not np.isfinite(target) or target <= 0
            or len(depths) == 0 or not np.isfinite(depths).all()
            or abs(depths[0]) > tolerance or np.any(np.diff(depths) <= 0)
            or depths[-1] > target + tolerance):
        raise ValueError("Invalid accepted loading schedule")
    if (require_complete or report.get("completed") is True) and abs(depths[-1] - target) > tolerance:
        raise ValueError("History does not span zero to target")
    expected = np.r_[0., np.arange(increment, target - 1e-12, increment), target]
    control = report.get("load_control")
    if control is None:
        if len(depths) > len(expected) or not np.allclose(depths, expected[:len(depths)],
                                                         atol=tolerance, rtol=0):
            raise ValueError("Accepted depths do not match the fixed loading schedule")
    else:
        if not isinstance(control, dict) or control.get("mode") != "adaptive_bisection":
            raise ValueError("Unknown loading control")
        minimum = float(control["minimum_increment_mm"])
        maximum = float(control["maximum_increment_mm"])
        output_grid = float(control["output_grid_increment_mm"])
        if (not np.isfinite([minimum, maximum, output_grid]).all() or not 0 < minimum <= maximum
                or abs(maximum - increment) > tolerance or abs(output_grid - increment) > tolerance):
            raise ValueError("Invalid adaptive loading control")
        if np.any(np.diff(depths) > maximum + tolerance):
            raise ValueError("Accepted substep exceeds the maximum increment")
        for anchor in expected[expected <= depths[-1] + tolerance]:
            if not np.isclose(depths, anchor, atol=tolerance, rtol=0).any():
                raise ValueError(f"Missing accepted output-grid depth {anchor:g} mm")
        # Do not impose minimum on accepted intervals: a target/anchor remainder
        # can be smaller. The solver controls how many failed trials are retried.
    return depths


def record_digest(value):
    """Canonical JSON fingerprint, shared with the existing independent audit."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def geometry_metadata(report):
    """Reconcile explicit geometry sources; no Maxwell facet-count default."""
    sig, resolved = report.get('signature', {}), report.get('resolved_model', {})
    settings, simple = report.get('settings', {}), report.get('simplifications', {})
    def agree(name, values, required=True):
        values = [v for v in values if v is not None]
        if not values:
            if required:
                raise ValueError('Missing intact geometry metadata: '+name)
            return None
        first = values[0]
        if any(not np.isfinite(v) or not np.isclose(v, first, rtol=0, atol=1e-12) for v in values):
            raise ValueError('Conflicting geometry metadata: '+name)
        return first
    configs = [c for c in (sig.get('config'), resolved.get('config'), report.get('config')) if c]
    if not configs:
        raise ValueError('Missing intact geometry metadata: config')
    config = configs[0]
    for other in configs[1:]:
        for key in config.keys() & other.keys():
            if config[key] != other[key]:
                raise ValueError('Conflicting geometry config: '+key)
    outline = report.get('blade_discretization', {}).get('outline_vertex_count')
    facets = agree('nose_facets', [resolved.get('geometry', {}).get('nose_facets'),
        settings.get('nose_facets'), simple.get('nose_facets'), sig.get('nose_facets'),
        outline-5 if outline is not None else None])
    if isinstance(facets, bool) or int(facets) != facets:
        raise ValueError('Invalid nose facet count')
    dhat = agree('dhat', [sig.get('dhat'), report.get('dhat_mm'), settings.get('dhat'),
                          resolved.get('contact', {}).get('dhat_mm')])
    start = agree('initial_depth_mm', [resolved.get('initial_depth_mm'),
        -settings['preparation_clearance_mm'] if settings.get('preparation_clearance_mm') is not None else None], False)
    return config, float(dhat), int(facets), start


def audit_coverage(source, report, audit_path=None, manifest_path=None):
    """Authenticate an existing contiguous audit, including legacy manifests.

    A new export fingerprint is provenance only. A legacy audit without source
    binding is usable only with a prior manifest binding result AND audit bytes.
    """
    from workbench.results import _qualified_prefix
    path = Path(audit_path) if audit_path else source/'audit.json'
    if not path.is_file():
        return 0, dict(reason='independent audit absent')
    audit_bytes = path.read_bytes()
    audit = json.loads(audit_bytes)
    prefix, reason = _qualified_prefix(report, audit)
    binding = audit.get('source_binding')
    provenance = dict(audit_path=str(path), audit_sha256=hashlib.sha256(audit_bytes).hexdigest(), reason=reason)
    if binding:
        for field, key in (('signature_sha256', 'signature'), ('producer_sources_sha256', 'source_sha256'),
                           ('settings_sha256', 'settings')):
            if binding.get(field) != record_digest(report.get(key)):
                raise ValueError('Audit source binding mismatch: '+field)
        for i in range(prefix):
            if audit['rows'][i].get('accepted_record_sha256') != record_digest(report['steps'][i]):
                raise ValueError(f'Audit accepted record mismatch: {i}')
        if audit.get('source_result_sha256') != hashlib.sha256((source/'result.json').read_bytes()).hexdigest():
            raise ValueError('Audit lacks matching original final result bytes; use an independently bound sidecar')
        provenance['binding'] = 'original audit final result bytes, signature, sources, settings and each accepted record'
    elif manifest_path:
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_bytes())
        entries = {e['path']: e['sha256'] for e in manifest['files']}
        for target in (source/'result.json', path.resolve()):
            relative = target.relative_to(manifest_path.parent).as_posix()
            if entries.get(relative) != hashlib.sha256(target.read_bytes()).hexdigest():
                raise ValueError('Legacy manifest binding mismatch: '+relative)
        provenance.update(binding='prior manifest binds original result and audit bytes',
                          manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    else:
        return 0, dict(provenance, reason='legacy audit lacks source binding; supply its prior manifest or re-audit')
    provenance['rows'] = audit['rows'][:prefix]
    return prefix, provenance


def accumulate_motion(tracker, motion, contact, prior_contact, prior_motion, cumulative, dt, *, continuity=None):
    """Endpoint/right-rule proximity increments; no integration across gaps."""
    result = dict(motion, segment_id=None, break_reason=None, continuity_check=continuity,
                  signed_segment_mm=None, absolute_segment_mm=None, migration_rate_mm_s=None)
    if not motion['valid']:
        result['break_reason'] = motion['status']
        return result
    reason = None
    if not prior_contact or prior_contact.get('status') != 'unique' or not prior_motion or not prior_motion['valid']:
        reason = 'first_valid_endpoint_or_contact_birth'
    else:
        old_edge, edge = prior_contact['edge_nodes'], contact['edge_nodes']
        if (old_edge != edge and not set(old_edge).intersection(edge)
                and not (continuity and continuity['continuous'])):
            reason = 'nonadjacent_projection_jump'
    if reason:
        cumulative.update(segment_id=cumulative.get('segment_id', -1)+1, signed=0., absolute=0.)
    else:
        increment = motion['transported_mm_s']*dt
        cumulative['signed'] += increment
        cumulative['absolute'] += abs(increment)
        result['migration_rate_mm_s'] = (contact['reference_s_mm']-prior_contact['reference_s_mm'])/dt
    result.update(segment_id=cumulative['segment_id'], break_reason=reason,
                  signed_segment_mm=cumulative['signed'], absolute_segment_mm=cumulative['absolute'])
    return result


def export_trace(source, output, *, qualified_prefix=False, audit_path=None, manifest_path=None):
    """Read all selected accepted states. Never mutate or promote a source path."""
    source, output = Path(source).resolve(), Path(output).resolve()
    report_bytes = (source/'result.json').read_bytes()
    report = json.loads(report_bytes)
    sig = report.get('signature', {})
    maxwell = sig.get('material', {}).get('viscous_flow') == 'isochoric_maxwell'
    intact = (maxwell or (report.get('case') == 'oblique' and report.get('simplifications', {}).get('fracture') is False))
    if (not intact or sig.get('fracture') or any('fracture' in s for s in report['steps'])
            or report.get('resolved_model', {}).get('damage_off_semantics', 'continuous_intact') != 'continuous_intact'):
        raise ValueError('Offline export requires a completed intact history or explicit qualified intact prefix')
    if not qualified_prefix and (report.get('status') != 'completed' or report.get('completed') is not True):
        raise ValueError('Offline export requires a completed intact history; select --qualified-prefix explicitly')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Export must be separate from the original run')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Export directory must be empty')
    config, dhat, facets, declared_start = geometry_metadata(report)
    validate_loading_schedule(report, require_complete=not qualified_prefix)
    prefix, qualification = audit_coverage(source, report, audit_path, manifest_path)
    if qualified_prefix and not prefix:
        raise ValueError('No authenticated qualified prefix: '+str(qualification.get('reason')))
    steps = report['steps'][:prefix] if qualified_prefix else report['steps']
    audit_rows = qualification.pop('rows', [])
    tracker, prior, prior_metric, prior_depth = None, None, None, None
    hashes, observed, snapshots = {}, [], []
    display_indices = set(np.linspace(0, len(steps)-1, min(4, len(steps))).round().astype(int))
    prior_contact, prior_motion, cumulative = None, None, {}
    time_from_initial = 0.
    first_valid = None
    for i, step in enumerate(steps):
        if (not np.isfinite(step['residual_n']) or not np.isfinite(step['min_j'])
                or step['residual_n'] > min(report['force_tolerance_n'], 1e-4) or step['min_j'] <= 0):
            raise ValueError(f'State {i} has unaccepted reported diagnostics')
        relative = Path('states')/f'{i:04d}.npz'
        path = source/relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative.as_posix()] = digest
        if i < prefix and audit_rows[i]['checkpoint_sha256'] != digest:
            raise ValueError(f'Audit checkpoint fingerprint mismatch: {i}')
        with np.load(path, allow_pickle=False) as state:
            x, previous = state['positions'], state['previous_positions']
            n = int(state['body_nodes'])
            if tracker is None:
                reference, triangles = state['reference_xy'].copy(), state['triangles'].copy()
                tracker = TipContactTracker(reference, triangles, n, config, dhat, facets)
                prior, body_nodes = reference, n
                zero_knife = x[n:].copy()
                shifts = zero_knife-reference[n:]
                start = float(shifts[:, 1].mean())
                np.testing.assert_allclose(shifts, np.tile([0., start], (len(shifts), 1)), atol=1e-9, rtol=0)
                if declared_start is None:
                    # Legacy compatibility is proved by saved rigid positions and
                    # the old documented construction, not an assumed facet count.
                    legacy = -2*dhat/max(np.cos(np.deg2rad(config['inclination_deg'])), .1)
                    np.testing.assert_allclose(start, legacy, atol=1e-9, rtol=0)
                else:
                    np.testing.assert_allclose(start, declared_start, atol=1e-9, rtol=0)
                np.testing.assert_allclose(start, tracker.metadata['initial_depth_mm'], atol=1e-9, rtol=0)
                prior_depth = start
                dm = np.stack((reference[triangles[:, 1]]-reference[triangles[:, 0]],
                               reference[triangles[:, 2]]-reference[triangles[:, 0]]), axis=2)
                reference_det = np.linalg.det(dm)
                if np.any(reference_det == 0):
                    raise ValueError('Degenerate reference triangle')
                definition = tracker.fixed_material_definition()
                prior_contact = tracker.observe(prior, start)
                prior_metric = np.tile(np.eye(3), (len(triangles), 1, 1)) if maxwell else None
                geometry = report.get('resolved_model', {}).get('geometry', {})
                for key, array in (('blade_reference_sha256', reference[n:]),
                                   ('body_reference_sha256', reference[:n]), ('triangles_sha256', triangles)):
                    if key in geometry and hashlib.sha256(array.tobytes()).hexdigest() != geometry[key]:
                        raise ValueError('Resolved topology hash mismatch: '+key)
            np.testing.assert_array_equal(state['reference_xy'], reference)
            np.testing.assert_array_equal(state['triangles'], triangles)
            np.testing.assert_array_equal(previous, prior)
            if n != body_nodes or x.shape != reference.shape or not np.isfinite(x).all():
                raise ValueError(f'State {i} has invalid geometry')
            np.testing.assert_allclose(float(state['depth_mm']), step['depth_mm'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(x[n:], zero_knife-[0., step['depth_mm']], atol=1e-9, rtol=0)
            ds = np.stack((x[triangles[:, 1]]-x[triangles[:, 0]], x[triangles[:, 2]]-x[triangles[:, 0]]), axis=2)
            minimum_j = float(np.min(np.linalg.det(ds)/reference_det))
            if minimum_j <= 0:
                raise ValueError(f'State {i} contains an inverted body element')
            np.testing.assert_allclose(minimum_j, step['min_j'], atol=1e-9, rtol=1e-7)
            dt = None
            if maxwell:
                dt = float(state['material_dt_s'])
                rate = float(sig['loading_rate'])
                np.testing.assert_allclose(dt, (step['depth_mm']-prior_depth)/rate, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(dt, step['material']['dt_s'], atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(state['previous_material_metric'], prior_metric)
                prior_metric = state['material_metric'].copy()
                time_from_initial += dt
                if 'elapsed_from_initial_s' in step['material']:
                    np.testing.assert_allclose(time_from_initial, step['material']['elapsed_from_initial_s'], atol=1e-11, rtol=1e-12)
            contact = tracker.observe(x, step['depth_mm'])
            footprint = tracker.observe_footprint(x, step['depth_mm'])
            # Old observations used the same definitions. New metadata is not
            # silently inferred by display code, and parity covers full objects.
            for key, value in (('tip_contact', contact), ('blade_contact_footprint', footprint)):
                if step.get(key, {}).get('status') != 'error' and key in step and step[key] != value:
                    raise ValueError(f'Online/offline observation mismatch: {i}/{key}')
            motion = dict(valid=False, status='physical_time_unavailable') if dt is None else tracker.relative_motion(x, previous, contact, dt)
            continuity = None
            if (motion['valid'] and prior_motion and prior_motion['valid']
                    and prior_contact['status'] == 'unique'
                    and not set(contact['edge_nodes']).intersection(prior_contact['edge_nodes'])):
                continuity = tracker.projection_continuity(previous, x, prior_contact, contact)
            motion = accumulate_motion(tracker, motion, contact, prior_contact, prior_motion, cumulative, dt,
                                       continuity=continuity)
            if first_valid is None and contact['status'] == 'unique':
                first_valid = dict(index=i, bracket_depth_mm=[prior_depth, step['depth_mm']], reference_s_mm=contact['reference_s_mm'])
            epsilon = report.get('resolved_model', {}).get('friction', {}).get('epsilon_u_mm',
                report.get('settings', {}).get('friction_smoothing_mm', sig.get('friction_smoothing_mm', 1e-5))) if maxwell else None
            observed.append(dict(index=i, depth_mm=step['depth_mm'], qualified=i < prefix,
                time_from_initial_s=time_from_initial if dt is not None else None, dt_s=dt,
                epsilon_u_mm=epsilon, epsilon_u_over_dt_mm_s=epsilon/dt if dt else None,
                tip_contact=contact, blade_contact_footprint=footprint, relative_motion=motion,
                fixed_material=tracker.fixed_material_motion(x, definition),
                force={k: step.get(k) for k in ('reaction_fz_n', 'reaction_x_n', 'reaction_normal_fz_n', 'reaction_friction_fz_n')},
                material_markers=dict(nodes=sorted(set(sum(definition['gauge_nodes'].values(), []))),
                    positions_mm=x[sorted(set(sum(definition['gauge_nodes'].values(), [])))].tolist())))
            if i in display_indices:
                nodes = sorted(set(tracker.edges.ravel()), key=lambda j: tracker.s[j])
                snapshots.append(dict(index=i, depth_mm=step['depth_mm'], top_positions_mm=x[nodes].tolist(),
                    blade_outline_mm=x[n:n+facets+5].tolist(), markers=observed[-1]['material_markers'],
                    pole_mm=contact['knife_tip_mm']))
            prior, prior_depth = x.copy(), step['depth_mm']
            prior_contact, prior_motion = contact, motion
    if tracker is None:
        raise ValueError('No accepted states')
    output.mkdir(parents=True, exist_ok=True)
    result = dict(schema='softcut-contact-kinematics/2', status='completed_postprocessing', mechanical_solve_performed=False,
        source_directory=str(source), source_status=report['status'], source_completed=report['completed'],
        source_result_sha256=hashlib.sha256(report_bytes).hexdigest(), source_checkpoints_sha256=hashes,
        source_solver_sha256=report.get('source_sha256'), source_signature=sig,
        resolved_model=report.get('resolved_model'), qualification=dict(qualification, qualified_states=min(prefix, len(steps))),
        selection='qualified_accepted_prefix' if qualified_prefix else 'completed_accepted_history',
        postprocessor_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(__file__).with_name('tip_contact.py'))},
        checks='saved acceptance, original audit binding if available, topology, complete previous and metric history, physical dt, knife motion, positive J; no force re-solve',
        tip_contact_tracking=tracker.metadata, first_geometrically_valid=first_valid,
        fixed_material_definition=definition,
        relative_motion_definition=dict(raw='(v_blade_fixed_pole-v_material_same_current_edge_weights) dot current tangent',
            transported='previous material point carried by finite rigid blade motion minus current material point, dot current tangent / dt',
            trajectory_identity='fixed knife pole and contiguous original top-material projection branch; material points along this branch can differ',
            continuity='adjacent reference edges remain connected; fast multi-edge crossings use at most 32 sampled subintervals on the accepted-state geometric chord. These are not equilibrium states or proof of physical contact throughout the step; gaps, ambiguity and unresolved changes cut the segment.',
            integration='right endpoint increment, per valid contiguous endpoint segment; birth increment excluded; endpoint validity does not prove continuous physical contact',
            qualification='geometric proximity; no local pressure or traction qualification; not physical accumulated slip'),
        contact_events=proximity_events(observed, tracker, start),
        snapshots=snapshots, steps=observed)
    write_trace_csv(output/'contact_trace.csv', observed)
    write_kinematics_table(output/'kinematics.csv', observed)
    result['visualization'] = dict(status='running')
    write_json(output/'contact_trace.json', result)
    try:
        from tools.visualize_ipc import plot_tip_contact
        plot_tip_contact(observed, output/'tip_contact_offset.png')
        write_research_report(output, result)
        result['visualization']['status'] = 'completed'
    except Exception as exc:
        result['visualization'].update(status='failed', error=f'{type(exc).__name__}: {exc}')
    write_json(output/'contact_trace.json', result)
    return result


def proximity_events(steps, tracker, start):
    """Geometric status changes are bracketed by actual saved endpoints."""
    initial = tracker.observe_footprint(tracker.reference, start)
    prior = dict(pole=tracker.observe(tracker.reference, start)['status'],
                 **{name: part['status'] for name, part in initial['parts'].items()})
    previous_depth, events = start, []
    for step in steps:
        current = dict(pole=step['tip_contact']['status'],
                       **{name: part['status'] for name, part in step['blade_contact_footprint']['parts'].items()})
        for name, status in current.items():
            if prior[name] != status:
                events.append(dict(probe=name, previous_status=prior[name], current_status=status,
                    bracket_depth_mm=[previous_depth, step['depth_mm']],
                    meaning='geometric proximity status transition; not an exact event time or mechanical first touch'))
        prior, previous_depth = current, step['depth_mm']
    return events


def write_kinematics_table(path, steps):
    rows = []
    for step in steps:
        row = {k: step[k] for k in ('index', 'depth_mm', 'qualified', 'time_from_initial_s', 'dt_s', 'epsilon_u_mm', 'epsilon_u_over_dt_mm_s')}
        row.update({k: step['tip_contact'].get(k) for k in ('status', 'reference_s_mm', 'reference_offset_mm', 'gap_mm', 'edge_nodes', 'edge_alpha')})
        row.update({k: step['relative_motion'].get(k) for k in ('raw_mm_s', 'transported_mm_s', 'transport_correction_mm_s', 'segment_id', 'break_reason', 'signed_segment_mm', 'absolute_segment_mm', 'local_edge_stretch', 'migration_rate_mm_s')})
        row.update(motion_valid=step['relative_motion']['valid'], motion_status=step['relative_motion']['status'])
        check = step['relative_motion'].get('continuity_check') or {}
        row.update(continuity_status=check.get('status'), continuity_chord_intervals=check.get('intervals'))
        row.update(step['force'])
        for name, roi in step['fixed_material']['rois'].items():
            row.update({name+'_'+k: roi[k] for k in ('u_s_mm', 'u_n_mm', 'count')})
        row.update({name+'_stretch': v for name, v in step['fixed_material']['gauge_stretch'].items()})
        for name, part in dict(whole=step['blade_contact_footprint'], **step['blade_contact_footprint']['parts']).items():
            for key in ('supported_current_length_mm', 'supported_reference_length_mm', 'reference_s_intervals_mm'):
                row[name+'_'+key] = part[key]
        rows.append(row)
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def write_research_report(output, result):
    """One existing-export HTML report; full-rate CSV/JSON remain authoritative."""
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps = result['steps']
    d = [s['depth_mm'] for s in steps]
    def numbers(values):
        return [np.nan if v is None else v for v in values]
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), layout='constrained')
    migration = numbers([s['tip_contact']['reference_offset_mm'] for s in steps])
    connected = [np.nan if s['relative_motion'].get('break_reason') == 'nonadjacent_projection_jump' else v
                 for s, v in zip(steps, migration)]
    line = axes[0, 0].plot(d, connected, label='M = S* - S_drop (mm)')[0]
    axes[0, 0].scatter(d, migration, s=12, color=line.get_color())
    gap_axis = axes[0, 0].twinx()
    gap_axis.plot(d, [s['tip_contact']['gap_mm'] for s in steps], color='tab:orange', label='pole gap (mm)')
    gap_axis.axhline(result['tip_contact_tracking']['dhat_mm'], color='tab:orange', ls=':', label='dhat (mm)')
    gap_axis.set_ylabel('pole gap (mm)')
    gap_axis.legend(loc='lower left', fontsize=7)
    for name in ('nose', 'left_flank', 'right_flank'):
        axes[0, 1].plot(d, [s['blade_contact_footprint']['parts'][name]['supported_current_length_mm'] for s in steps], label=name)
    for name in result['fixed_material_definition']['roi_nodes']:
        for component, style in (('u_s_mm', '-'), ('u_n_mm', '--')):
            axes[1, 0].plot(d, numbers([s['fixed_material']['rois'][name][component] for s in steps]), style, label=name+' '+component)
    for name in result['fixed_material_definition']['gauge_nodes']:
        axes[1, 1].plot(d, [s['fixed_material']['gauge_stretch'][name] for s in steps], label=name+' stretch')
    for key in ('raw_mm_s', 'transported_mm_s', 'signed_segment_mm', 'absolute_segment_mm'):
        # Separate segments visually; never draw a line across missing support.
        y = numbers([s['relative_motion'].get(key) for s in steps])
        for i, s in enumerate(steps):
            if s['relative_motion'].get('break_reason'):
                y[i] = np.nan
        axis = axes[2 if key.endswith('mm_s') else 3, 0]
        line = axis.plot(d, y, label=key)[0]
        # A segment boundary excludes an integral interval, not its well-defined
        # endpoint diagnostic. Show EVERY finite diagnostic, including births
        # and unresolved nonadjacent changes, without connecting across gaps.
        axis.scatter(d, numbers([s['relative_motion'].get(key) for s in steps]), s=12, color=line.get_color())
    axes[3, 1].plot(d, numbers([s['epsilon_u_over_dt_mm_s'] for s in steps]), label='epsilon_u/dt (mm/s)')
    for key in steps[0]['force']:
        axes[2, 1].plot(d, numbers([s['force'][key] for s in steps]), label=key)
    for ax in axes.flat:
        ax.set_xlabel('nominal downward depth (mm)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output/'kinematics.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(result['snapshots']), figsize=(14, 4), squeeze=False, layout='constrained')
    for ax, frame in zip(axes[0], result['snapshots']):
        top, blade = np.asarray(frame['top_positions_mm']), np.asarray(frame['blade_outline_mm'])
        ax.plot(top[:, 0], top[:, 1], color='steelblue', label='original material top')
        blade = np.vstack((blade, blade[0]))
        ax.plot(blade[:, 0], blade[:, 1], color='black', label='blade')
        pts = np.asarray(frame['markers']['positions_mm'])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], c=frame['markers']['nodes'], cmap='tab10', s=12, label='fixed material IDs')
        pole = frame['pole_mm']
        ax.set(xlim=(pole[0]-2, pole[0]+2), ylim=(pole[1]-2, pole[1]+2),
               xlabel='lab x (mm)', ylabel='lab z (mm)', title=f"d={frame['depth_mm']:.4g} mm")
        ax.set_aspect('equal');ax.legend(fontsize=6)
    fig.savefig(output/'material_states.png', dpi=150);plt.close(fig)
    text = html.escape(json.dumps({k: result[k] for k in ('source_status', 'source_completed', 'source_result_sha256',
        'qualification', 'tip_contact_tracking', 'fixed_material_definition', 'relative_motion_definition')}, indent=2))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Material migration and deformation</title>'
        '<h1>Material migration, deformation, proximity motion and force</h1><p>Laboratory x right, z up; material S along initial top. '
        'All accepted states participate. Geometric footprint is not a pressure-bearing area. '
        'A migration plateau does not establish cessation of interface slip. Experimental validation remains unavailable.</p>'
        '<p><a href="kinematics.csv">Full-rate machine table</a> · <a href="contact_trace.json">Definitions and all samples</a></p>'
        '<img src="kinematics.png" style="max-width:100%"><img src="material_states.png" style="max-width:100%">'
        '<h2>Source and qualification</h2><pre>'+text+'</pre>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--qualified-prefix", action="store_true",
                        help="Select only the independently authenticated continuous accepted prefix; preserve source failure status")
    export.add_argument("--audit", type=Path, help="Existing independent audit sidecar")
    export.add_argument("--manifest", type=Path, help="Prior manifest binding result and audit for legacy outputs")
    query = sub.add_parser("query")
    query.add_argument("--source", type=Path, required=True)
    query.add_argument("--depth", type=float, required=True)
    args = parser.parse_args()
    if args.action == "export":
        result = export_trace(args.source, args.output, qualified_prefix=args.qualified_prefix,
                              audit_path=args.audit, manifest_path=args.manifest)
        print(json.dumps(dict(status=result["status"], accepted_states=len(result["steps"]), output=str(args.output))))
    else:
        path = args.source / "contact_trace.json"
        if not path.exists():
            path = args.source / "result.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("tip_contact_tracking"):
            parser.error("Contact was not tracked; export this run first")
        print(json.dumps(query_depth(report["steps"], args.depth), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
