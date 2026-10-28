"""Read-only audit of permanent compression pairs in saved network states.

This module never constructs a finite-element problem, runs equilibrium, or
reads experimental data. Only the pure mixed-interface constitutive assembly
is imported. Authenticated input states are not a proof of contact adequacy or
of converged discretization; unauthenticated states remain diagnostics.
"""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from prototype.cohesive_mixed import MixedModeLaw, interface_response


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def read_json(path):
    def nonfinite(value):
        raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=nonfinite)


def source_check(recorded, root):
    root = Path(root).resolve()
    missing, mismatched = [], []
    for name, expected in recorded.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            missing.append(name)
        elif digest(path) != expected:
            mismatched.append(name)
    required = {'prototype/cohesive_mixed.py', 'prototype/cohesive_cut.py',
                'prototype/ipc_reference.py', 'prototype/ipc_viscoelastic.py', 'tools/run_cohesive_cut.py'}
    missing.extend(sorted(required-set(recorded)))
    law_path = 'prototype/cohesive_mixed.py'
    active_law_matches = bool(recorded.get(law_path) == digest(PROJECT/law_path))
    return dict(status='match' if recorded and not missing and not mismatched and active_law_matches else 'unverified',
                missing=missing, mismatched=mismatched, active_law_matches=active_law_matches,
                source_root=str(root))


def point_segment_distance(point, start, end):
    edge = end-start
    length2 = float(edge@edge)
    if length2 <= 0.:
        raise ValueError('Positive finite face lengths required')
    parameter = np.clip(float((point-start)@edge)/length2, 0., 1.)
    return float(np.linalg.norm(point-(start+parameter*edge)))


def cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def segment_distance(a, b, c, d):
    """Minimum Euclidean distance between finite 2-D segments, including crossing."""
    u, v, offset = b-a, d-c, c-a
    denominator = float(cross(u, v))
    if denominator != 0.:
        t, s = float(cross(offset, v))/denominator, float(cross(offset, u))/denominator
        if 0. <= t <= 1. and 0. <= s <= 1.:
            return 0.
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def pairing_metrics(positions, segments, gaps, areas, slips, history, law, dhat,
                    *, traction_tolerance_mpa=1e-12):
    """Evaluate saved geometry; never advance histories or solve a load increment."""
    x, seg = np.asarray(positions, float), np.asarray(segments)
    gaps, areas, slips = np.asarray(gaps), np.asarray(areas), np.asarray(slips)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError('Finite saved positions required')
    if (seg.ndim != 2 or seg.shape[1] != 4 or seg.dtype.kind not in 'iu'
            or np.any(seg < 0) or np.any(seg >= len(x))):
        raise ValueError('Integer four-node saved interface topology required')
    h = np.asarray(history, float)
    if h.shape != (len(seg), 2):
        raise ValueError('Two accepted Gauss histories per segment required')
    if not np.isfinite(dhat) or dhat <= 0. or not np.isfinite(traction_tolerance_mpa) or traction_tolerance_mpa < 0.:
        raise ValueError('Positive dhat and nonnegative traction tolerance required')
    q = interface_response(x, seg, gaps, areas, h, law, tangent=False, reference_slips=slips)
    history_consistent = bool(np.allclose(q['history'], h, rtol=1e-10, atol=1e-12))
    damage = law.history_response(h)['damage']
    full = np.all(damage >= 1.-1e-10, axis=1)
    ids = np.flatnonzero(full)
    records = []
    if len(ids):
        local_x = x[seg[ids]]
        local_segments = np.arange(len(ids)*4).reshape(-1, 4)
        local_q = interface_response(local_x.reshape(-1, 2), local_segments,
            np.asarray(gaps)[ids], np.asarray(areas)[ids], h[ids], law,
            tangent=False, reference_slips=np.asarray(slips)[ids])
        local_g = local_q['compression_gradient'].reshape(-1, 4, 2)
        for j, interface_id in enumerate(ids):
            xx, gg = local_x[j], local_g[j]
            left, right = xx[[0, 2]], xx[[1, 3]]
            le, re = left[1]-left[0], right[1]-right[0]
            mean = .5*(le+re)
            director = mean/np.linalg.norm(mean)
            mean_length = .5*(np.linalg.norm(le)+np.linalg.norm(re))
            offset = float((right.mean(axis=0)-left.mean(axis=0))@director)
            lp, rp = left@director, right@director
            projected_gap = max(float(lp.min()-rp.max()), float(rp.min()-lp.max()), 0.)
            distance = segment_distance(*left, *right)
            scale = max(1., float(np.max(np.abs(xx))), float(mean_length))
            geometry_tol = 128*np.finfo(float).eps*scale
            tc = local_q['compression_traction'][j]
            active = bool(np.any(np.abs(tc) > traction_tolerance_mpa))
            forces = -gg
            moments = np.array([np.sum(cross(left-left.mean(axis=0), forces[[0, 2]])),
                                np.sum(cross(right-right.mean(axis=0), forces[[1, 3]]))])
            records.append(dict(interface_id=int(interface_id), gauss_damage=damage[interface_id].tolist(),
                finite_face_distance_mm=distance, geometry_tolerance_mm=geometry_tol,
                geometrically_disjoint=bool(distance > geometry_tol),
                outside_ipc_activation_distance=bool(distance > dhat+geometry_tol),
                mean_face_length_mm=float(mean_length), mean_director_length_mm=float(np.linalg.norm(mean)),
                centroid_tangential_offset_mm=offset, tangential_offset_over_mean_length=abs(offset)/float(mean_length),
                finite_face_projected_tangential_gap_mm=projected_gap,
                projected_compression_opening_mm=local_q['opening'][j].tolist(),
                compression_traction_mpa=tc.tolist(), compression_active=active,
                compression_gradient_n=gg.tolist(), compression_gradient_norm_n=float(np.linalg.norm(gg)),
                maximum_nodal_compression_force_n=float(np.linalg.norm(gg, axis=1).max()),
                face_resultant_internal_force_n=[forces[[0, 2]].sum(axis=0).tolist(), forces[[1, 3]].sum(axis=0).tolist()],
                face_centroid_internal_moments_n_mm=moments.tolist(),
                face_centroid_couple_norm_n_mm=float(np.linalg.norm(moments)),
                total_internal_force_balance_norm_n=float(np.linalg.norm(forces.sum(axis=0))),
                total_internal_moment_about_pair_centroid_n_mm=float(np.sum(cross(xx-xx.mean(axis=0), forces))),
                compression_stored_energy_n_mm=float(.5*areas[interface_id]*np.sum(local_q['compression_stored'][j]))))
    remote = [r for r in records if r['outside_ipc_activation_distance'] and r['compression_active']]
    return dict(history_consistent_with_saved_geometry=history_consistent,
        candidate_interfaces=len(seg), fully_failed_interfaces=len(records),
        partially_failed_interfaces=int(np.count_nonzero(np.any(damage >= 1.-1e-10, axis=1)&~full)),
        fully_failed_compression_active_pairs=sum(r['compression_active'] for r in records),
        geometrically_disjoint_compression_active_pairs=sum(r['geometrically_disjoint'] and r['compression_active'] for r in records),
        outside_ipc_activation_compression_active_pairs=len(remote),
        maximum_remote_compression_traction_mpa=max((max(map(abs, r['compression_traction_mpa'])) for r in remote), default=0.),
        maximum_remote_local_gradient_norm_n=max((r['compression_gradient_norm_n'] for r in remote), default=0.),
        maximum_remote_face_couple_norm_n_mm=max((r['face_centroid_couple_norm_n_mm'] for r in remote), default=0.),
        compression_stored_energy_n_mm=q['compression_stored_energy_n_mm'],
        compression_assembled_gradient_norm_n=float(np.linalg.norm(q['compression_gradient'])),
        fully_failed_compression_stored_energy_n_mm=sum(r['compression_stored_energy_n_mm'] for r in records),
        outside_ipc_activation_compression_stored_energy_n_mm=sum(r['compression_stored_energy_n_mm'] for r in remote),
        interfaces=records)


def audit_result(result, *, source_root=PROJECT, input_config=None, traction_tolerance_mpa=1e-12):
    """Authenticate an existing audit and annotate each accepted saved state."""
    if not np.isfinite(traction_tolerance_mpa) or traction_tolerance_mpa < 0.:
        raise ValueError('Nonnegative finite traction tolerance required')
    result = Path(result).resolve()
    result = result/'result.json' if result.is_dir() else result
    directory = result.parent
    result_hash = digest(result)
    report = read_json(result)
    config = report.get('configuration', {})
    if config.get('fracture_topology') != 'edge_network' or config.get('cohesive_kind') != 'mixed_mode_energy':
        raise ValueError('Saved mixed-mode edge_network result required')
    config_path, audit_path = directory/'config.json', directory/'audit.json'
    config_hash = digest(config_path) if config_path.is_file() else None
    audit_hash = digest(audit_path) if audit_path.is_file() else None
    config_matches = config_path.is_file() and read_json(config_path) == config
    law_keys = {f.name for f in fields(MixedModeLaw)}
    if set(config['cohesive'])-law_keys:
        raise ValueError('Unknown cohesive parameter in saved configuration')
    law = MixedModeLaw(**config['cohesive'])
    law_dict = law.to_dict()
    law_identity = report.get('cohesive_law') == law_dict and report.get('signature', {}).get('fracture', {}).get('law') == law_dict
    source = source_check(report.get('source_sha256', {}), source_root)
    original_input = 'not_supplied'
    if input_config is not None:
        original_input = 'match' if digest(input_config) == report.get('input_sha256') else 'mismatch'
    audit = read_json(audit_path) if audit_path.is_file() else {}
    steps, audit_rows = report.get('steps', []), audit.get('rows', [])
    complete_audit = bool(steps and audit.get('passed') is True and audit.get('cohesive_history_chain_passed') is True
                          and len(steps) == len(audit_rows))
    provenance = bool(config_matches and law_identity and source['status'] == 'match' and original_input != 'mismatch')
    rows, previous, topology = [], None, None
    chain_valid = True
    for i, step in enumerate(steps):
        checkpoint = directory/'states'/f'{i:04d}.npz'
        reasons = []
        row = audit_rows[i] if i < len(audit_rows) else {}
        if not checkpoint.is_file():
            rows.append(dict(state_index=i, depth_mm=step.get('depth_mm'), qualification='diagnostic_only',
                             reasons=['missing_checkpoint'], metrics=None))
            chain_valid = False
            continue
        checkpoint_hash = digest(checkpoint)
        with np.load(checkpoint, allow_pickle=False) as saved:
            required = {'positions', 'depth_mm', 'body_nodes', 'cohesive_maximum', 'previous_cohesive_maximum',
                'interface_segments', 'reference_interface_gaps_mm', 'reference_interface_areas_mm2',
                'reference_interface_slips_mm', 'cohesive_law_parameters_json', 'fracture_topology'}
            if not required <= set(saved.files):
                raise ValueError(f'State {i}: missing saved fields {sorted(required-set(saved.files))}')
            positions, segments = saved['positions'], saved['interface_segments']
            gaps, areas, slips = saved['reference_interface_gaps_mm'], saved['reference_interface_areas_mm2'], saved['reference_interface_slips_mm']
            history, old = saved['cohesive_maximum'], saved['previous_cohesive_maximum']
            if int(saved['body_nodes']) <= 0 or np.any(segments >= int(saved['body_nodes'])):
                raise ValueError('Interface contains blade or nonbody node')
            if str(saved['fracture_topology']) != 'edge_network' or json.loads(str(saved['cohesive_law_parameters_json'])) != law_dict:
                reasons.append('checkpoint_law_or_topology_identity_mismatch')
            current_topology = (segments, gaps, areas, slips)
            if topology is None:
                topology = tuple(a.copy() for a in current_topology)
            elif not all(np.array_equal(a, b) for a, b in zip(topology, current_topology)):
                reasons.append('checkpoint_topology_changed')
                chain_valid = False
            expected_old = np.zeros_like(history) if previous is None else previous
            if (history.shape != old.shape or not np.isfinite(history).all() or not np.isfinite(old).all()
                    or np.any(history < old) or np.any(old < 0.) or not np.array_equal(old, expected_old)):
                reasons.append('cohesive_history_chain_mismatch')
                chain_valid = False
            previous = history.copy()
            metrics = pairing_metrics(positions, segments, gaps, areas, slips, history, law,
                config['dhat_mm'], traction_tolerance_mpa=traction_tolerance_mpa)
            if not metrics['history_consistent_with_saved_geometry']:
                reasons.append('saved_history_behind_geometry')
            if not np.isclose(float(saved['depth_mm']), step.get('depth_mm', np.nan), rtol=0., atol=1e-12):
                reasons.append('checkpoint_depth_mismatch')
            energy = metrics['compression_stored_energy_n_mm']
            for name, saved_value in [('checkpoint', saved['compression_stored_energy_n_mm'] if 'compression_stored_energy_n_mm' in saved else np.nan),
                                      ('record', step.get('fracture', {}).get('compression_stored_energy_n_mm', np.nan))]:
                if not np.isclose(energy, saved_value, rtol=1e-9, atol=1e-12):
                    reasons.append(name+'_compression_energy_mismatch_or_missing')
        if digest(checkpoint) != checkpoint_hash:
            reasons.append('checkpoint_changed_during_read')
        native_ok = bool(row.get('passed') is True and row.get('cohesive_history_passed') is True
            and row.get('checkpoint_sha256') == checkpoint_hash and row.get('accepted_record_sha256') == record_digest(step)
            and np.isfinite([row.get('depth_mm', np.nan), row.get('residual_n', np.nan), row.get('min_j', np.nan)]).all()
            and abs(row['depth_mm']-step.get('depth_mm', np.inf)) <= 1e-12
            and 0. <= row['residual_n'] <= 1e-4 and row['min_j'] > 0.)
        if not native_ok:
            reasons.append('missing_or_invalid_fresh_audit_hash_or_equilibrium')
        if not complete_audit:
            reasons.append('incomplete_existing_audit')
        if not provenance:
            reasons.append('unverified_config_law_or_source_identity')
        if not chain_valid:
            reasons.append('unverified_history_prefix')
        rows.append(dict(state_index=i, depth_mm=step.get('depth_mm'), checkpoint_sha256=checkpoint_hash,
            qualification='authenticated_saved_state' if not reasons else 'diagnostic_only', reasons=reasons, metrics=metrics))
    inputs_stable = bool(digest(result) == result_hash
        and (digest(config_path) if config_path.is_file() else None) == config_hash
        and (digest(audit_path) if audit_path.is_file() else None) == audit_hash
        and source_check(report.get('source_sha256', {}), source_root) == source)
    if not inputs_stable:
        for row in rows:
            row['qualification'] = 'diagnostic_only'
            row['reasons'].append('input_files_changed_during_postprocessing')
    authenticated = bool(rows and all(r['qualification'] == 'authenticated_saved_state' for r in rows))
    available = [r['metrics'] for r in rows if r['metrics'] is not None]
    return dict(schema='softcut-network-pairing-audit/1', result_path=str(result), result_sha256=result_hash,
        config_sha256=config_hash, input_files_stable_during_read=inputs_stable,
        config_matches_result=config_matches, law_identity_matches_result=law_identity,
        original_input_hash_status=original_input, source_check=source,
        audit_sha256=audit_hash,
        qualification='authenticated_saved_prefix' if authenticated else 'diagnostic_only',
        completed_target=bool(authenticated and report.get('completed') is True
            and abs(steps[0].get('depth_mm', np.inf)) <= 1e-12
            and np.isclose(steps[-1].get('depth_mm', np.nan), config.get('target_mm', np.nan), rtol=0., atol=1e-12)),
        report_status=report.get('status'), examined_accepted_states=len(rows),
        audited_state_count=sum(r['qualification'] == 'authenticated_saved_state' for r in rows),
        history_and_input_authentication_only=True, physical_pairing_adequacy_qualified=False,
        experiment_data_read=False, equilibrium_solve_performed=False,
        tool_sha256=digest(__file__), interpreter=sys.executable,
        law=law_dict, dhat_mm=config['dhat_mm'], traction_tolerance_mpa=traction_tolerance_mpa,
        maximum_remote_compression_traction_mpa=max((m['maximum_remote_compression_traction_mpa'] for m in available), default=0.),
        maximum_remote_local_gradient_norm_n=max((m['maximum_remote_local_gradient_norm_n'] for m in available), default=0.),
        maximum_remote_face_couple_norm_n_mm=max((m['maximum_remote_face_couple_norm_n_mm'] for m in available), default=0.),
        maximum_outside_ipc_activation_compression_active_pairs=max((m['outside_ipc_activation_compression_active_pairs'] for m in available), default=0),
        scope_notes=[
            'Only both Gauss points with damage >= 1-1e-10 identify a fully failed interface; a single failed point is not promoted.',
            'Outside IPC activation refers only to the distance between these two finite paired faces, not every other surface or the blade.',
            'Compression gradients and face-centroid couples are internal interface quantities, not a partition or percentage of blade reaction.',
            'A constitutive scalar pressure cap does not bound nodal forces: the rotating director Jacobian contains tangential-offset/length factors.',
            'This authenticates previously saved equilibrium audit evidence; it does not rerun IPC, certify contact adequacy, or establish mesh/gap convergence.'],
        rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path, help='Existing result.json or its directory')
    parser.add_argument('--output', type=Path, required=True, help='New standalone JSON output; existing files are never overwritten')
    parser.add_argument('--source-root', type=Path, default=PROJECT)
    parser.add_argument('--input-config', type=Path, help='Optional original input whose bytes match result.input_sha256')
    parser.add_argument('--traction-tolerance-mpa', type=float, default=1e-12)
    args = parser.parse_args(argv)
    result = audit_result(args.result, source_root=args.source_root, input_config=args.input_config,
                          traction_tolerance_mpa=args.traction_tolerance_mpa)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('qualification', 'examined_accepted_states',
        'maximum_outside_ipc_activation_compression_active_pairs', 'maximum_remote_local_gradient_norm_n')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
