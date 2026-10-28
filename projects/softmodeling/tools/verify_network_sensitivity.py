"""Explicit bounded studies and read-only comparisons of network cutting results.

``template`` only writes a manifest; ``compare`` never imports a solver. Only
``runstudy`` launches the existing runner, once per explicitly listed case.
This tool compares numerical evidence; it never upgrades energy qualification.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = 'softcut-network-sensitivity/1'
AXES = {
    'gap': {'reference_gap_mm'},
    'contact': {'dhat_mm'},
    'ccd': {'ccd_tolerance_mm'},
    'normal_stiffness': {'cohesive.normal_stiffness_mpa_per_mm'},
    'compression_stiffness': {'cohesive.compression_stiffness_mpa_per_mm'},
    'time': {'increment_mm'},
    'damage_control': {'cohesive.damage_enabled'},
    'mesh': {'mesh.local_spacing_mm', 'mesh.surface_spacing_mm', 'mesh.coarse_spacing_mm',
             'mesh.seam_spacing_mm'},
}
IGNORED = {'label', 'parameter_source'}
BOUNDED_COMPRESSION = 'bounded_reversible_compression_penalty'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def canonical_contact_stencils(stencils):
    """Canonical contact identity without changing roles, ownership or count."""
    if not isinstance(stencils, list) or any(not isinstance(s, dict) for s in stencils):
        raise ValueError('Contact stencils must be a list of dictionaries')
    canonical = []
    for original in stencils:
        stencil = dict(original)
        if stencil.get('kind') == 'EV' and 'global_edge_nodes' in stencil:
            stencil['global_edge_nodes'] = sorted(stencil['global_edge_nodes'])
        if stencil.get('kind') == 'VV' and 'global_nodes' in stencil:
            stencil['global_nodes'] = sorted(stencil['global_nodes'])
        canonical.append(stencil)
    return sorted(canonical, key=lambda s: (
        s['kind'], tuple(s.get('global_edge_nodes', s.get('global_nodes', []))),
        s.get('global_vertex', -1), tuple(s['segment_ids'])))


def evidence_equal(actual, expected):
    """Compare independently audited nested fields without requiring new keys."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if key not in actual:
                return False
            if key == 'contact_stencils':
                try:
                    if not evidence_equal(canonical_contact_stencils(actual[key]), canonical_contact_stencils(value)):
                        return False
                except (KeyError, TypeError, ValueError):
                    return False
            elif not evidence_equal(actual[key], value):
                return False
        return True
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(evidence_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)):
        return not isinstance(actual, bool) and isinstance(actual, (int, float)) and bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12))
    return actual == expected


def read_json(path):
    def nonfinite(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=nonfinite)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf-8')


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def finite_number(value, lower, upper, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f'{name} must be finite in [{lower},{upper}]')


def validate_manifest(manifest):
    """Reject confounded families before loading results or starting processes."""
    if manifest.get('schema') != SCHEMA or manifest.get('no_inverse_fitting') is not True:
        raise ValueError('Explicit schema and no_inverse_fitting=true required')
    cases = manifest.get('cases', [])
    if not cases or len(cases) > 100:
        raise ValueError('List between 1 and 100 explicit cases')
    ids = [c.get('id') for c in cases]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids):
        raise ValueError('Unique simple case identifiers required')
    by_id = {c['id']: c for c in cases}
    for case in cases:
        finite_number(case.get('seconds'), 1, 3600, 'seconds')
        config = case.get('config', {})
        required = {'schema', 'cohesive_kind', 'fracture_topology', 'angle_deg', 'material',
                    'cohesive', 'friction_coefficient', 'preparation_clearance_mm',
                    'loading_rate_mm_s', 'target_mm', 'increment_mm', 'mesh',
                    'reference_gap_mm', 'dhat_mm', 'tip_radius_mm', 'network_predictor',
                    'network_metric_floor_n_per_mm', 'parameter_status'}
        if not required <= config.keys():
            raise ValueError(f'{case["id"]}: missing explicit configuration: {sorted(required-config.keys())}')
        if config.get('no_experimental_force_fitting') is not True or config['loading_rate_mm_s'] != .2:
            raise ValueError('No fitting and fixed speed 0.2 mm/s required')
        if config['fracture_topology'] != 'edge_network':
            raise ValueError('This runner adapter supports edge_network only; intact comparison is a separate control')
        if config['schema'] not in ('softcut-cohesive/2', 'softcut-cohesive/3') or config['cohesive_kind'] != 'mixed_mode_energy':
            raise ValueError('Explicit mixed-mode network configuration required')
        finite_number(config['target_mm'], np.nextafter(0., 1.), 2., 'target')
        finite_number(config['increment_mm'], .000625, .05, 'increment')
        finite_number(config['preparation_clearance_mm'], np.nextafter(0., 1.), 1., 'preparation clearance')
        law_required = {'normal_strength_mpa', 'shear_strength_mpa', 'toughness_n_per_mm',
                        'normal_stiffness_mpa_per_mm', 'compression_stiffness_mpa_per_mm', 'damage_enabled'}
        if not law_required <= config['cohesive'].keys():
            raise ValueError('Declare every cohesive parameter, including compression stiffness')
        for key in law_required-{'damage_enabled', 'compression_stiffness_mpa_per_mm'}:
            finite_number(config['cohesive'][key], np.nextafter(0., 1.), 1e12, key)
        finite_number(config['cohesive']['compression_stiffness_mpa_per_mm'], 0., 1e12, 'compression stiffness')
        if not isinstance(config['cohesive']['damage_enabled'], bool):
            raise ValueError('damage_enabled must be a boolean')
        finite_number(config['reference_gap_mm'], np.nextafter(0., 1.), 1., 'gap')
        finite_number(config['dhat_mm'], np.nextafter(0., 1.), 1., 'dhat')
        if config['reference_gap_mm'] <= 2*config['dhat_mm']:
            raise ValueError('Require gap > 2*dhat')
        if 'minimum_increment_mm' in config:
            finite_number(config['minimum_increment_mm'], np.nextafter(0., 1.), config['increment_mm'], 'minimum increment')
        mode = config.get('closure_regularization')
        ell = config['cohesive'].get('compression_regularization_mm')
        if mode == BOUNDED_COMPRESSION:
            finite_number(ell, np.nextafter(0., 1.), 1., 'compression regularization length')
            if ell != config['reference_gap_mm']:
                raise ValueError('Bounded compression_regularization_mm must equal reference_gap_mm exactly')
        elif ell is not None:
            raise ValueError('Linear/legacy compression requires compression_regularization_mm=None')
        if config['schema'] == 'softcut-cohesive/3' and (mode not in {'reversible_compression_penalty', BOUNDED_COMPRESSION}
                or config['cohesive']['compression_stiffness_mpa_per_mm'] <= 0 or 'minimum_increment_mm' not in config):
            raise ValueError('Version 3 requires explicit positive reversible compression and minimum increment')
        if config['schema'] == 'softcut-cohesive/3':
            finite_number(config.get('ccd_tolerance_mm'), np.nextafter(0., 1.),
                          min(1e-6, config['reference_gap_mm']/10, config['dhat_mm']/10), 'CCD tolerance')
        elif config.get('closure_regularization') is not None or config['cohesive']['compression_stiffness_mpa_per_mm'] != 0 or 'minimum_increment_mm' in config:
            raise ValueError('Compression regularization and custom minimum increment require schema 3')
    families = manifest.get('families', [])
    if not families or len({f['id'] for f in families}) != len(families):
        raise ValueError('Unique explicit families required')
    used = set()
    for family in families:
        kind = family.get('kind')
        members = family.get('cases', [])
        if kind not in AXES or len(members) < 2 or len(set(members)) != len(members) or not set(members) <= by_id.keys():
            raise ValueError('Each supported family needs at least two distinct listed cases')
        if family.get('reference') not in members:
            raise ValueError('Family reference must be a member')
        required_depth = family.get('required_depth_mm')
        finite_number(required_depth, np.nextafter(0., 1.), 2., 'required depth')
        base = flatten(by_id[members[0]]['config'])
        allowed = set(AXES[kind])
        if kind == 'gap' and all(by_id[m]['config'].get('closure_regularization') == BOUNDED_COMPRESSION for m in members):
            # This is a mandatory derived numerical scale, not an additional
            # independently fitted axis. Every case was checked for ell==gap.
            allowed.add('cohesive.compression_regularization_mm')
        varied = set()
        for member in members:
            config = by_id[member]['config']
            if not np.isclose(required_depth, config['target_mm'], rtol=0, atol=1e-12):
                raise ValueError('Family required depth must equal every case target; no shortened-prefix qualification')
            other = flatten(config)
            changed = {k for k in base.keys() | other.keys() if base.get(k) != other.get(k)}-IGNORED
            if changed-allowed:
                raise ValueError(f'Confounded {family["id"]}: {sorted(changed-allowed)}')
            varied |= changed
        if not varied:
            raise ValueError('A sensitivity family must actually vary its declared axis')
        used.update(members)
    if used != set(ids):
        raise ValueError('Every case must belong to a family')
    if not isinstance(manifest.get('stop_on_failure'), bool):
        raise ValueError('Declare stop_on_failure explicitly')
    cost = sum(c['seconds']+60 for c in cases)
    finite_number(manifest.get('total_wall_budget_s'), cost, 366000, 'total wall budget')
    return manifest


def source_check(recorded, source_root):
    if not recorded:
        return {'status': 'unknown', 'mismatches': [], 'missing': []}
    root = Path(source_root).resolve()
    mismatches, missing = [], []
    for name, expected in recorded.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            missing.append(name)
        elif digest(path) != expected:
            mismatches.append(name)
    return {'status': 'match' if not mismatches and not missing else 'mismatch',
            'mismatches': mismatches, 'missing': missing}


def stats(values):
    a = np.asarray(values, dtype=float)
    if not a.size or not np.isfinite(a).all():
        return None
    return dict(zip(('min', 'p05', 'median', 'p95', 'max'),
                    map(float, np.quantile(a, [0, .05, .5, .95, 1]))))


def geometry_metrics(path, config, topology, signature):
    """Use saved reference cells only, without constructing or loading FE."""
    if not path.is_file():
        return {'status': 'unknown_missing_checkpoint'}
    with np.load(path, allow_pickle=False) as state:
        if not {'reference_xy', 'triangles', 'interface_segments'} <= set(state.files):
            return {'status': 'unknown_missing_geometry'}
        xy, tri, seg = state['reference_xy'], state['triangles'], state['interface_segments']
        points = xy[tri]
        u, v = points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]
        areas = .5*np.abs(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0])
        owner = np.full(len(xy), -1, dtype=int)
        owner[tri.ravel()] = np.repeat(np.arange(len(tri)), 3)
        if len(seg) and np.any(owner[seg] < 0):
            return {'status': 'invalid_interface_ownership'}
        left = np.linalg.norm(xy[seg[:, 2]]-xy[seg[:, 0]], axis=1)
        right = np.linalg.norm(xy[seg[:, 3]]-xy[seg[:, 1]], axis=1)
        hl = 2*areas[owner[seg[:, 0]]]/left
        hr = 2*areas[owner[seg[:, 1]]]/right
        h = 2/(1/hl+1/hr)
        reference_area = float(state['reference_interface_areas_mm2'].sum()) if 'reference_interface_areas_mm2' in state else None
    material, law = config['material'], config['cohesive']
    g0 = material['equilibrium_shear_mpa']+material['maxwell_shear_mpa']
    bulk = material['bulk_mpa']
    e0 = 9*bulk*g0/(3*bulk+g0)
    return {'status': 'computed_saved_reference_geometry', 'cell_area_mm2': stats(areas),
            'retained_cell_area_mm2': float(areas.sum()), 'bulk_area_ratio': topology.get('bulk_area_ratio'),
            'minimum_triangle_area_ratio': topology.get('minimum_triangle_area_ratio'),
            'reference_interface_area_mm2': reference_area,
            'h_definition': 'Harmonic mean of retained adjacent triangle altitudes to each actual inset interface face; not original uncut cell h.',
            'h_mm': stats(h), 'g_over_h': stats(config['reference_gap_mm']/h),
            'Kn_h_over_E0': stats(law['normal_stiffness_mpa_per_mm']*h/e0),
            'Kc_h_over_E0': stats(law.get('compression_stiffness_mpa_per_mm', 0.)*h/e0),
            'Kc_gap_over_normal_strength': law.get('compression_stiffness_mpa_per_mm', 0.)*config['reference_gap_mm']/law['normal_strength_mpa'],
            'Kc_gap_scale_note': 'Nominal reversible stress scale for closing the reference gap, not measured blade load or contact force partition.',
            'E0_mpa': e0, 'E0_definition': '3D isotropic small-strain scale, not the finite-strain plane-strain tangent',
            'dhat_over_gap': config['dhat_mm']/config['reference_gap_mm'],
            'specimen_width_mm': signature.get('config', {}).get('width_mm')}


def audit_prefix(directory, report):
    """Authenticate an existing independent audit; do not call it a new solve."""
    audit_path = directory/'audit.json'
    if not audit_path.is_file():
        return {'status': 'unknown_missing_audit', 'qualified_states': 0}
    audit = read_json(audit_path)
    steps = report.get('steps', [])
    rows = audit.get('rows', [])
    count, reason = 0, None
    for i, (step, row) in enumerate(zip(steps, rows)):
        checkpoint = directory/'states'/f'{i:04d}.npz'
        valid = (row.get('passed') is True and row.get('cohesive_history_passed') is True
                 and row.get('checkpoint_sha256') and checkpoint.is_file()
                 and digest(checkpoint) == row['checkpoint_sha256']
                 and np.isfinite([row.get('depth_mm', np.nan), row.get('residual_n', np.nan), row.get('min_j', np.nan)]).all()
                 and abs(row['depth_mm']-step.get('depth_mm', np.inf)) <= 1e-12
                 and np.isclose(row.get('reaction_fz_n', np.nan), step.get('reaction_fz_n', np.nan), rtol=1e-8, atol=1e-10)
                 and 0 <= row['residual_n'] <= 1e-4 and row['min_j'] > 0)
        if valid and 'accepted_record_sha256' in row:
            valid = row['accepted_record_sha256'] == record_digest(step)
        if valid and 'max_damage' in row:
            valid = np.isclose(row['max_damage'], step.get('fracture', {}).get('max_damage', np.nan), rtol=1e-10, atol=1e-12)
        if valid and row.get('cohesive_events_passed') is True:
            valid = isinstance(row.get('fracture'), dict) and evidence_equal(step.get('fracture'), row['fracture'])
        if valid and 'material' in row:
            valid = evidence_equal(step.get('material'), row['material'])
        if valid and 'friction_work_n_mm' in row:
            valid = evidence_equal(step.get('friction_work_n_mm'), row['friction_work_n_mm'])
        if not valid:
            reason = f'Audit/checkpoint mismatch at state {i}'
            break
        count += 1
    passed = bool(steps and count == len(steps) == len(rows) and audit.get('passed') is True
                  and audit.get('cohesive_history_chain_passed') is True)
    return {'status': 'authenticated_existing_fresh_audit' if passed else 'incomplete_or_invalid_audit',
            'qualified_states': count, 'passed': passed, 'audit_sha256': digest(audit_path), 'reason': reason,
            'scope': 'Existing runner fresh-reconstruction audit authenticated against checkpoint hashes; comparator does not rerun mechanical equilibrium.'}


def checkpoint_fracture_evidence(directory, steps, law):
    """Independent history/topology reduction; no contact solve or law import.

    A checkpoint hash authenticates the saved arrays, not annotations in the
    separate result JSON. Recompute annotations before using them as events.
    """
    rows, previous, mismatch = [], None, []
    required = {'cohesive_maximum', 'previous_cohesive_maximum', 'original_interface_edges',
                'interface_exterior_seed', 'reference_interface_lengths_mm',
                'reference_interface_areas_mm2'}
    d0 = law['normal_strength_mpa']/law['normal_stiffness_mpa_per_mm']
    df = 2*law['toughness_n_per_mm']/law['normal_strength_mpa']
    if df <= d0:
        raise ValueError('Invalid cohesive peak/final opening ordering')
    for i, step in enumerate(steps):
        with np.load(directory/'states'/f'{i:04d}.npz', allow_pickle=False) as state:
            if not required <= set(state.files):
                return {'status': 'unknown_missing_checkpoint_fracture_fields', 'rows': [], 'mismatches': []}
            history, old = state['cohesive_maximum'], state['previous_cohesive_maximum']
            edges, seed = state['original_interface_edges'], state['interface_exterior_seed']
            length, area = state['reference_interface_lengths_mm'], state['reference_interface_areas_mm2']
            count = len(edges)
            if history.shape != (count, 2) or old.shape != history.shape or edges.shape != (count, 2) or seed.shape != (count,) or length.shape != (count,) or area.shape != (count,):
                raise ValueError('Invalid saved fracture shapes')
            if not np.isfinite(history).all() or not np.isfinite(old).all() or np.any(history < old) or np.any(old < 0) or not np.all(length > 0) or not np.all(area > 0):
                raise ValueError('Invalid irreversible fracture history/geometry')
            expected_old = np.zeros_like(history) if previous is None else previous
            if not np.array_equal(old, expected_old):
                raise ValueError('Saved cohesive history chain differs')
            previous = history.copy()
            damage = np.zeros_like(history)
            if law['damage_enabled']:
                soft = (history > d0)&(history < df)
                damage[soft] = df*(history[soft]-d0)/(history[soft]*(df-d0))
                damage[history >= df] = 1.
            broken = np.all(damage >= 1-1e-10, axis=1)
            incident = {}
            for edge_id, vertices in enumerate(edges):
                for vertex in vertices:
                    incident.setdefault(int(vertex), []).append(edge_id)
            connected = set(map(int, np.flatnonzero(broken & seed.astype(bool))))
            pending = list(connected)
            while pending:
                edge_id = pending.pop()
                for vertex in edges[edge_id]:
                    for neighbor in incident[int(vertex)]:
                        if broken[neighbor] and neighbor not in connected:
                            connected.add(neighbor)
                            pending.append(neighbor)
            connected = sorted(connected)
            dissipation = float(np.sum(.5*area[:, None]*law['toughness_n_per_mm']*np.clip((history-d0)/(df-d0), 0., 1.))) if law['damage_enabled'] else 0.
            fresh = dict(max_damage=float(damage.max(initial=0.)), gauss_damage=damage.tolist(),
                         damaged_gauss_points=int(np.count_nonzero(damage > 1e-8)),
                         separated_gauss_points=int(np.count_nonzero(damage >= 1-1e-10)),
                         damaged_segments=int(np.count_nonzero(np.any(damage > 1e-8, axis=1))),
                         separated_segments=int(broken.sum()), separated_bonds=int(broken.sum()),
                         separated_reference_length_mm=float(length@broken),
                         surface_connected_failed_segments=connected,
                         surface_connected_crack_length_mm=float(length[connected].sum()),
                         fracture_dissipation_n_mm=dissipation)
            saved = step.get('fracture', {})
            mandatory = {'max_damage', 'separated_segments', 'surface_connected_failed_segments',
                         'surface_connected_crack_length_mm', 'fracture_dissipation_n_mm'}
            for key, value in fresh.items():
                if key not in saved and key not in mandatory:
                    continue
                actual = saved.get(key)
                try:
                    equal = (actual == value if key == 'surface_connected_failed_segments' else
                             np.asarray(actual).shape == np.asarray(value).shape and
                             np.allclose(actual, value, rtol=1e-10, atol=1e-12))
                except (TypeError, ValueError):
                    equal = False
                if not equal:
                    mismatch.append({'state': i, 'field': key})
            for key in ('cohesive_stored_energy_n_mm', 'damageable_stored_energy_n_mm', 'compression_stored_energy_n_mm'):
                if key in state and (key not in saved or not np.isclose(float(state[key]), saved[key], rtol=1e-10, atol=1e-12)):
                    mismatch.append({'state': i, 'field': key})
            rows.append(fresh)
    return {'status': 'verified_checkpoint_history_and_topology' if steps and not mismatch else 'inconsistent_result_fracture',
            'rows': rows, 'mismatches': mismatch,
            'scope': 'Damage, complete two-point failure, reference-area dissipation and connectivity independently reduced from authenticated NPZ; no IPC/contact recomputation.'}


def authenticated_events(steps, evidence, reported, audit_rows):
    """Derive onset brackets; do not promote unaudited contact annotations."""
    if evidence['status'] != 'verified_checkpoint_history_and_topology':
        return {'events': None, 'consistent': False, 'mismatches': [], 'contact_status': 'unknown'}
    predicates = {'damage_onset': lambda r: r['max_damage'] > 1e-8,
                  'first_fully_separated_segment': lambda r: r['separated_segments'] > 0}
    events, mismatches = {}, []
    for name, predicate in predicates.items():
        index = next((i for i, row in enumerate(evidence['rows']) if predicate(row)), None)
        event = {'status': 'not_observed_in_accepted_prefix'} if index is None else {
            'status': 'observed', 'state_index': index,
            'depth_interval_mm': [steps[index-1]['depth_mm'] if index else None, steps[index]['depth_mm']],
            'fz_n': steps[index]['reaction_fz_n']}
        events[name] = event
        original = (reported or {}).get(name)
        if original is not None and any(original.get(key) != value for key, value in event.items()):
            mismatches.append(name)
    # Future audit rows may publish their independently recomputed fracture
    # diagnostics. A record digest alone cannot establish contact attribution.
    contact_rows = [row.get('fracture') if row.get('cohesive_events_passed') is True else None for row in audit_rows]
    contact_available = len(contact_rows) == len(steps) and bool(steps) and all(
        isinstance(r, dict) and 'blade_surface_connected_crack_contacts' in r for r in contact_rows)
    contact_name = 'blade_in_surface_connected_crack'
    if contact_available:
        for i, fresh in enumerate(contact_rows):
            if fresh['blade_surface_connected_crack_contacts'] != steps[i].get('fracture', {}).get('blade_surface_connected_crack_contacts'):
                mismatches.append(f'contact_state_{i}')
            evidence['rows'][i]['blade_surface_connected_crack_contacts'] = fresh['blade_surface_connected_crack_contacts']
        index = next((i for i, fresh in enumerate(contact_rows) if fresh['blade_surface_connected_crack_contacts'] > 0
                      and evidence['rows'][i]['surface_connected_crack_length_mm'] > 0), None)
        event = {'status': 'not_observed_in_accepted_prefix'} if index is None else {
            'status': 'observed', 'state_index': index,
            'depth_interval_mm': [steps[index-1]['depth_mm'] if index else None, steps[index]['depth_mm']],
            'fz_n': steps[index]['reaction_fz_n']}
        events[contact_name] = event
        original = (reported or {}).get(contact_name)
        if original is not None and any(original.get(key) != value for key, value in event.items()):
            mismatches.append(contact_name)
    else:
        events[contact_name] = {'status': 'unknown_no_independent_contact_evidence'}
    return {'events': events, 'consistent': not mismatches, 'mismatches': mismatches,
            'contact_status': 'independent_audit_evidence' if contact_available else 'unknown'}


def path_segments(directory, step, index):
    ids = step.get('fracture', {}).get('surface_connected_failed_segments')
    checkpoint = directory/'states'/f'{index:04d}.npz'
    if ids is None or not checkpoint.is_file():
        return None
    with np.load(checkpoint, allow_pickle=False) as state:
        if not {'reference_xy', 'interface_segments'} <= set(state.files):
            return None
        seg = state['interface_segments'][np.asarray(ids, dtype=int)]
        coordinates = state['reference_xy'][seg].reshape(-1, 2, 2, 2).mean(axis=2)
    return coordinates.tolist()


def compression_regularization_metadata(config):
    """Declared constitutive stress bound, never a measured/contact force bound."""
    law = config['cohesive']
    ell = law.get('compression_regularization_mm')
    bounded = config.get('closure_regularization') == BOUNDED_COMPRESSION
    return {'mode': config.get('closure_regularization'),
            'compression_regularization_mm': ell,
            'regularization_length_role': 'Derived reference gap; not a fitted independent parameter' if bounded else None,
            'compression_branch_traction_magnitude_bound_mpa': law['compression_stiffness_mpa_per_mm']*ell if bounded else None,
            'bound_scope': 'Magnitude of this reversible cohesive compression traction only; not a bound on IPC contact traction, total contact force, nodal force or blade force.',
            'linear_mode_note': None if bounded else 'The linear/legacy branch has no finite global traction bound declared here.'}


def _summarize_case(case, manifest_dir, source_root=PROJECT):
    result_path = Path(case.get('result_directory', ''))
    directory = (manifest_dir/result_path).resolve() if not result_path.is_absolute() else result_path
    summary = {'id': case['id'], 'result_directory': str(directory), 'status': 'missing_result',
               'qualified_states': 0, 'completed_target_and_audit': False, 'source_sha256': {},
               'law_regularization': compression_regularization_metadata(case['config'])}
    if not (directory/'result.json').is_file():
        return summary
    report = read_json(directory/'result.json')
    config_match = report.get('configuration') == case['config']
    audit = audit_prefix(directory, report)
    n = audit['qualified_states']
    steps = report.get('steps', [])[:n]
    depth = np.array([s.get('depth_mm', np.nan) for s in steps])
    force = np.array([s.get('reaction_fz_n', np.nan) for s in steps])
    curve_valid = bool(len(depth) and np.isfinite(depth).all() and np.isfinite(force).all()
                       and (len(depth) == 1 or np.all(np.diff(depth) > 0)))
    preparation = report.get('fracture_topology_metadata', {}).get('preparation') or (steps[0].get('preparation') if steps else None)
    clearance = case['config']['preparation_clearance_mm']
    preparation_match = bool(preparation and preparation.get('mode') == 'explicit_vertical_clearance'
                             and np.isclose(preparation.get('initial_depth_mm', np.nan), -clearance, rtol=0, atol=1e-12)
                             and np.isclose(preparation.get('vertical_clearance_mm', np.nan), clearance, rtol=0, atol=1e-12))
    expected_dt = np.diff(np.r_[-clearance, depth])/.2
    actual_dt = np.asarray([s.get('material', {}).get('dt_s', np.nan) for s in steps])
    schedule_match = bool(len(depth) and np.isfinite(actual_dt).all() and np.all(actual_dt > 0)
                          and np.allclose(actual_dt, expected_dt, rtol=1e-10, atol=1e-12))
    sources = report.get('source_sha256', {})
    source = source_check(sources, source_root)
    audit_rows = read_json(directory/'audit.json').get('rows', [])[:n] if (directory/'audit.json').is_file() else []
    fracture_evidence = checkpoint_fracture_evidence(directory, steps, case['config']['cohesive'])
    event_evidence = authenticated_events(steps, fracture_evidence, report.get('events'), audit_rows)
    record_bound = bool(steps and len(audit_rows) == len(steps) and all(
        row.get('accepted_record_sha256') == record_digest(step) for row, step in zip(audit_rows, steps)))
    completed = bool(config_match and preparation_match and schedule_match and audit.get('passed') and curve_valid and report.get('completed') is True
                     and abs(depth[-1]-case['config']['target_mm']) < 1e-12
                     and abs(depth[0]) < 1e-12 and fracture_evidence['status'] == 'verified_checkpoint_history_and_topology'
                     and event_evidence['consistent'])
    last = steps[-1] if steps else {}
    energy = report.get('energy_accounting', {})
    fractions = []
    for step in steps if record_bound else []:
        fracture = step.get('fracture', {})
        c = fracture.get('compression_stored_energy_n_mm')
        b = step.get('material', {}).get('stored_energy_n_mm')
        total = fracture.get('cohesive_stored_energy_n_mm')
        ipc = fracture.get('ipc_barrier_stored_energy_n_mm')
        if c is not None and b is not None and total is not None and ipc is not None:
            denominator = b+total+ipc
            if denominator > 1e-12:
                fractions.append(c/denominator)
    summary.update(status=report.get('status'), result_sha256=digest(directory/'result.json'),
                   config_matches_manifest=config_match, audit=audit, qualified_states=n,
                   preparation=preparation, preparation_matches_manifest=preparation_match,
                   physical_schedule_matches_depth_and_speed=schedule_match,
                   completed_target_and_audit=completed, source_sha256=sources, current_source_check=source,
                   interpreter=report.get('interpreter'), target_mm=case['config']['target_mm'],
                   accepted_states=len(report.get('steps', [])), last_qualified_depth_mm=float(depth[-1]) if len(depth) else None,
                   curve_valid=curve_valid, qualified_curve={'depth_mm': depth.tolist(), 'fz_n': force.tolist()},
                   energy_accounting=energy, energy_qualification_preserved=energy.get('energy_balance_qualified'),
                   energy_qualification_supported_by_this_tool=False,
                   discretization_qualification_preserved=energy.get('discretization_converged'),
                   fracture_checkpoint_evidence=fracture_evidence,
                   accepted_records_bound_to_audit=record_bound,
                   events=event_evidence['events'], event_evidence=event_evidence,
                   event_qualification='independent_checkpoint_history_and_topology_only',
                   energy_report_scope='Original runner accepted-path ledger; not recomputed by comparator. Unauthenticated states do not qualify its entries.',
                   geometry=geometry_metrics(directory/'states'/'0000.npz', case['config'],
                                             report.get('fracture_topology_metadata', {}), report.get('signature', {})),
                   maximum_virtual_compression_stored_fraction=max(fractions) if fractions else None,
                   virtual_compression_fraction_authentication='accepted_record_hash_bound_to_existing_audit' if record_bound else 'unknown_missing_record_binding',
                   virtual_compression_fraction_definition='Recoverable compression storage / (bulk + total cohesive + IPC storage); not a force share or fracture dissipation.',
                   endpoint_fracture={k: (fracture_evidence['rows'][-1] if fracture_evidence['rows'] else {}).get(k) for k in
                                      ('max_damage', 'separated_segments', 'surface_connected_crack_length_mm',
                                       'fracture_dissipation_n_mm', 'blade_surface_connected_crack_contacts')},
                   reference_path_segments_mm=path_segments(directory, {'fracture': fracture_evidence['rows'][-1]}, n-1) if steps and fracture_evidence['status'] == 'verified_checkpoint_history_and_topology' else None,
                   path_definition='Midlines of paired saved inset reference faces, only fully failed surface-connected interfaces; not exact original unsplit edge coordinates.',
                   actual_physical_schedule=[{'depth_mm': s['depth_mm'], 'dt_s': s.get('material', {}).get('dt_s'),
                                              'load_step': s.get('load_step')} for s in steps])
    return summary


def summarize_case(case, manifest_dir, source_root=PROJECT):
    try:
        return _summarize_case(case, manifest_dir, source_root)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return {'id': case['id'], 'status': 'invalid_result', 'qualified_states': 0,
                'completed_target_and_audit': False, 'source_sha256': {},
                'error': f'{type(error).__name__}: {error}'}


def integral(y, x):
    return float(np.sum(.5*(np.asarray(y)[1:]+np.asarray(y)[:-1])*np.diff(x)))


def square_integral(y, x):
    """Exact squared L2 integral of the interpolated piecewise-linear curve."""
    y = np.asarray(y)
    return float(np.sum((y[:-1]**2+y[:-1]*y[1:]+y[1:]**2)*np.diff(x)/3))


def curve_comparison(candidate, reference, required_depth, resolution=1e-4):
    result = {'status': 'inconclusive', 'relative_L2': None, 'endpoint_difference_n': None,
              'external_work_difference_n_mm': None, 'no_extrapolation': True}
    if not candidate.get('curve_valid') or not reference.get('curve_valid'):
        return result
    a, b = candidate['qualified_curve'], reference['qualified_curve']
    xa, xb = np.asarray(a['depth_mm']), np.asarray(b['depth_mm'])
    lo, hi = max(xa[0], xb[0]), min(xa[-1], xb[-1], required_depth)
    if hi <= lo:
        return result
    grid = np.unique(np.r_[xa[(xa >= lo)&(xa <= hi)], xb[(xb >= lo)&(xb <= hi)], lo, hi])
    fa, fb = np.interp(grid, xa, a['fz_n']), np.interp(grid, xb, b['fz_n'])
    norm = square_integral(fb, grid)
    resolved = bool(norm > resolution**2*(hi-lo))
    work_a, work_b = integral(fa, grid), integral(fb, grid)
    result.update(common_qualified_interval_mm=[float(lo), float(hi)],
                  relative_L2=float(np.sqrt(square_integral(fa-fb, grid)/norm)) if resolved else None,
                  reaction_reference_resolved=resolved, overlap_endpoint_difference_n=float(fa[-1]-fb[-1]),
                  overlap_external_work_n_mm={'candidate': work_a, 'reference': work_b},
                  overlap_work_difference_n_mm=work_a-work_b)
    full = (candidate['completed_target_and_audit'] and reference['completed_target_and_audit']
            and abs(lo) < 1e-12 and abs(hi-required_depth) < 1e-12)
    if full:
        result.update(status='comparison_available' if resolved else 'inconclusive_near_zero_reference',
                      target_endpoint_reaction_n={'candidate': float(fa[-1]), 'reference': float(fb[-1])},
                      endpoint_difference_n=float(fa[-1]-fb[-1]), external_work_difference_n_mm=work_a-work_b,
                      relative_work_difference=abs(work_a-work_b)/abs(work_b) if abs(work_b) > resolution*(hi-lo) else None)
    return result


def sampled_path_distance(a, b):
    if a is None or b is None or not len(a) or not len(b):
        return None
    a, b = np.asarray(a), np.asarray(b)
    def directed(edges, other):
        points = np.concatenate((edges[:, 0], edges.mean(axis=1), edges[:, 1]))
        p, v = other[:, 0], other[:, 1]-other[:, 0]
        vv = np.sum(v*v, axis=1)
        distance = 0.
        for point in points:
            t = np.clip(np.sum((point-p)*v, axis=1)/np.maximum(vv, 1e-30), 0., 1.)
            distance = max(distance, float(np.min(np.linalg.norm(point-p-t[:, None]*v, axis=1))))
        return distance
    return max(directed(a, b), directed(b, a))


def compare_manifest(manifest, manifest_dir, source_root=PROJECT):
    validate_manifest(manifest)
    cases = [summarize_case(c, Path(manifest_dir), source_root) for c in manifest['cases']]
    by_id = {c['id']: c for c in cases}
    families = []
    for family in manifest['families']:
        members = [by_id[i] for i in family['cases']]
        ref = by_id[family['reference']]
        same_sources = bool(ref['source_sha256'] and all(c['source_sha256'] == ref['source_sha256'] for c in members))
        current = all(c.get('current_source_check', {}).get('status') == 'match' for c in members)
        complete = all(c['completed_target_and_audit'] for c in members)
        pairs = []
        for case in members:
            if case['id'] == ref['id']:
                continue
            pair = curve_comparison(case, ref, family['required_depth_mm'])
            pair.update(candidate=case['id'], reference=ref['id'],
                        sampled_reference_path_distance_mm=sampled_path_distance(case.get('reference_path_segments_mm'), ref.get('reference_path_segments_mm')) if case['completed_target_and_audit'] and ref['completed_target_and_audit'] else None,
                        onset_intervals={'candidate': case.get('events', {}).get('damage_onset') if case.get('events') else None,
                                         'reference': ref.get('events', {}).get('damage_onset') if ref.get('events') else None})
            if not same_sources or not current:
                pair['status'] = 'inconclusive_source_identity'
            pairs.append(pair)
        families.append(dict(id=family['id'], kind=family['kind'],
                             status='comparison_available' if complete and same_sources and current and all(p['status']=='comparison_available' for p in pairs) else 'inconclusive',
                             source_identity_same=same_sources, current_sources_match=current,
                             all_complete_and_audited=complete, pairs=pairs,
                             convergence_qualified=False,
                             note='Pairwise differences are evidence, not a convergence pass. Require three levels, meaningful controls and energy/path trends in a declared protocol.'))
    return {'schema': 'softcut-network-sensitivity-result/1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'comparison_available' if all(f['status']=='comparison_available' for f in families) else 'inconclusive',
            'no_inverse_fitting': True, 'experimental_inputs_used': [], 'new_fe_runs_started_by_comparison': 0,
            'study_manifest': copy.deepcopy(manifest), 'family_metadata_verified': True,
            'comparator_interpreter': sys.executable, 'comparator_sha256': digest(__file__),
            'cases': cases, 'families': families, 'convergence_qualified': False,
            'limitation': 'Saved-state postprocessing, not a new equilibrium audit, material calibration or physical onset prediction.'}


def save_report(output, report):
    write_json(output/'sensitivity.json', report)
    rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(c.get(k)))}</td>' for k in
                   ('id', 'status', 'accepted_states', 'qualified_states', 'last_qualified_depth_mm', 'completed_target_and_audit',
                    'energy_qualification_preserved', 'maximum_virtual_compression_stored_fraction'))+'</tr>' for c in report['cases'])
    page = '<!doctype html><meta charset="utf-8"><title>网络切割数值敏感性</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:20px;color:#173041}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ccd}pre{white-space:pre-wrap;background:#eef3f6;padding:16px}</style>'
    page += '<h1>网络切割数值敏感性</h1><p>状态：'+html.escape(report['status'])+'。仅比较已接受且可认证的前缀；未完成目标、缺审计、源码不一致或信号接近残量时，结论为 inconclusive。不会把能量资格由 false 改为 true。</p>'
    page += '<p>压缩项占比是可恢复储能占比，不是断裂耗散或刀反力占比。相同 gap 不代表连续完整体；本适配器的 damage_control 仍是分裂网络。</p><table><tr><th>案例</th><th>运行</th><th>接受态</th><th>认证态</th><th>认证深度/mm</th><th>完成及审计</th><th>原能量资格</th><th>最大压缩储能占比</th></tr>'+rows+'</table>'
    page += '<p>有界压缩模式强制 ell=gap；JSON 中的 Kc·ell 仅是该可恢复内聚压缩支路的牵引幅值上界（MPa），不是 IPC 压力或刀具总接触力上界。旧线性模式不具有该有限全局上界。</p>'
    page += '<h2>控制族比较</h2><pre>'+html.escape(json.dumps(report['families'],ensure_ascii=False,indent=2))+'</pre><p><a href="sensitivity.json">完整机器可读记录</a>；缺失量保留 null，不外插、不补齐。</p>'
    (output/'index.html').write_text(page, encoding='utf-8')


def numerical_hashes():
    paths = list((PROJECT/'prototype').rglob('*.py'))+list((PROJECT/'mechanics').rglob('*.py'))+[PROJECT/'tools/run_cohesive_cut.py']
    return {str(p.relative_to(PROJECT)).replace('\\', '/'): digest(p) for p in sorted(paths)}


def runstudy(manifest, output, python=sys.executable):
    """One attempt for each explicit case, one subprocess at a time, hard bounds."""
    validate_manifest(manifest)
    executable = Path(python).resolve()
    if not executable.is_file():
        raise ValueError('Interpreter must be an existing executable path')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    resolved = copy.deepcopy(manifest)
    before = numerical_hashes()
    write_json(output/'manifest.json', resolved)
    started = time.monotonic()
    executions = []
    for case in resolved['cases']:
        case['result_directory'] = str(output/case['id']/'result')
    for case in resolved['cases']:
        if numerical_hashes() != before:
            executions.append({'case': case['id'], 'status': 'not_started_source_changed'})
            break
        if time.monotonic()-started+case['seconds']+60 > manifest['total_wall_budget_s']:
            executions.append({'case': case['id'], 'status': 'not_started_total_budget'})
            break
        directory = output/case['id']
        directory.mkdir()
        write_json(directory/'config.json', case['config'])
        args = [str(executable), '-m', 'tools.run_cohesive_cut', '--config', str(directory/'config.json'),
                '--output', case['result_directory'], '--seconds', str(case['seconds'])]
        record = {'case': case['id'], 'args': args, 'cwd': str(PROJECT), 'timeout_s': case['seconds']+60,
                  'source_sha256_before': before, 'retries': 0}
        try:
            flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            with (directory/'launcher.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(args, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=case['seconds']+60, check=False, creationflags=flags)
            record.update(status='process_ended', returncode=result.returncode)
        except subprocess.TimeoutExpired:
            record.update(status='launcher_timeout', returncode=None)
        except OSError as error:
            record.update(status='launch_error', returncode=None, error=str(error))
        record['source_sha256_after'] = numerical_hashes()
        executions.append(record)
        write_json(output/'execution.json', executions)
        result_file = Path(case['result_directory'])/'result.json'
        success = record.get('returncode') == 0 and result_file.is_file() and read_json(result_file).get('completed') is True
        if record['source_sha256_after'] != before or (not success and manifest['stop_on_failure']):
            break
    write_json(output/'manifest.json', resolved)
    report = compare_manifest(resolved, output)
    report['execution'] = executions
    report['new_fe_runs_started'] = sum('args' in e for e in executions)
    save_report(output, report)
    return report


def template(config, kind, values, seconds):
    if kind not in AXES or len(values) < 2:
        raise ValueError('Choose a supported family and at least two values')
    if config.get('closure_regularization') == BOUNDED_COMPRESSION and config['cohesive'].get('compression_regularization_mm') != config['reference_gap_mm']:
        raise ValueError('Template input must already satisfy compression_regularization_mm == reference_gap_mm')
    cases = []
    for i, value in enumerate(values):
        c = copy.deepcopy(config)
        if kind == 'mesh':
            finite_number(value, np.nextafter(0., 1.), 100, 'mesh scale')
            for name in AXES[kind]:
                key = name.split('.')[1]
                if key in c['mesh']:
                    c['mesh'][key] *= value
        else:
            key = next(iter(AXES[kind]))
            if kind == 'damage_control':
                if value not in (0, 1):
                    raise ValueError('damage_control values must be 0 or 1')
                value = bool(value)
            if key.startswith('cohesive.'):
                c['cohesive'][key.split('.')[1]] = value
            else:
                c[key] = value
            if kind == 'gap' and c.get('closure_regularization') == BOUNDED_COMPRESSION:
                c['cohesive']['compression_regularization_mm'] = c['reference_gap_mm']
        cases.append({'id': f'{kind}_{i}', 'config': c, 'seconds': seconds,
                      'result_directory': f'{kind}_{i}/result'})
    result = {'schema': SCHEMA, 'no_inverse_fitting': True, 'stop_on_failure': True,
              'total_wall_budget_s': sum(seconds+60 for _ in cases), 'cases': cases,
              'families': [{'id': kind, 'kind': kind, 'cases': [c['id'] for c in cases],
                            'reference': cases[-1]['id'], 'required_depth_mm': config['target_mm']}]}
    return validate_manifest(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('compare', 'runstudy'):
        p = sub.add_parser(name)
        p.add_argument('--manifest', required=True, type=Path)
        p.add_argument('--output', required=True, type=Path)
        if name == 'runstudy':
            p.add_argument('--python', default=sys.executable)
    p = sub.add_parser('template')
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--family', required=True, choices=tuple(AXES))
    p.add_argument('--values', required=True, nargs='+', type=float)
    p.add_argument('--seconds', required=True, type=float)
    p.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'template':
        result = template(read_json(args.config), args.family, args.values, args.seconds)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return 0
    manifest = read_json(args.manifest)
    if args.command == 'runstudy':
        report = runstudy(manifest, args.output, args.python)
    else:
        validate_manifest(manifest)
        args.output.mkdir(parents=True, exist_ok=False)
        report = compare_manifest(manifest, args.manifest.resolve().parent)
        report['manifest_sha256'] = digest(args.manifest)
        save_report(args.output, report)
    return 0 if report['status'] == 'comparison_available' else 2


if __name__ == '__main__':
    raise SystemExit(main())
