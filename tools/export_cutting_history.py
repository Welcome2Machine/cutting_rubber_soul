"""Export accepted cutting-study motion and forces without a mechanics solver.

Supports intact IPC and directional-rational-at1-v1 archives. No interpolation,
stress recomputation, experimental time assignment, or physical crack detection.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, positive=False):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and np.isfinite(value) and (not positive or value > 0), f'Invalid {name}')
    return float(value)


def load_history(source, *, allow_incomplete_prefix=False, allow_failed_prefix=False):
    """Validate the whole accepted prefix before exporting any numerical data."""
    source = Path(source).resolve()
    payload = (source/'result.json').read_bytes()
    report = json.loads(payload.decode('utf-8-sig'))
    complete = report.get('status') == 'completed' and report.get('completed') is True
    _require(complete or ((allow_incomplete_prefix or allow_failed_prefix) and report.get('completed') is False
                          and report.get('status') in ('failed', 'model_boundary', 'stopped')),
             'Require completed history or explicit --allow-incomplete-prefix for failed/model_boundary/stopped history')
    directional = report.get('framework') == 'directional-rational-at1-v1'
    _require(directional or (not report.get('framework')
                            and report.get('simplifications', {}).get('fracture') is False),
             'Unsupported history: expected intact IPC or directional-rational-at1-v1')
    config = report.get('config')
    _require(isinstance(config, dict), 'Missing source config')
    for key in ('length_mm', 'height_mm', 'width_mm'):
        _number(config.get(key), key, positive=True)
    angle = np.deg2rad(_number(config.get('inclination_deg'), 'inclination_deg'))
    source_hashes = report.get('source_sha256')
    _require(isinstance(source_hashes, dict) and bool(source_hashes), 'Missing source fingerprints')
    verified, missing = [], []
    for name, value in source_hashes.items():
        _require(isinstance(name, str) and isinstance(value, str) and len(value) == 64
                 and all(c in '0123456789abcdef' for c in value.lower()), 'Invalid source fingerprint')
        frozen = (source/'source'/name).resolve()
        _require(frozen.is_relative_to(source/'source'), 'Source fingerprint path leaves snapshot')
        if frozen.is_file():
            _require(digest(frozen) == value.lower(), f'Frozen source hash mismatch: {name}')
            verified.append(name)
        else:
            missing.append(name)
    steps = report.get('steps')
    _require(isinstance(steps, list) and len(steps) > 0, 'No accepted states')
    target = _number(report.get('target_mm'), 'target_mm', positive=True)
    depths = np.array([_number(s.get('depth_mm'), 'step depth') for s in steps])
    _require(abs(depths[0]) <= 1e-12 and np.all(np.diff(depths) > 0)
             and depths[-1] <= target+1e-12, 'Depth history must start at zero and increase to at most target')
    if complete:
        _require(abs(depths[-1]-target) <= 1e-12, 'Completed history does not reach target')
    increment = _number((report.get('parameters', {}).get('increment') if directional
                         else report.get('loading_increment_mm')), 'increment', positive=True)
    if directional:
        _require(np.all(np.diff(depths) <= increment+1e-10), 'Directional history skips a loading increment')
    else:
        expected = np.r_[0., np.arange(increment, target-1e-12, increment), target]
        _require(len(depths) <= len(expected) and np.allclose(depths, expected[:len(depths)],
                                                            atol=1e-12, rtol=0),
                 'Intact history is not a consecutive loading prefix')
    _require(abs(_number(report.get('last_depth_mm'), 'last_depth_mm')-depths[-1]) <= 1e-12,
             'Last accepted depth mismatch')
    _require(report.get('last_checkpoint') == f'states/{len(steps)-1:04d}.npz',
             'Last accepted checkpoint mismatch')
    expected_files = {f'{i:04d}.npz' for i in range(len(steps))}
    _require({p.name for p in (source/'states').glob('*.npz')} == expected_files,
             'Missing or unreported accepted state files')
    tolerance = _number(report.get('force_tolerance_n'), 'force_tolerance_n', positive=True)
    reference = triangles = previous = body_count = None
    positions, knives, damage, old_damage, hashes, rows = [], [], [], [], {}, []
    previous_damage = None
    for i, step in enumerate(steps):
        relative = f'states/{i:04d}.npz'; path = source/relative
        hashes[relative] = digest(path)
        with np.load(path, allow_pickle=False) as state:
            required = {'positions', 'previous_positions', 'reference_xy', 'triangles', 'depth_mm', 'body_nodes'}
            _require(required.issubset(state.files), f'Missing checkpoint fields: {relative}')
            X, x = state['reference_xy'].copy(), state['positions'].copy()
            tri, count = state['triangles'].copy(), state['body_nodes']
            _require(count.shape == () and np.issubdtype(count.dtype, np.integer), 'Invalid body node count')
            count = int(count)
            _require(X.ndim == 2 and X.shape[1] == 2 and x.shape == X.shape
                     and np.isfinite(X).all() and np.isfinite(x).all() and 0 < count < len(X),
                     f'Invalid reference/current coordinates: {relative}')
            _require(tri.ndim == 2 and tri.shape[1] == 3 and len(tri) > 0
                     and np.issubdtype(tri.dtype, np.integer) and tri.min() >= 0 and tri.max() < count,
                     f'Invalid body topology: {relative}')
            if i == 0:
                reference, triangles, body_count = X, tri, count
                previous = X
                reference_edges = np.stack((X[tri[:, 1]]-X[tri[:, 0]], X[tri[:, 2]]-X[tri[:, 0]]), axis=-1)
                reference_det = np.linalg.det(reference_edges)
                _require(np.all(reference_det > 0), 'Nonpositive reference element orientation')
            _require(count == body_count and np.array_equal(X, reference) and np.array_equal(tri, triangles),
                     f'Topology/reference changed: {relative}')
            _require(np.array_equal(state['previous_positions'], previous), f'Broken accepted-state chain: {relative}')
            _require(state['depth_mm'].shape == () and abs(float(state['depth_mm'])-depths[i]) <= 1e-12,
                     f'Checkpoint depth mismatch: {relative}')
            edges = np.stack((x[tri[:, 1]]-x[tri[:, 0]], x[tri[:, 2]]-x[tri[:, 0]]), axis=-1)
            J = np.linalg.det(edges)/reference_det
            _require(np.isfinite(J).all() and np.all(J > 0), f'Invalid accepted Jacobian: {relative}')
            _require(abs(_number(step.get('min_j'), 'min_j')-J.min()) <= 1e-8, 'Reported Jacobian mismatch')
            residual = _number(step.get('residual_n'), 'residual_n')
            _require(0 <= residual <= tolerance, f'Unaccepted force residual: {relative}')
            fz = _number(step.get('reaction_fz_n'), 'reaction_fz_n')
            fx = None if step.get('reaction_fx_n') is None else _number(step['reaction_fx_n'], 'reaction_fx_n')
            max_damage = None
            if directional:
                _require({'damage', 'old_damage'}.issubset(state.files), 'Missing diffuse-damage history')
                d, old = state['damage'].copy(), state['old_damage'].copy()
                _require(d.shape == old.shape == (count,) and np.isfinite(d).all() and np.isfinite(old).all()
                         and np.all((0 <= old) & (old <= d) & (d <= 1)), 'Invalid irreversible damage history')
                _require(np.array_equal(old, np.zeros(count) if i == 0 else previous_damage),
                         'Broken accepted damage chain')
                max_damage = float(d.max())
                _require(abs(_number(step.get('max_damage'), 'max_damage')-max_damage) <= 1e-12,
                         'Reported damage maximum mismatch')
                damage.append(d); old_damage.append(old); previous_damage = d
            else:
                _require('damage' not in state.files, 'Intact report contains damage')
            positions.append(x[:count]); knives.append(x[count:]); previous = x
            rows.append(dict(frame=i, solver_depth_mm=float(depths[i]), reaction_fx_n=fx,
                             reaction_fz_n=fz, fx_available=fx is not None, force_residual_n=residual,
                             min_J=float(J.min()), max_diffuse_damage=max_damage))
    es, en = np.array([np.cos(angle), -np.sin(angle)]), np.array([np.sin(angle), np.cos(angle)])
    local = reference[:body_count]@np.column_stack((es, en))
    top = np.flatnonzero(np.isclose(local[:, 1], config['height_mm'], atol=1e-8, rtol=0))
    _require(len(top) >= 2, 'Cannot identify original top material surface from source geometry')
    top = top[np.argsort(local[top, 0], kind='stable')]
    _require(abs(local[top[0], 0]) <= 1e-8 and abs(local[top[-1], 0]-config['length_mm']) <= 1e-8,
             'Top surface does not span source specimen length')
    edges = np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = {tuple(edge) for edge in edges[counts == 1]}
    _require(all(tuple(sorted(pair)) in boundary for pair in zip(top[:-1], top[1:])),
             'Top material nodes do not form a consecutive boundary')
    knife_local = np.array(knives[0])@np.column_stack((es, en))
    lowest = np.isclose(knife_local[:, 1], knife_local[:, 1].min(), atol=1e-10, rtol=0)
    can_reference = en[1] > 1e-8 and np.all((knife_local[lowest, 0] >= 0)
                                                         & (knife_local[lowest, 0] <= config['length_mm']))
    plane_offset = float((knife_local[:, 1].min()-config['height_mm'])/en[1]) if can_reference else None
    plane_reference = dict(available=bool(can_reference), solver_depth_offset_mm=plane_offset,
        definition='First geometric tangency of the saved rigid knife polygon translating vertically downward to the undeformed original top plane; not first force or experimental time zero.',
        limitation='Rigid polygon/undeformed-plane reference only; IPC acts at positive gap and may deform the surface before this depth.')
    for row in rows:
        row['geometric_plane_depth_mm'] = None if plane_offset is None else row['solver_depth_mm']-plane_offset
    _require((source/'result.json').read_bytes() == payload, 'Source report changed while reading')
    return dict(source=source, report=report, report_sha256=hashlib.sha256(payload).hexdigest(),
                directional=directional, complete=complete, reference=reference[:body_count],
                triangles=triangles, positions=np.array(positions), top=top, local=local,
                knives=np.array(knives),
                plane_reference=plane_reference,
                damage=np.array(damage), old_damage=np.array(old_damage), rows=rows,
                hashes=hashes, source_verified=verified, source_missing=missing)


def bottom_force(reference, triangles, positions, cauchy, width, bottom):
    """Material bulk support resultant, using the stress from that exact state."""
    X, x = reference[triangles], positions[triangles]
    Dm = np.stack((X[:, 1]-X[:, 0], X[:, 2]-X[:, 0]), axis=-1)
    Ds = np.stack((x[:, 1]-x[:, 0], x[:, 2]-x[:, 0]), axis=-1)
    inverse = np.linalg.inv(Dm); F = Ds@inverse; J = np.linalg.det(F)
    cauchy = np.asarray(cauchy)
    _require(cauchy.shape == (len(triangles), 3, 3) and np.isfinite(cauchy).all()
             and np.allclose(cauchy, cauchy.swapaxes(1, 2), atol=1e-9, rtol=1e-9), 'Invalid saved Cauchy stress')
    P = J[:, None, None]*cauchy[:, :2, :2]@np.linalg.inv(F).transpose(0, 2, 1)
    gradients = np.array([[-1., -1.], [1., 0.], [0., 1.]])[None]@inverse
    nodal = np.einsum('e,eiJ,eaJ->eai', .5*width*np.linalg.det(Dm), P, gradients)
    force = np.zeros_like(reference); np.add.at(force, triangles.ravel(), nodal.reshape(-1, 2))
    return force[bottom].sum(axis=0)


def recover_forces(data):
    """Recover a historical resultant only after an independent saved-Fz check.

    This is not a reconstructed contact-force distribution. It uses body force
    balance, only for the oblique host's fixed-bottom, body/knife-only contact.
    """
    report, source = data['report'], data['source']
    case = report.get('parameters', {}).get('case') if data['directional'] else report.get('case')
    _require(case == 'oblique', 'Force recovery supports fixed-bottom oblique histories only')
    X, positions = data['reference'], data['positions']
    bottom = np.flatnonzero(np.isclose(data['local'][:, 1], 0., atol=1e-8, rtol=0))
    _require(len(bottom) > 1 and np.allclose(positions[:, bottom], X[bottom], atol=1e-10, rtol=0),
             'Force recovery requires stationary bottom support')
    dhat = report.get('dhat_mm')
    if dhat is None and data['directional']:
        filename = 'prototype/ipc_cutting.py'
        _require(filename in data['source_verified'], 'Need verified frozen knife-contact distance')
        tree = ast.parse((source/'source'/filename).read_text(encoding='utf-8-sig'))
        values = [ast.literal_eval(k.value) for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == 'DirectionalFractureProblem'
                  for k in node.keywords if k.arg == 'dhat']
        _require(len(values) == 1, 'Cannot establish frozen contact distance')
        dhat = values[0]
    dhat = _number(dhat, 'saved contact distance', positive=True)
    angle = np.deg2rad(report['config']['inclination_deg']); en = np.array([np.sin(angle), np.cos(angle)])
    # Every straight knife edge is above this plane when all its vertices are.
    _require(np.min(data['knives']@en) > dhat+1e-9, 'Knife contact can reach bottom support; recovery invalid')
    saved = []
    for relative in data['hashes']:
        with np.load(source/relative, allow_pickle=False) as state:
            saved.append(state['cauchy'].copy() if 'cauchy' in state.files else None)
    if all(stress is not None for stress in saved):
        forces = np.array([bottom_force(X, data['triangles'], x, stress,
                           report['config']['width_mm'], bottom) for x, stress in zip(positions, saved)])
        origin = 'archived accepted Cauchy stress; no constitutive recomputation'
    else:
        _require(not any(stress is not None for stress in saved), 'Partial saved-stress history')
        _require(not data['directional'] and report.get('volumetric_discretization', 'element') == 'element',
                 'Mixed/damaged force recovery requires its saved consistent stress')
        root = Path(__file__).resolve().parents[1]
        for name in ('prototype/material.py', 'mechanics/finite_strain.py'):
            _require(digest(root/name) == report['source_sha256'].get(name),
                     f'Cannot recover intact force: current material source differs from archive ({name})')
        sys.path.insert(0, str(root))
        from prototype.material import response
        config = SimpleNamespace(**report['config'])
        T = X[data['triangles']]
        inverse = np.linalg.inv(np.stack((T[:, 1]-T[:, 0], T[:, 2]-T[:, 0]), axis=-1))
        forces = []
        for x in positions:
            t = x[data['triangles']]
            F = np.stack((t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=-1)@inverse
            forces.append(bottom_force(X, data['triangles'], x, response(F, config, tangent=False)['cauchy'],
                                       config.width_mm, bottom))
        forces = np.array(forces)
        origin = 'intact stress recomputed with material/mechanics source SHA256 identical to archive, using saved config; no IPC'
    _require(forces.shape == (len(data['rows']), 2) and np.isfinite(forces).all(), 'Invalid recovered forces')
    free_nodes = len(X)-len(bottom)
    for row, force in zip(data['rows'], forces):
        bound = np.sqrt(free_nodes)*row['force_residual_n']+1e-8*max(1., abs(row['reaction_fz_n']), float(np.linalg.norm(force)))
        defect = abs(force[1]-row['reaction_fz_n'])
        _require(defect <= bound, f'Historical force recovery disagrees with native Fz at frame {row["frame"]}: {defect:g}>{bound:g} N')
        row.update(recovered_fx_n=float(force[0]), recovered_fz_n=float(force[1]),
                   recovery_fz_defect_n=float(defect), recovery_force_error_bound_n=float(bound))
        if row['fx_available']:
            _require(abs(force[0]-row['reaction_fx_n']) <= bound, 'Recovered/native Fx mismatch')
    return dict(method='sum of assembled bulk nodal gradients on the fixed bottom', stress_source=origin,
                force_convention='rubber_on_knife_lab_xz', bottom_material_node_count=len(bottom),
                validity='Oblique host: only bottom fixed, no body forces, body/knife-only contact; knife farther than dhat from bottom.',
                accepted_check='Recovered Fz must match native saved Fz within sqrt(free body nodes)*L2 residual plus roundoff.',
                recovered_Fx_is_native=False)


def export_history(source, output, *, allow_incomplete_prefix=False, allow_failed_prefix=False, recover_fx=False, include_body=False):
    data = load_history(source, allow_incomplete_prefix=allow_incomplete_prefix, allow_failed_prefix=allow_failed_prefix)
    recovery = recover_forces(data) if recover_fx else None
    for relative, value in data['hashes'].items():
        _require(digest(data['source']/relative) == value, 'Source state changed during export')
    _require(digest(data['source']/'result.json') == data['report_sha256'], 'Source report changed during export')
    output = Path(output).resolve()
    _require(not output.exists() or not any(output.iterdir()), 'Output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    rows, X, positions, top = data['rows'], data['reference'], data['positions'], data['top']
    with (output/'forces.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output/'top_surface.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'solver_depth_mm', 'geometric_plane_depth_mm', 'material_node_id', 'reference_s_mm', 'reference_n_mm',
                         'reference_x_mm', 'reference_z_mm', 'current_x_mm', 'current_z_mm',
                         'displacement_x_mm', 'displacement_z_mm', 'diffuse_damage'])
        for i, row in enumerate(rows):
            for node in top:
                d = data['damage'][i, node] if data['directional'] else ''
                writer.writerow([i, row['solver_depth_mm'], row['geometric_plane_depth_mm'], node, *data['local'][node], *X[node],
                                 *positions[i, node], *(positions[i, node]-X[node]), d])
    depths = np.array([r['solver_depth_mm'] for r in rows])
    geometric_depths = np.array([np.nan if r['geometric_plane_depth_mm'] is None else r['geometric_plane_depth_mm'] for r in rows])
    arrays = dict(surface_node_ids=top, reference_xy=X[top],
                  surface_positions=positions[:, top], surface_displacements=positions[:, top]-X[top][None],
                  depths_mm=depths, geometric_depths_mm=geometric_depths,
                  reaction_fz_n=np.array([r['reaction_fz_n'] for r in rows]),
                  reaction_fx_n=np.array([np.nan if r['reaction_fx_n'] is None else r['reaction_fx_n'] for r in rows]),
                  fx_available=np.array([r['fx_available'] for r in rows]))
    if data['directional']:
        arrays.update(diffuse_damage=data['damage'][:, top])
    if recovery:
        arrays.update(recovered_fx_n=np.array([r['recovered_fx_n'] for r in rows]),
                      recovered_fz_n=np.array([r['recovered_fz_n'] for r in rows]),
                      recovery_force_error_bound_n=np.array([r['recovery_force_error_bound_n'] for r in rows]))
    np.savez_compressed(output/'surface_history.npz', **arrays)
    generated = ['forces.csv', 'top_surface.csv', 'surface_history.npz']
    if include_body:
        body = dict(material_node_ids=np.arange(len(X)), reference_xy_mm=X, triangles=data['triangles'],
                    positions_xy_mm=positions, solver_depth_mm=depths)
        if data['directional']:
            body.update(diffuse_damage=data['damage'], previous_diffuse_damage=data['old_damage'])
        np.savez_compressed(output/'body_history.npz', **body)
        generated.append('body_history.npz')
    report = data['report']
    manifest = dict(status='exported', export_complete=True, accepted_frames=len(rows),
        source=str(data['source']), source_status=report['status'], source_completed=data['complete'],
        coverage='complete solver history' if data['complete'] else 'accepted prefix only; rejected/boundary trial excluded',
        first_depth_mm=rows[0]['solver_depth_mm'], last_depth_mm=rows[-1]['solver_depth_mm'],
        source_target_mm=report['target_mm'], source_error=report.get('error'),
        model='continuous directional diffuse damage' if data['directional'] else 'intact elastic indentation',
        physical_crack_event_inferred=False, experimental_validation=False,
        use_limit='Motion/force conditional on the source model; intact continuation beyond actual fracture is not post-cutting force.',
        depth_definition='Original solver depth, positive downward. No contact-zero offset or experimental clock inferred.',
        geometric_plane_reference=data['plane_reference'],
        all_body_positions=dict(source=str(data['source']), accepted_state_pattern='states/{frame:04d}.npz',
                                field='positions[:body_nodes]', reference_field='reference_xy[:body_nodes]',
                                material_id='original node index', exported_copy=bool(include_body)),
        coordinates='Laboratory x,z (array columns 0,1), z upward; displacement=current-reference; no amplification.',
        surface_definition='Original inclined top boundary, fixed material node IDs ordered by reference s; not a reidentified current contour.',
        force_definition='Native reaction_fx_n/reaction_fz_n, N, unchanged source signs; missing Fx is empty CSV/NaN NPZ with availability mask.',
        force_sign_convention=report.get('reaction_force_convention',
            'IPC reaction: force of rubber on prescribed knife, laboratory x right, z up; positive Fz resists downward motion.'),
        fx_available_frames=sum(r['fx_available'] for r in rows),
        recovered_force_provenance=recovery,
        source_config=report['config'], source_parameters=report.get('parameters'),
        source_simplifications=report.get('simplifications'), source_framework=report.get('framework'),
        source_volumetric_discretization=report.get('volumetric_discretization', 'element'),
        source_first_damage_interval_mm=report.get('first_damage_interval_mm'),
        source_first_damage_note=report.get('first_damage_note'),
        source_calibration=report.get('calibration'), source_energy_balance_resolved=report.get('energy_balance_resolved'),
        source_result_sha256=data['report_sha256'], source_sha256=report['source_sha256'],
        source_snapshot_files_verified=data['source_verified'], source_snapshot_files_unavailable=data['source_missing'],
        checkpoints_sha256=data['hashes'], exporter_sha256=digest(__file__),
        files_sha256={name:digest(output/name) for name in generated})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete-prefix', '--allow-failed-prefix', action='store_true',
                        help='Export only accepted states of a failed/model_boundary/stopped run; never a live run')
    parser.add_argument('--recover-fx', action='store_true', help='Recover historical bulk-support force, gated by native Fz agreement')
    parser.add_argument('--include-body', action='store_true', help='Also duplicate full body positions/damage; default links original accepted states')
    args = parser.parse_args()
    result = export_history(args.source, args.output, allow_incomplete_prefix=args.allow_incomplete_prefix,
                            recover_fx=args.recover_fx, include_body=args.include_body)
    print(json.dumps({key:result[key] for key in ('status', 'accepted_frames', 'last_depth_mm', 'source_status', 'fx_available_frames')}))


if __name__ == '__main__':
    main()
