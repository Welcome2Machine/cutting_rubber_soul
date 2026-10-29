"""Checked archived-equation diagnostics, never a physical continuation loader.

Execution/solver code may evolve. Constitutive/contact equations must still
match the recorded Git producer, and saved histories/forces are rechecked.
Every differing file is reported; this does NOT claim an exact producer replay.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .run_zero_gap_cut import problem, hashes
from .diagnose_finite_increment import restore_failed_seed, virtual_state
from .audit_finite_increment import stored_equilibrium
from prototype.finite_increment_cut import FrozenVirtualCrack


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def force_equation_ast(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FrozenVirtualCrack')
    names = {'__init__', 'material_bulk', 'lag', 'coupled_friction_tangent', 'evaluate', 'stored'}
    selected = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError('Frozen force equation definition incomplete')
    return ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False)


def check_archived_equations(raw, root):
    """Allow only explicit execution/solver changes, not changed force physics."""
    current = hashes()
    original = raw['source_hashes']
    ref = raw['execution_identity']['git_commit']
    if not isinstance(ref, str) or len(ref) != 40 or any(c not in '0123456789abcdef' for c in ref):
        raise ValueError('An explicit producer Git commit is required')
    differences = {}
    execution = {'tools/run_zero_gap_cut.py', 'numerical_runtime.py',
                 'virtual_path_geometry.py', 'virtual_path_probe.py', 'virtual_friction_work.py',
                 'virtual_constrained_root.py'}
    for name in sorted(original.keys() | current.keys()):
        if original.get(name) == current.get(name):
            continue
        differences[name] = dict(original=original.get(name), current=current.get(name))
        if name in execution:
            continue
        if name not in ('prototype/finite_increment_cut.py', 'prototype/force_globalization.py'):
            raise ValueError(f'Archived force-equation source differs: {name}')
        source = subprocess.check_output(['git', 'show', f'{ref}:projects/softmodeling/{name}'], cwd=root)
        if hashlib.sha256(source).hexdigest() != original[name]:
            raise ValueError('Recorded Git source does not reproduce archived producer hash')
        if name.endswith('finite_increment_cut.py'):
            if force_equation_ast(source.decode()) != force_equation_ast((root/name).read_text()):
                raise ValueError('Frozen physical force equation changed')
    # The factory is execution code but must construct the same physical problem.
    name = 'tools/run_zero_gap_cut.py'
    source = subprocess.check_output(['git', 'show', f'{ref}:projects/softmodeling/{name}'], cwd=root)
    if hashlib.sha256(source).hexdigest() != original[name]:
        raise ValueError('Archived problem factory hash differs from producer Git')
    def factory(text, normalize=False):
        node = next(n for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef) and n.name == 'problem')
        # Only an additional optional solver partition selector; archived config
        # must not contain it, so the original constructor receives no new value.
        if normalize:
            if 'ffm_chord_partition' in raw['config']:
                raise ValueError('Changed factory requires original partition configuration')
            for child in ast.walk(node):
                if isinstance(child, (ast.Set, ast.Tuple)):
                    child.elts = [e for e in child.elts if not (isinstance(e, ast.Constant) and e.value == 'ffm_chord_partition')]
        return ast.dump(node, include_attributes=False)
    old_factory, new_factory = source.decode(), (root/name).read_text()
    if (factory(old_factory) != factory(new_factory) and
            factory(old_factory, True) != factory(new_factory, True)):
        raise ValueError('Archived physical problem factory changed')
    return dict(mode='equation_compatible_diagnostic_only', exact_producer_source=current == original,
        producer_commit=ref, producer_source_hashes=original, current_source_hashes=current,
        source_differences=differences, physical_continuation_permitted=False)


def load_scene(directory, endpoint='production'):
    directory = Path(directory).resolve()
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((directory/'result.json').read_text())
    provenance = check_archived_equations(raw, root)
    p = problem(raw['config'])
    inputs = {'result.json': digest(directory/'result.json')}
    def arrays(path):
        inputs[str(path.relative_to(directory.parent))] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            return {k:z[k].copy() for k in z.files}
    physical = arrays(directory/'failed_trial.npz')
    seed, depth = restore_failed_seed(p, physical)
    _, g, _, j = p.evaluate(seed, p.lag(seed), tangent=False)
    p.update_constraint_free(seed, g)
    real = dict(residual_n=float(np.sqrt(2)*np.linalg.norm(g[p.free])), min_j=float(j),
                chord_safe=p.accepted_chord_is_safe(seed), depth_mm=depth, dt_s=p.material_dt)
    if real['residual_n'] > 1e-4 or j <= 0 or not real['chord_safe']:
        raise ValueError('Archived physical seed is not admissible/equilibrated')
    if endpoint == 'production':
        failure = raw['attempts'][-1]['virtual_failure']
        if failure['seam_indices'] != [48]:
            raise ValueError('Final producer failure is not candidate 48; select diagnostic pair explicitly')
        path = directory/failure['evidence_file']
        if digest(path) != failure['evidence_sha256']:
            raise ValueError('Failed virtual evidence hash differs')
        fields = arrays(path)
        np.testing.assert_array_equal(fields['seed_positions'], seed)
        np.testing.assert_array_equal(fields['seed_increment'], physical['increment'])
        before, after = fields['virtual_anchor'], fields['virtual_trial']
    elif endpoint == 'diagnostic':
        folder = directory.parent/(directory.name+'_diagnostic')
        meta = json.loads((folder/'diagnostic.json').read_text())
        if meta['result_sha256'] != inputs['result.json'] or meta['failed_sha256'] != digest(directory/'failed_trial.npz'):
            raise ValueError('Independent diagnostic does not describe this physical seed')
        record = next(c for c in meta['candidates'] if c['seam_indices'] == [48])
        fields = arrays(folder/'context.npz')
        np.testing.assert_array_equal(fields['seed_positions'], seed)
        np.testing.assert_array_equal(fields['seed_increment'], physical['increment'])
        path = folder/record['evidence_file']
        if digest(path) != record['evidence_sha256']:
            raise ValueError('Independent endpoint hash differs')
        result = arrays(path)
        before, after = fields['before_positions'], result['positions']
        fields['removed'] = result['removed']
    else:
        raise ValueError('Choose production or diagnostic endpoints')
    np.testing.assert_array_equal(fields['previous_released'], p.released)
    np.testing.assert_allclose(fields['frozen_metric'], p.material_response(seed)['metric'], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(fields['frozen_history'], p.cohesive_response(seed)['history'], rtol=0, atol=1e-12)
    base = FrozenVirtualCrack(p, seed, fields['frozen_metric'], fields['frozen_history'],
        fields['previous_released'], np.zeros_like(p.released), p.ffm_epsilon)
    _, pre_residual = stored_equilibrium(base, before)
    post = FrozenVirtualCrack(p, before, fields['frozen_metric'], fields['frozen_history'],
        fields['previous_released'], fields['removed'], p.ffm_epsilon)
    for value in (before, after):
        np.testing.assert_array_equal(value[p.n:], seed[p.n:])
        np.testing.assert_array_equal(value[p.mesh.bottom_nodes], p.rest[p.mesh.bottom_nodes])
    info = dict(provenance=provenance, inputs_sha256=inputs, endpoint_origin=endpoint,
        physical_seed=real, before_residual_n=pre_residual, endpoint=virtual_state(post, after),
        loading_started=False, physical_history_committed=False)
    return p, base, post, before, after, info
