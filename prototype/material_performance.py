"""Opt-in execution controls for the native intact Maxwell model.

The material law, residual, tangent, loading policy and physical acceptance are
inherited unchanged. Performance policy is separate from the physical signature.
"""
import numpy as np


def performance_policy(*, material_batch_size=0, linear_backend='native', linear_threads=4,
                       ipc_threads=2):
    if type(material_batch_size) is not int or material_batch_size < 0:
        raise ValueError('Material batch size must be a nonnegative integer')
    if linear_backend not in ('native', 'pardiso'):
        raise ValueError('Linear backend must be native or pardiso')
    if type(linear_threads) is not int or linear_threads < 1:
        raise ValueError('Linear threads must be a positive integer')
    if type(ipc_threads) is not int or ipc_threads < 1:
        raise ValueError('IPC threads must be a positive integer')
    accelerated = material_batch_size > 0 or linear_backend != 'native'
    return dict(version=1, material_batch_size=material_batch_size,
                linear_backend=linear_backend,
                linear_threads=linear_threads if linear_backend == 'pardiso' else None,
                ipc_threads=ipc_threads if accelerated else None,
                blas_threads=1 if accelerated else None)


def require_same_performance(old, current):
    """Never silently resume a history with different execution controls."""
    if old.get('performance_policy', performance_policy()) != current:
        raise ValueError('Resume performance policy differs; start an independent path')


class MaterialPerformance:
    def linear_direction(self, matrix, rhs):
        value = self._performance_backend.solve(matrix, rhs)
        # PARDISO has a guarded SuperLU fallback. Check that fallback too, using
        # the ORIGINAL matrix, just as in the full-increment qualification.
        error = float(np.linalg.norm(matrix @ value - rhs))
        limit = max(1e-8 * np.linalg.norm(rhs), 1e-10)
        if not np.isfinite(value).all() or not np.isfinite(error) or error > limit:
            raise RuntimeError('Original-matrix linear residual failed')
        return value


def enable_performance(problem, policy):
    """Enable only after native resume audit; do not replace other mixins."""
    expected = performance_policy(material_batch_size=policy['material_batch_size'],
        linear_backend=policy['linear_backend'], linear_threads=policy['linear_threads'] or 4,
        ipc_threads=policy['ipc_threads'] or 2)
    if policy != expected:
        raise ValueError('Unsupported performance policy')
    if getattr(problem, '_performance_backend', None) is not None:
        raise ValueError('Performance backend already enabled')
    if policy['linear_backend'] == 'pardiso':
        from .pardiso import PardisoDirection
        backend = PardisoDirection(policy['linear_threads'])
        problem.__class__ = type('AcceleratedMaterialProblem',
                                (MaterialPerformance, type(problem)), {})
        problem._performance_backend = backend
    problem.material_batch_size = policy['material_batch_size']
    if policy['ipc_threads'] is not None:
        problem.ipc.set_num_threads(policy['ipc_threads'])


def performance_summary(problem):
    backend = getattr(problem, '_performance_backend', None)
    rows = backend.rows if backend is not None else []
    return dict(linear_calls=len(rows), linear_seconds=sum(r['seconds'] for r in rows),
                fallback_count=sum(r['fallback_reason'] is not None for r in rows),
                max_linear_residual_n=max((r['residual_n'] for r in rows), default=None))


def close_performance(problem):
    backend = getattr(problem, '_performance_backend', None)
    if backend is not None:
        backend.close()
        problem._performance_backend = None


def native_audit(make_native, out, report):
    """Reassemble the whole saved path on a separate unbatched native object."""
    import json
    from .material_case import signature, audit
    problem = make_native()
    if (getattr(problem, 'material_batch_size', 0) != 0
            or getattr(problem, '_performance_backend', None) is not None):
        raise ValueError('Independent audit requires an unbatched native problem')
    if json.loads(json.dumps(signature(problem))) != report['signature']:
        raise ValueError('Independent audit physical signature differs')
    result = audit(problem, out, report)
    report['independent_audit_policy'] = dict(material_batch_size=0,
        linear_backend='native', fresh_problem=True, states=result['states'],
        scope='Whole accepted path; exact dual histories, physical dt and dissipation chain')
    return result
