"""Verified current-matrix solves with bounded reuse of an older LU factor.

An instance belongs to one loading increment. The caller may change the tangent
and its stabilization shift between calls; an old factor is ONLY a GMRES
preconditioner. Every returned direction passes the current matrix's residual
test, independently of the iterative solver's success flag.
"""
import time

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, gmres, splu


class _ReuseAbandoned(RuntimeError):
    pass


class ReusedLUSolver:
    """Reuse within an increment, with a fresh COLAMD factorization fallback."""

    def __init__(self):
        self._lu = None
        self._shape = None
        self._factor_seconds = None

    def solve(self, matrix, rhs):
        """Return ``(solution, statistics)`` for THIS ``matrix @ x = rhs``.

        Timings separate factorization from triangular/iterative solves. A
        rejected reuse attempt contributes to ``solve_s``; ``total_s`` also
        includes input preparation and explicit residual checks.
        """
        started = time.perf_counter()
        A = csc_matrix(matrix, dtype=float)
        b = np.asarray(rhs, dtype=float)
        if (A.ndim != 2 or A.shape[0] != A.shape[1] or b.ndim != 1
                or b.shape[0] != A.shape[0]):
            raise ValueError('Linear solve requires a square matrix and matching vector')
        if not np.all(np.isfinite(A.data)) or not np.all(np.isfinite(b)):
            raise ValueError('Linear system contains nonfinite values')
        norm_b = float(np.linalg.norm(b))
        if not np.isfinite(norm_b):
            raise ValueError('Linear right-hand-side norm exceeds floating-point range')
        limit = max(1e-10*norm_b, 1e-12)
        stats = dict(method=None, iterations=0, true_residual=None,
                     residual_limit=limit, factor_s=0., solve_s=0.,
                     fallback_reason=None, refinement_steps=0)

        def finish(x, method, residual):
            stats.update(method=method, true_residual=residual,
                         total_s=time.perf_counter()-started)
            return x, stats

        def residual(x):
            if x.shape != b.shape or not np.all(np.isfinite(x)):
                return float('inf')
            return float(np.linalg.norm(b-A@x))

        if not np.any(b):
            return finish(np.zeros_like(b), 'zero_rhs', 0.)

        if self._lu is not None and self._shape == A.shape:
            reuse_start = time.perf_counter()
            budget = max(.05, .5*self._factor_seconds)
            stats['reuse_budget_s'] = budget

            def callback(preconditioned_norm):
                stats['iterations'] += 1
                if not np.isfinite(preconditioned_norm):
                    raise _ReuseAbandoned('gmres_nonfinite')
                if time.perf_counter()-reuse_start > budget:
                    raise _ReuseAbandoned('gmres_time_budget')

            inverse = LinearOperator(A.shape, matvec=self._lu.solve, dtype=float)
            try:
                x, info = gmres(A, b, M=inverse, rtol=1e-11, atol=1e-13,
                                restart=16, maxiter=2, callback=callback,
                                callback_type='pr_norm')
                x = np.asarray(x, dtype=float)
                error = residual(x)
                if time.perf_counter()-reuse_start > budget:
                    stats['fallback_reason'] = 'gmres_time_budget'
                elif info != 0:
                    stats['fallback_reason'] = f'gmres_info_{info}'
                elif not np.isfinite(error) or error > limit:
                    stats['fallback_reason'] = 'gmres_true_residual'
                else:
                    stats['solve_s'] = time.perf_counter()-reuse_start
                    return finish(x, 'gmres_reused_lu', error)
            except _ReuseAbandoned as exc:
                stats['fallback_reason'] = str(exc)
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                stats['fallback_reason'] = f'gmres_exception_{type(exc).__name__}'
            stats['solve_s'] = time.perf_counter()-reuse_start
        elif self._lu is not None:
            stats['fallback_reason'] = 'matrix_shape_changed'

        factor_start = time.perf_counter()
        try:
            factor = splu(A, permc_spec='COLAMD')
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError('Current-matrix LU factorization failed') from exc
        stats['factor_s'] = time.perf_counter()-factor_start
        solve_start = time.perf_counter()
        x = np.asarray(factor.solve(b), dtype=float)
        error = residual(x)
        for _ in range(2):
            if np.isfinite(error) and error <= limit:
                break
            if not np.all(np.isfinite(x)):
                break
            x += factor.solve(b-A@x)
            stats['refinement_steps'] += 1
            error = residual(x)
        stats['solve_s'] += time.perf_counter()-solve_start
        if not np.isfinite(error) or error > limit:
            raise RuntimeError(f'Current-matrix linear residual failed: '
                               f'residual={error:.9g}, limit={limit:.9g}')
        self._lu, self._shape = factor, A.shape
        self._factor_seconds = stats['factor_s']
        return finish(x, 'direct_lu', error)
