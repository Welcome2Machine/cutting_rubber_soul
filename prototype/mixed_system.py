"""Sparse mixed Newton systems with exact element-bubble elimination.

The pressure is an auxiliary linear unknown; the nonlinear host solves its
consistent mass equation exactly at each residual evaluation. No dense global
pressure Schur complement is formed. Nonsymmetric friction blocks are allowed.
"""
import numpy as np
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import splu


def solve_mixed(local, metric, B, mass, mass_solve, shift, rhs, bubbles=None, linear_solver=None,
                *, _refine=True):
    if not np.isfinite(shift) or shift < 0:
        raise ValueError('Invalid mixed PTC shift')
    L = (local+shift*metric).tocsc()
    system = bmat([[L, B.T], [B, -mass/(1+shift)]], format='csc')
    system.eliminate_zeros()
    full_rhs = np.r_[rhs, np.zeros(B.shape[0])]
    fallback = None
    condensed = system
    linear_info = None

    def solve(matrix, vector):
        nonlocal linear_info
        if linear_solver is not None:
            value, linear_info = linear_solver.solve(matrix, vector)
            return value
        factor = splu(matrix)
        linear_info = dict(method='direct_lu', factor_nnz=factor.L.nnz+factor.U.nnz)
        return factor.solve(vector)
    if bubbles is not None and len(bubbles):
        bubbles = np.asarray(bubbles, dtype=int).reshape(-1, 2)
        internal = bubbles.ravel()
        if len(np.unique(internal)) != len(internal) or np.any(internal < 0) or np.any(internal >= len(rhs)):
            raise ValueError('Invalid bubble partition')
        kept = np.setdiff1d(np.arange(system.shape[0]), internal)
        D = system[internal][:, internal].tocoo()
        if np.any(D.row//2 != D.col//2):
            raise ValueError('Bubble block is not element-local')
        blocks = np.zeros((len(bubbles), 2, 2))
        blocks[D.row//2, D.row%2, D.col%2] = D.data
        # An indefinite block is allowed. A numerically singular block requires
        # the full saddle solve, never a silent pseudoinverse or discarded mode.
        if np.any(~np.isfinite(blocks)):
            raise FloatingPointError('Nonfinite bubble block')
        if np.max(np.linalg.cond(blocks)) > 1e12:
            fallback = 'ill_conditioned_local_bubble_block'
        else:
            inv = np.linalg.inv(blocks)
            ids = np.arange(len(internal)).reshape(-1, 2)
            Di = coo_matrix((inv.ravel(), (
                np.broadcast_to(ids[:, :, None], inv.shape).ravel(),
                np.broadcast_to(ids[:, None, :], inv.shape).ravel())), shape=D.shape).tocsc()
            C = system[internal][:, kept].tocsc()
            R = system[kept][:, internal].tocsc()
            DC = Di@C
            dr = Di@full_rhs[internal]
            condensed = (system[kept][:, kept]-R@DC).tocsc()
            condensed.eliminate_zeros()
            solution = np.empty(len(full_rhs))
            solution[kept] = solve(condensed, full_rhs[kept]-R@dr)
            solution[internal] = dr-DC@solution[kept]
    if bubbles is None or not len(bubbles) or fallback:
        solution = solve(system, full_rhs)
    du = solution[:len(rhs)]
    residual = L@du+(1+shift)*(B.T@mass_solve(B@du))-rhs
    error = float(np.linalg.norm(residual))
    limit = 1e-8*max(1., float(np.linalg.norm(rhs)))
    refinements = []
    if _refine and np.all(np.isfinite(du)) and np.isfinite(error):
        # A small mixed residual is NOT a small displacement-Schur residual:
        # the pressure mass scales with element volume and can amplify its
        # backward error. Correct the ACTUAL condensed physical residual using
        # the same sparse system, without relaxing its acceptance threshold.
        for _ in range(2):
            if error <= limit:
                break
            correction, info = solve_mixed(local, metric, B, mass, mass_solve,
                shift, -residual, bubbles, linear_solver=None, _refine=False)
            du += correction
            residual = L@du+(1+shift)*(B.T@mass_solve(B@du))-rhs
            updated = float(np.linalg.norm(residual))
            refinements.append(dict(before_n=error,after_n=updated,linear=info))
            error = updated
    if not np.all(np.isfinite(solution)) or not np.isfinite(error) or error > limit:
        raise np.linalg.LinAlgError(f'Mixed linear residual {error:g} exceeds {limit:g}')
    return du, dict(method='mixed_bubble_condensation' if bubbles is not None else 'mixed_sparse',
        fallback=fallback, full_unknowns=system.shape[0], factored_unknowns=condensed.shape[0],
        matrix_nnz=condensed.nnz, linear=linear_info,
        physical_residual_refinements=refinements,
        physical_linear_residual_n=error, residual_limit_n=limit)
