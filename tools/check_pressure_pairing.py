"""Bounded displacement/pressure benchmarks, never a cutting production launch.

Run from repository root: python -m tools.check_pressure_pairing --output ...
The Cook-type problem is a local diagnostic, not a reproduction of a paper.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import splu, spsolve
from threadpoolctl import threadpool_limits
from prototype.model_config import FEMMesh, FEMConfig
from prototype.ipc_p2 import InteriorP2
from prototype.ipc_mini import MiniElement
from prototype.pressure_pairing import P2P1Volume
from prototype.macro_element import refine_macro_mesh
from prototype.material import NodalVolume, bulk
from tests.test_ipc_macro import sparse_local
from run_support import compute_process_policy


def base_mesh(n, cook=False):
    uv = np.array([(i/n, j/n) for j in range(n+1) for i in range(n+1)])
    if cook:
        xy = np.column_stack((48*uv[:, 0], 44*uv[:, 0]+(44-28*uv[:, 0])*uv[:, 1]))
    else:
        bump = np.sin(np.pi*uv[:, 0])*np.sin(np.pi*uv[:, 1])
        xy = uv+bump[:, None]*[.06, -.04]
    tri = []
    for j in range(n):
        for i in range(n):
            a = j*(n+1)+i
            tri.extend([[a, a+1, a+n+2], [a, a+n+2, a+n+1]])
    return FEMMesh(xy, xy, np.array(tri), np.empty((0, 2), int), np.empty(0),
                   np.empty(0, int), np.empty(0, int), np.zeros(len(xy), int))


def evaluate(el, nv, u, tangent=True):
    r, k, dofs, E, mj = el.bulk(u, with_tangent=tangent)
    B = None
    if nv is not None:
        r, k, B, E = nv.correct(u, r, k, E)
    g = np.zeros(len(u)); np.add.at(g, dofs.ravel(), r.ravel())
    A = sparse_local(k, dofs, len(u)) if tangent else None
    return E, g, A, B, mj


def cook(n, mode, budget=180.):
    cfg = FEMConfig(width_mm=1., shear_modulus_n_mm2=.47, poisson_ratio=.49)
    el = (MiniElement if mode.startswith('mini') else InteriorP2)(base_mesh(n, True), cfg, 4)
    nv = None if mode in ('full', 'mini-full') else P2P1Volume(el, lumped=mode == 'lumped')
    xy = el.mesh.reference_xy; size = el.mesh.dof_count
    fixed = (2*np.flatnonzero(xy[:el.base_nodes, 0] == 0)[:, None]+np.arange(2)).ravel()
    free = np.setdiff1d(np.arange(size), fixed)
    right = np.flatnonzero(xy[:, 0] == 48); right = right[np.argsort(xy[right, 1])]
    weights = np.ones(len(right)); weights[[0, -1]] = .5; weights *= 16/weights.sum()
    force = np.zeros(size); force[2*right+1] = .025*weights
    tip = np.argmin(np.linalg.norm(xy-[48, 60], axis=1))
    u = np.zeros(size); start = time.perf_counter(); iterations = 0
    for load in np.linspace(.1, 1., 10):
        for iteration in range(40):
            E, g, A, B, mj = evaluate(el, nv, u)
            residual = g-load*force
            if np.linalg.norm(residual[free]) <= 1e-8:
                break
            du = np.zeros(size)
            du[free] = (spsolve(A[free][:, free], -residual[free]) if nv is None else
                        nv.solve(A, B, free, -residual[free]))
            slope = residual@du
            if slope >= 0:
                raise RuntimeError('Non-descent Newton direction')
            a = el.safe_step(u, du)
            for cut in range(30):
                Et, gt, _, _, jt = evaluate(el, nv, u+a*du, False)
                old = E-load*force@u; new = Et-load*force@(u+a*du)
                eps = 64*np.finfo(float).eps*max(1., abs(E), abs(Et))
                if new <= old+1e-4*a*slope or (new <= old+eps and
                        np.linalg.norm((gt-load*force)[free]) < .5*np.linalg.norm(residual[free])):
                    break
                a *= .5
            else:
                raise RuntimeError('Cook line search failed')
            u += a*du; iterations += 1
            if time.perf_counter()-start > budget:
                raise TimeoutError('Cook bounded solve budget')
        else:
            raise RuntimeError('Cook nonlinear iteration budget')
    # Fresh assembly, not the success flag or the previous line-search residual.
    E, g, A, B, mj = evaluate(el, nv, u)
    residual = float(np.linalg.norm((g-force)[free]))
    if residual > 1e-8 or mj <= 0:
        raise RuntimeError('Cook fresh physical acceptance failed')
    return dict(n=n, mode=mode, nodes=len(xy), tip_uy_mm=float(u[2*tip+1]),
        min_j=mj, force_residual_n=residual, iterations=iterations,
        elapsed_s=time.perf_counter()-start)


def spectrum(n, kind, F):
    """Finite mesh inf-sup diagnostic with full Dirichlet perturbations.

    H1 seminorm for displacement; consistent L2 norm for pressure; one constant
    pressure null mode expected at homogeneous F. Not a uniform stability proof.
    """
    cfg = FEMConfig(width_mm=1., shear_modulus_n_mm2=.47, poisson_ratio=.49)
    base = base_mesh(n)
    if kind in ('p2p1', 'mini'):
        el = (MiniElement if kind == 'mini' else InteriorP2)(base, cfg, 4); mesh = el.mesh
        nv = P2P1Volume(el)
        u = (mesh.reference_xy@(F-np.eye(2)).T).ravel()
        _, _, _, B, _ = evaluate(el, nv, u)
        scalar = np.einsum('e,q,eqaJ,eqbJ->eab', el.area, el.weights, el.grad, el.grad)
        nd = el.dofs.shape[1]
        k = np.einsum('eab,ik->eaibk', scalar, np.eye(2)).reshape(-1, nd, nd)
        A = sparse_local(k, el.dofs, mesh.dof_count)
        mass = nv.mass.toarray()
    else:
        mesh = refine_macro_mesh(base, np.empty((0, 2), int))[0]
        nv = NodalVolume(mesh, cfg)
        u = (mesh.reference_xy@(F-np.eye(2)).T).ravel()
        r, k, d, E, _ = bulk(mesh, u, cfg)
        nv.correct(u, r, k, E, condensed=False); B = nv.pressure_coupling
        area, grad, dofs = mesh.bulk_data
        scalar = np.einsum('e,eaJ,ebJ->eab', area, grad, grad)
        k = np.einsum('eab,ik->eaibk', scalar, np.eye(2)).reshape(-1, 6, 6)
        A = sparse_local(k, dofs, mesh.dof_count)
        # Same consistent pressure norm, independent of the lumped energy.
        mass = P2P1Volume(InteriorP2(base, cfg)).mass.toarray()
    xy = mesh.reference_xy
    boundary = np.any(np.isclose(xy, 0.) | np.isclose(xy, 1.), axis=1)
    if kind == 'mini':
        boundary[el.base_nodes:] = False
    free = (2*np.flatnonzero(~boundary)[:, None]+np.arange(2)).ravel()
    b = B[:, free].toarray()
    S = b@splu(A[free][:, free].tocsc()).solve(b.T)
    ev = eigvalsh((S+S.T)/2, mass)
    threshold = 1e-10*max(1., ev[-1])
    return dict(n=n, kind=kind, F=F.tolist(), nullity=int(np.sum(ev <= threshold)),
        beta=float(np.sqrt(max(ev[1], 0.))), eigenvalues=ev.tolist())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path('prototype/pressure_pairing.py'), Path('prototype/ipc_p2.py'),
             Path('prototype/material.py'), Path('mechanics/finite_strain.py'),
             Path('prototype/model_config.py'), Path('prototype/macro_element.py'),
             Path('tests/test_ipc_macro.py'), Path('run_support.py')]
    manifest = {}
    for f in files:
        dst = out/'source'/f.resolve().relative_to(Path.cwd().resolve())
        dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(f.read_bytes())
        manifest[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    report = dict(scope='bounded pairing and Cook-type tests; no cutting production run',
                  sources=manifest, spectra=[], cook=[], status='running')
    def save():
        (out/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    with compute_process_policy('compute') as policy, threadpool_limits(2):
        for n in (3, 6):
            for kind in ('macro', 'p2p1'):
                for F in (np.eye(2), np.array([[1.25, .5], [.08, .7]])):
                    row = spectrum(n, kind, F); report['spectra'].append(row)
                    print('SPECTRUM', n, kind, row['nullity'], row['beta'], flush=True); save()
        for n in (8, 16, 32):
            for mode in ('full', 'consistent', 'lumped'):
                try:
                    row = cook(n, mode)
                except Exception as exc:
                    row = dict(n=n, mode=mode, status='failed', error=repr(exc))
                report['cook'].append(row); print(json.dumps(row), flush=True); save()
    report['runtime_policy'] = policy
    report['status'] = 'bounded_checks_ended'; save()


if __name__ == '__main__':
    main()
