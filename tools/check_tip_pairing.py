"""Conditional local material relaxation under a saved P1 boundary displacement.

All exterior patch nodes, including the material surface, are prescribed. No
contact or friction is re-solved, and no global Fz is inferred. P2 enrichment
preserves precisely the same P1 boundary. A larger patch checks truncation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import bmat, diags
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits
from prototype.model_config import FEMMesh, FEMConfig
from prototype.macro_element import refine_macro_mesh
from prototype.material import bulk, NodalVolume
from prototype.ipc_p2 import InteriorP2
from prototype.quadratic import quadratic_minimum
from prototype.pressure_pairing import P2P1Volume
from tests.test_ipc_macro import sparse_local
from run_support import compute_process_policy


def solve(source, config, radius, mode, budget=90.):
    with np.load(source, allow_pickle=False) as z:
        n = int(z['body_nodes']); ref = z['reference_xy'][:n]; old = z['positions'][:n]-ref
        tri = z['triangles']; parent = z['macro_triangles']
        x = ref[tri]; dm = np.stack((x[:, 1]-x[:, 0], x[:, 2]-x[:, 0]), axis=2)
        y = (ref+old)[tri]; ds = np.stack((y[:, 1]-y[:, 0], y[:, 2]-y[:, 0]), axis=2)
        j = np.linalg.det(ds@np.linalg.inv(dm)); center = x[j.argmin()].mean(axis=0)
        chosen = np.linalg.norm(ref[parent].mean(axis=1)-center, axis=1) <= radius
        ids = np.unique(parent[chosen]); mapping = np.full(n, -1, int); mapping[ids] = np.arange(len(ids))
        xy = ref[ids]
        base = FEMMesh(xy, xy, mapping[parent[chosen]], np.empty((0, 2), int), np.empty(0),
                       np.empty(0, int), np.empty(0, int), np.zeros(len(ids), int))
        fine, _ = refine_macro_mesh(base, np.empty((0, 2), int))
        dist, index = cKDTree(ref).query(fine.reference_xy)
        if dist.max() > 1e-10:
            raise ValueError('Patch not exactly nested in saved material mesh')
        seed = old[index]
    cfg = FEMConfig(**config)
    el = InteriorP2(fine, cfg, 4) if mode.startswith('p2') else None
    mesh = fine if el is None else el.mesh
    u = seed.ravel().copy() if el is None else np.vstack((seed, seed[el.interior_edges].mean(axis=1))).ravel()
    nv = None
    if mode in ('macro', 'macro-a4'):
        nv = NodalVolume(fine, cfg, stabilization_shear=4. if mode == 'macro-a4' else 0.)
    elif mode in ('p2-consistent', 'p2-lumped'):
        nv = P2P1Volume(el, lumped=mode == 'p2-lumped')
    edges = np.sort(fine.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = np.unique(edges[counts == 1]); fixed = (2*boundary[:, None]+np.arange(2)).ravel()
    free = np.setdiff1d(np.arange(len(u)), fixed); prescribed = u[fixed].copy()
    def ev(v, tangent=True):
        r, k, dofs, energy, mj = (bulk(mesh, v, cfg, with_tangent=tangent) if el is None else
                                 el.bulk(v, with_tangent=tangent))
        B = mass = None
        if isinstance(nv, P2P1Volume):
            r, k, B, energy = nv.correct(v, r, k, energy); mass = nv.pressure_mass
        elif nv is not None:
            r, k, _, energy = nv.correct(v, r, k, energy, condensed=False)
            if tangent:
                B = np.sqrt(nv.projected_fraction)*nv.pressure_coupling
                mass = diags(nv.pressure_mass)
        g = np.zeros(len(v)); np.add.at(g, dofs.ravel(), r.ravel())
        A = sparse_local(k, dofs, len(v)) if tangent else None
        return energy, g, A, B, mass, mj
    start = time.perf_counter()
    for iteration in range(80):
        E, g, A, B, mass, mj = ev(u)
        norm = np.linalg.norm(g[free])
        if norm <= 1e-6:
            break
        a0 = A[free][:, free]; b = None if B is None else B[:, free]
        for shift in (0., .001, .01, .1, 1., 10., 100., 1000.):
            a = a0+shift*diags(np.maximum(abs(a0.diagonal()), 1.))
            mat = a if b is None else bmat([[a, b.T], [b, -mass]], format='csc')
            rhs = -g[free] if b is None else np.r_[-g[free], np.zeros(b.shape[0])]
            answer = spsolve(mat, rhs)
            if not np.all(np.isfinite(answer)):
                continue
            if np.linalg.norm(mat@answer-rhs) > 1e-7*max(1., np.linalg.norm(rhs)):
                continue
            du = np.zeros(len(u)); du[free] = answer[:len(free)]
            slope = g@du
            if slope >= 0:
                continue
            alpha = 1. if el is None else el.safe_step(u, du)
            for cut in range(28):
                try:
                    Et, gt, _, _, _, jt = ev(u+alpha*du, False)
                except (ValueError, FloatingPointError):
                    alpha *= .5; continue
                roundoff = 64*np.finfo(float).eps*max(1., abs(E), abs(Et))
                if jt > 0 and (Et <= E+1e-4*alpha*slope or (Et <= E+roundoff and
                    np.linalg.norm(gt[free]) < .5*norm)):
                    break
                alpha *= .5
            else:
                continue
            break
        else:
            raise RuntimeError('No acceptable local descent step')
        u += alpha*du
        if time.perf_counter()-start > budget:
            raise TimeoutError('Local patch 90s budget')
    else:
        raise RuntimeError('Local patch nonlinear iteration budget')
    E, g, A, B, mass, mj = ev(u)
    np.testing.assert_array_equal(u[fixed], prescribed)
    residual = float(np.linalg.norm(g[free]))
    if residual > 1e-6 or mj <= 0:
        raise RuntimeError('Local patch fresh acceptance failed')
    if el is None:
        area, grad, _ = mesh.bulk_data
        J = np.linalg.det(np.eye(2)+np.einsum('eai,eaJ->eiJ', u.reshape(-1, 2)[mesh.triangles], grad))
        centroid = mesh.reference_xy[mesh.triangles].mean(axis=1)
        central = J[np.linalg.norm(centroid-center, axis=1) <= .03]
        rms = np.sqrt(np.average((J-1)**2, weights=area))
    else:
        J = np.linalg.det(el.deformation(u))
        exact = quadratic_minimum(np.linalg.det(el.deformation(u, el.vertex_grad)))
        centroid = mesh.reference_xy[el.elements[:, :3]].mean(axis=1)
        central = exact[np.linalg.norm(centroid-center, axis=1) <= .03]
        rms = np.sqrt(np.sum(el.area[:, None]*el.weights*(J-1)**2)/el.area.sum())
    row = dict(radius_mm=radius, mode=mode, nodes=mesh.node_count, prescribed_nodes=len(boundary),
        parent_triangles=int(chosen.sum()), min_j=mj, central_min_j=float(central.min()),
        volume_rms=float(rms), energy_n_mm=E, residual_n=residual, iterations=iteration,
        elapsed_s=time.perf_counter()-start, max_interior_change_mm=float(np.max(np.linalg.norm(
            u[:seed.size].reshape(-1, 2)-seed, axis=1))))
    return row, dict(reference_xy=mesh.reference_xy, displacement=u,
        triangles=fine.triangles, base_nodes=fine.node_count, pressure_nodes=(
        nv.last['pressure_nodes'] if isinstance(nv, P2P1Volume) else np.empty(0)))


def main():
    ap = argparse.ArgumentParser(__doc__); ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    report = json.loads((args.run/'result.json').read_text(encoding='utf-8'))
    index = next(i for i, s in enumerate(report['steps']) if abs(s['depth_mm']-.12) < 1e-10)
    source = args.run/'states'/f'{index:04d}.npz'
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    data = dict(scope='all-boundary prescribed material patch; not contact equilibrium or Fz',
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        config=report['config'], cases=[], status='running', source_hashes={})
    for f in [Path(__file__), *Path('prototype').glob('*.py'), *Path('mechanics').glob('*.py'), Path('run_support.py')]:
        relative = f.resolve().relative_to(Path.cwd().resolve()); dst = out/'source'/relative
        dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(f.read_bytes())
        data['source_hashes'][relative.as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    def save():
        (out/'results.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    with compute_process_policy('compute') as policy, threadpool_limits(2):
        for radius in (.06, .12):
            for mode in ('p1-full', 'macro', 'macro-a4', 'p2-full', 'p2-consistent', 'p2-lumped'):
                try:
                    row, field = solve(source, report['config'], radius, mode)
                    np.savez_compressed(out/f'{radius:g}_{mode}.npz', **field)
                except Exception as exc:
                    row = dict(radius_mm=radius, mode=mode, status='failed', error=repr(exc))
                data['cases'].append(row); print(json.dumps(row), flush=True); save()
    data.update(runtime_policy=policy, status='bounded_checks_ended'); save()


if __name__ == '__main__':
    main()
