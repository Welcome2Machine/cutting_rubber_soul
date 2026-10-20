"""Isolated common-law material and nonuniform FE coupling verification.

Not a cutting simulation. Does not import or alter IPC, production defaults or
accepted cutting histories. See docs/history/topics/cross_mode_verification.md for gates.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from mechanics.finite_strain import Material, response, elastic_response
from prototype.model_config import FEMMesh, FEMConfig
from prototype.ipc_p2 import InteriorP2

K0 = 23.34333333333331
G_INF, G_V, TAU = .14, 1.24, 1.7
KINEMATICS = ('plane_strain', 'plane_stress')


def material(k=K0):
    nu = (3*k-2*G_INF)/(2*(3*k+G_INF))
    return Material((G_INF,), (2.,), (G_V,), (2.,), poisson=nu,
                    relaxation_s=TAU, viscous_flow='isochoric_maxwell')


def invariant_checks(q, kinematics):
    metric = q['metric']
    errors = dict(metric_det_error=float(np.max(np.abs(np.linalg.det(metric)-1))),
        metric_min_eigenvalue=float(np.linalg.eigvalsh(metric).min()),
        min_j=float(q['J'].min()), min_dissipation=float(q['viscous_rate'].min()),
        ps_scaled_residual=float(np.max(q['plane_stress_scaled_residual'])))
    assert errors['metric_det_error'] < 1e-8
    assert errors['metric_min_eigenvalue'] > 0 and errors['min_j'] > 0
    assert errors['min_dissipation'] >= -1e-10
    if kinematics == 'plane_stress':
        assert errors['ps_scaled_residual'] <= 1e-8
    return errors


def material_point(mode, amount, A, dt, kin, lateral=1.):
    F = np.eye(2)[None]
    if mode == 'shear':
        F[0, 0, 1] = .6*amount
    else:
        F[0, 1, 1] = 1 + (.3 if mode == 'tension' else -.3)*amount
    free_lateral = mode in ('tension', 'compression')
    if free_lateral:
        F[0, 0, 0] = lateral
    for _ in range(25):
        q = response(F, A, dt, material(), kinematics=kin, tangent=free_lateral)
        if not free_lateral or abs(q['P'][0, 0, 0]) < 1e-10:
            return F, q
        change = -q['P'][0, 0, 0]/q['tangent'][0, 0, 0]
        F[0, 0, 0] += np.clip(change, -.15*F[0, 0, 0], .15*F[0, 0, 0])
    raise RuntimeError('Uniaxial lateral zero-traction solve failed')


def point_path(mode, kin, ramp_s, dt):
    A = np.eye(3)[None]; lateral = 1.; rows = []; worst = 0.
    ramp_n = round(ramp_s/dt); hold_n = round(5*TAU/dt)
    for i in range(1, ramp_n+hold_n+1):
        F, q = material_point(mode, min(i/ramp_n, 1), A, dt, kin, lateral)
        inv = invariant_checks(q, kin); worst = max(worst, inv['metric_det_error'])
        lateral = F[0, 0, 0]; A = q['metric'].copy()
        stress = q['P'][0, 0, 1] if mode == 'shear' else q['P'][0, 1, 1]
        rows.append(dict(time_s=i*dt, nominal_stress_mpa=float(stress),
            J=float(q['J'][0]), lateral_stretch=float(lateral),
            out_stretch=float(q['lambda_out'][0]), energy=float(q['energy'][0]),
            dissipation_rate=float(q['viscous_rate'][0])))
    held_energy = np.array([x['energy'] for x in rows[ramp_n-1:]])
    assert np.max(np.diff(held_energy)) < 1e-9
    return dict(mode=mode, kinematics=kin, ramp_s=ramp_s, dt_s=dt,
        end_ramp=rows[ramp_n-1], end_hold=rows[-1], rows=rows,
        metric_det_error=worst, hold_energy_monotone=True)


def analytic_checks():
    m = material(); dt = .05; eps = 1e-6
    F = np.array([[[1., eps], [0., 1.]]]); A = np.eye(3)[None]
    errors = []
    for i in range(1, 41):
        q = response(F, A, dt, m, kinematics='plane_strain', tangent=False)
        expect = eps*(G_INF+G_V/(1+dt/TAU)**i)
        errors.append(abs(q['P'][0, 0, 1]/expect-1)); A = q['metric']
    # Independent isotropic small-strain matrix with backward-Euler G_alg.
    alg = G_INF+G_V/(1+dt/TAU); lame = K0-2*alg/3
    probe = np.array([[.2, .3], [-.1, -.15]])*eps
    strain = (probe+probe.T)/2
    expected = 2*alg*strain+lame*np.trace(strain)*np.eye(2)
    q = response((np.eye(2)+probe)[None], np.eye(3)[None], dt, m,
                 kinematics='plane_strain')
    pe_error = float(np.linalg.norm(q['P'][0]-expected)/np.linalg.norm(expected))
    lame_ps = 2*alg*lame/(lame+2*alg)
    expected_ps = 2*alg*strain+lame_ps*np.trace(strain)*np.eye(2)
    qs = response((np.eye(2)+probe)[None], np.eye(3)[None], dt, m,
                  kinematics='plane_stress')
    ps_error = float(np.linalg.norm(qs['P'][0]-expected_ps)/np.linalg.norm(expected_ps))
    assert max(errors+[pe_error, ps_error]) < 1e-4
    return dict(maxwell_relative_error=max(errors), pe_linear_error=pe_error,
        ps_linear_error=ps_error, G0=G_INF+G_V, Ginf=G_INF, K=K0,
        nu0=(3*K0-2*(G_INF+G_V))/(2*(3*K0+G_INF+G_V)),
        nuinf=m.poisson, bulk_relaxation=False)


def rectangular_mesh(n):
    xy = np.array([(2*i/n, 2*j/n) for j in range(n+1) for i in range(2*n+1)])
    triangles = []
    for j in range(n):
        for i in range(2*n):
            a = j*(2*n+1)+i
            # Alternating diagonals avoid a single global diagonal preference.
            if (i+j) % 2:
                triangles.extend([[a, a+1, a+2*n+1], [a+1, a+2*n+2, a+2*n+1]])
            else:
                triangles.extend([[a, a+1, a+2*n+2], [a, a+2*n+2, a+2*n+1]])
    return FEMMesh(xy, xy.copy(), np.array(triangles), np.empty((0, 2), int),
        np.empty(0), np.empty(0, int), np.empty(0, int), np.zeros(len(xy), int))


class Block:
    def __init__(self, n, kind, kin, kfactor=1., order=3):
        base = rectangular_mesh(n)
        self.base, self.kind, self.kin = base, kind, kin
        self.material = material(K0*kfactor)
        self.el = None
        if kind == 'p2':
            self.el = InteriorP2(base, FEMConfig(width_mm=20.), order)
            self.mesh = self.el.mesh; self.elements = self.el.elements
            self.grad = self.el.grad; self.weights = self.el.weights
            self.area = self.el.area
        elif kind == 'p1':
            self.mesh = base; self.elements = base.triangles
            self.area, g, _ = base.bulk_data
            self.grad = g[:, None]; self.weights = np.ones(1)
        else:
            raise ValueError(kind)
        self.dofs = (2*self.elements[:, :, None]+np.arange(2)).reshape(len(self.elements), -1)
        xy = self.mesh.reference_xy
        self.bottom = np.flatnonzero(np.isclose(xy[:, 1], 0))
        self.patch = np.flatnonzero(np.isclose(xy[:, 1], 2)&(xy[:, 0]>=1.5)&(xy[:, 0]<=2.5))
        self.fixed = (2*np.r_[self.bottom, self.patch,][:, None]+np.arange(2)).ravel()
        self.free = np.setdiff1d(np.arange(self.mesh.dof_count), self.fixed)
        self.nq = len(self.weights); self.vol = self.area[:, None]*20*self.weights[None]

    def F(self, u):
        return np.eye(2)+np.einsum('eai,eqaJ->eqiJ', u.reshape(-1, 2)[self.elements], self.grad)

    def min_j(self, u):
        return self.el.minimum_j(u) if self.el else float(np.linalg.det(self.F(u)).min())

    def safe_step(self, u, direction):
        if self.el:
            return self.el.safe_step(u, direction)
        F = self.F(u); D = self.F(direction)-np.eye(2)
        c = np.linalg.det(F); a = np.linalg.det(D)
        b = F[..., 0, 0]*D[..., 1, 1]+D[..., 0, 0]*F[..., 1, 1]-F[..., 0, 1]*D[..., 1, 0]-D[..., 0, 1]*F[..., 1, 0]
        alpha = 1.
        for _ in range(60):
            critical = np.zeros_like(a)
            np.divide(-b, 2*a, out=critical, where=a>0)
            critical = np.clip(critical, 0, alpha)
            if min(c.min(), (c+b*alpha+a*alpha**2).min(),
                   (c+b*critical+a*critical**2).min()) > 0:
                return alpha
            alpha *= .5
        raise FloatingPointError('No positive P1 deformation path')

    def evaluate(self, u, A, dt, tangent=True):
        if self.min_j(u) <= 0:
            raise FloatingPointError('Nonpositive pointwise in-plane Jacobian')
        q = response(self.F(u).reshape(-1, 2, 2), A, dt, self.material,
                     kinematics=self.kin, tangent=tangent)
        nq = self.nq; ne = len(self.elements); nd = self.dofs.shape[1]
        local = np.einsum('eq,eqiJ,eqaJ->eai', self.vol,
            q['P'].reshape(ne, nq, 2, 2), self.grad, optimize=True).reshape(ne, nd)
        g = np.zeros(self.mesh.dof_count); np.add.at(g, self.dofs.ravel(), local.ravel())
        H = None
        if tangent:
            C = q['tangent'].reshape(ne, nq, 2, 2, 2, 2)
            z = np.einsum('eqiJkL,eqbL->eqiJbk', C, self.grad, optimize=True)
            h = np.einsum('eq,eqaJ,eqiJbk->eaibk', self.vol, self.grad, z,
                          optimize=True).reshape(ne, nd, nd)
            H = coo_matrix((h.ravel(), (np.broadcast_to(self.dofs[:, :, None], h.shape).ravel(),
                np.broadcast_to(self.dofs[:, None, :], h.shape).ravel())),
                shape=(len(u), len(u))).tocsc()
        energy = float(np.sum(self.vol.ravel()*q['incremental_potential']))
        return energy, g, H, q


def derivative_checks():
    rows = []
    for kind in ('p1', 'p2'):
        for kin in KINEMATICS:
            b = Block(2, kind, kin); xy = b.mesh.reference_xy
            u = np.column_stack((.02*xy[:, 0]*xy[:, 1], -.03*xy[:, 1]**2)).ravel()
            A = np.tile(np.eye(3), (len(b.elements)*b.nq, 1, 1))
            A = b.evaluate(.5*u, A, .13, False)[3]['metric']
            direction = np.random.default_rng(219).normal(size=len(u)); direction /= np.linalg.norm(direction)
            E, g, H, q = b.evaluate(u, A, .12)
            eps = 2e-6
            ep, gp, _, _ = b.evaluate(u+eps*direction, A, .12, False)
            em, gm, _, _ = b.evaluate(u-eps*direction, A, .12, False)
            force = float(np.linalg.norm((gp-gm)/(2*eps)-H@direction)/np.linalg.norm(H@direction))
            work = float(abs((ep-em)/(2*eps)-g@direction)/max(1., abs(g@direction)))
            resultant = float(np.linalg.norm(g.reshape(-1, 2).sum(axis=0)))
            assert max(force, work) < 2e-5 and resultant < 1e-9
            rows.append(dict(kind=kind, kinematics=kin, tangent_relative_error=force,
                potential_work_error=work, resultant_n=resultant, **invariant_checks(q, kin)))
    return rows


def block_path(out, n, kind, kin, kfactor, angle, step=.025, order=3, budget=120.):
    label = f'n{n}_{kind}_{kin}_k{kfactor:g}_a{angle}_d{step:g}_q{order}'
    b = Block(n, kind, kin, kfactor, order); start = time.perf_counter()
    u = np.zeros(b.mesh.dof_count); A = np.tile(np.eye(3), (len(b.elements)*b.nq, 1, 1))
    theta = np.deg2rad(angle); direction = np.array([np.sin(theta), -np.cos(theta)])
    rows = []; displacements = [u.copy()]; metrics = [A.copy()]
    report = dict(id=label, n=n, kind=kind, kinematics=kin, kfactor=kfactor,
        angle_deg=angle, step_mm=step, order=order, nodes=b.mesh.node_count,
        triangles=len(b.elements), quadrature_points=len(A), rows=rows, status='running')
    saved = out/(label+'.npz'); dt = step/.2
    try:
        for depth in np.linspace(step, .2, round(.2/step)):
            trial = u.copy(); trial.reshape(-1, 2)[b.patch] = depth*direction
            for iteration in range(30):
                if time.perf_counter()-start > budget:
                    raise TimeoutError('Independent block path budget')
                E, g, H, q = b.evaluate(trial, A, dt)
                residual = float(np.linalg.norm(g[b.free]))
                if residual <= 1e-7:
                    break
                du = np.zeros_like(u); du[b.free] = spsolve(H[b.free][:, b.free], -g[b.free])
                lin = np.linalg.norm((H@du+g)[b.free])
                if not np.all(np.isfinite(du)) or lin > max(1e-9, 1e-7*residual):
                    raise RuntimeError('Physical linear residual failed')
                slope = g@du
                if slope >= 0:
                    raise RuntimeError('Non-descent incremental-potential direction')
                alpha = b.safe_step(trial, du)
                for _ in range(24):
                    Et, gt, _, _ = b.evaluate(trial+alpha*du, A, dt, False)
                    eps = 64*np.finfo(float).eps*max(1., abs(E), abs(Et))
                    if Et <= E+1e-4*alpha*slope or (Et <= E+eps and
                            np.linalg.norm(gt[b.free]) < .5*residual):
                        break
                    alpha *= .5
                else:
                    raise RuntimeError('Block energy line search failed')
                trial += alpha*du
            else:
                raise RuntimeError('Block Newton iteration budget')
            # Fresh assembly from PREVIOUS accepted metric; commit only afterwards.
            E, g, _, q = b.evaluate(trial, A, dt, False)
            inv = invariant_checks(q, kin)
            fresh = float(np.linalg.norm(g[b.free])); assert fresh <= 1e-7
            nodal = g.reshape(-1, 2); patch_force = nodal[b.patch].sum(axis=0)
            balance = float(np.linalg.norm(patch_force+nodal[b.bottom].sum(axis=0)))
            assert balance <= 1e-6
            xy = b.mesh.reference_xy[:b.base.node_count]; top = np.flatnonzero(np.isclose(xy[:, 1], 2))
            probe_x = np.array([.5, 1., 3., 3.5])
            ux = trial.reshape(-1, 2)
            probes = np.column_stack([np.interp(probe_x, xy[top, 0], ux[top, k]) for k in range(2)])
            meanJ = float(np.sum(b.vol.ravel()*q['J'])/b.vol.sum())
            row = dict(depth_mm=float(depth), time_s=float(depth/.2), force_n=float(patch_force@direction),
                force_xy_n=patch_force.tolist(), residual_n=fresh, balance_n=balance,
                iterations=iteration, true_inplane_min_j=b.min_j(trial), mean_J=meanJ,
                out_stretch_min=float(q['lambda_out'].min()), out_stretch_max=float(q['lambda_out'].max()),
                surface_probe_displacement_mm=probes.tolist(), **inv)
            rows.append(row); u = trial; A = q['metric'].copy()
            displacements.append(u.copy()); metrics.append(A.copy())
        report['status'] = 'completed'
    except (RuntimeError, FloatingPointError, TimeoutError, AssertionError) as exc:
        report.update(status='failed', error=repr(exc))
    np.savez_compressed(saved, displacement=np.array(displacements), metric=np.array(metrics),
        reference_xy=b.mesh.reference_xy, triangles=b.base.triangles)
    report.update(elapsed_s=time.perf_counter()-start, checkpoint=str(saved),
        checkpoint_sha256=hashlib.sha256(saved.read_bytes()).hexdigest())
    return report


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('preflight', 'material', 'blocks'), required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = out/'sources.json'
    sources = [Path(__file__), Path('mechanics/finite_strain.py'), Path('prototype/model_config.py'),
               Path('prototype/ipc_p2.py'), Path('prototype/quadratic.py'), Path('prototype/material.py')]
    hashes = {str(f.resolve().relative_to(Path.cwd())): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    if manifest.exists():
        assert json.loads(manifest.read_text()) == hashes, 'Source changed between stages'
    else:
        save_json(manifest, hashes)
        for f in sources:
            target = out/'source'/f.resolve().relative_to(Path.cwd()); target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f.read_bytes())
    with threadpool_limits(2):
        if args.stage == 'preflight':
            r = dict(analytic=analytic_checks(), assembly=derivative_checks())
            r['pilot'] = block_path(out, 4, 'p2', 'plane_strain', 1., 33)
            save_json(out/'preflight.json', r); print(json.dumps(r), flush=True)
        elif args.stage == 'material':
            assert (out/'preflight.json').exists()
            r = dict(analytic=analytic_checks(), paths=[], status='running')
            for kin in KINEMATICS:
                for mode in ('tension', 'compression', 'constrained_compression', 'shear'):
                    for ramp, dt in ((.5, .05), (2.5, .05), (10., .05), (2.5, .025)):
                        row = point_path(mode, kin, ramp, dt); r['paths'].append(row)
                        print('MATERIAL', kin, mode, ramp, dt, row['end_ramp']['nominal_stress_mpa'], flush=True)
                        save_json(out/'material.json', r)
            r['status'] = 'completed'; save_json(out/'material.json', r)
        else:
            pre = json.loads((out/'preflight.json').read_text())
            assert pre['pilot']['status'] == 'completed'
            r = dict(status='running', paths=[])
            target = out/'blocks.json'
            if target.exists():
                raise FileExistsError('Do not overwrite a bounded block study')
            cases = [(n, kind, kin, k, angle, .025, 3)
                for n in (4, 8) for kind in ('p1', 'p2') for kin in KINEMATICS
                for k in (1/3, 1., 3.) for angle in (33, 57)]
            # Independently integrated controls, not endpoint reequilibrations.
            cases += [(8, kind, kin, 1., a, .0125, 3) for kind in ('p1', 'p2')
                      for kin in KINEMATICS for a in (33, 57)]
            cases += [(8, 'p2', kin, 1., a, .025, 4) for kin in KINEMATICS for a in (33, 57)]
            for n, kind, kin, k, angle, step, order in cases:
                if (n, kind, kin, k, angle, step, order) == (4, 'p2', 'plane_strain', 1., 33, .025, 3):
                    row = pre['pilot']; assert hashlib.sha256(Path(row['checkpoint']).read_bytes()).hexdigest() == row['checkpoint_sha256']
                else:
                    row = block_path(out, n, kind, kin, k, angle, step, order)
                r['paths'].append(row); save_json(target, r)
                print('BLOCK', row['id'], row['status'], round(row['elapsed_s'], 3),
                      row['rows'][-1]['force_n'] if row['rows'] else None, flush=True)
            r['status'] = 'bounded_checks_ended'; save_json(target, r)


if __name__ == '__main__':
    main()
