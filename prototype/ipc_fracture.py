"""Two-way AT1/IPC checkpoint controls, not a calibrated crack-growth model.

Total elastic energy drives damage, including compression. Original IPC boundary
topology remains intact. These fixed-depth branches isolate feedback; they are
NOT a continuous damage history from zero or a prediction of first cracking.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from run_support import save_npz, source_hashes, write_json
from .ipc_reference import BASELINE, IPCProblem, ROOT
from .material import response
from .phase_field import FixedDisplacementAT1


class CoupledAT1Problem(IPCProblem):
    def __init__(self, *args, ell_mm=.1, **kwargs):
        if kwargs.get('volumetric_discretization', 'element') != 'element':
            raise ValueError('AT1 control currently requires element volumetric discretization; '
                             'mixed damage/pressure coupling is not implemented')
        super().__init__(*args, **kwargs)
        self.phase = FixedDisplacementAT1(self.mesh.reference_xy, self.mesh.triangles,
                                         self.config.width_mm, self.config.fracture_energy_n_mm, ell_mm)
        self.damage = np.zeros(self.n)

    def elastic_density(self, x):
        F = np.eye(2)+np.einsum('eai,eaJ->eiJ',
                               (x[:self.n]-self.mesh.reference_xy)[self.mesh.triangles], self.grad)
        return response(F, self.config, tangent=False)['energy']

    def bulk_terms(self, x, tangent):
        r, k, dofs, energy, minj = super().bulk_terms(x, tangent)
        if not np.any(self.damage):
            return r, k, dofs, energy, minj
        weights = self.phase.degradation(self.damage)
        # Same element average in energy, force and tangent. The extra stress
        # evaluation avoids changing the archived intact material adapter.
        energy = float((self.phase.volumes*weights)@self.elastic_density(x))
        return r*weights[:, None], k*weights[:, None, None] if k is not None else None, dofs, energy, minj

    def coupled_friction_tangent(self, x, lag, frozen_hessian):
        if lag is None:
            return frozen_hessian
        return super().coupled_friction_tangent(x, lag, frozen_hessian)

    def recouple(self, x, depth, previous, old_damage, *, deadline, max_outer=30,
                 phase_tolerance_mpa=1e-6):
        """Stagger phase and fresh contact equilibrium at one physical load.

        Both histories are fixed for ALL alternating/PTC iterations. Only the
        caller may commit the returned pair as the next physical history.
        """
        lower = self.phase.validate_damage(old_damage).copy()
        if max_outer < 1 or not np.isfinite(phase_tolerance_mpa) or phase_tolerance_mpa <= 0:
            raise ValueError('Positive coupling budget and phase tolerance required')
        x = np.asarray(x, float).copy(); previous = np.asarray(previous, float).copy()
        if x.shape != self.rest.shape or previous.shape != x.shape or not np.isfinite([x, previous]).all():
            raise ValueError('Invalid mechanical state or history')
        saved_damage, saved_previous = self.damage.copy(), self.previous.copy()
        self.previous = previous; self.damage = lower.copy(); self.coupling_history = []
        self.trial = x.copy()
        try:
            for outer in range(max_outer):
                if time.monotonic() >= deadline: raise TimeoutError('AT1/IPC coupling wall budget')
                psi = self.elastic_density(x)
                before = self.damage.copy()
                updated, info = self.phase.solve(psi, old=lower, tolerance_mpa=phase_tolerance_mpa,
                                                seconds=min(60., deadline-time.monotonic()))
                before_release, before_fracture = self.phase.energy(before, psi)
                decrease = info['energy_change_n_mm']-(before_fracture-before_release)
                if decrease > 1e-8*max(1., abs(before_fracture), abs(before_release)):
                    raise RuntimeError('AT1 subproblem increased fixed-displacement energy')
                self.damage = updated
                # Never use step() here: it would reset the physical slip history.
                x, iterations = self.solve_friction(x, depth, deadline)
                psi = self.elastic_density(x)
                kkt = self.phase.kkt_residual(self.damage, psi, lower)
                balance = self.acceptance(x, depth, iterations+1, outer+1)
                released, fracture = self.phase.energy(self.damage, psi)
                record = dict(outer=outer, max_damage=float(self.damage.max()),
                              damage_update_inf=float(np.max(np.abs(self.damage-before))),
                              phase_kkt_residual_mpa=kkt, residual_n=balance['residual_n'],
                              reaction_fz_n=balance['reaction_fz_n'],
                              fixed_u_phase_energy_change_n_mm=decrease,
                              fracture_energy_n_mm=fracture,
                              elastic_release_at_current_u_n_mm=released)
                self.coupling_history.append(record)
                print('COUPLING '+json.dumps(record), flush=True)
                if kkt <= phase_tolerance_mpa:
                    if not np.array_equal(self.previous, previous):
                        raise RuntimeError('Physical friction history changed during staggering')
                    balance.update(record, mechanics_recoupled=True,
                                   coupled_energy_n_mm=balance['energy_n_mm']+fracture,
                                   energy_note='includes fresh friction pseudo-potential; not global dissipative energy balance',
                                   phase_tolerance_mpa=phase_tolerance_mpa)
                    return x, self.damage.copy(), balance
            raise RuntimeError(f'AT1/IPC coupling iteration budget: phase KKT={kkt:g} MPa')
        except Exception:
            self.failed_damage = self.damage.copy()
            self.damage, self.previous = saved_damage, saved_previous
            raise


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_control(source, output, depth, ell, seconds, threads):
    """Rebalance one verified intact checkpoint after introducing free damage."""
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', completed=False, pid=os.getpid(), depth_mm=depth,
                  ell_mm=ell, seconds=seconds, threads=threads, model='coupled total-energy AT1 control',
                  history='damage introduced at fixed depth from zero; not continuous loading',
                  compression_drives_damage=True, independent_strength=False,
                  new_crack_contact=False, source_sha256=source_hashes(snapshot=output/'source'))
    shutil.copyfile(ROOT/'prototype/ipc_baseline.json', output/'source/prototype/ipc_baseline.json')
    write_json(output/'result.json', report)
    with (output/'run.log').open('w', encoding='utf-8', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), threadpool_limits(limits=threads):
        p = None
        try:
            origin = json.loads((source/'result.json').read_text(encoding='utf-8'))
            if origin['status'] not in ('completed', 'failed') or origin['simplifications']['fracture']:
                raise ValueError('Expected a stopped, intact source run')
            if origin.get('volumetric_discretization', 'element') != 'element':
                raise ValueError('Cannot import mixed pressure history into element AT1 control')
            baseline_hash = digest(ROOT/'prototype/ipc_baseline.json')
            if baseline_hash != origin['baseline_config_sha256']:
                raise ValueError('Archived IPC baseline settings mismatch')
            report['baseline_config_sha256'] = baseline_hash
            for name in ('prototype/material.py', 'mechanics/finite_strain.py'):
                if digest(ROOT/name) != origin['source_sha256'][name]:
                    raise ValueError('Material mismatch: '+name)
            index = next(i for i, s in enumerate(origin['steps']) if abs(s['depth_mm']-depth) < 1e-12)
            checkpoint = source/'states'/f'{index:04d}.npz'
            report.update(source_result_sha256=digest(source/'result.json'), source_checkpoint_sha256=digest(checkpoint),
                          source_completed=origin['completed'], source_last_depth_mm=origin['steps'][-1]['depth_mm'])
            mesh, sim = origin['mesh_settings'], origin['simplifications']
            p = CoupledAT1Problem(case=origin['case'], mu=origin['config']['friction_coefficient'],
                                  dhat=origin['dhat_mm'], stiffness=origin['barrier_stiffness_mpa'],
                                  max_iterations=origin['max_inner_iterations'], ell_mm=ell,
                                  local_spacing=mesh['local_spacing_mm'], surface_spacing=mesh['surface_spacing_mm'],
                                  fine_halfwidth=mesh['fine_halfwidth_mm'], nose_facets=sim['nose_facets'],
                                  tip_radius=sim['tip_radius_mm'], inclination=origin['config']['inclination_deg'],
                                  hessian_mode=origin['hessian_mode'])
            p.ipc.set_num_threads(threads)
            if digest(p.ipc.__file__) != origin['ipc_binary_sha256']:
                raise ValueError('IPC binary mismatch')
            for key, value in origin['config'].items():
                if getattr(p.config, key) != value: raise ValueError('Config mismatch: '+key)
            with np.load(checkpoint) as state:
                np.testing.assert_array_equal(state['reference_xy'], p.rest)
                np.testing.assert_array_equal(state['triangles'], p.mesh.triangles)
                np.testing.assert_allclose(float(state['depth_mm']), depth, rtol=0, atol=1e-12)
                if int(state['body_nodes']) != p.n: raise ValueError('Body count mismatch')
                x, previous = state['positions'].copy(), state['previous_positions'].copy()
            previous_file = source/'states'/f'{index-1:04d}.npz' if index else None
            if previous_file:
                with np.load(previous_file) as state:
                    np.testing.assert_array_equal(previous, state['positions'])
                report['previous_checkpoint_sha256'] = digest(previous_file)
            else:
                np.testing.assert_array_equal(previous, p.rest)
            p.previous = previous.copy()
            intact = p.acceptance(x, depth, 0, 0)
            np.testing.assert_allclose(intact['reaction_fz_n'], origin['steps'][index]['reaction_fz_n'], rtol=1e-8, atol=1e-8)
            report.update(config=p.config.to_dict(), mesh_settings=mesh, source_simplifications=sim, intact=intact,
                          residual_stiffness=p.phase.residual, force_tolerance_n=BASELINE['force_tolerance_n'])
            write_json(output/'result.json', report)
            print(f'START depth_mm={depth:g} ell_mm={ell:g} intact_fz_N={intact["reaction_fz_n"]:.9g}', flush=True)
            final, damage, balance = p.recouple(x, depth, previous, np.zeros(p.n), deadline=time.monotonic()+seconds)
            tip = final[p.n:][np.argmin(final[p.n:, 1])]
            peak = int(np.argmax(damage)) if damage.max() > 1e-12 else None
            report.update(status='completed', completed=True, balance=balance,
                          reaction_change_n=balance['reaction_fz_n']-intact['reaction_fz_n'],
                          max_displacement_correction_mm=float(np.max(np.linalg.norm(final-x, axis=1))),
                          peak_node=peak, peak_offset_from_tip_xz_mm=(final[peak]-tip).tolist() if peak is not None else None,
                          peak_reference_xz_mm=p.rest[peak].tolist() if peak is not None else None)
            save_npz(output/'accepted.npz', positions=final, previous_positions=previous, old_damage=np.zeros(p.n),
                     damage=damage, reference_xy=p.rest, triangles=p.mesh.triangles, body_nodes=p.n, depth_mm=depth)
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            print(report['error'], flush=True)
            if p is not None and hasattr(p, 'failed_damage'):
                save_npz(output/'failed_trial.npz', positions=p.trial, damage=p.failed_damage,
                         previous_positions=previous, old_damage=np.zeros(p.n),
                         reference_xy=p.rest, triangles=p.mesh.triangles, body_nodes=p.n, depth_mm=depth)
        if p is not None: report['coupling_history'] = getattr(p, 'coupling_history', [])
        write_json(output/'result.json', report)
    return report['completed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depths', type=float, nargs='+', default=[1., 2.])
    parser.add_argument('--ell', type=float, default=.1)
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if (args.threads < 1 or not all(np.isfinite(v) and v > 0 for v in (args.ell, args.seconds))
            or not all(np.isfinite(d) and d >= 0 for d in args.depths)
            or len(set(args.depths)) != len(args.depths)):
        parser.error('Invalid budget, ell, threads or depths')
    if args.output.exists(): parser.error('Output must not already exist')
    args.output.mkdir(parents=True)
    if args.detach:
        output = args.output.resolve(); frozen = output/'source'
        source_hashes(snapshot=frozen)
        shutil.copyfile(ROOT/'prototype/ipc_baseline.json', frozen/'prototype/ipc_baseline.json')
        command = [sys.executable, '-u', '-m', 'prototype.ipc_fracture', '--source', str(args.source.resolve()),
                   '--output', str(output/'controls'), '--depths', *map(str, args.depths),
                   '--ell', str(args.ell), '--seconds', str(args.seconds), '--threads', str(args.threads)]
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT/'.runtime/ipc-python')+os.pathsep+env.get('PYTHONPATH', '')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): env[key] = str(args.threads)
        with (output/'stdout.log').open('wb') as stdout, (output/'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, cwd=frozen, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                       start_new_session=os.name != 'nt')
        write_json(output/'launch.json', dict(pid=process.pid, command=command, source_root=str(frozen)))
        print(process.pid)
    else:
        statuses = []
        for depth in args.depths:
            ok = checkpoint_control(args.source, args.output/f'depth{depth:g}', depth, args.ell, args.seconds, args.threads)
            statuses.append(dict(depth_mm=depth, completed=ok))
            write_json(args.output/'status.json', dict(status='completed' if len(statuses) == len(args.depths) else 'running',
                                                      all_controls_passed=all(s['completed'] for s in statuses), cases=statuses))
        sys.exit(0 if all(s['completed'] for s in statuses) else 1)


if __name__ == '__main__': main()
