"""Bounded, free-sided pre-fracture compliance checks against an intact patch.

Uses the production finite-strain P1 bulk, mixed interface and native IPC/CCD.
Damage is disabled. Geometry, material, boundary motion and physical duration
are common to intact and split meshes. No experimental inputs are read.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.cohesive_mixed import MixedModeLaw, interface_response
from prototype.cohesive_network import network_mesh
from prototype.ipc_reference import determinant_step, toolkit
from prototype.material import bulk
from prototype.model_config import FEMConfig, FEMMesh


FORCE_TOLERANCE_N = 1e-8
DHAT_MM = 2e-5
CCD_MAX_ITERATIONS = 100000
LOADING_RATE_MM_S = .2
MATERIAL = dict(shear_modulus_n_mm2=4., poisson_ratio=.25)


def _tick(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Compliance case budget exhausted')


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ['tools/verify_network_compliance.py', 'prototype/cohesive_mixed.py',
             'prototype/cohesive_network.py', 'prototype/material.py',
             'prototype/model_config.py', 'prototype/ipc_reference.py',
             'mechanics/finite_strain.py']
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def square_mesh(cells, config):
    coordinates = np.array([[i/cells, j/cells] for j in range(cells+1) for i in range(cells+1)])
    triangles = []
    for j in range(cells):
        for i in range(cells):
            a = j*(cells+1)+i
            triangles.extend(([a, a+1, a+cells+2], [a, a+cells+2, a+cells+1]))
    return FEMMesh(coordinates.copy(), coordinates, np.asarray(triangles, int),
        np.empty((0, 2), int), np.empty(0), np.flatnonzero(coordinates[:, 1] == 0.),
        np.empty(0, int), np.zeros(len(coordinates), int))


class CompliancePatch:
    def __init__(self, cells=3, *, network=True, gap_mm=.0005,
                 stiffness_mpa_per_mm=400., compression_stiffness_mpa_per_mm=None,
                 compression_regularization='linear'):
        if type(cells) is not int or not 1 <= cells <= 8:
            raise ValueError('cells must be an integer in [1,8]')
        if not np.isfinite(stiffness_mpa_per_mm) or stiffness_mpa_per_mm <= 0.:
            raise ValueError('Positive finite interface stiffness required')
        if not np.isfinite(gap_mm) or gap_mm <= 2*DHAT_MM:
            raise ValueError('gap must exceed twice the common fixed IPC distance')
        if compression_regularization not in ('linear', 'gap_bounded'):
            raise ValueError('compression_regularization must be linear or gap_bounded')
        self.config = FEMConfig(length_mm=1., height_mm=1., width_mm=1.,
                                inclination_deg=0., **MATERIAL)
        original = square_mesh(cells, self.config)
        self.mesh, self.topology = network_mesh(original, self.config, gap_mm) if network else (original, None)
        self.reference = self.mesh.reference_xy.copy()
        kc = stiffness_mpa_per_mm if compression_stiffness_mpa_per_mm is None else compression_stiffness_mpa_per_mm
        self.law = MixedModeLaw(normal_strength_mpa=.2, shear_strength_mpa=.2,
            normal_stiffness_mpa_per_mm=stiffness_mpa_per_mm, toughness_n_per_mm=.006,
            damage_enabled=False, compression_stiffness_mpa_per_mm=kc,
            compression_regularization_mm=gap_mm if compression_regularization == 'gap_bounded' else None)
        self.history = np.zeros((len(self.topology.segments), 2)) if network else None
        tol = 1e-12
        self.bottom = np.flatnonzero(np.abs(self.reference[:, 1]) < tol)
        self.top = np.flatnonzero(np.abs(self.reference[:, 1]-1.) < tol)
        if not len(self.bottom) or not len(self.top):
            raise ValueError('Both physical support lines must contain nodes')
        self.reference[self.bottom, 1] = 0.
        self.reference[self.top, 1] = 1.
        self.horizontal_pin = int(self.bottom[np.argmin(self.reference[self.bottom, 0])])
        self.fixed = np.unique(np.r_[2*self.bottom+1, 2*self.top+1, 2*self.horizontal_pin])
        self.free = np.setdiff1d(np.arange(self.reference.size), self.fixed)
        self.top_dofs = 2*self.top+1
        tri = self.mesh.triangles
        candidate_edges = np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
        unique, counts = np.unique(candidate_edges, axis=0, return_counts=True)
        edges = unique[counts == 1]
        self.surface = np.unique(edges)
        self.surface_dofs = (2*self.surface[:, None]+np.arange(2)).ravel()
        self.ipc = toolkit()
        self.ccd_tolerance_mm = min(1e-6, gap_mm/10., DHAT_MM/10.)
        self.narrow_phase_ccd = self.ipc.TightInclusionCCD(
            tolerance=self.ccd_tolerance_mm,
            max_iterations=CCD_MAX_ITERATIONS,
            conservative_rescaling=self.ipc.TightInclusionCCD.DEFAULT_CONSERVATIVE_RESCALING)
        self.cmesh = self.ipc.CollisionMesh(self.reference[self.surface], np.searchsorted(self.surface, edges))
        self.cmesh.can_collide = self.ipc.make_static_obstacle_filter(len(self.surface))
        self.barrier = self.ipc.BarrierPotential(DHAT_MM, 6.6, True)
        if self.ipc.has_intersections(self.cmesh, self.reference[self.surface]):
            raise ValueError('Reference patch intersects')

    def boundary(self, displacement):
        x = self.reference.copy()
        x[self.top, 1] += displacement
        return x.ravel()[self.fixed]

    def collisions(self, positions):
        contacts = self.ipc.NormalCollisions()
        contacts.use_area_weighting = True
        contacts.collision_set_type = self.ipc.NormalCollisions.IMPROVED_MAX_APPROX
        contacts.build(self.cmesh, positions[self.surface], DHAT_MM)
        return contacts

    def evaluate(self, positions, *, tangent=True):
        residual, stiffness, dofs, stored, minj = bulk(
            self.mesh, (positions-self.reference).ravel(), self.config, with_tangent=tangent)
        gradient = np.zeros(positions.size)
        np.add.at(gradient, dofs.ravel(), residual.ravel())
        q = None
        if self.topology is not None:
            t = self.topology
            q = interface_response(positions, t.segments, t.reference_gaps,
                t.reference_areas, self.history, self.law, tangent=tangent,
                reference_slips=t.reference_slips)
            gradient += q['gradient']
        contacts = self.collisions(positions)
        sx = positions[self.surface]
        ipc_energy = float(self.barrier(contacts, self.cmesh, sx))
        ipc_gradient = self.barrier.gradient(contacts, self.cmesh, sx)
        gradient[self.surface_dofs] += ipc_gradient
        hessian = None
        if tangent:
            hessian = np.zeros((positions.size, positions.size))
            for local, ids in zip(stiffness, dofs):
                hessian[np.ix_(ids, ids)] += local
            if q is not None:
                hessian += q['hessian'].toarray()
            ipc_hessian = self.barrier.hessian(contacts, self.cmesh, sx,
                                               self.ipc.PSDProjectionMethod.NONE).toarray()
            hessian[np.ix_(self.surface_dofs, self.surface_dofs)] += ipc_hessian
        interface_energy = q['stored_energy_n_mm'] if q is not None else 0.
        return dict(energy=float(stored+interface_energy+ipc_energy), bulk_stored=float(stored),
            interface_stored=float(interface_energy),
            compression_stored=q['compression_stored_energy_n_mm'] if q is not None else 0.,
            damageable_stored=q['damageable_stored_energy_n_mm'] if q is not None else 0.,
            ipc_stored=ipc_energy, gradient=gradient, hessian=hessian, min_j=float(minj),
            reaction_n=float(gradient[self.top_dofs].sum()),
            maximum_damage=float(q['damage'].max()) if q is not None else 0.,
            fracture_dissipation=float(q['fracture_dissipation_n_mm']) if q is not None else 0.,
            ipc_ev_contacts=len(contacts.ev_collisions), ipc_vv_contacts=len(contacts.vv_collisions),
            ipc_force_norm_n=float(np.linalg.norm(ipc_gradient)))

    def safe_step(self, positions, direction):
        gradient = self.mesh.bulk_data[1]
        deformation = np.eye(2)+np.einsum('eai,eaJ->eiJ',
            (positions-self.reference)[self.mesh.triangles], gradient)
        change = np.einsum('eai,eaJ->eiJ', direction[self.mesh.triangles], gradient)
        return min(float(determinant_step(deformation, change)),
            float(self.ipc.compute_collision_free_stepsize(self.cmesh,
                positions[self.surface], (positions+direction)[self.surface],
                narrow_phase_ccd=self.narrow_phase_ccd)))

    def solve(self, accepted, previous, displacement, deadline):
        trial = accepted.copy()
        trial[:, 1] += (displacement-previous)*self.reference[:, 1]
        trial.ravel()[self.fixed] = self.boundary(displacement)
        if self.safe_step(accepted, trial-accepted) < 1.-1e-12:
            raise RuntimeError('Boundary predictor violates CCD/J; refine load increment')
        for iteration in range(100):
            _tick(deadline)
            self.last_trial = trial.copy()
            value = self.evaluate(trial)
            residual = value['gradient'][self.free]
            norm = float(np.linalg.norm(residual))
            matrix = value['hessian'][np.ix_(self.free, self.free)]
            minimum_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
            if norm <= FORCE_TOLERANCE_N:
                return trial, value, dict(iterations=iteration, residual_n=norm,
                    minimum_free_hessian_eigenvalue_n_per_mm=minimum_eigenvalue)
            shift = max(0., 1e-8-minimum_eigenvalue)
            direction = np.zeros(trial.size)
            direction[self.free] = np.linalg.solve(matrix+shift*np.eye(len(self.free)), -residual)
            direction = direction.reshape(trial.shape)
            alpha = min(1., self.safe_step(trial, direction))
            slope = float(value['gradient']@direction.ravel())
            for search in range(35):
                _tick(deadline)
                candidate = trial+alpha*direction
                tested = self.evaluate(candidate, tangent=False)
                residual_improved = (np.linalg.norm(tested['gradient'][self.free]) < .5*norm
                    and tested['energy'] <= value['energy']+64*np.finfo(float).eps*max(1., abs(value['energy'])))
                if tested['energy'] <= value['energy']+1e-4*alpha*slope+1e-16 or residual_improved:
                    trial = candidate
                    break
                alpha *= .5
            else:
                raise RuntimeError('Compliance Newton line search failed')
        raise RuntimeError('Compliance Newton iteration limit reached')


def _row(displacement, value, diagnostics):
    return dict(displacement_mm=float(displacement), physical_time_s=abs(float(displacement))/LOADING_RATE_MM_S,
        reaction_n=value['reaction_n'], total_stored_n_mm=value['energy'],
        bulk_stored_n_mm=value['bulk_stored'], interface_stored_n_mm=value['interface_stored'],
        compression_stored_n_mm=value['compression_stored'], damageable_stored_n_mm=value['damageable_stored'],
        ipc_stored_n_mm=value['ipc_stored'], min_j=value['min_j'], maximum_damage=value['maximum_damage'],
        fracture_dissipation_n_mm=value['fracture_dissipation'], ipc_force_norm_n=value['ipc_force_norm_n'],
        ipc_ev_contacts=value['ipc_ev_contacts'], ipc_vv_contacts=value['ipc_vv_contacts'], **diagnostics)


def run_case(spec, *, strain=.0002, steps=8, seconds=45.):
    if not np.isfinite([strain, seconds]).all() or not 0 < strain <= .01 or not 1 <= seconds <= 180:
        raise ValueError('strain in (0,.01], case seconds in [1,180] required')
    if type(steps) is not int or not 2 <= steps <= 128 or spec['mode'] not in ('tension', 'compression'):
        raise ValueError('Require 2--128 steps and tension/compression mode')
    spec = dict(spec)
    spec.setdefault('compression_regularization', 'linear')
    started = time.monotonic()
    deadline = started+seconds
    patch = CompliancePatch(spec['cells'], network=spec['network'], gap_mm=spec['gap_mm'],
        stiffness_mpa_per_mm=spec['stiffness_mpa_per_mm'],
        compression_stiffness_mpa_per_mm=spec['compression_stiffness_mpa_per_mm'],
        compression_regularization=spec['compression_regularization'])
    positions = [patch.reference.copy()]
    initial = patch.evaluate(positions[0])
    rows = [_row(0., initial, dict(iterations=0, residual_n=float(np.linalg.norm(initial['gradient'][patch.free])),
        minimum_free_hessian_eigenvalue_n_per_mm=float(np.linalg.eigvalsh(initial['hessian'][np.ix_(patch.free, patch.free)]).min())))]
    completed = False
    failure = None
    target = strain if spec['mode'] == 'tension' else -strain
    try:
        for displacement in np.linspace(0., target, steps+1)[1:]:
            trial, value, diagnostics = patch.solve(positions[-1], rows[-1]['displacement_mm'], float(displacement), deadline)
            if value['min_j'] <= 0 or patch.ipc.has_intersections(patch.cmesh, trial[patch.surface]):
                raise RuntimeError('Equilibrium has invalid geometry')
            positions.append(trial.copy())
            rows.append(_row(displacement, value, diagnostics))
        completed = True
    except Exception as error:
        failure = dict(type=type(error).__name__, error=str(error), attempted_displacement_mm=float(displacement))
    # Independent patch reconstruction, fresh residuals and exact prescribed DOFs.
    audit = dict(passed=False, maximum_residual_n=None, min_j=None,
                 minimum_saved_transition_safe_fraction=None, error=None)
    try:
        fresh = CompliancePatch(spec['cells'], network=spec['network'], gap_mm=spec['gap_mm'],
            stiffness_mpa_per_mm=spec['stiffness_mpa_per_mm'],
            compression_stiffness_mpa_per_mm=spec['compression_stiffness_mpa_per_mm'],
            compression_regularization=spec['compression_regularization'])
        maximum, minimum, transition_fraction = 0., 1., 1.
        for index, (x, row) in enumerate(zip(positions, rows)):
            _tick(deadline)
            q = fresh.evaluate(x, tangent=False)
            maximum = max(maximum, float(np.linalg.norm(q['gradient'][fresh.free])))
            minimum = min(minimum, q['min_j'])
            if not np.array_equal(x.ravel()[fresh.fixed], fresh.boundary(row['displacement_mm'])):
                raise RuntimeError('Prescribed boundary mismatch')
            if fresh.ipc.has_intersections(fresh.cmesh, x[fresh.surface]):
                raise RuntimeError('Independent intersection audit failed')
            if q['maximum_damage'] != 0. or q['fracture_dissipation'] != 0.:
                raise RuntimeError('Damage entered the elastic compliance control')
            if index:
                transition_fraction = min(transition_fraction,
                    fresh.safe_step(positions[index-1], x-positions[index-1]))
        audit.update(passed=maximum <= FORCE_TOLERANCE_N and minimum > 0.
                     and transition_fraction >= 1.-1e-12,
                     maximum_residual_n=maximum, min_j=minimum,
                     minimum_saved_transition_safe_fraction=transition_fraction)
    except Exception as error:
        audit['error'] = f'{type(error).__name__}: {error}'
    load = np.array([r['displacement_mm'] for r in rows])
    force = np.array([r['reaction_n'] for r in rows])
    work = np.r_[0., np.cumsum(.5*(force[1:]+force[:-1])*np.diff(load))]
    stored = np.array([r['total_stored_n_mm'] for r in rows])
    remainder = work-(stored-stored[0])
    energy_scale = max(float(np.max(np.abs(work))), 1e-14)
    energy_error = float(np.max(np.abs(remainder)))
    work_resolved = energy_scale > 1e-10
    energy_qualified = bool(completed and audit['passed'] and energy_error <= max(1e-10, .005*energy_scale))
    for row, external, error in zip(rows, work, remainder):
        row.update(external_work_n_mm=float(external), energy_remainder_n_mm=float(error))
    young = 2*MATERIAL['shear_modulus_n_mm2']*(1+MATERIAL['poisson_ratio'])
    plane_strain_modulus = young/(1-MATERIAL['poisson_ratio']**2)
    h = 1/spec['cells']
    reaction_resolution = 10*FORCE_TOLERANCE_N
    reaction_resolved = bool(completed and abs(force[-1]) > reaction_resolution)
    result = dict(**spec, completed=completed, qualified=bool(completed and audit['passed'] and energy_qualified),
        failure=failure, audit=audit, energy=dict(qualified=energy_qualified,
            maximum_absolute_remainder_n_mm=energy_error,
            maximum_relative_remainder=energy_error/energy_scale if work_resolved else None,
            work_resolved=work_resolved,
            qualification_basis='resolved_work_with_absolute_floor' if work_resolved else 'absolute_remainder_only_unresolved_work',
            relative_tolerance=.005, absolute_floor_n_mm=1e-10),
        rows=rows, cells_per_side=spec['cells'], h_mm=h, gap_over_h=spec['gap_mm']/h if spec['network'] else 0.,
        E_over_Kh=young/(spec['stiffness_mpa_per_mm']*h) if spec['network'] else 0.,
        E_plane_strain_over_Kh=plane_strain_modulus/(spec['stiffness_mpa_per_mm']*h) if spec['network'] else 0.,
        apparent_modulus_mpa=float(force[-1]/load[-1]) if reaction_resolved else None,
        reaction_resolved=reaction_resolved, reaction_resolution_n=reaction_resolution,
        modulus_resolution_mpa=reaction_resolution/abs(target),
        response_resolution='resolved' if reaction_resolved else 'below_absolute_force_resolution_or_incomplete',
        continuum_small_strain_modulus_mpa=plane_strain_modulus,
        compression_regularization_mm=patch.law.compression_regularization_mm,
        compression_traction_cap_mpa=patch.law.to_dict()['compression_traction_cap_mpa'],
        constitutive_identity=patch.law.to_dict(),
        retained_area_ratio=patch.topology.bulk_area_ratio if patch.topology is not None else 1.,
        free_dofs=len(patch.free), body_nodes=len(patch.reference), triangles=len(patch.mesh.triangles),
        steps_requested=steps, accepted_states=len(rows), target_strain=target,
        duration_s=strain/LOADING_RATE_MM_S, loading_rate_mm_s=LOADING_RATE_MM_S,
        case_budget_s=seconds, elapsed_s=time.monotonic()-started,
        contact=dict(full_native_EV_VV=True, full_CCD=True, frictionless=True,
            dhat_mm=DHAT_MM, ipc_version=patch.ipc.__version__,
            ccd_tolerance_mm=patch.ccd_tolerance_mm,
            ccd_conservative_rescaling=float(patch.ipc.TightInclusionCCD.DEFAULT_CONSERVATIVE_RESCALING),
            ccd_max_iterations=CCD_MAX_ITERATIONS),
        damage_enabled=False, numerical_regularization='Newton direction only; no artificial force or dissipation')
    fields = dict(reference=patch.reference, positions=np.asarray(positions), triangles=patch.mesh.triangles,
        fixed_dofs=patch.fixed, top_nodes=patch.top, surface=patch.surface,
        failed_trial=getattr(patch, 'last_trial', positions[-1]) if failure else np.empty((0, 2)))
    return result, fields


def build_matrix(compression_regularization='linear'):
    """Three one-factor sweeps plus a joint limit and zero-compression control."""
    if compression_regularization not in ('linear', 'gap_bounded'):
        raise ValueError('compression_regularization must be linear or gap_bounded')
    cases = {}
    def add(group, cells, gap, stiffness, kc=None, network=True, modes=('tension', 'compression')):
        kc = stiffness if kc is None else kc
        for mode in modes:
            key = (cells, gap, stiffness, kc, network, mode)
            if key not in cases:
                cases[key] = dict(id=f'case_{len(cases):02d}', groups=[], cells=cells, gap_mm=gap,
                    stiffness_mpa_per_mm=stiffness, compression_stiffness_mpa_per_mm=kc,
                    network=network, mode=mode, compression_regularization=compression_regularization)
            cases[key]['groups'].append(group)
    for n in (2, 3, 4):
        add('intact', n, .0005, 400., network=False)
        add('mesh', n, .0005, 400.)
    for gap in (.00025, .0005, .001):
        add('gap', 3, gap, 400.)
    for stiffness in (100., 400., 1600.):
        add('stiffness', 3, .0005, stiffness)
    for n in (2, 3, 4):
        add('joint_limit', n, .001*(2/n)**2, 1600.*(n/2)**2)
    add('zero_compression_control', 3, .0005, 400., kc=0., modes=('compression',))
    return list(cases.values())


def _render(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, group, key, label in zip(axes, ('mesh', 'gap', 'stiffness'),
                                    ('h_mm', 'gap_mm', 'stiffness_mpa_per_mm'),
                                    ('Mesh spacing h (mm)', 'Reference gap (mm)', 'Kn = Kc (MPa/mm)')):
        for mode in ('tension', 'compression'):
            rows = sorted([c for c in report['cases'] if group in c['groups'] and c['mode'] == mode and c['qualified']], key=lambda c: c[key])
            ax.plot([c[key] for c in rows], [c['modulus_ratio_to_intact'] for c in rows], 'o-', label=mode)
        ax.axhline(1., color='black', linewidth=.8, linestyle='--')
        ax.set_xlabel(label)
        ax.set_ylabel('Apparent modulus / intact')
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output/'compliance.png', dpi=160)
    plt.close(fig)
    table = ''.join('<tr>'+''.join(f'<td>{html.escape(str(c.get(k)))}</td>' for k in
        ('id', 'groups', 'mode', 'cells', 'gap_mm', 'stiffness_mpa_per_mm',
         'compression_stiffness_mpa_per_mm', 'modulus_ratio_to_intact', 'response_resolution', 'qualified'))+'</tr>' for c in report['cases'])
    page = f'''<!doctype html><meta charset="utf-8"><title>Network compliance verification</title>
<style>body{{font:15px system-ui;max-width:1320px;margin:30px auto;line-height:1.55}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:6px}}img{{max-width:100%}}.note{{background:#fff1d6;padding:15px}}</style>
<h1>网络切前柔度验证</h1><p class="note">同一 1×1 mm 方形、自由侧面、生产有限应变 P1 体材料、无损伤混合界面及完整 IPC/CCD。此报告只验证局部切前柔度；不认证完整切割、裂后路径或实验预测。压缩支路为可恢复虚拟闭合惩罚，本轮模式：{html.escape(str(report['compression_regularization']))}。linear 为旧线性模式；gap_bounded 固定 ell_c=参考 gap，压力上限为 Kc×gap，不增加独立拟合参数。</p>
<p>总体计算资格：{report['passed']}。全部算例固定 dhat={DHAT_MM} mm；共同物理时长 {report['duration_s']} s。网格、间隙、刚度独立扫描；joint_limit 同时减小 g/h 与 E/(Kh)，单独标记。固定 K 时细化网格可增加串联界面柔度，不应将其解释成收敛。端点反力小于绝对分辨率 {10*FORCE_TOLERANCE_N} N 时，模量及比值留空；数值平衡通过不代表近零响应被精确测定。</p>
<p>体材料为无 Maxwell 支路的有限应变平面应变弹性验证材料，时长仅记录共同加载计划。实际最大接触数：{report['contact_scope']['maximum_actual_contacts']}；本报告不认证闭合接触或摩擦。近零外功只审查绝对功账余项，相对误差留空。</p>
<img src="compliance.png"><p><a href="verification.json">完整 JSON</a> · <a href="summary.csv">汇总 CSV</a></p>
<table><tr><th>ID</th><th>组</th><th>加载</th><th>每边单元</th><th>gap/mm</th><th>Kn</th><th>Kc</th><th>模量/连续体</th><th>力分辨率</th><th>资格</th></tr>{table}</table>'''
    (output/'index.html').write_text(page, encoding='utf-8')


def run(output, *, seconds=45., steps=8, strain=.0002, specs=None, compression_regularization='linear'):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    specs = build_matrix(compression_regularization) if specs is None else specs
    source = _source_hashes()
    started = time.monotonic()
    report = dict(schema='network-compliance-verification/1', created_utc=datetime.now(timezone.utc).isoformat(),
        interpreter=sys.executable, source_sha256=source, cases=[], material=MATERIAL,
        no_experimental_force_fitting=True, experimental_inputs_used=[], physical_onset_prediction=None,
        scope='local damage-disabled free-side elasticity/compliance only; not full cutting qualification',
        dhat_mm=DHAT_MM, duration_s=strain/LOADING_RATE_MM_S,
        compression_regularization=sorted({s.get('compression_regularization', 'linear') for s in specs}),
        boundary='bottom uy=0; top uy=prescribed; one bottom ux pin; every other lateral DOF free',
        numerical_tolerance_n=FORCE_TOLERANCE_N, per_case_budget_s=seconds,
        nominal_total_case_budget_s=seconds*len(specs), steps=steps)
    for spec in specs:
        spec_path = output/f"{spec['id']}.spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
        arguments = [sys.executable, '-u', '-m', 'tools.verify_network_compliance',
            '--case-json', str(spec_path), '--output', str(output), '--seconds', str(seconds),
            '--steps', str(steps), '--strain', str(strain)]
        child_error = None
        start_case = time.monotonic()
        with (output/f"{spec['id']}.log").open('w', encoding='utf-8') as log:
            try:
                child = subprocess.run(arguments, cwd=Path(__file__).resolve().parents[1],
                    stdout=log, stderr=subprocess.STDOUT, check=False, timeout=seconds+5.,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if child.returncode != 0:
                    child_error = dict(type='WorkerFailure', error=f'Worker exited {child.returncode}')
            except subprocess.TimeoutExpired:
                child_error = dict(type='HardBudgetExceeded', error='Native solve exceeded case wall limit; worker terminated')
        if child_error is None:
            result = json.loads((output/f"{spec['id']}.json").read_text(encoding='utf-8'))
        else:
            result = dict(**spec, completed=False, qualified=False, failure=child_error, rows=[],
                audit=dict(passed=False), energy=dict(qualified=False), apparent_modulus_mpa=None,
                case_budget_s=seconds, elapsed_s=time.monotonic()-start_case,
                duration_s=strain/LOADING_RATE_MM_S)
            (output/f"{spec['id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        result['execution'] = dict(args=arguments, hard_wall_budget_s=seconds+5.,
                                   interpreter_startup_allowance_s=5.)
        report['cases'].append(result)
        (output/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"{spec['id']} {spec['mode']} n={spec['cells']} g={spec['gap_mm']:.6g} K={spec['stiffness_mpa_per_mm']:.6g} qualified={result['qualified']} E={result['apparent_modulus_mpa']}", flush=True)
    baselines = {(c['cells'], c['mode']): c for c in report['cases'] if not c['network'] and c['qualified']}
    for case in report['cases']:
        reference = baselines.get((case['cells'], case['mode']))
        case['modulus_ratio_to_intact'] = (case['apparent_modulus_mpa']/reference['apparent_modulus_mpa']
            if reference and case['qualified'] and case.get('reaction_resolved') and reference.get('reaction_resolved') else None)
        case['intact_reference_case_id'] = reference['id'] if reference else None
        (output/f"{case['id']}.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding='utf-8')
    report['source_sha256_at_end'] = _source_hashes()
    report['passed'] = bool(all(c['qualified'] for c in report['cases']) and source == report['source_sha256_at_end'])
    report['elapsed_s'] = time.monotonic()-started
    report['interpretation'] = dict(joint_refinement_is_not_independent_axes=True,
        complete_cutting_qualified=False, fracture_path_convergence_qualified=False,
        virtual_compression_is_not_physical_contact_equivalence=True)
    report['contact_scope'] = dict(native_contact_enabled=True,
        maximum_actual_contacts=max((r['ipc_ev_contacts']+r['ipc_vv_contacts']
            for c in report['cases'] for r in c['rows']), default=0),
        closed_contact_or_friction_qualified=False)
    (output/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    columns = ['id', 'mode', 'cells', 'gap_mm', 'stiffness_mpa_per_mm', 'compression_stiffness_mpa_per_mm',
        'compression_regularization', 'compression_regularization_mm', 'compression_traction_cap_mpa',
        'network', 'qualified', 'h_mm', 'gap_over_h', 'E_over_Kh', 'E_plane_strain_over_Kh',
        'retained_area_ratio', 'apparent_modulus_mpa', 'modulus_ratio_to_intact', 'response_resolution', 'elapsed_s']
    with (output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows({k: case.get(k) for k in columns} for case in report['cases'])
    _render(output, report)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=45., help='Per-case construction/solve/audit budget')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--strain', type=float, default=.0002)
    parser.add_argument('--compression-regularization', choices=('linear', 'gap_bounded'), default='linear')
    parser.add_argument('--case-json', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case_json:
        spec = json.loads(Path(args.case_json).read_text(encoding='utf-8'))
        with threadpool_limits(limits=1):
            result, fields = run_case(spec, seconds=args.seconds, steps=args.steps, strain=args.strain)
        output = Path(args.output)
        np.savez_compressed(output/f"{spec['id']}.npz", **fields)
        (output/f"{spec['id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return 0
    return run(args.output, seconds=args.seconds, steps=args.steps, strain=args.strain,
               compression_regularization=args.compression_regularization)


if __name__ == '__main__':
    raise SystemExit(main())
