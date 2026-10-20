"""Bounded native P1 interior-refinement diagnostic with unchanged contact trace.

Every mesh starts with its own rest histories. A continuation may resume only
an identical signature/source prefix. This is local interior sensitivity, not
a mixed-element locking cure or a complete contact-mesh convergence claim.
"""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from prototype.ipc_viscoelastic import ViscoelasticProblem
from prototype.ipc_reference import BASELINE, adaptive_load_steps, record_tip_contact, export_contact_table
from prototype.tip_contact import TipContactTracker
from run_support import source_hashes, write_json, save_npz, compute_process_policy
from tools.run_material_candidate import (make_problem as baseline_problem, signature,
                                          audit, resume, resume_compatibility)


COMMON = dict(equilibrium_shear=.14, maxwell_shear=1.24, relaxation_s=1.7,
              loading_rate=.2, bulk_modulus=23.34333333333331, mu=.35)


def make_problem(angle, level=1, radius=1.):
    """Same native material/contact/acceptance as the completed angle queue."""
    if angle not in (0., 33., 57.) or level not in (0, 1, 2):
        raise ValueError('Require actual angle 0/33/57 and level 0/1/2')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Refinement radius must be finite and positive')
    p = ViscoelasticProblem(maxwell_shear=COMMON['maxwell_shear'],
        relaxation_s=COMMON['relaxation_s'], loading_rate=COMMON['loading_rate'],
        bulk_modulus=COMMON['bulk_modulus'], case='oblique', mu=COMMON['mu'],
        shear_modulus=COMMON['equilibrium_shear'], ogden_alpha=2., dhat=.0025,
        local_spacing=.05, surface_spacing=.005, nose_facets=64,
        inclination=angle, tip_radius=.02, fine_halfwidth=3., max_iterations=500,
        predictor='secant', hessian_mode='regularized', element='p1',
        interior_refinement_levels=level, interior_refinement_radius=radius)
    p.stagnation_window = 40
    return p


def boundary_edges(mesh):
    t = mesh.triangles
    edges, counts = np.unique(np.sort(np.vstack((t[:, [0, 1]], t[:, [1, 2]],
        t[:, [2, 0]])), axis=1), axis=0, return_counts=True)
    if np.max(counts) > 2:
        raise ValueError('Nonmanifold interior-refined mesh')
    return edges[counts == 1]


def geometry_audit(p, base, *, level, radius):
    """Exact trace, fixed support and physics checks; no equilibrium claim."""
    if p.config.to_dict() != base.config.to_dict() or p.material != base.material:
        raise ValueError('Refinement changed constitutive/configuration parameters')
    if (p.dhat != base.dhat or p.contact_formulation != base.contact_formulation
            or p.loading_rate != base.loading_rate or p.max_iterations != base.max_iterations):
        raise ValueError('Refinement changed contact/time/iteration settings')
    np.testing.assert_array_equal(p.mesh.reference_xy[:base.n], base.mesh.reference_xy)
    np.testing.assert_array_equal(p.mesh.bottom_nodes, base.mesh.bottom_nodes)
    np.testing.assert_array_equal(boundary_edges(p.mesh), boundary_edges(base.mesh))
    np.testing.assert_array_equal(p.rest[p.surface], base.rest[base.surface])
    np.testing.assert_array_equal(p.cmesh.edges, base.cmesh.edges)
    np.testing.assert_array_equal(p.rest[p.n:], base.rest[base.n:])
    np.testing.assert_array_equal(p.material_metric,
                                  np.tile(np.eye(3), (len(p.mesh.triangles), 1, 1)))
    np.testing.assert_array_equal(p.previous, p.rest)
    area = p.mesh.bulk_data[0]; reference_area = base.mesh.bulk_data[0].sum()
    if np.any(area <= 0):
        raise ValueError('Refined mesh contains inverted/zero-area triangle')
    np.testing.assert_allclose(area.sum(), reference_area, rtol=2e-14, atol=1e-11)
    tri = p.mesh.reference_xy[p.mesh.triangles]
    squared = sum(np.sum((tri[:, i]-tri[:, j])**2, axis=1) for i, j in ((0,1),(1,2),(2,0)))
    quality = 4*np.sqrt(3)*area/squared
    return dict(passed=True, level=level, radius_mm=radius,
        original_nodes=base.n, refined_nodes=p.n, original_elements=len(base.mesh.triangles),
        refined_elements=len(p.mesh.triangles), original_free_dofs=len(base.free),
        refined_free_dofs=len(p.free), boundary_edges=len(boundary_edges(p.mesh)),
        contact_coordinates_and_connectivity_identical=True, fixed_bottom_identical=True,
        blade_identical=True, material_identical=True, rest_histories_fresh=True,
        area_mm2=float(area.sum()), minimum_triangle_quality=float(quality.min()),
        scope='Interior-only local refinement; boundary-adjacent tangential resolution unchanged. '
              'This does not establish absence of volumetric locking or spatial convergence.')


def snapshot(out):
    hashes = source_hashes(packages=('prototype', 'mechanics'), snapshot=out/'source')
    root = Path(__file__).resolve().parents[1]
    for name in ('tools/check_native_mesh.py', 'tools/run_material_candidate.py',
                 'prototype/ipc_baseline.json'):
        target = out/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, target)
        hashes[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    return hashes


def validate_resume(previous, current):
    if previous.get('mesh_diagnostic') != current['mesh_diagnostic']:
        raise ValueError('Resume requires this exact interior-refinement diagnostic')
    return resume_compatibility(previous, current)


def run(args, *, problem_factory=None, geometry_checker=None, source_snapshot=None,
        problem_signature=None, diagnostic_metadata=None):
    """Shared bounded runner; explicit diagnostics may supply verified adapters."""
    factory = make_problem if problem_factory is None else problem_factory
    geometry = geometry_audit if geometry_checker is None else geometry_checker
    freeze = snapshot if source_snapshot is None else source_snapshot
    identify = signature if problem_signature is None else problem_signature
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'states').mkdir()
    report = dict(status='initializing', completed=False, steps=[],
        source_sha256=freeze(out), target_mm=args.depth, rejected_load_attempts=[],
        force_tolerance_n=BASELINE['force_tolerance_n'], tip_contact_tracking=dict(errors=[]),
        mesh_diagnostic=(diagnostic_metadata if diagnostic_metadata is not None else
            dict(level=args.level, radius_mm=args.radius,
                 fresh_start_required=True, no_cross_mesh_history_transfer=True)),
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(out/'result.json', report)
    p = None; x = None; began = time.monotonic()
    with (out/'run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), \
            threadpool_limits(args.threads), compute_process_policy() as policy:
        report['runtime_process_policy'] = policy
        try:
            p = factory(args.angle, args.level, args.radius)
            base = baseline_problem(angle=args.angle, **COMMON)
            report['geometry_audit'] = geometry(p, base, level=args.level, radius=args.radius)
            del base
            report['signature'] = json.loads(json.dumps(identify(p)))
            report['reference_contact_check'] = p.reference_contact_check
            report['mesh'] = dict(nodes=p.n, elements=len(p.mesh.triangles))
            x = p.rest.copy(); old = p.start
            if args.phase == 'geometry':
                report.update(status='geometry_passed', completed=False)
            else:
                if args.resume:
                    previous = json.loads((args.resume/'result.json').read_text(encoding='utf-8'))
                    validate_resume(previous, report)
                    x, old = resume(p, args.resume, out, report)
                tracker = TipContactTracker(p.rest, p.mesh.triangles, p.n, p.config, p.dhat, p.nose_facets)
                report['status'] = 'running'; write_json(out/'result.json', report)
                deadline = time.monotonic()+args.seconds
                targets = np.r_[0., np.arange(args.increment, args.depth-1e-12, args.increment), args.depth]
                def rejected(event):
                    report['rejected_load_attempts'].append(event)
                    write_json(out/'result.json', report)
                for depth in targets[targets > old+1e-12]:
                    for x, record in adaptive_load_steps(p, x, old, float(depth), deadline, .000625, rejected):
                        old = record['depth_mm']; index = len(report['steps'])
                        save_npz(out/'states'/f'{index:04d}.npz', reference_xy=p.rest,
                            triangles=p.mesh.triangles, positions=x, previous_positions=p.previous,
                            depth_mm=old, body_nodes=p.n, **p.checkpoint_data())
                        record_tip_contact(tracker, x, record, report)
                        report['steps'].append(record); report['last_depth_mm'] = old
                        write_json(out/'result.json', report)
                        print(f'ACCEPT depth_mm={old:.6f} Fz_N={record["reaction_fz_n"]:.9g} '
                              f'residual_N={record["residual_n"]:.4g}', flush=True)
                report.update(status='completed', completed=True)
        except Exception as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else 'failed',
                          error=f'{type(exc).__name__}: {exc}', completed=False)
            traceback.print_exc(file=log)
            if p is not None and x is not None and p.material_dt is not None:
                save_npz(out/'failed_trial.npz', positions=getattr(p, 'trial', x),
                         previous_positions=p.previous, **p.checkpoint_data())
        report['elapsed_s'] = time.monotonic()-began
        try:
            if p is not None and report['steps']:
                audit(p, out, report)
        except Exception as exc:
            report.update(status='audit_failed', completed=False,
                          audit_error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc(file=log)
    export_contact_table(out, report)
    with (out/'fz_depth.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['depth_mm', 'fz_n'])
        writer.writerows((r['depth_mm'], r['reaction_fz_n']) for r in report['steps'])
    write_json(out/'result.json', report)
    write_json(out/'outcome.json', {k: report.get(k) for k in
        ('status', 'completed', 'last_depth_mm', 'elapsed_s', 'error', 'audit_error')})
    print(json.dumps({k: report.get(k) for k in ('status', 'last_depth_mm', 'elapsed_s', 'error')}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('geometry', 'run'), default='geometry')
    parser.add_argument('--angle', type=float, choices=(0., 33., 57.), required=True)
    parser.add_argument('--level', type=int, choices=(0, 1, 2), default=1)
    parser.add_argument('--radius', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth', type=float, default=.01)
    parser.add_argument('--increment', type=float, default=.01)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if (not all(np.isfinite(v) and v > 0 for v in
                (args.radius, args.depth, args.increment, args.seconds)) or args.threads < 1):
        parser.error('Require positive finite radius/depth/increment/budget and positive threads')
    if args.depth > .5+1e-12:
        parser.error('This bounded diagnostic is limited to 0.5 mm')
    if args.resume and args.phase != 'run':
        parser.error('Resume is only available for a solve with identical mesh and histories')
    run(args)


if __name__ == '__main__':
    main()
