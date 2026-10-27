"""Bounded explicit launcher for the finite-gap radial cohesive cutting case."""
import argparse
import contextlib
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.cohesive_cut import make_cohesive_problem
from prototype.ipc_reference import adaptive_load_steps
from prototype.material_case import signature, audit, failure_checkpoint
from run_support import write_json, save_npz, source_hashes


def read_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if config.get('schema') != 'softcut-cohesive/1':
        raise ValueError('Expected softcut-cohesive/1 config')
    if config.get('angle_deg') != 0 or config.get('loading_rate_mm_s') != .2:
        raise ValueError('This restricted model requires 0 degrees and 0.2 mm/s')
    target, increment = config['target_mm'], config['increment_mm']
    if not np.isfinite([target, increment]).all() or not 0 < target <= 2 or not .000625 <= increment <= .05:
        raise ValueError('Require target (0,2] mm and increment [.000625,.05] mm')
    if config.get('parameter_status') != 'demonstration_only_not_experimentally_calibrated':
        raise ValueError('This model cannot claim experimentally calibrated fracture parameters')
    return config


def events(steps, output=None, law=None):
    result = {}
    for name, predicate in (
        ('damage_onset', lambda s: s['fracture']['max_damage'] > 1e-8),
        ('first_fully_separated_bond', lambda s: s['fracture']['separated_bonds'] > 0),
        ('blade_in_surface_connected_crack', lambda s: s['fracture'].get('surface_connected_crack_length_mm',0) > 0
         and s['fracture'].get('blade_surface_connected_crack_contacts',0) > 0)):
        i = next((i for i, s in enumerate(steps) if predicate(s)), None)
        result[name] = dict(status='not_observed_in_accepted_prefix') if i is None else dict(
            status='observed', state_index=i, depth_interval_mm=[steps[i-1]['depth_mm'] if i else None,
                                                              steps[i]['depth_mm']],
            fz_n=steps[i]['reaction_fz_n'])
        if i is not None and output is not None:
            with np.load(output/'states'/f'{i:04d}.npz') as state:
                maximum=state['cohesive_maximum']
                damage=law.response(maximum, maximum)['damage']
                mask=damage > 1e-8 if name == 'damage_onset' else damage >= 1-1e-10
                pairs=state['seam_pairs'][mask]
                result[name]['bond_ids']=np.flatnonzero(mask).tolist()
                result[name]['reference_midpoints_mm']=state['reference_xy'][pairs].mean(axis=1).tolist()
                result[name]['current_midpoints_mm']=state['positions'][pairs].mean(axis=1).tolist()
    return result


def render(out, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps = report['steps']
    images = []
    if steps:
        depth = np.array([s['depth_mm'] for s in steps])
        force = np.array([s['reaction_fz_n'] for s in steps])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].plot(depth, force, 'o-', ms=3)
        if not report.get('verification', {}).get('accepted_prefix_audited'):
            axes[0].set_title('UNQUALIFIED: diagnostic accepted history only')
        axes[0].set(xlabel='Vertical blade travel (mm)', ylabel='Raw Fz (N)')
        axes[1].plot(depth, [s['fracture']['max_damage'] for s in steps], label='Maximum damage')
        axes[1].plot(depth, [s['fracture']['separated_reference_length_mm'] for s in steps], label='Separated length (mm)')
        axes[1].set(xlabel='Vertical blade travel (mm)');axes[1].legend()
        fig.tight_layout();fig.savefig(out/'curves.svg');plt.close(fig);images.append('curves.svg')
        for i in np.unique(np.linspace(0, len(steps)-1, min(4, len(steps))).astype(int)):
            with np.load(out/'states'/f'{i:04d}.npz') as a:
                x, tri, n, pairs = a['positions'], a['triangles'], int(a['body_nodes']), a['seam_pairs']
                maximum, gaps = a['cohesive_maximum'], a['reference_bond_lengths_mm']
                from prototype.cohesive import CohesiveLaw
                law = CohesiveLaw(**report['configuration']['cohesive'])
                damage = law.response(maximum, maximum)['damage']
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.triplot(x[:n, 0], x[:n, 1], tri, lw=.35, color='#8996a0')
                blade = x[n:];ax.fill(blade[:, 0], blade[:, 1], color='#37474f')
                centers = x[pairs].mean(axis=1)
                points = ax.scatter(centers[:, 0], centers[:, 1], c=damage, cmap='inferno', vmin=0, vmax=1, s=18)
                fig.colorbar(points, ax=ax, label='Accepted bond damage')
                ax.set(xlim=(6.3, 7.7), ylim=(4.1, 5.2), xlabel='y (mm)', ylabel='z (mm)',
                       title=f"Accepted state: d={steps[i]['depth_mm']:.4f} mm")
                ax.set_aspect('equal');fig.tight_layout()
                name=f'state_{i:04d}.svg';fig.savefig(out/name);plt.close(fig);images.append(name)
    text = '''<!doctype html><meta charset="utf-8"><title>Cohesive cutting evidence</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px}img{max-width:100%}pre{white-space:pre-wrap;background:#f3f5f6;padding:16px}</style>
<h1>Finite-gap radial cohesive cutting</h1><p>30 × 5 mm section, width 20 mm; 0°; 0.2 mm/s. Prescribed reference seam, demonstration parameters. This is a restricted mode-I interface, not an experimentally validated general mixed-mode fracture model.</p>
<p>The initial gap is numerical geometry and is subtracted from constitutive opening. Full IPC and CCD remain active on both new faces. Cohesive damage and Maxwell history are committed only for accepted steps. The figure scale is 1.</p>'''
    summary = {key: report.get(key) for key in ('status', 'completed', 'last_depth_mm', 'error', 'audit_error', 'events', 'verification', 'energy_accounting')}
    text += '<pre>'+html.escape(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre>'
    text += ''.join('<p><img src="'+name+'"></p>' for name in images)
    text += '<p><a href="result.json">Full metadata</a> · <a href="curves.csv">Accepted curves</a> · <a href="audit.json">Independent history/equilibrium audit</a></p>'
    (out/'index.html').write_text(text, encoding='utf-8')


def energy_ledger(steps):
    """Explicit increment work account; a branch jump is not silently dissipated."""
    rows=[]
    for before, after in zip(steps[:-1], steps[1:]):
        b, a=before['material'], after['material']
        fb, fa=before['fracture'], after['fracture']
        dd=after['depth_mm']-before['depth_mm']
        external=.5*(before['reaction_fz_n']+after['reaction_fz_n'])*dd
        eq=a['equilibrium_energy_n_mm']-b['equilibrium_energy_n_mm']
        neq=a['overstress_energy_n_mm']-b['overstress_energy_n_mm']
        cohesive=fa['cohesive_stored_energy_n_mm']-fb['cohesive_stored_energy_n_mm']
        viscous=a['cumulative_viscous_dissipation_n_mm']-b['cumulative_viscous_dissipation_n_mm']
        fracture=fa['fracture_dissipation_n_mm']-fb['fracture_dissipation_n_mm']
        friction=after['friction_work_n_mm']
        barrier=fa['ipc_barrier_stored_energy_n_mm']-fb['ipc_barrier_stored_energy_n_mm']
        residual=external-eq-neq-cohesive-viscous-fracture-friction-barrier
        motion=fa['max_body_step_motion_mm']
        rows.append(dict(from_depth_mm=before['depth_mm'], to_depth_mm=after['depth_mm'],
            external_work_n_mm=external, equilibrium_stored_change_n_mm=eq,
            nonequilibrium_stored_change_n_mm=neq, cohesive_stored_change_n_mm=cohesive,
            viscous_dissipation_n_mm=viscous, fracture_dissipation_n_mm=fracture,
            friction_work_n_mm=friction, ipc_barrier_stored_change_n_mm=barrier,
            unaccounted_energy_n_mm=residual, force_change_n=after['reaction_fz_n']-before['reaction_fz_n'],
            max_body_step_motion_mm=motion, displacement_amplification=motion/dd,
            large_displacement_step_diagnostic=motion/dd > 10))
    return rows


def run(config_path, output, seconds):
    config = read_config(config_path)
    if not np.isfinite(seconds) or not 1 <= seconds <= 3600:
        raise ValueError('seconds must be in [1,3600]')
    out = Path(output).resolve();out.mkdir(parents=True, exist_ok=False);(out/'states').mkdir()
    write_json(out/'config.json', config)
    sources = source_hashes(packages=('prototype', 'mechanics'))
    sources['tools/run_cohesive_cut.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    reserve=min(60., max(5., .2*seconds), .5*seconds)
    solve_budget=seconds-reserve
    report = dict(schema='softcut-cohesive-result/1', configuration=config,
        status='initializing', completed=False, steps=[], source_sha256=sources,
        interpreter=sys.executable, input_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        execution_budget_s=seconds, solve_budget_s=solve_budget, reserved_audit_budget_s=reserve,
        budget_scope='construction, loading and audit within execution budget; bounded plotting/serialization uses launcher grace',
        target_mm=config['target_mm'], rejected_load_attempts=[],
        parameter_status=config['parameter_status'],
        assumptions=['prescribed vertical reference seam at drop_s=7 mm, only 0 degrees',
          'finite reference void strip: not a zero-thickness intact specimen',
          'objective radial opening bonds; ZERO linear shear stiffness at the stress-free reference, no independent mode-II law',
          '0-degree geometry is not globally symmetric: drop_s=7 mm in a 30 mm specimen',
          'Coulomb IPC on every boundary including both seam faces; no adhesive blade contact',
          'coarse mechanism mesh; requires gap, penalty, space/time convergence before quantitative prediction'])
    write_json(out/'result.json', report)
    started = time.monotonic();p = None;x = None
    with (out/'run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), threadpool_limits(limits=1):
        try:
            p = make_cohesive_problem(config)
            report.update(signature=signature(p), reference_contact_check=p.reference_contact_check,
                          mesh=dict(nodes=p.n, elements=len(p.mesh.triangles)),
                          cohesive_law=p.cohesive_law.to_dict(),
                          reference_interface_area_mm2=float(p.bond_areas.sum()))
            x=p.rest.copy();old=p.start;deadline=started+solve_budget
            targets=np.r_[0., np.arange(config['increment_mm'], config['target_mm']-1e-12,
                                       config['increment_mm']), config['target_mm']]
            def rejected(event):
                report['rejected_load_attempts'].append(event)
                write_json(out/'result.json', report)
            report['status']='running'
            for depth in targets:
                for x, record in adaptive_load_steps(p, x, old, float(depth), deadline, .000625, rejected):
                    i=len(report['steps']);old=record['depth_mm']
                    save_npz(out/'states'/f'{i:04d}.npz', reference_xy=p.rest,
                        triangles=p.mesh.triangles, positions=x, previous_positions=p.previous,
                        depth_mm=old, body_nodes=p.n, **p.checkpoint_data())
                    report['steps'].append(record);report['last_depth_mm']=old
                    write_json(out/'result.json', report)
                    print(f'ACCEPT depth={old:.6g} Fz={record["reaction_fz_n"]:.6g} damage={record["fracture"]["max_damage"]:.6g}', flush=True)
            report.update(status='completed', completed=True)
        except Exception as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else 'failed',
                          error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc(file=log)
            if p is not None and x is not None and p.material_dt is not None:
                failed, context=failure_checkpoint(p, x)
                save_npz(out/'failed_trial.npz', **failed);report['failed_attempt']=context
        if p is not None and report['steps']:
            try:
                # Reconstruct a new problem; never certify by merely reusing Newton residuals.
                fresh = make_cohesive_problem(config)
                audit_report=audit(fresh, out, report, destination=out/'audit.pending.json', deadline=started+seconds)
                previous=np.zeros(len(fresh.bond_lengths))
                for i, row in enumerate(report['steps']):
                    with np.load(out/'states'/f'{i:04d}.npz') as state:
                        np.testing.assert_array_equal(state['previous_cohesive_maximum'], previous)
                        previous=state['cohesive_maximum'].copy()
                audit_report['cohesive_history_chain_passed']=True
                write_json(out/'audit.json', audit_report)
                (out/'audit.pending.json').unlink(missing_ok=True)
                report['verification']=dict(accepted_prefix_audited=True, accepted_states=len(report['steps']),
                    scope='fresh residual, positive J, IPC intersection, Maxwell and cohesive history; not experimental or discretization validation')
            except Exception as exc:
                report.update(status='audit_failed', completed=False, audit_error=f'{type(exc).__name__}: {exc}')
                report['verification']=dict(accepted_prefix_audited=False, qualified_states=0)
                write_json(out/'audit.json', dict(passed=False, cohesive_history_chain_passed=False,
                    qualified_prefix_states=0, rows=[], error=report['audit_error']))
                (out/'audit.pending.json').unlink(missing_ok=True)
                traceback.print_exc(file=log)
    qualified=bool(report.get('verification',{}).get('accepted_prefix_audited'))
    report['diagnostic_events']=events(report['steps'], out, p.cohesive_law if p is not None else None)
    report['events']=report['diagnostic_events'] if qualified else {
        name:dict(status='unavailable_unqualified_history') for name in report['diagnostic_events']}
    report['event_qualification']='independently_audited_prefix' if qualified else 'unqualified'
    report['elapsed_s']=time.monotonic()-started
    steps=report['steps'];work=0.
    for before, after in zip(steps[:-1], steps[1:]):
        work += .5*(before['reaction_fz_n']+after['reaction_fz_n'])*(after['depth_mm']-before['depth_mm'])
    if steps:
        first, last=steps[0], steps[-1]
        bulk=last['material']['stored_energy_n_mm']-first['material']['stored_energy_n_mm']
        cohesive=last['fracture']['cohesive_stored_energy_n_mm']-first['fracture']['cohesive_stored_energy_n_mm']
        viscous=last['material']['cumulative_viscous_dissipation_n_mm']-first['material']['cumulative_viscous_dissipation_n_mm']
        fracture=last['fracture']['fracture_dissipation_n_mm']-first['fracture']['fracture_dissipation_n_mm']
        friction=sum(s['friction_work_n_mm'] for s in steps[1:])
        report['energy_accounting']=dict(reference='first accepted state; preload history retained separately',
            external_work_trapezoid_n_mm=work, bulk_stored_change_n_mm=bulk,
            equilibrium_stored_change_n_mm=last['material']['equilibrium_energy_n_mm']-first['material']['equilibrium_energy_n_mm'],
            nonequilibrium_stored_change_n_mm=last['material']['overstress_energy_n_mm']-first['material']['overstress_energy_n_mm'],
            cohesive_stored_change_n_mm=cohesive, viscous_dissipation_n_mm=viscous,
            fracture_dissipation_n_mm=fracture, friction_work_n_mm=friction,
            remainder_n_mm=work-bulk-cohesive-viscous-fracture-friction,
            remainder_note='This pre-barrier subtotal is supplemented by the explicit increment ledger. Unaccounted release at an equilibrium branch jump is not assigned to friction or fracture.',
            energy_balance_qualified=False,
            qualification_note='Quasistatic equilibrium only. No inertial or physical jump-dissipation evolution is implemented; large snap releases require a dynamic/energetic continuation before quantitative post-fracture prediction.')
        ledger=energy_ledger(steps)
        with (out/'energy_ledger.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            if ledger:
                writer=csv.DictWriter(stream, list(ledger[0]));writer.writeheader();writer.writerows(ledger)
        report['energy_accounting']['ipc_barrier_change_n_mm']=sum(row['ipc_barrier_stored_change_n_mm'] for row in ledger)
        report['energy_accounting']['unaccounted_energy_after_barrier_n_mm']=sum(row['unaccounted_energy_n_mm'] for row in ledger)
        report['energy_accounting']['largest_unaccounted_increment']=max(ledger, key=lambda row: abs(row['unaccounted_energy_n_mm'])) if ledger else None
        report['energy_accounting']['large_displacement_steps']=[row for row in ledger if row['large_displacement_step_diagnostic']]
    write_json(out/'result.json', report)
    with (out/'curves.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields=['depth_mm','reaction_fz_n','residual_n','min_j','max_damage','separated_bonds',
                'separated_reference_length_mm','fracture_dissipation_n_mm','cohesive_stored_energy_n_mm',
                'minimum_paired_face_distance_mm']
        fields+=['blade_new_face_ev_contacts','blade_new_face_vv_contacts']
        fields+=['blade_fully_separated_face_contacts','blade_surface_connected_crack_contacts',
                 'surface_connected_crack_length_mm','qualified','raw_reaction_fz_n']
        writer=csv.DictWriter(stream, fields);writer.writeheader()
        for step in steps:
            row=dict(step, **step['fracture'], qualified=qualified, raw_reaction_fz_n=step['reaction_fz_n'])
            writer.writerow({k:row[k] if qualified or k in ('depth_mm','qualified','raw_reaction_fz_n') else None for k in fields})
    render(out, report)
    print(json.dumps(dict(output=str(out), status=report['status'], completed=report['completed'],
                          accepted_states=len(steps), events=report['events']), ensure_ascii=False))
    return 0 if report['completed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=300.)
    args=parser.parse_args()
    return run(args.config, args.output, args.seconds)


if __name__=='__main__':
    raise SystemExit(main())
