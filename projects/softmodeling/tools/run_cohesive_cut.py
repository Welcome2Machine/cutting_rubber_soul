"""Bounded explicit launcher for radial and mixed-mode cohesive cutting."""
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


def law_from_config(config):
    if config.get('cohesive_kind', 'radial_mode_i') == 'mixed_mode_energy':
        from prototype.cohesive_mixed import MixedModeLaw
        return MixedModeLaw(**config['cohesive'])
    from prototype.cohesive import CohesiveLaw
    return CohesiveLaw(**config['cohesive'])


def read_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if config.get('schema') not in {'softcut-cohesive/1', 'softcut-cohesive/2', 'softcut-cohesive/3'}:
        raise ValueError('Expected softcut-cohesive/1, /2 or /3 config')
    regularized = config.get('schema') == 'softcut-cohesive/3'
    mixed = config.get('schema') in {'softcut-cohesive/2', 'softcut-cohesive/3'}
    fine_width = config.get('mesh', {}).get('fine_halfwidth_mm')
    if fine_width is not None and (not regularized or isinstance(fine_width, bool)
            or not isinstance(fine_width, (int, float)) or not np.isfinite(fine_width) or fine_width <= 0):
        raise ValueError('Explicit fine_halfwidth_mm requires version 3 and a positive finite value')
    if mixed:
        if config.get('cohesive_kind') != 'mixed_mode_energy':
            raise ValueError('Mixed versions require explicit mixed_mode_energy')
        if config.get('fracture_topology') not in {'prescribed_seam', 'edge_network'}:
            raise ValueError('Choose prescribed_seam or edge_network explicitly')
        if config['fracture_topology'] == 'edge_network':
            floor=config.get('network_metric_floor_n_per_mm')
            if isinstance(floor, bool) or not isinstance(floor, (int,float)) or not np.isfinite(floor) or floor <= 0:
                raise ValueError('Declare a positive finite network numerical metric floor')
            if config.get('network_predictor') not in {'body_stationary_if_safe', 'baseline', 'safe_affine_blend'}:
                raise ValueError('Declare the network loading predictor policy explicitly')
        angle = config.get('angle_deg')
        if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not np.isfinite(angle) or not 0 <= angle < 90:
            raise ValueError('Require finite angle in [0,90) degrees')
        if config.get('no_experimental_force_fitting') is not True:
            raise ValueError('Explicit independent-parameter provenance declaration required')
        required = {'normal_strength_mpa', 'shear_strength_mpa', 'normal_stiffness_mpa_per_mm',
                    'toughness_n_per_mm', 'damage_enabled'}
        if not required <= set(config.get('cohesive', {})):
            raise ValueError('All mixed-mode material parameters must be explicit')
    elif config.get('angle_deg') != 0 or config.get('cohesive_kind', 'radial_mode_i') != 'radial_mode_i':
        raise ValueError('Legacy radial model requires 0 degrees')
    if config.get('loading_rate_mm_s') != .2:
        raise ValueError('Requires 0.2 mm/s')
    target, increment = config['target_mm'], config['increment_mm']
    if not np.isfinite([target, increment]).all() or not 0 < target <= 2 or not .000625 <= increment <= .05:
        raise ValueError('Require target (0,2] mm and increment [.000625,.05] mm')
    if config.get('parameter_status') != 'demonstration_only_not_experimentally_calibrated':
        raise ValueError('This model cannot claim experimentally calibrated fracture parameters')
    if regularized:
        if config.get('fracture_topology') != 'edge_network':
            raise ValueError('Version 3 currently requires edge_network')
        closure = config.get('closure_regularization')
        if closure not in {'reversible_compression_penalty', 'bounded_reversible_compression_penalty',
                           'finite_face_bounded_compression_penalty'}:
            raise ValueError('Version 3 requires explicit reversible compression regularization')
        length = config.get('cohesive', {}).get('compression_regularization_mm')
        finite_faces = closure == 'finite_face_bounded_compression_penalty'
        kinematics = config.get('cohesive', {}).get('compression_kinematics', 'projected_gap')
        if kinematics != ('finite_faces' if finite_faces else 'projected_gap'):
            raise ValueError('Closure mode and compression kinematics must match explicitly')
        if closure in {'bounded_reversible_compression_penalty', 'finite_face_bounded_compression_penalty'}:
            if (isinstance(length, bool) or not isinstance(length, (int, float))
                    or not np.isfinite(length) or length <= 0 or length != config['reference_gap_mm']):
                raise ValueError('Bounded compression length must equal the reference gap, not a fitted parameter')
        elif length is not None:
            raise ValueError('Linear compression mode must not specify a bounded length')
        for name, value in (
                ('compression stiffness', config.get('cohesive', {}).get('compression_stiffness_mpa_per_mm')),
                ('preparation clearance', config.get('preparation_clearance_mm')),
                ('CCD tolerance', config.get('ccd_tolerance_mm')),
                ('minimum increment', config.get('minimum_increment_mm'))):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'Declare positive finite {name}')
        if config['minimum_increment_mm'] > increment:
            raise ValueError('Minimum increment must not exceed nominal increment')
        if config['ccd_tolerance_mm'] > min(1e-6, config['reference_gap_mm']/10, config['dhat_mm']/10):
            raise ValueError('CCD tolerance must resolve gap/contact scale and never exceed legacy tolerance')
        backtracks = config.get('ptc_model_backtracks', 0)
        if type(backtracks) is not int or not 0 <= backtracks <= 30:
            raise ValueError('ptc_model_backtracks must be an integer in [0,30]')
        guard = config.get('ptc_residual_guard', False)
        if type(guard) is not bool or guard and backtracks == 0:
            raise ValueError('ptc_residual_guard must be boolean and requires positive backtracks')
        shift_policy = config.get('ptc_shift_policy', 'residual_ratio')
        if (not isinstance(shift_policy, str) or shift_policy not in {'residual_ratio', 'backtrack_aware'}
                or shift_policy == 'backtrack_aware' and backtracks == 0):
            raise ValueError('ptc_shift_policy requires residual_ratio or backtrack_aware with positive backtracks')
    elif (config.get('closure_regularization') is not None
          or config.get('cohesive', {}).get('compression_stiffness_mpa_per_mm', 0.) != 0.
          or config.get('cohesive', {}).get('compression_regularization_mm') is not None
          or config.get('ptc_model_backtracks', 0) != 0
          or config.get('ptc_residual_guard', False) is not False
          or config.get('ptc_shift_policy', 'residual_ratio') != 'residual_ratio'
          or 'minimum_increment_mm' in config):
        raise ValueError('Compression regularization or custom minimum increment requires version 3')
    solver = config.get('nonlinear_solver', 'ptc')
    if solver not in ('ptc', 'newton_lm') or (solver != 'ptc' and not regularized):
        raise ValueError('newton_lm requires version 3')
    law_from_config(config)
    return config


def gauss_centers(positions, segments):
    xi = np.array([-1., 1.])/np.sqrt(3.)
    weights = np.column_stack(((1-xi)/2, (1+xi)/2))
    ends = positions[segments].reshape(-1, 2, 2, 2).mean(axis=2)
    return np.einsum('ga,sad->sgd', weights, ends)


def saved_damage(state, law):
    maximum = state['cohesive_maximum']
    if maximum.ndim == 2:
        return law.history_response(maximum)['damage']
    return law.response(maximum, maximum)['damage']


def regularization_scales(problem, config):
    """Dimensionless diagnostics, not a substitute for a full-field comparison."""
    if not problem.is_mixed_mode:
        return None
    lengths = np.asarray(problem.interface_lengths)
    if not lengths.size:
        return None
    material = config['material']
    shear = material['equilibrium_shear_mpa']+material['maxwell_shear_mpa']
    bulk = material['bulk_mpa']
    young = 9*bulk*shear/(3*bulk+shear)
    poisson = (3*bulk-2*shear)/(2*(3*bulk+shear))
    plane_strain = young/(1-poisson**2)
    law = problem.cohesive_law
    h = float(np.min(lengths))
    return dict(reference_edge_min_mm=h, reference_edge_median_mm=float(np.median(lengths)),
        maximum_gap_over_reference_edge=config['reference_gap_mm']/h,
        instantaneous_young_mpa=young, instantaneous_plane_strain_scale_mpa=plane_strain,
        maximum_Eprime_over_Kn_h=plane_strain/(law.normal_stiffness_mpa_per_mm*h),
        Kc_gap_over_normal_strength=law.compression_stiffness_mpa_per_mm*config['reference_gap_mm']/law.normal_strength_mpa,
        compression_traction_bound_mpa=(None if law.compression_regularization_mm is None else
                                        law.compression_stiffness_mpa_per_mm*law.compression_regularization_mm),
        compression_bound_scope='paired constitutive normal traction only; not a nodal-force, total contact-load or friction-error bound',
        cohesive_characteristic_length_mm=plane_strain*law.toughness_n_per_mm/law.normal_strength_mpa**2,
        qualification='small-strain dimensional scales only; no universal network compliance or process-zone error bound',
        compression_ipc_normal_load_share=None,
        load_partition_status='unavailable; compression internal-force norm is not a contact normal-load share')


def events(steps, output=None, law=None):
    result = {}
    mixed = law is not None and hasattr(law, 'history_response')
    for name, predicate in (
        ('damage_onset', lambda s: s['fracture']['max_damage'] > 1e-8),
        ('first_fully_separated_segment' if mixed else 'first_fully_separated_bond',
         lambda s: s['fracture']['separated_bonds'] > 0),
        ('blade_in_surface_connected_crack', lambda s: s['fracture'].get('surface_connected_crack_length_mm',0) > 0
         and s['fracture'].get('blade_surface_connected_crack_contacts',0) > 0)):
        i = next((i for i, s in enumerate(steps) if predicate(s)), None)
        result[name] = dict(status='not_observed_in_accepted_prefix') if i is None else dict(
            status='observed', state_index=i, depth_interval_mm=[steps[i-1]['depth_mm'] if i else None,
                                                              steps[i]['depth_mm']],
            fz_n=steps[i]['reaction_fz_n'])
        if i is not None and output is not None:
            with np.load(output/'states'/f'{i:04d}.npz') as state:
                damage=saved_damage(state, law)
                if damage.ndim == 2:
                    mask = damage > 1e-8 if name == 'damage_onset' else np.broadcast_to(
                        np.all(damage >= 1-1e-10, axis=1)[:,None], damage.shape)
                    segments = state['interface_segments']
                    result[name]['interface_segment_and_gauss_ids']=np.argwhere(mask).tolist()
                    result[name]['reference_midpoints_mm']=gauss_centers(state['reference_xy'], segments)[mask].tolist()
                    result[name]['current_midpoints_mm']=gauss_centers(state['positions'], segments)[mask].tolist()
                    continue
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
                law = law_from_config(report['configuration'])
                damage = saved_damage(a, law)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.triplot(x[:n, 0], x[:n, 1], tri, lw=.35, color='#8996a0')
                blade = x[n:];ax.fill(blade[:, 0], blade[:, 1], color='#37474f')
                centers = (gauss_centers(x, a['interface_segments']).reshape(-1,2)
                           if damage.ndim == 2 else x[pairs].mean(axis=1))
                points = ax.scatter(centers[:, 0], centers[:, 1], c=damage.ravel(), cmap='inferno', vmin=0, vmax=1, s=18)
                fig.colorbar(points, ax=ax, label='Accepted interface damage')
                tip = blade[np.argmin(blade[:,1])]
                ax.set(xlim=(tip[0]-1.3, tip[0]+1.3), ylim=(tip[1]-1., tip[1]+1.1), xlabel='y (mm)', ylabel='z (mm)',
                       title=f"Accepted state: d={steps[i]['depth_mm']:.4f} mm")
                ax.set_aspect('equal');fig.tight_layout()
                name=f'state_{i:04d}.svg';fig.savefig(out/name);plt.close(fig);images.append(name)
    text = '''<!doctype html><meta charset="utf-8"><title>Cohesive cutting evidence</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px}img{max-width:100%}pre{white-space:pre-wrap;background:#f3f5f6;padding:16px}</style>
<h1>Cohesive cutting: numerical research evidence</h1><p>30 × 5 mm section, width 20 mm; 0.2 mm/s. Independently prescribed demonstration parameters; no inverse fitting of experimental cutting-force data. Mechanical qualification does not certify a physical initiation depth.</p>
<p>The initial gap is numerical geometry and is subtracted from constitutive opening. Full IPC and CCD remain active on both new faces. Cohesive damage and Maxwell history are committed only for accepted steps. The figure scale is 1.</p>'''
    summary = {key: report.get(key) for key in ('status', 'completed', 'configuration', 'assumptions', 'mesh', 'last_depth_mm', 'error', 'audit_error', 'events', 'verification', 'energy_accounting')}
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
        compression=fa.get('compression_stored_energy_n_mm',0.)-fb.get('compression_stored_energy_n_mm',0.)
        viscous=a['cumulative_viscous_dissipation_n_mm']-b['cumulative_viscous_dissipation_n_mm']
        fracture=fa['fracture_dissipation_n_mm']-fb['fracture_dissipation_n_mm']
        friction=after['friction_work_n_mm']
        barrier=fa['ipc_barrier_stored_energy_n_mm']-fb['ipc_barrier_stored_energy_n_mm']
        residual=external-eq-neq-cohesive-viscous-fracture-friction-barrier
        motion=fa['max_body_step_motion_mm']
        rows.append(dict(from_depth_mm=before['depth_mm'], to_depth_mm=after['depth_mm'],
            external_work_n_mm=external, equilibrium_stored_change_n_mm=eq,
            nonequilibrium_stored_change_n_mm=neq, cohesive_stored_change_n_mm=cohesive,
            compression_stored_change_n_mm=compression,
            damageable_cohesive_stored_change_n_mm=cohesive-compression,
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
    if config.get('cohesive_kind') == 'mixed_mode_energy':
        report['assumptions'] = [
            'objective corotational four-node mixed-mode interface with two Gauss points and reference-area energy',
            'one shared Gc; Ks=Kn*(Ts/Tn)^2; no independently specified GIc/GIIc',
            'rate-independent interface coupled to finite-strain Maxwell bulk with physical time',
            'finite numerical gaps; not exactly the original intact continuum',
            'full IPC/CCD and additive Coulomb contact/friction on every exposed interface face',
            ('all interior mesh edges are candidate cracks; path emerges on this discrete network, with mesh-direction bias'
             if config['fracture_topology'] == 'edge_network' else 'single prescribed reference seam; not free-path selection'),
            'no cutting-force inverse fitting; no threshold selection from reported no-fracture observations',
            'quasistatic branch jumps require independent energy qualification; no hidden stabilizing dissipation']
        if config.get('schema') == 'softcut-cohesive/3':
            report['assumptions'].extend([
                'reversible compression penalty acts relative to the positive reference gap and persists after failure; not an intact-equivalence proof',
                'IPC Coulomb friction uses IPC normal loads only, not the additional reversible compression force; contact-load partition remains unqualified'])
    write_json(out/'result.json', report)
    started = time.monotonic();p = None;x = None
    with (out/'run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), threadpool_limits(limits=1):
        try:
            p = make_cohesive_problem(config)
            physical_signature=signature(p)
            physical_signature['fracture']=dict(kind=config.get('cohesive_kind','radial_mode_i'),
                topology=config.get('fracture_topology','prescribed_seam'), law=p.cohesive_law.to_dict(),
                history_shape=list(p.cohesive_maximum.shape), reference_gap_mm=config['reference_gap_mm'],
                preparation_clearance_mm=config.get('preparation_clearance_mm'), initial_depth_mm=p.start)
            report['solver_policy']=dict(network_metric_floor_n_per_mm=config.get('network_metric_floor_n_per_mm',0.),
                network_predictor=config.get('network_predictor','baseline'),
                minimum_increment_mm=config.get('minimum_increment_mm', .000625),
                ccd_tolerance_mm=config.get('ccd_tolerance_mm'),
                ptc_model_backtracks=config.get('ptc_model_backtracks', 0),
                ptc_residual_guard=config.get('ptc_residual_guard', False),
                ptc_shift_policy=config.get('ptc_shift_policy', 'residual_ratio'),
                role='iteration-direction regularization only; not a force, energy or physical dissipation')
            report.update(signature=physical_signature, reference_contact_check=p.reference_contact_check,
                          mesh=dict(nodes=p.n, elements=len(p.mesh.triangles)),
                          cohesive_law=p.cohesive_law.to_dict(),
                          reference_interface_area_mm2=float((p.interface_areas if p.is_mixed_mode else p.bond_areas).sum()),
                          fracture_topology=getattr(p, 'fracture_topology', 'prescribed_seam'),
                          fracture_topology_metadata=getattr(p, 'fracture_topology_metadata', {}))
            report['regularization_scales']=regularization_scales(p, config)
            x=p.rest.copy();old=p.start;deadline=started+solve_budget
            targets=np.r_[0., np.arange(config['increment_mm'], config['target_mm']-1e-12,
                                       config['increment_mm']), config['target_mm']]
            def rejected(event):
                report['rejected_load_attempts'].append(event)
                write_json(out/'result.json', report)
            report['status']='running'
            for depth in targets:
                for x, record in adaptive_load_steps(p, x, old, float(depth), deadline,
                                                    config.get('minimum_increment_mm', .000625), rejected):
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
                previous=np.zeros_like(fresh.cohesive_maximum)
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
        compression=last['fracture'].get('compression_stored_energy_n_mm',0.)-first['fracture'].get('compression_stored_energy_n_mm',0.)
        viscous=last['material']['cumulative_viscous_dissipation_n_mm']-first['material']['cumulative_viscous_dissipation_n_mm']
        fracture=last['fracture']['fracture_dissipation_n_mm']-first['fracture']['fracture_dissipation_n_mm']
        friction=sum(s['friction_work_n_mm'] for s in steps[1:])
        report['energy_accounting']=dict(reference='first accepted state; preload history retained separately',
            external_work_trapezoid_n_mm=work, bulk_stored_change_n_mm=bulk,
            equilibrium_stored_change_n_mm=last['material']['equilibrium_energy_n_mm']-first['material']['equilibrium_energy_n_mm'],
            nonequilibrium_stored_change_n_mm=last['material']['overstress_energy_n_mm']-first['material']['overstress_energy_n_mm'],
            cohesive_stored_change_n_mm=cohesive, viscous_dissipation_n_mm=viscous,
            compression_stored_change_n_mm=compression,
            damageable_cohesive_stored_change_n_mm=cohesive-compression,
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
        scale=max(abs(work), abs(bulk)+abs(cohesive)+abs(viscous)+abs(fracture)+abs(friction), 1e-12)
        remainder=report['energy_accounting']['unaccounted_energy_after_barrier_n_mm']
        report['energy_accounting']['relative_energy_remainder']=abs(remainder)/scale
        report['energy_accounting']['diagnostic_tolerance']=.02
        report['energy_accounting']['within_diagnostic_tolerance']=bool(ledger and abs(remainder)/scale <= .02)
        report['energy_accounting']['discretization_converged']=False
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
