"""Independent saved finite-increment evidence audit; never starts loading."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from numerical_runtime import runtime_entry, add_thread_arguments, thread_arguments
from run_support import write_json
from .run_zero_gap_cut import problem, hashes
from prototype.finite_increment_cut import FrozenVirtualCrack, coupled_margin
from virtual_friction_work import interval_decision
from .virtual_work_audit import audit_work


def stored_equilibrium(p, x):
    p.set_primary_increment(x, x-p.previous)
    _, g, _, j = p.evaluate(x, p.lag(x), tangent=False)
    p.update_constraint_free(x, g)
    residual = np.sqrt(2)*float(np.linalg.norm(g[p.free]))
    if residual > 1e-4 or j <= 0 or not p.accepted_chord_is_safe(x):
        raise ValueError('Saved virtual equilibrium failed force/J/CCD')
    if (not np.array_equal(x[p.mesh.bottom_nodes], p.rest[p.mesh.bottom_nodes])
            or not np.array_equal(x[p.seam[~p.released], 0], p.rest[p.seam[~p.released], 0])):
        raise ValueError('Saved virtual boundary differs')
    return g, residual


def audit_evidence(p, state, check, fields, deadline=float('inf')):
    for key in ('seed_positions', 'before_positions', 'after_positions'):
        if not np.array_equal(fields[key][p.n:], state['positions'][p.n:]):
            raise ValueError('Finite-crack virtual blade position differs')
    p.previous = state['previous_positions'].copy()
    p.previous_velocity = state['previous_velocity'].copy()
    p.previous_material_metric = state['previous_material_metric'].copy()
    p.previous_history = state['previous_history'].copy()
    p.material_dt = float(state['material_dt_s'])
    p.released = fields['previous_released'].copy()
    p.rebuild_contact()
    p.friction = p.ipc.FrictionPotential(p.friction_velocity_epsilon*p.material_dt)
    seed = fields['seed_positions']
    seed_increment_present = 'seed_increment' in fields
    p.set_primary_increment(seed, fields.get('seed_increment', seed-p.previous))
    np.testing.assert_allclose(p.material_response(seed)['metric'], fields['frozen_metric'], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(p.cohesive_response(seed)['history'], fields['frozen_history'], rtol=0, atol=1e-12)
    _,gradient,_,seed_j = p.evaluate(seed, p.lag(seed), tangent=False)
    p.update_constraint_free(seed,gradient)
    seed_residual=np.sqrt(2)*float(np.linalg.norm(gradient[p.free]))
    if seed_residual>1e-4 or seed_j<=0 or not p.accepted_chord_is_safe(seed):
        raise ValueError('Actual insertion seed failed force/J/CCD')
    if (not np.array_equal(seed[p.mesh.bottom_nodes],p.rest[p.mesh.bottom_nodes])
            or not np.array_equal(seed[p.seam[~p.released],0],p.rest[p.seam[~p.released],0])):
        raise ValueError('Actual insertion seed boundary differs')
    np.testing.assert_array_equal(fields['pre_anchor'],seed)
    np.testing.assert_allclose(fields['depth_mm'],state['depth_mm'],rtol=0,atol=1e-14)
    g = gradient.reshape(-1,2)
    physical = -g[p.seam,0]/p.seam_area
    np.testing.assert_allclose(physical, fields['physical_traction'], rtol=1e-8, atol=1e-6)
    removed = fields['removed']
    if np.any(removed & fields['previous_released']):
        raise ValueError('Candidate is not virgin')
    indices = np.flatnonzero(removed)
    if (not len(indices) or np.any(np.diff(indices)!=1)
            or np.any(p.rest[p.seam[indices],1] <= 0)):
        raise ValueError('Candidate not a connected interior finite patch')
    np.testing.assert_array_equal(check['nodes'],p.seam[indices])
    np.testing.assert_array_equal(check['seam_indices'],indices)
    empty = np.zeros_like(removed)
    base = FrozenVirtualCrack(p, fields['pre_anchor'], fields['frozen_metric'], fields['frozen_history'],
        fields['previous_released'], empty, p.ffm_epsilon)
    before = fields['before_positions']
    bg, r0 = stored_equilibrium(base, before)
    post = FrozenVirtualCrack(p, before, fields['frozen_metric'], fields['frozen_history'],
        fields['previous_released'], removed, p.ffm_epsilon)
    after = fields['after_positions']
    _, r1 = stored_equilibrium(post, after)
    stress = np.minimum(physical, -bg.reshape(-1,2)[p.seam,0]/p.seam_area)
    u0, u1 = base.stored(before), post.stored(after)
    required = p.toughness*float(p.seam_area[indices].sum())
    independent = pre_independent = None
    if 'friction_integration' in check:
        independent = audit_work(post, before, after, deadline)
        pre_independent = audit_work(base, seed, before, deadline)
        recorded = check['friction_integration']
        if not recorded['converged'] or recorded['method'] != 'geometry_partitioned_adaptive_gauss_kronrod':
            raise ValueError('Accepted insertion lacks completed production quadrature')
        d9 = check['virtual_friction_work_n_mm']
        d17 = independent['work_n_mm']
        uncertainty = recorded['empirical_uncertainty_n_mm']
        if abs(d9-d17) > uncertainty+independent['empirical_uncertainty_n_mm']+1e-7:
            raise ValueError('Independent friction integration disagrees with production interval')
        decision = interval_decision(u0-u1, required, d17,
            independent['empirical_uncertainty_n_mm'], independent['converged'])
        refined = decision['status'] == 'accepted'
        refined_margin = decision['margin_lower_n_mm']
        passed, margin = coupled_margin(stress[indices],p.strength,u0-u1,d9,required,uncertainty)
        pre17 = pre_independent['work_n_mm']
        seed_stored = base.stored(seed)
        seed_margin = seed_stored-u1-pre17-d17-required-independent['empirical_uncertainty_n_mm']-pre_independent['empirical_uncertainty_n_mm']
    else:
        # Historical bundles retain their original arithmetic, marked separately.
        d5, d9, d17 = [post.friction_path_work(before, after, n) for n in (5,9,17)]
        pre5,pre9,pre17=[base.friction_path_work(seed,before,n) for n in (5,9,17)]
        seed_stored=base.stored(seed)
        passed, margin = coupled_margin(stress[indices],p.strength,u0-u1,d9,required,2*abs(d9-d5))
        refined, refined_margin = coupled_margin(stress[indices],p.strength,u0-u1,d17,required,2*abs(d17-d9))
        seed_margin=seed_stored-u1-pre17-d17-required-2*abs(pre17-pre9)-2*abs(d17-d9)
    for key, value in dict(virtual_stored_before_n_mm=u0, virtual_stored_after_n_mm=u1,
        virtual_friction_work_n_mm=d9, required_fracture_work_n_mm=required, energy_margin_n_mm=margin).items():
        np.testing.assert_allclose(check[key], value, rtol=1e-8, atol=1e-8)
    if not passed or not check['accepted']:
        raise ValueError('Stored accepted crack does not satisfy both conditions')
    return dict(passed=True, stronger_quadrature_passed=refined,
        independent_friction_integration=independent,pre_independent_friction_integration=pre_independent,
        quadrature_convergence_validated=independent is not None and independent['converged'],
        physical_seed_residual_n=seed_residual,physical_seed_min_j=seed_j,
        primary_seed_increment_saved=seed_increment_present, virtual_pre_residual_n=r0,
        virtual_post_residual_n=r1, energy_margin_n_mm=margin, refined_energy_margin_n_mm=refined_margin,
        virtual_pre_chord_certificate=base.virtual_chord_certificate,
        virtual_post_chord_certificate=post.virtual_chord_certificate,
        extra_quadrature_work_n_mm=d17, virtual_friction_work_n_mm=d9,
        seed_stored_n_mm=seed_stored,pre_virtual_stored_change_n_mm=u0-seed_stored,
        pre_virtual_friction_work_n_mm=pre17,
        seed_path_energy_margin_n_mm=seed_margin,seed_path_energy_supported=bool(seed_margin>=0 and (pre_independent is None or pre_independent['converged'])),
        reference_tributary_area_mm2=float(p.seam_area[indices].sum()))


@runtime_entry
def inspect(directory, seconds, *, runtime=None):
    if not np.isfinite(seconds) or not 1 <= seconds <= 3600:
        raise ValueError('Explicit audit budget 1..3600 s required')
    directory=Path(directory).resolve()
    path=directory/'result.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    source=hashes();start=time.monotonic()
    helper=Path(__file__).with_name('virtual_work_audit.py')
    helper_hash=hashlib.sha256(helper.read_bytes()).hexdigest()
    report=dict(schema='softcut-finite-increment-audit/1',loading_started=False,
        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_hashes=source,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        auditor_dependency_sha256={'tools/virtual_work_audit.py':helper_hash},
        producer_source_matches_current=source==result['source_hashes'],
        interpreter=sys.executable,budget_s=seconds,thread_runtime=runtime,passed=False,checks=[])
    p=problem(result['config']);p.ipc.set_num_threads(runtime['requested']['ipc_threads'])
    with threadpool_limits(limits=runtime['requested']['blas_threads'], user_api='blas'):
        try:
            for rec in result['steps']:
                if time.monotonic()-start >= seconds:raise TimeoutError('Finite-increment audit budget')
                with np.load(directory/rec['state_file'],allow_pickle=False) as data:
                    state={k:data[k].copy() for k in data.files}
                insertion=set(n for event in rec['insertion_events'] for n in event['nodes'])
                audited=set()
                for check in rec.get('finite_increment_checks',[]):
                    if not check['accepted']:continue
                    ep=(directory/check['evidence_file']).resolve()
                    if ep.parent!=directory or hashlib.sha256(ep.read_bytes()).hexdigest()!=check['evidence_sha256']:
                        raise ValueError('Virtual evidence identity differs')
                    with np.load(ep,allow_pickle=False) as data:fields={k:data[k].copy() for k in data.files}
                    audit=audit_evidence(p,state,check,fields,start+seconds)
                    audit.update(depth_mm=rec['depth_mm'],evidence_file=ep.name)
                    report['checks'].append(audit)
                    audited.update(check['nodes'])
                if insertion!=audited:raise ValueError('Physical insertion lacks paired finite-crack evidence')
            report['passed']=bool(result['steps']) and all(c['stronger_quadrature_passed'] for c in report['checks'])
        except (AssertionError,RuntimeError,ValueError,TimeoutError,KeyError) as exc:
            report['error']=str(exc)
    report['auditor_source_unchanged']=report['auditor_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['auditor_dependencies_unchanged']=helper_hash==hashlib.sha256(helper.read_bytes()).hexdigest()
    report['source_unchanged']=source==hashes() and report['auditor_source_unchanged'] and report['auditor_dependencies_unchanged']
    report['elapsed_s']=time.monotonic()-start
    report['passed'] &= report['source_unchanged']
    report['primary_seed_increment_complete']=all(c['primary_seed_increment_saved'] for c in report['checks'])
    report['seed_path_energy_support_complete']=all(c['seed_path_energy_supported'] for c in report['checks'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,required=True)
    add_thread_arguments(parser)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing audit')
    report=inspect(args.directory,args.seconds,**thread_arguments(args));write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','checks')}))


if __name__=='__main__':main()
