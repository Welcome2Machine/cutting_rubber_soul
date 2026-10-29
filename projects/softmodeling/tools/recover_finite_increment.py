"""Recover only logged, completely written accepted states after an explicit stop.

Rebuild physical records and accepted virtual-crack evidence with unchanged
producer equations. Rejected candidate details and iteration counts are lost;
the recovered interrupted prefix is never labelled a completed trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numerical_runtime import runtime_entry, add_thread_arguments, thread_arguments
from virtual_friction_work import virtual_work, interval_decision
from run_support import write_json
from . import run_zero_gap_cut as driver
from .audit_finite_increment import stored_equilibrium, audit_evidence
from .audit_zero_gap_cut import inspect
from prototype.finite_increment_cut import FrozenVirtualCrack


@runtime_entry
def recover(directory,log,seconds,*,runtime=None):
    if not 1<=seconds<=3600:raise ValueError('Explicit recovery audit budget required')
    directory=directory.resolve();start=time.monotonic();source=driver.hashes()
    if (directory/'result.json').exists():raise FileExistsError('Never overwrite producer result')
    result=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    stopped=json.loads((directory/'stop.json').read_text(encoding='utf-8'))
    if source!=result['source_hashes'] or source!=stopped['source_hashes']:
        raise ValueError('Recovery requires unchanged producer equations')
    prefix=len(result['steps'])
    logged=[json.loads(line[len('ACCEPTED '):]) for line in log.read_text().splitlines() if line.startswith('ACCEPTED ')]
    p=driver.problem(result['config']);p.ipc.set_num_threads(runtime['requested']['ipc_threads'])
    for index,recorded in enumerate(logged,prefix):
        if time.monotonic()-start>seconds:raise TimeoutError('Recovery audit budget')
        path=directory/f'state_{index:05d}.npz'
        with np.load(path,allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
        for a,b in [('previous','previous_positions'),('previous_velocity','previous_velocity'),
                ('previous_material_metric','previous_material_metric'),('previous_history','previous_history'),
                ('released','released'),('active_contact','active_contact')]:setattr(p,a,state[b].copy())
        p.material_dt=float(state['material_dt_s']);p.previous_barrier_energy=float(state['previous_barrier_energy_n_mm'])
        p.rebuild_contact();p.set_free();p.friction=p.ipc.FrictionPotential(p.friction_velocity_epsilon*p.material_dt)
        p.set_primary_increment(state['positions'],state['increment']);p.insertion_events=[]
        rec=p.acceptance(state['positions'],float(state['depth_mm']),0,0)
        for k,v in recorded.items():np.testing.assert_allclose(rec[k],v,rtol=1e-8,atol=1e-8)
        checks=[];events=[]
        for ep in sorted(directory.glob(f'finite_increment_{index:05d}_*.npz')):
            with np.load(ep,allow_pickle=False) as z:fields={k:z[k].copy() for k in z.files}
            mask=fields['removed'];ids=np.flatnonzero(mask)
            base=FrozenVirtualCrack(p,fields['pre_anchor'],fields['frozen_metric'],fields['frozen_history'],
                fields['previous_released'],np.zeros_like(mask),p.ffm_epsilon)
            before=fields['before_positions'];after=fields['after_positions']
            post=FrozenVirtualCrack(p,before,fields['frozen_metric'],fields['frozen_history'],fields['previous_released'],mask,p.ffm_epsilon)
            _,r0=stored_equilibrium(base,before);_,r1=stored_equilibrium(post,after)
            u0,u1=base.stored(before),post.stored(after)
            work=virtual_work(post,before,after,start+seconds)
            area=float(p.seam_area[ids].sum());required=p.toughness*area
            d9=work['work_n_mm'];uncertainty=work['empirical_uncertainty_n_mm']
            if d9 is None:raise ValueError('Recovered virtual work is unresolved')
            decision=interval_decision(u0-u1,required,d9,uncertainty,work['converged'])
            if decision['status']!='accepted':raise ValueError('Recovered insertion has no resolved energy support')
            check=dict(nodes=p.seam[ids].tolist(),seam_indices=ids.tolist(),accepted=True,
                reference_tributary_area_mm2=area,stress_min_mpa=float(fields['stress'][ids].min()),
                virtual_stored_before_n_mm=u0,virtual_stored_after_n_mm=u1,virtual_friction_work_n_mm=d9,
                required_fracture_work_n_mm=required,friction_quadrature_uncertainty_n_mm=uncertainty,friction_integration=work,energy_interval=decision,
                energy_margin_n_mm=decision['margin_lower_n_mm'],
                virtual_pre_residual_n=r0,virtual_post_residual_n=r1,
                evidence_file=ep.name,evidence_sha256=hashlib.sha256(ep.read_bytes()).hexdigest(),recovered=True)
            audit_evidence(p,state,check,fields,start+seconds)
            checks.append(check);events.append(dict(nodes=check['nodes'],reference_y_mm=p.rest[p.seam[ids],1].tolist(),
                outgoing_traction_mpa=fields['physical_traction'][ids].tolist(),
                overshoot_mpa=(fields['physical_traction'][ids]-p.strength).tolist()))
        expected=set(p.seam[state['released'] & ~state['previous_released']].tolist())
        if expected!=set(n for c in checks for n in c['nodes']):raise ValueError('Unaccounted physical insertion')
        rec.update(finite_increment_checks=checks,insertion_events=events,iterations=None,
            friction_outer_iterations=None,state_file=path.name,state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            recovered_physical_record=True,accepted_chord_ccd_safe=True)
        result['steps'].append(rec)
    result.update(status='interrupted_recovered_prefix',error=stopped['reason'],audits=[],thread_runtime=runtime,
        accepted_prefix_audited=False,source_unchanged=source==driver.hashes(),elapsed_s=time.monotonic()-start,
        recovery=dict(original_run_elapsed_s_unavailable=True,nonlinear_iterations_unavailable=True,
            rejected_virtual_checks_unavailable=True,loading_started=False,logged_appended_states=len(logged),
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    driver.report(directory,result)
    proof=inspect(directory,max(1.,seconds-(time.monotonic()-start)),
        ipc_threads=runtime['requested']['ipc_threads'],blas_threads=runtime['requested']['blas_threads'])
    if not proof['passed']:raise ValueError('Recovered path failed independent chain audit: '+str(proof.get('error')))
    result['audits']=proof['audits'];result['accepted_prefix_audited']=True
    result['elapsed_s']=time.monotonic()-start
    driver.report(directory,result)
    write_json(directory/'recovery_audit.json',proof)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--log',required=True,type=Path)
    parser.add_argument('--seconds',required=True,type=float);add_thread_arguments(parser);args=parser.parse_args()
    result=recover(args.directory,args.log,args.seconds,**thread_arguments(args))
    write_json(args.directory/'result.json',result)
    print(json.dumps(result['summary']))


if __name__=='__main__':main()
