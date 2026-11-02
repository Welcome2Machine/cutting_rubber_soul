"""Actual coupled branches from a complete audited parent, never extrapolated states.

The old additional time quota is an observation here, not an acceptance gate.
All original physical gates and actual material/topology commits remain active.
Each branch has a fresh problem, predictors and witness store. No branch is a
production continuation or a global error bound.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from numerical_runtime import numerical_runtime
from step_control import failure_kind
from . import run_coupled_growth as growth, run_fracture_route as route
from .continue_coupled_growth import restore
from .audit_coupled_contact import independent_balance
from .audit_precrack_entry import strict_contacts
from .face_witness import WitnessStore, accepted_fields, array_hash, native
from .frozen_scene import digest
from .run_coupled_contact import rejected_fields
from .event_resolution import measure


def read_state(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def signature(p, s):
    failed = s['active_faces'] & np.all(s['history'] >= 2*p.toughness/p.strength, axis=1)
    return dict(mirror_active=np.flatnonzero(s['active_contact']).tolist(),
                active_faces=np.flatnonzero(s['active_faces']).tolist(),
                failed_faces=np.flatnonzero(failed).tolist(),
                original_interior_contacts=strict_contacts(p, s['positions'], p.connected_length(failed)))


def branch(config, state, predictors, parent_row, out, h, pieces, deadline, resolution=None):
    """Commit each real substep independently; keep rejected trials and exact rollback."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    p = growth.problem(config); x = restore(p, state, predictors)
    parent_hash = array_hash(accepted_fields(p, x))
    np.savez_compressed(out/'parent_state.npz', **state)
    store = WitnessStore(out, dict(config=config, source_hashes=growth.sources(),
                                  execution_identity=route.driver.execution_identity()))
    start = time.monotonic(); depth0 = float(state['depth_mm']); depth = depth0
    previous = state; previous_row = parent_row; rows = []; audits = []; failures = []
    initial_signature = signature(p, state)
    for i in range(pieces):
        target = depth0 + p.loading_rate*h*(i+1)/pieces
        before = array_hash(accepted_fields(p, x)); released = p.released.copy()
        parent_file = 'parent_state.npz' if not rows else rows[-1]['state_file']
        store.begin(p, x, dict(parent=dict(file=parent_file, sha256=digest(out/parent_file)),
                              from_depth_mm=depth, target_depth_mm=target,
                              physical_dt_s=(target-depth)/p.loading_rate))
        p.global_compute_deadline = deadline
        try:
            y, row = p.step(x, depth, target, min(deadline, time.monotonic()+45))
            store.accepted(row)
            row.update(time_s=(target-p.start)/p.loading_rate, dt_s=p.material_dt)
            route.observables(p, y, row)
            s = route.snapshot(p, y, released, target)
            for key in ('positions', 'velocity', 'material_metric', 'history', 'released', 'active_faces'):
                np.testing.assert_array_equal(s['previous_'+key], previous[key])
            row['event_signature'] = signature(p, s)
            if resolution is not None:
                old_a=(previous['velocity'][:p.n]-previous['previous_velocity'][:p.n])/float(previous['material_dt_s'])
                row['observed_time_budget']=measure(p.mass,s['velocity'][:p.n],previous['velocity'][:p.n],old_a,p.material_dt,
                    [row[k] for k in ('reaction_normal_fz_n','reaction_friction_fz_n')],
                    [previous_row[k] for k in ('reaction_normal_fz_n','reaction_friction_fz_n')],row['energy'],resolution)
            name=f'state_{i:05d}.npz';np.savez_compressed(out/name,**s)
            row.update(state_file=name,state_sha256=digest(out/name));rows.append(native(row))
            # A fresh object, not the live branch, independently checks each chord,
            # original force equation, Maxwell/cohesive histories and energy.
            fresh=growth.problem(config)
            checks=route.restore_audit(fresh,s,row)
            balance=independent_balance(fresh,s)
            audits.append(dict(checks, **balance, state_sha256=digest(out/name)))
            x=y;depth=target;previous=s;previous_row=row
        except (RuntimeError,ValueError,TimeoutError,AssertionError,FloatingPointError) as exc:
            rolled_back=array_hash(accepted_fields(p,x))==before
            fields,energy=rejected_fields(p,getattr(p,'failed_positions',x))
            np.savez_compressed(out/'rejected_trial.npz',**fields)
            witness=store.rejected(p,x,exc,failure_kind(exc,deadline))
            failures.append(dict(error=str(exc),failure_kind=failure_kind(exc,deadline),rollback_verified=rolled_back,
                target_depth_mm=target,dt_s=(target-depth)/p.loading_rate,energy=energy,witness=witness,
                trial_sha256=digest(out/'rejected_trial.npz')))
            if not rolled_back:raise RuntimeError('Branch failure committed history') from exc
            break
    result=dict(scope='conditional actual step-doubling diagnostic; no production suffix',
        config=config,source_hashes=growth.sources(),parent_hash=parent_hash,h_s=h,pieces=pieces,
        initial_depth_mm=depth0,initial_signature=initial_signature,steps=rows,failures=failures,
        completed=len(rows)==pieces,elapsed_s=time.monotonic()-start,independent_audit=audits)
    (out/'result.json').write_text(json.dumps(native(result),indent=2))
    return result


def compare(a, b):
    """No interpolation or event alignment; endpoints must actually coincide."""
    da,db=Path(a),Path(b)
    ra=json.loads((da/'result.json').read_text());rb=json.loads((db/'result.json').read_text())
    if not ra['completed'] or not rb['completed']:return dict(comparable=False,reason='Incomplete actual branch')
    if ra['parent_hash']!=rb['parent_hash'] or ra['config']!=rb['config'] or ra['h_s']!=rb['h_s']:
        raise ValueError('Different branch parent, equation or physical endpoint')
    sa=read_state(da/ra['steps'][-1]['state_file']);sb=read_state(db/rb['steps'][-1]['state_file'])
    np.testing.assert_array_equal(sa['depth_mm'],sb['depth_mm'])
    m=sa['mass'];n=len(m);dv=sa['velocity'][:n]-sb['velocity'][:n]
    norm=lambda v:float(np.sqrt(m@np.sum(v*v,axis=1)))
    impulses=lambda r:np.sum([[s['dt_s']*s[k] for k in ('reaction_normal_fz_n','reaction_friction_fz_n')] for s in r['steps']],axis=0)
    return dict(comparable=True,velocity_absolute_sqrt_nmm=norm(dv),velocity_scale_sqrt_nmm=norm(sb['velocity'][:n]),
        velocity_relative=norm(dv)/max(norm(sb['velocity'][:n]),1e-30),position_max_mm=float(np.max(np.abs(sa['positions']-sb['positions']))),
        force_difference_n={k:ra['steps'][-1][k]-rb['steps'][-1][k] for k in ('reaction_fz_n','reaction_normal_fz_n','reaction_friction_fz_n')},
        impulse_components_n_s=[impulses(ra).tolist(),impulses(rb).tolist()],
        history_max_difference={k:float(np.max(np.abs(sa[k].astype(float)-sb[k].astype(float)))) for k in ('material_metric','history','active_faces','active_contact','released')},
        absolute_numerical_n_mm=[sum(s['absolute_numerical_including_equation_work_n_mm'] for s in r['independent_audit']) for r in (ra,rb)],
        elapsed_s=[ra['elapsed_s'],rb['elapsed_s']],strict_error_bound=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());args.out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.plan,args.out/'plan.json');answers=[]
    deadline=time.monotonic()+min(plan['seconds'],float(os.environ.get('SOFTCUT_WORKER_DEADLINE_EPOCH',time.time()+plan['seconds']))-time.time()-5)
    with numerical_runtime(1,1,True) as runtime:
        for case in plan['cases']:
            directory=Path(case['directory']);raw=json.loads((directory/'result.json').read_text())
            proof=json.loads(Path(case['audit']).read_text())
            if not proof['passed'] or proof['result_sha256']!=digest(directory/'result.json') or proof['source_hashes']!=raw['source_hashes']:
                raise ValueError('Unaudited parent result')
            # Changes are allowed only in explicit numerical policy/observation files.
            current=growth.sources();changed={k for k in set(current)|set(raw['source_hashes']) if current.get(k)!=raw['source_hashes'].get(k)}
            if changed-set(plan.get('allowed_source_changes',[])):raise ValueError('Unregistered source change: '+str(changed))
            i=case['index'];predictors=[]
            for row in raw['steps'][i-1:i+1]:
                file=directory/row['state_file']
                if digest(file)!=row['state_sha256']:raise ValueError('Changed parent/predictor')
                state=read_state(file);predictors.append((row['depth_mm'],state['positions']))
            item=dict(case=case,parent_state_sha256=row['state_sha256'],source_changes=sorted(changed),branches=[])
            for pieces in case.get('pieces',[1,2]):
                out=args.out/case['name']/str(pieces)
                q=branch(raw['config'],state,predictors,row,out,case['h_s'],pieces,deadline,plan.get('resolution'))
                item['branches'].append(dict(pieces=pieces,completed=q['completed'],elapsed_s=q['elapsed_s']))
            item['comparison']=compare(args.out/case['name']/'1',args.out/case['name']/'2')
            answers.append(item)
            (args.out/'summary.json').write_text(json.dumps(native(dict(cases=answers,runtime=runtime,plan_sha256=digest(args.plan))),indent=2))
            print(json.dumps(native(item)),flush=True)


if __name__=='__main__':main()
